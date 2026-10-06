"""Evaluate a trained memory module: generate with a frozen vLLM decoder, then score.

Registered benchmarks follow their evaluation protocol (--benchmark <name|all>); any other dataset split can be
evaluated with --config <name> --split <split> --scoring <scoring>.
"""

import argparse
from dataclasses import replace

from datasets import Dataset
from tqdm import tqdm
from transformers import AutoConfig

from memorilla.benchmarks import (
    ACCURACY,
    ACCURACY_HARD,
    CHOICE,
    GENERATION,
    METRICS_BY_SCORING,
    SCORINGS,
    Benchmark,
    get_benchmarks,
)
from memorilla.data import (
    DEFAULT_SYSTEM_PROMPT,
    EVAL_COLUMNS,
    HARD_COLUMN,
    TEST_SPLIT,
    MemoryCollator,
    MemoryDataset,
    ensure_data,
)
from memorilla.embeddings import EmbeddingStore, ensure_embeddings, load_split_with_embeddings
from memorilla.memory import MemoryModule
from memorilla.metrics import SCORER_GPU_MEMORY_UTILIZATION, check_scoring_inputs, clean_generation, score_predictions
from memorilla.paths import DATA_DIR, DATA_REPO, DEFAULT_DECODER, DEFAULT_ENCODER, MODEL_CACHE_DIR, RESULTS_DIR, resolve
from memorilla.utils import DEFAULT_SEED, read_recipe, release_gpu_memory, seed_everything
from memorilla.vllm import DECODER_GPU_MEMORY_UTILIZATION, MemoryVLLM

CUSTOM_MAX_LENGTH = 2048
CUSTOM_MAX_NEW_TOKENS = 128


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--benchmark", type=str, help="Registered benchmark name, or 'all'.")
    target.add_argument("--config", type=str, help="Dataset config to evaluate instead of a registered benchmark.")
    parser.add_argument("--split", type=str, default=TEST_SPLIT, help="Split to evaluate.")
    parser.add_argument(
        "--scoring", type=str, choices=SCORINGS, default=GENERATION, help="Scoring method for --config."
    )

    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint directory or memory.pt.")
    parser.add_argument("--decoder", type=str, default=DEFAULT_DECODER, help="Decoder the checkpoint was trained with.")
    parser.add_argument(
        "--encoder", type=str, default=DEFAULT_ENCODER, help="Encoder of the stored embeddings and of choice scoring."
    )
    prompt = parser.add_mutually_exclusive_group()
    prompt.add_argument("--system_prompt", type=str, default=None, help="Override the system prompt.")
    prompt.add_argument(
        "--system_prompt_from",
        type=str,
        default=None,
        help="Use the system prompt a training recipe (YAML) trains with.",
    )
    parser.add_argument("--max_length", type=int, default=None, help="Override the prompt token limit.")
    parser.add_argument("--max_new_tokens", type=int, default=None, help="Override the generation budget.")

    parser.add_argument("--batch_size", type=int, default=32, help="Rows per generation call.")
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=DECODER_GPU_MEMORY_UTILIZATION,
        help="Fraction of GPU memory for the decoder.",
    )
    parser.add_argument(
        "--scorer_gpu_memory_utilization",
        type=float,
        default=SCORER_GPU_MEMORY_UTILIZATION,
        help="Fraction of GPU memory for the choice-scoring encoder.",
    )
    parser.add_argument("--data_repo", type=str, default=DATA_REPO, help="Hub dataset repo or local directory.")
    parser.add_argument("--data_dir", type=str, default=str(DATA_DIR), help="Local mirror of a Hub data repo.")
    parser.add_argument("--cache_dir", type=str, default=MODEL_CACHE_DIR, help="Model cache directory.")
    parser.add_argument(
        "--output_dir", type=str, default=str(RESULTS_DIR), help="Results directory (one folder per benchmark)."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    return parser.parse_args()


def resolve_targets(args: argparse.Namespace) -> list[Benchmark]:
    """Turn the arguments into the list of benchmarks to evaluate, with overrides applied.

    Args:
        args: Parsed command-line arguments.

    Returns:
        The benchmarks.
    """
    if args.benchmark:
        benchmarks = get_benchmarks(args.benchmark)
    else:
        benchmarks = [
            Benchmark(
                name=args.config,
                config=args.config,
                split=TEST_SPLIT,
                system_prompt=DEFAULT_SYSTEM_PROMPT,
                max_length=CUSTOM_MAX_LENGTH,
                max_new_tokens=CUSTOM_MAX_NEW_TOKENS,
                scoring=args.scoring,
                metrics=METRICS_BY_SCORING[args.scoring],
            )
        ]
    if args.split != TEST_SPLIT:
        benchmarks = [
            replace(benchmark, name=f"{benchmark.name}-{args.split}", split=args.split) for benchmark in benchmarks
        ]

    system_prompt = args.system_prompt
    if args.system_prompt_from:
        system_prompt = read_recipe(args.system_prompt_from).get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    overrides = {
        "system_prompt": system_prompt,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
    }
    overrides = {key: value for key, value in overrides.items() if value is not None}
    return [replace(benchmark, **overrides) for benchmark in benchmarks]


def load_target(benchmark: Benchmark, args: argparse.Namespace) -> tuple[Benchmark, Dataset, EmbeddingStore]:
    """Fetch a benchmark's split and embeddings and check that the split can be scored.

    Args:
        benchmark: The benchmark.
        args: Parsed command-line arguments.

    Returns:
        The benchmark (choice scoring reports ``accuracy_hard`` exactly when the split flags hard rows), its split and
        its embedding store.
    """
    root = ensure_data(benchmark.config, [benchmark.split], repo=args.data_repo, data_dir=args.data_dir)
    ensure_embeddings(
        benchmark.config,
        [benchmark.split],
        repo=args.data_repo,
        data_dir=args.data_dir,
        encoder=args.encoder,
        cache_dir=args.cache_dir,
    )
    dataset, store = load_split_with_embeddings(benchmark.config, benchmark.split, root, EVAL_COLUMNS, args.encoder)
    if benchmark.scoring == CHOICE:
        has_hard = HARD_COLUMN in dataset.column_names and any(dataset[HARD_COLUMN])
        benchmark = replace(benchmark, metrics=(ACCURACY, ACCURACY_HARD) if has_hard else (ACCURACY,))
    check_scoring_inputs(benchmark, dataset)
    return benchmark, dataset, store


def generate(
    model: MemoryVLLM,
    benchmark: Benchmark,
    dataset: Dataset,
    store: EmbeddingStore,
    batch_size: int,
) -> list[str]:
    """Generate one answer per row of a split.

    Args:
        model: The memory-conditioned decoder.
        benchmark: Benchmark that sets the prompt and generation budget.
        dataset: The split.
        store: Its embedding store.
        batch_size: Rows per generation call.

    Returns:
        The cleaned generations in row order.
    """
    collator = MemoryCollator(
        model.tokenizer,
        {(benchmark.config, benchmark.split): store},
        num_memories=model.memory.num_memories,
        max_length=benchmark.max_length,
        system_prompt=benchmark.system_prompt,
        suffix=benchmark.suffix,
        train=False,
    )
    items = MemoryDataset([(benchmark.config, benchmark.split, dataset)])
    predictions = []
    for start in tqdm(range(0, len(items), batch_size), desc=benchmark.name):
        batch = collator([items[index] for index in range(start, min(start + batch_size, len(items)))])
        texts = model.generate(**batch, max_new_tokens=benchmark.max_new_tokens)
        predictions.extend(clean_generation(text) for text in texts)
    return predictions


def main(args: argparse.Namespace) -> None:
    """Generate and score every requested target.

    Args:
        args: Parsed command-line arguments.

    Raises:
        ValueError: If the checkpoint's output width does not match the decoder's hidden size.
    """
    seed_everything(args.seed)
    targets = [load_target(benchmark, args) for benchmark in resolve_targets(args)]

    memory = MemoryModule.from_pretrained(resolve(args.checkpoint))
    hidden_size = AutoConfig.from_pretrained(args.decoder, cache_dir=args.cache_dir).hidden_size
    if memory.config.output_dim != hidden_size:
        raise ValueError(
            f"The checkpoint produces {memory.config.output_dim}-dimensional memory tokens but {args.decoder} has "
            f"hidden size {hidden_size}; pass the decoder the checkpoint was trained with."
        )

    model = MemoryVLLM(
        args.decoder,
        memory,
        max_model_len=max(benchmark.max_length for benchmark, _, _ in targets),
        gpu_memory_utilization=args.gpu_memory_utilization,
        cache_dir=args.cache_dir,
    )
    output_dir = resolve(args.output_dir)
    jobs = []
    for benchmark, dataset, store in targets:
        predictions = generate(model, benchmark, dataset, store, args.batch_size)
        jobs.append((benchmark, dataset, predictions, output_dir / benchmark.name))
    del model, memory
    release_gpu_memory()

    score_predictions(
        jobs, encoder=args.encoder, cache_dir=args.cache_dir, gpu_memory_utilization=args.scorer_gpu_memory_utilization
    )


if __name__ == "__main__":
    main(parse_args())
