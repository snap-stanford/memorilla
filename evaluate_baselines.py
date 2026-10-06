"""Evaluate the text-only baselines: closed-book, retrieval (RAG-k) and full context.

Documents go into the user turn as text (most similar first for RAG, stored order for full context), then the question
and suffix; over-long prompts are truncated from the front. Outputs and scoring match evaluate.py.
"""

import argparse
from collections import defaultdict
from dataclasses import dataclass
import os
from pathlib import Path
from typing import TYPE_CHECKING

from datasets import Dataset
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from memorilla.benchmarks import (
    RETRIEVAL_CONTEXT_LENGTH,
    YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS,
    Benchmark,
    get_benchmarks,
)
from memorilla.data import (
    COLLECTION_COLUMN,
    DOCUMENTS_COLUMN,
    EVAL_COLUMNS,
    QUESTION_COLUMN,
    apply_chat_template,
    ensure_data,
    load_split,
)
from memorilla.embeddings import EmbeddingStore, ensure_embeddings, load_split_with_embeddings
from memorilla.metrics import SCORER_GPU_MEMORY_UTILIZATION, check_scoring_inputs, clean_generation, score_predictions
from memorilla.paths import DATA_DIR, DATA_REPO, DEFAULT_DECODER, DEFAULT_ENCODER, MODEL_CACHE_DIR, RESULTS_DIR, resolve
from memorilla.utils import DEFAULT_SEED, local_model_path, release_gpu_memory, seed_everything, write_json

if TYPE_CHECKING:
    from vllm import LLM

CLOSED_BOOK = "closed_book"
RAG = "rag"
FULL_CONTEXT = "full_context"
METHODS = (CLOSED_BOOK, RAG, FULL_CONTEXT)
GENERATION_RESERVE = 32
DOCUMENT_SEPARATOR = "\n\n"
CONTEXT_FILE = "context.json"
PERCENTILES = (50, 90, 95, 99)
VLLM_ENGINE_VARIABLE = "VLLM_USE_V1"
VLLM_MULTIPROC_VARIABLE = "VLLM_WORKER_MULTIPROC_METHOD"
BASELINE_GPU_MEMORY_UTILIZATION = 0.85

ContextSummary = dict[str, float | int]
EvaluationJob = tuple[Benchmark, Dataset, list[str], Path]


@dataclass(frozen=True)
class PromptSet:
    """The tokenised baseline prompts of one benchmark.

    Attributes:
        benchmark: The benchmark.
        dataset: Its evaluated split.
        prompts: Prompt token ids per row, truncated from the front to the prompt budget.
        context: Fit-rate summary, written to ``context.json``.
        max_model_len: Context length of the engine that serves the prompts.
        yarn_factor: YaRN RoPE scaling factor of that engine, or None.
    """

    benchmark: Benchmark
    dataset: Dataset
    prompts: list[list[int]]
    context: ContextSummary
    max_model_len: int
    yarn_factor: float | None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--benchmark", type=str, required=True, help="Registered benchmark name, or 'all'.")
    parser.add_argument("--method", type=str, choices=METHODS, required=True, help="Baseline to run.")
    parser.add_argument("--top_k", type=int, default=5, help="Documents retrieved per question with --method rag.")
    parser.add_argument("--decoder", type=str, default=DEFAULT_DECODER, help="Decoder (Hub id or local path).")
    parser.add_argument(
        "--encoder", type=str, default=DEFAULT_ENCODER, help="Encoder of the stored embeddings and of choice scoring."
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=BASELINE_GPU_MEMORY_UTILIZATION,
        help="Fraction of GPU memory for the decoder.",
    )
    parser.add_argument(
        "--scorer_gpu_memory_utilization",
        type=float,
        default=SCORER_GPU_MEMORY_UTILIZATION,
        help="Fraction of GPU memory for the choice-scoring encoder.",
    )
    parser.add_argument("--max_num_seqs", type=int, default=None, help="vLLM concurrency cap for long contexts.")
    parser.add_argument("--data_repo", type=str, default=DATA_REPO, help="Hub dataset repo or local directory.")
    parser.add_argument("--data_dir", type=str, default=str(DATA_DIR), help="Local mirror of a Hub data repo.")
    parser.add_argument("--cache_dir", type=str, default=MODEL_CACHE_DIR, help="Model cache directory.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(RESULTS_DIR / "baselines"),
        help="Results directory (one folder per method and benchmark).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    return parser.parse_args()


def method_label(method: str, top_k: int) -> str:
    """Name of the output folder for a method.

    Args:
        method: One of ``METHODS``.
        top_k: Retrieval depth.

    Returns:
        ``closed_book``, ``rag_top<k>`` or ``full_context``.
    """
    return f"{RAG}_top{top_k}" if method == RAG else method


def context_window(benchmark: Benchmark, method: str) -> tuple[int, float | None]:
    """Context length and YaRN factor of the decoder for a benchmark and method.

    Args:
        benchmark: The benchmark.
        method: One of ``METHODS``.

    Returns:
        ``(max_model_len, yarn_factor)``.
    """
    if method == FULL_CONTEXT:
        return benchmark.baseline.full_context_length, benchmark.baseline.yarn_factor
    return RETRIEVAL_CONTEXT_LENGTH, None


def retrieve(dataset: Dataset, store: EmbeddingStore, top_k: int) -> list[list[str]]:
    """Select each row's ``top_k`` documents by cosine similarity to its question, most similar first.

    Args:
        dataset: Split with ``collection_id`` and ``documents``.
        store: Its embedding store.
        top_k: Documents kept per row.

    Returns:
        The retrieved documents per row.
    """
    retrieved = []
    for row, item in enumerate(dataset.select_columns([COLLECTION_COLUMN, DOCUMENTS_COLUMN])):
        documents = F.normalize(store.documents(item[COLLECTION_COLUMN]).float(), dim=1)
        question = F.normalize(store.question(row).float().unsqueeze(0), dim=1)
        similarities = (documents @ question.T).squeeze(1)
        order = similarities.argsort(descending=True)[: min(top_k, len(item[DOCUMENTS_COLUMN]))].tolist()
        retrieved.append([item[DOCUMENTS_COLUMN][index] for index in order])
    return retrieved


def user_message(documents: list[str], question: str, suffix: str) -> str:
    """Render the user turn.

    Args:
        documents: Documents placed before the question (may be empty).
        question: The question.
        suffix: Text appended after the question.

    Returns:
        The user message.
    """
    if documents:
        return DOCUMENT_SEPARATOR.join(documents) + DOCUMENT_SEPARATOR + question + suffix
    return question + suffix


def build_prompts(
    tokenizer: PreTrainedTokenizerBase,
    benchmark: Benchmark,
    dataset: Dataset,
    documents: list[list[str]],
    max_input_tokens: int,
) -> tuple[list[list[int]], list[int]]:
    """Tokenise the chat prompts, truncating from the front to ``max_input_tokens``.

    Args:
        tokenizer: Decoder tokenizer.
        benchmark: The benchmark.
        dataset: The split.
        documents: Documents per row.
        max_input_tokens: Prompt token budget.

    Returns:
        The (possibly truncated) prompt ids and the untruncated prompt lengths.
    """
    prompts, lengths = [], []
    for question, docs in zip(dataset[QUESTION_COLUMN], documents, strict=True):
        messages = [
            {"role": "system", "content": benchmark.baseline.system_prompt},
            {"role": "user", "content": user_message(docs, str(question), benchmark.baseline.suffix)},
        ]
        ids = tokenizer.encode(
            apply_chat_template(tokenizer, messages, add_generation_prompt=True), add_special_tokens=False
        )
        lengths.append(len(ids))
        prompts.append(ids[-max_input_tokens:])
    return prompts, lengths


def context_summary(lengths: list[int], max_model_len: int, max_input_tokens: int) -> ContextSummary:
    """Describe how much of the intended context reached the decoder.

    Args:
        lengths: Untruncated prompt lengths.
        max_model_len: Decoder context length.
        max_input_tokens: Prompt token budget.

    Returns:
        Fit rate (percent of prompts kept whole), truncation counts and prompt-length statistics.
    """
    array = np.asarray(lengths, dtype=np.int64)
    truncated = array > max_input_tokens
    summary = {
        "max_model_len": max_model_len,
        "max_input_tokens": max_input_tokens,
        "num_prompts": int(array.size),
        "num_truncated": int(truncated.sum()),
        "fit_rate": round(100 * float((~truncated).mean()), 2),
        "mean_prompt_tokens": round(float(array.mean()), 1),
        "max_prompt_tokens": int(array.max()),
        "tokens_dropped": int(np.clip(array - max_input_tokens, 0, None).sum()),
    }
    summary.update({f"p{q}_prompt_tokens": float(np.percentile(array, q)) for q in PERCENTILES})
    return summary


def prepare_prompts(benchmark: Benchmark, args: argparse.Namespace, tokenizer: PreTrainedTokenizerBase) -> PromptSet:
    """Load a benchmark's split, select its documents and tokenise the baseline prompts.

    Args:
        benchmark: The benchmark.
        args: Parsed command-line arguments.
        tokenizer: Decoder tokenizer.

    Returns:
        The prompts with their split, fit-rate summary and engine settings.
    """
    root = ensure_data(benchmark.config, [benchmark.split], repo=args.data_repo, data_dir=args.data_dir)
    columns = EVAL_COLUMNS if args.method == CLOSED_BOOK else [*EVAL_COLUMNS, DOCUMENTS_COLUMN]
    if args.method == RAG:
        ensure_embeddings(
            benchmark.config,
            [benchmark.split],
            repo=args.data_repo,
            data_dir=args.data_dir,
            encoder=args.encoder,
            cache_dir=args.cache_dir,
        )
        dataset, store = load_split_with_embeddings(benchmark.config, benchmark.split, root, columns, args.encoder)
        documents = retrieve(dataset, store, args.top_k)
    else:
        dataset = load_split(benchmark.config, benchmark.split, root, columns)
        if args.method == FULL_CONTEXT:
            documents = [list(docs) for docs in dataset[DOCUMENTS_COLUMN]]
        else:
            documents = [[] for _ in range(len(dataset))]
    check_scoring_inputs(benchmark, dataset)

    max_model_len, yarn_factor = context_window(benchmark, args.method)
    max_input_tokens = max(max_model_len - benchmark.baseline.max_new_tokens - GENERATION_RESERVE, 1)
    prompts, lengths = build_prompts(tokenizer, benchmark, dataset, documents, max_input_tokens)
    context = context_summary(lengths, max_model_len, max_input_tokens)
    return PromptSet(benchmark, dataset, prompts, context, max_model_len, yarn_factor)


def start_engine(args: argparse.Namespace, max_model_len: int, yarn_factor: float | None) -> "LLM":
    """Start a text-only vLLM engine for the decoder.

    Args:
        args: Parsed command-line arguments.
        max_model_len: Context length.
        yarn_factor: YaRN RoPE scaling factor, or None for the native RoPE.

    Returns:
        The engine.
    """
    from vllm import LLM

    options = {}
    if yarn_factor is not None:
        options["hf_overrides"] = {
            "rope_scaling": {
                "rope_type": "yarn",
                "factor": yarn_factor,
                "original_max_position_embeddings": YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS,
            },
            "max_position_embeddings": max_model_len,
        }
    if args.max_num_seqs:
        options["max_num_seqs"] = args.max_num_seqs
    return LLM(
        model=local_model_path(args.decoder, args.cache_dir),
        download_dir=args.cache_dir,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        **options,
    )


def generate_baseline(engine: "LLM", prompt_sets: list[PromptSet], output_dir: Path) -> list[EvaluationJob]:
    """Generate greedily for every prompt set served by one engine and write their ``context.json``.

    Args:
        engine: Text-only engine started for the sets' context window.
        prompt_sets: Prompt sets that share that window.
        output_dir: Method output directory; each benchmark writes to ``output_dir/<benchmark>``.

    Returns:
        ``(benchmark, dataset, predictions, output_dir)`` per prompt set, for ``score_predictions``.
    """
    from vllm import SamplingParams

    jobs = []
    for prompt_set in prompt_sets:
        benchmark = prompt_set.benchmark
        sampling = SamplingParams(
            max_tokens=benchmark.baseline.max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            truncate_prompt_tokens=prompt_set.context["max_input_tokens"],
        )
        requests = [{"prompt_token_ids": ids} for ids in prompt_set.prompts]
        outputs = engine.generate(requests, sampling_params=sampling)
        predictions = [clean_generation(output.outputs[0].text) for output in outputs]
        jobs.append((benchmark, prompt_set.dataset, predictions, output_dir / benchmark.name))
        write_json(output_dir / benchmark.name / CONTEXT_FILE, prompt_set.context)
    return jobs


def main(args: argparse.Namespace) -> None:
    """Generate and score the baseline on every requested benchmark.

    Benchmarks that need the same context window share one engine. vLLM records the engine version it picks for the
    decoder in ``VLLM_USE_V1``; the variable is restored before scoring so that the embedding engine used for
    ``choice`` scoring picks its own. Engine processes are spawned rather than forked: retrieval runs PyTorch CPU
    kernels in this process first, and a forked engine process can deadlock on their thread pool.

    Args:
        args: Parsed command-line arguments.
    """
    os.environ.setdefault(VLLM_MULTIPROC_VARIABLE, "spawn")
    seed_everything(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.decoder, cache_dir=args.cache_dir, trust_remote_code=True)
    output_dir = resolve(args.output_dir) / method_label(args.method, args.top_k)

    windows: dict[tuple[int, float | None], list[PromptSet]] = defaultdict(list)
    for benchmark in get_benchmarks(args.benchmark):
        prompt_set = prepare_prompts(benchmark, args, tokenizer)
        windows[(prompt_set.max_model_len, prompt_set.yarn_factor)].append(prompt_set)

    engine_choice = os.environ.get(VLLM_ENGINE_VARIABLE)
    jobs = []
    for (max_model_len, yarn_factor), prompt_sets in windows.items():
        engine = start_engine(args, max_model_len, yarn_factor)
        jobs += generate_baseline(engine, prompt_sets, output_dir)
        del engine
        release_gpu_memory()
    if engine_choice is None:
        os.environ.pop(VLLM_ENGINE_VARIABLE, None)

    score_predictions(
        jobs, encoder=args.encoder, cache_dir=args.cache_dir, gpu_memory_utilization=args.scorer_gpu_memory_utilization
    )


if __name__ == "__main__":
    main(parse_args())
