"""Latency and GPU memory of the memory module: document and question embeddings in, memory tokens out.

Only ``MemoryModule`` is timed, on one device, for collections of N document embeddings and a batch of questions.
The encoder and the decoder are not part of the measurement. The cost does not depend on the embedding values, so
random unit-norm vectors (like encoder outputs) stand in for encoded documents and questions, and the module is
randomly initialised with the released architecture unless ``--checkpoint`` is given.

Sweep axes:

- ``--cache``, where and in what form the collection is kept between queries:
  ``gpu`` (embeddings on the GPU, a query is ``MemoryModule.forward``), ``host`` (embeddings in pinned host memory,
  copied to the GPU by every query), ``kv`` (keys and values of the cross-attention layers computed at write time,
  ``CachedMemory``) and ``kv_fp8`` (the same cache in float8, ``Fp8CachedMemory``, approximate).
- ``--modes``, how a query is executed: ``eager``, ``cuda_graph`` (captured once, replayed per query) or
  ``compile`` (``torch.compile``; the compile time lands in ``first_call``). ``host`` runs eager only.
- ``--shared``: 0 gives every question of a batch its own collection; 1 answers the whole batch from one collection.
- ``--splits``: chunks of the weighted sum over the documents (``kv`` only; 1 is a single product).
- ``--batch_sizes`` and ``--ns``.

Each call is timed with CUDA events after the L2 cache is overwritten, so a small collection is not served from a
cache that a real query arriving after other work would not find. A cell runs ``--warmup`` untimed calls, then
``--reps`` timed ones. Each cell also records the bytes of the stored collection or cache, the bandwidth a read
achieves if it reads them once (bytes / median time), PyTorch's peak allocation from compilation or capture through
the reads, above the allocation before them, and an output check for batch x N up to ``--check_max``. The check takes the output that the timed call
returns (after compilation or graph capture) and compares it with the default forward pass (``MemoryModule.forward``
in the run's dtype, eager, as ``MemoryVLLM`` serves it). A variant is within tolerance when it differs from the
default by at most ``--tolerance_factor`` times the default's own error against a float32 forward pass.

Results go to ``<out>/memory_module.jsonl``, one record per cell; a rerun skips the cells already there and retries
cells that failed. After a cell runs out of memory, the larger N of the same settings are skipped.
``python -m profiling.summarize <out>`` writes the tables.
"""

import argparse
from collections.abc import Callable, Iterable
import copy
from functools import partial
import gc
import itertools
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn.functional as F

from memorilla.memory import DEFAULT_NUM_CROSS_ATTN_LAYERS, DEFAULT_NUM_MEMORIES, MemoryModule
from memorilla.utils import DEFAULT_SEED, write_json
from profiling.cached_memory import CachedMemory, Fp8CachedMemory, tensor_bytes
from profiling.timing import GpuTimer, ResultLog, environment, gpu_memory, reset_peak_memory, run_reps, summarize

CACHES = ("gpu", "host", "kv", "kv_fp8")
MODES = ("eager", "cuda_graph", "compile")
DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
EMBEDDING_DIM = 2560  # Qwen3-Embedding-4B
DECODER_DIM = 4096  # Qwen3-8B
RESULTS_FILE = "memory_module.jsonl"
GRAPH_WARMUP = 3  # eager calls on a side stream before a CUDA graph is captured
RANDOM_CHUNK_ROWS = 65_536  # random embeddings are drawn in float32 this many rows at a time
# Peak HBM bandwidth in GB/s by device name (first match wins); --peak_bandwidth_gbs overrides it.
PEAK_BANDWIDTH_GBS = (
    ("B200", 8000.0),
    ("H200", 4800.0),
    ("H100 PCIe", 2000.0),
    ("H100", 3350.0),
    ("A100-SXM4-80GB", 2039.0),
    ("A100 80GB PCIe", 1935.0),
    ("A100", 1555.0),
    ("L40S", 864.0),
)


def int_list(text: str) -> list[int]:
    """Parse ``1,8,64`` into integers."""
    return [int(value) for value in text.split(",") if value]


def choice_list(choices: tuple[str, ...]) -> Callable[[str], list[str]]:
    """A parser of comma-separated values from ``choices``."""

    def parse(text: str) -> list[str]:
        values = [value for value in text.split(",") if value]
        unknown = set(values) - set(choices)
        if unknown:
            raise argparse.ArgumentTypeError(f"unknown values {sorted(unknown)}; choose from {','.join(choices)}")
        return values

    return parse


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help=f"Output directory; results go to <out>/{RESULTS_FILE}.")
    parser.add_argument(
        "--ns", type=int_list, default=int_list("1,512,4096,32768,262144,1048576"), help="Collection sizes N."
    )
    parser.add_argument("--batch_sizes", type=int_list, default=[1], help="Questions answered together.")
    parser.add_argument("--cache", type=choice_list(CACHES), default=list(CACHES), help=f"From {','.join(CACHES)}.")
    parser.add_argument("--modes", type=choice_list(MODES), default=["eager"], help=f"From {','.join(MODES)}.")
    parser.add_argument("--shared", type=int_list, default=[0], help="0, 1 or 0,1: one collection for the batch.")
    parser.add_argument("--splits", type=int_list, default=[1], help="Chunks of the weighted sum (kv only).")
    parser.add_argument("--warmup", type=int, default=10, help="Untimed calls per cell.")
    parser.add_argument("--reps", type=int, default=50, help="Timed calls per cell.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint directory or Hub repo (default: random init).")
    parser.add_argument("--subfolder", default=None, help="Checkpoint folder inside --checkpoint, e.g. pmv2.")
    parser.add_argument("--embedding_dim", type=int, default=EMBEDDING_DIM, help="Encoder width of a random module.")
    parser.add_argument("--output_dim", type=int, default=DECODER_DIM, help="Decoder width of a random module.")
    parser.add_argument("--num_memories", type=int, default=DEFAULT_NUM_MEMORIES, help="Slots of a random module.")
    parser.add_argument(
        "--num_cross_attn_layers", type=int, default=DEFAULT_NUM_CROSS_ATTN_LAYERS, help="Layers of a random module."
    )
    parser.add_argument("--write_reps", type=int, default=3, help="Timed cache writes per kv and kv_fp8 cell.")
    parser.add_argument("--host_max", type=int, default=1_048_576, help="Largest N of the host setting.")
    parser.add_argument("--check_max", type=int, default=262_144, help="Largest batch x N whose output is checked.")
    parser.add_argument(
        "--tolerance_factor",
        type=float,
        default=2.0,
        help="Allowed difference from the default forward pass, in multiples of its own error against float32.",
    )
    parser.add_argument("--tolerance_floor", type=float, default=1e-5, help="Smallest allowed relative difference.")
    parser.add_argument("--peak_bandwidth_gbs", type=float, default=None, help="Peak HBM bandwidth of the GPU.")
    parser.add_argument("--no_flush_l2", action="store_true", help="Keep the L2 cache warm between calls.")
    args = parser.parse_args(argv)
    if set(args.shared) - {0, 1}:
        parser.error("--shared takes 0 and 1.")
    if min(args.splits) < 1:
        parser.error("--splits must be at least 1.")
    return args


def grid(fixed: dict[str, Any], **axes: Iterable[Any]) -> list[dict[str, Any]]:
    """Every combination of the axes, with the last axis (N) varying fastest."""
    names = list(axes)
    return [{**fixed, **dict(zip(names, values, strict=True))} for values in itertools.product(*axes.values())]


def meaningful(key: dict[str, Any]) -> bool:
    """Whether a combination of settings is worth a cell (no duplicates, no unsupported pairs)."""
    if key["splits"] != 1 and key["cache"] != "kv":
        return False
    if key["mode"] != "eager" and key["cache"] == "host":
        return False
    return not (key["shared"] and key["batch"] == 1)


def is_oom(error: BaseException) -> bool:
    """Whether an exception means the device ran out of memory."""
    return isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower()


def sweep(
    log: ResultLog,
    keys: list[dict[str, Any]],
    cell: Callable[[dict[str, Any]], tuple[str, dict[str, Any]]],
    context: dict[str, Any],
) -> None:
    """Run every cell not yet in the log; after an out-of-memory cell, skip the larger N of the same settings.

    A cell that raises another exception is recorded as ``error`` and the sweep goes on; a rerun retries it.

    Args:
        log: The results log.
        keys: Cell settings, ordered with N ascending within each group of other settings.
        cell: Runs one cell and returns its status and fields.
        context: Fields recorded with every cell (environment and arguments).
    """
    failed: set[str] = set()
    for key in keys:
        group = ResultLog.key_id({name: value for name, value in key.items() if name != "n"})
        status = log.status(key)
        if status not in (None, "error"):
            failed |= {group} if status == "oom" else set()
            continue
        if group in failed:
            status, fields = "skipped", {"reason": "a smaller N ran out of memory"}
        else:
            try:
                status, fields = cell(key)
            except Exception as error:  # noqa: BLE001 - one failing setting must not end the sweep
                status = "oom" if is_oom(error) else "error"
                fields = {"reason": f"{type(error).__name__}: {str(error).splitlines()[0][:300] if str(error) else ''}"}
            if status == "oom":
                failed.add(group)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.write(key, status, **fields, **context)
        read = fields.get("timings", {}).get("read")
        detail = f"{read['median_ms']:.3f} ms [p25 {read['p25_ms']:.3f}, p75 {read['p75_ms']:.3f}]" if read else ""
        if "within_tolerance" in fields:
            verdict = "within" if fields["within_tolerance"] else "OUTSIDE"
            detail += f", diff vs default {fields['rel_diff_default']:.2e} ({verdict} {fields['tolerance']:.2e})"
        settings = " ".join(f"{name}={key[name]}" for name in ("cache", "mode", "shared", "splits", "batch"))
        print(f"{settings} N={key['n']:,}: {status} {detail or fields.get('reason', '')}", flush=True)


def load_memory(args: argparse.Namespace) -> MemoryModule:
    """The memory module of ``--checkpoint``, or a randomly initialised one; latency does not depend on the weights."""
    if args.checkpoint is None:
        torch.manual_seed(args.seed)
        module = MemoryModule(
            args.embedding_dim,
            args.output_dim,
            num_memories=args.num_memories,
            num_cross_attn_layers=args.num_cross_attn_layers,
        )
        return module.eval()
    path = Path(args.checkpoint).expanduser()
    if not path.exists():
        from huggingface_hub import snapshot_download

        patterns = [f"{args.subfolder}/*"] if args.subfolder else None
        path = Path(snapshot_download(repo_id=args.checkpoint, allow_patterns=patterns))
    return MemoryModule.from_pretrained(path / args.subfolder if args.subfolder else path)


def random_unit(shape: tuple[int, ...], generator: torch.Generator, device: torch.device, dtype: torch.dtype):
    """Random unit-norm vectors, like encoder outputs, built in chunks so that only the result is held in full."""
    values = torch.empty(shape, device=device, dtype=dtype)
    rows = values.view(-1, shape[-1])
    for start in range(0, rows.shape[0], RANDOM_CHUNK_ROWS):
        chunk = rows[start : start + RANDOM_CHUNK_ROWS]
        noise = torch.randn(chunk.shape, generator=generator, device=device, dtype=torch.float32)
        chunk.copy_(F.normalize(noise, dim=-1))
    return values


class ForwardReader:
    """The ``gpu`` setting: the stored embeddings go through ``MemoryModule.forward``."""

    def __init__(self, module: MemoryModule) -> None:
        """Wrap the module."""
        self.module = module

    def write(self, docs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Keep the embeddings and an all-false padding mask, as ``MemoryVLLM.answer`` passes it."""
        return docs, torch.zeros(docs.shape[:2], dtype=torch.bool, device=docs.device)

    def read(self, state: tuple[torch.Tensor, torch.Tensor], question: torch.Tensor) -> torch.Tensor:
        """Run the forward pass; one stored collection is broadcast to the whole batch."""
        docs, mask = state
        if docs.shape[0] != question.shape[0]:
            docs, mask = docs.expand(question.shape[0], -1, -1), mask.expand(question.shape[0], -1)
        return self.module(docs, mask, question)


class HostReader(ForwardReader):
    """The ``host`` setting: the embeddings live in pinned host memory and every query copies them to the GPU."""

    def write(self, docs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Move the embeddings to (pinned) host memory."""
        host = docs.cpu()
        host = host.pin_memory() if docs.is_cuda else host
        return host, torch.zeros(docs.shape[:2], dtype=torch.bool, device=docs.device)

    def read(self, state: tuple[torch.Tensor, torch.Tensor], question: torch.Tensor) -> torch.Tensor:
        """Copy the embeddings to the device, then run the forward pass."""
        host, mask = state
        return super().read((host.to(mask.device, non_blocking=True), mask), question)


def make_reader(cache: str, module: MemoryModule, splits: int) -> Any:
    """The object whose ``write`` stores a collection and whose ``read`` answers a batch of questions."""
    if cache == "gpu":
        return ForwardReader(module)
    if cache == "host":
        return HostReader(module)
    if cache == "kv":
        return CachedMemory(module, splits=splits)
    return Fp8CachedMemory(module)


def executable(reader: Any, state: Any, question: torch.Tensor, mode: str) -> Callable[[], torch.Tensor]:
    """A zero-argument call of ``reader.read(state, question)`` in the given execution mode; it returns the output."""
    if mode == "eager":
        return lambda: reader.read(state, question)
    if mode == "compile":
        # Every cell compiles anew; without the reset, recompilations add up across cells, and past Dynamo's
        # recompile limit later cells would silently run eager.
        torch._dynamo.reset()
        compiled = torch.compile(reader.read, dynamic=False)
        return lambda: compiled(state, question)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(GRAPH_WARMUP):
            reader.read(state, question)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = reader.read(state, question)

    def replay() -> torch.Tensor:
        graph.replay()
        return output

    return replay


def output_check(
    output: torch.Tensor, default: torch.Tensor, expected: torch.Tensor, factor: float, floor: float
) -> dict[str, Any]:
    """Compare a variant's output with the default forward pass and with the float32 forward pass.

    The default forward pass (``MemoryModule.forward`` in the run's dtype, eager, as ``MemoryVLLM`` serves it) has a
    rounding error of its own against float32. A variant is within tolerance when it differs from the default
    forward pass by at most ``factor`` times that error (and at least ``floor``, for float32 runs whose own error
    is near zero). Two computations that each round like the default differ by about 1.4 times its error.

    Args:
        output: The variant's output, as its timed call returns it.
        default: The default forward pass on the same inputs.
        expected: The forward pass of a float32 copy of the module on the same inputs.
        factor: Allowed difference from the default, in multiples of the default's own error.
        floor: Smallest allowed relative difference.

    Returns:
        ``rel_error`` (variant vs float32), ``default_rel_error`` (default vs float32), ``rel_diff_default`` and
        ``max_abs_diff_default`` (variant vs default), ``tolerance`` and ``within_tolerance``.
    """
    output, default = output.float(), default.float()
    default_error = ((default - expected).norm() / expected.norm()).item()
    difference = ((output - default).norm() / default.norm()).item()
    tolerance = max(factor * default_error, floor)
    return {
        "rel_error": ((output - expected).norm() / expected.norm()).item(),
        "default_rel_error": default_error,
        "rel_diff_default": difference,
        "max_abs_diff_default": (output - default).abs().max().item(),
        "tolerance": tolerance,
        "within_tolerance": difference <= tolerance,
    }


def peak_bandwidth(args: argparse.Namespace, device: torch.device) -> float | None:
    """Peak HBM bandwidth of the device in GB/s, if known."""
    if args.peak_bandwidth_gbs is not None or device.type != "cuda":
        return args.peak_bandwidth_gbs
    name = torch.cuda.get_device_name(device)
    return next((value for pattern, value in PEAK_BANDWIDTH_GBS if pattern in name), None)


def fp8_supported(device: torch.device) -> bool:
    """Whether ``torch._scaled_mm`` with per-row scales runs here (Hopper or newer; emulated without a GPU)."""
    return device.type != "cuda" or torch.cuda.get_device_capability(device) >= (9, 0)


def main(argv: list[str] | None = None) -> None:
    """Time the memory module over the sweep."""
    args = parse_args(argv)
    out = Path(args.out)
    env = environment()
    write_json(out / "memory_module.env.json", env)
    print(f"{env['gpu'] or args.device}, torch {env['torch']}, commit {env['git_commit']}", flush=True)

    device = torch.device(args.device)
    dtype = DTYPES[args.dtype]
    module = load_memory(args).to(device=device, dtype=dtype).eval()
    reference = copy.deepcopy(module).float() if args.check_max > 0 else None
    timer = GpuTimer(device, flush_l2=not args.no_flush_l2)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    config = module.config
    dim = config.embedding_dim
    peak = peak_bandwidth(args, device)

    def cell(key: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        n, batch, cache, mode, shared = key["n"], key["batch"], key["cache"], key["mode"], key["shared"]
        if cache == "host" and n > args.host_max:
            return "skipped", {"reason": f"N above --host_max {args.host_max}"}
        if mode == "cuda_graph" and device.type != "cuda":
            return "skipped", {"reason": "CUDA graphs need a GPU"}
        if cache == "kv_fp8" and not fp8_supported(device):
            return "skipped", {"reason": "float8 matrix products with per-row scales need a Hopper or newer GPU"}
        collections = 1 if shared else batch
        docs = random_unit((collections, n, dim), generator, device, dtype)
        question = random_unit((batch, dim), generator, device, dtype)
        reader = make_reader(cache, module, key["splits"])
        fields: dict[str, Any] = {}
        with torch.inference_mode():
            if cache.startswith("kv"):
                fields["write"] = summarize([timer(partial(reader.write, docs)) for _ in range(args.write_reps)])
            state = reader.write(docs)
            # The peak counts from here, so it includes what compilation or graph capture allocates (a CUDA graph
            # keeps its memory pool), but not the stored collection or cache, which is reported as state_bytes.
            baseline = reset_peak_memory()
            start = time.perf_counter()
            call = executable(reader, state, question, mode)
            output = call()
            fields["setup_ms"] = (time.perf_counter() - start) * 1000.0  # includes the first call (compilation)
            assert output.shape == (batch, config.num_memories, config.output_dim), output.shape
            fields.update(
                run_reps(lambda sink: sink.setdefault("read", []).append(timer(call)), args.warmup, args.reps)
            )
            fields["gpu_memory"] = gpu_memory(baseline)

            # The output checked is the one the timed call returns (after compilation or graph capture); it is
            # compared after timing, so the reference computations do not count towards the peak.
            if reference is not None and batch * n <= args.check_max:
                forward = ForwardReader(module)
                default = forward.read(forward.write(docs), question)
                expected = reference(docs.float().expand(batch, -1, -1), None, question.float())
                fields.update(output_check(output, default, expected, args.tolerance_factor, args.tolerance_floor))
                del default, expected
            del docs, output

        state_bytes = tensor_bytes(state)
        median_s = fields["timings"]["read"]["median_ms"] / 1000.0
        fields.update(
            state_bytes=state_bytes,
            per_query_ms=fields["timings"]["read"]["median_ms"] / batch,
            achieved_gbs=state_bytes / median_s / 1e9,
            peak_bandwidth_gbs=peak,
        )
        return "ok", fields

    fixed = {
        "device": str(device),
        "dtype": args.dtype,
        "flush_l2": not args.no_flush_l2,
        "embedding_dim": dim,
        "output_dim": config.output_dim,
        "num_memories": config.num_memories,
        "num_cross_attn_layers": config.num_cross_attn_layers,
        "num_self_attn_layers": config.num_self_attn_layers,
        "retrieval_init": config.retrieval_init,
    }
    context = {"env": env, "args": {name: value for name, value in vars(args).items() if name != "out"}}
    keys = grid(
        fixed,
        cache=args.cache,
        mode=args.modes,
        shared=args.shared,
        splits=args.splits,
        batch=args.batch_sizes,
        n=args.ns,
    )
    sweep(ResultLog(out / RESULTS_FILE), [key for key in keys if meaningful(key)], cell, context)


if __name__ == "__main__":
    main()
