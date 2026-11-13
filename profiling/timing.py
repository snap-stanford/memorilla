"""The CUDA-event timer, summary statistics, GPU memory snapshots, the run environment and the results log."""

from collections.abc import Callable
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch

L2_FLUSH_BYTES = 256 * 2**20  # larger than the L2 cache of current data-centre GPUs (B200: 126 MB)
PERCENTILES = (10, 25, 75, 90)


class GpuTimer:
    """Time a function with CUDA events, overwriting the L2 cache before each call.

    Falls back to the wall clock without a GPU.
    """

    def __init__(self, device: torch.device, flush_l2: bool = True) -> None:
        """Allocate the L2 flush buffer when ``flush_l2`` is set."""
        self.cuda = device.type == "cuda"
        self.flush = torch.empty(L2_FLUSH_BYTES, dtype=torch.uint8, device=device) if self.cuda and flush_l2 else None

    def __call__(self, fn: Callable[[], Any]) -> float:
        """Run ``fn`` once and return its time in milliseconds."""
        if not self.cuda:
            start = time.perf_counter()
            fn()
            return (time.perf_counter() - start) * 1000.0
        if self.flush is not None:
            self.flush.zero_()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        begin.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        return begin.elapsed_time(end)


def summarize(values: list[float]) -> dict[str, Any]:
    """Median, mean, standard deviation, extremes, percentiles and raw samples of one timing."""
    ordered = sorted(values)
    result: dict[str, Any] = {
        "median_ms": statistics.median(ordered),
        "mean_ms": statistics.fmean(ordered),
        "std_ms": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "n": len(ordered),
    }
    for q, value in zip(PERCENTILES, np.percentile(ordered, PERCENTILES).tolist(), strict=True):
        result[f"p{q}_ms"] = value
    result["raw_ms"] = list(values)
    return result


def run_reps(step: Callable[[dict[str, list[float]]], None], warmup: int, reps: int) -> dict[str, Any]:
    """Run ``step(sink)`` for ``warmup`` untimed and ``reps`` timed repetitions and summarise each timing.

    Returns:
        ``{"timings": {name: stats}, "first_call": {name: ms}}``, where ``first_call`` is the first warmup call.
    """
    first: dict[str, list[float]] = {}
    for rep in range(warmup):
        step(first if rep == 0 else {})
    sinks: dict[str, list[float]] = {}
    for _ in range(reps):
        step(sinks)
    return {
        "timings": {name: summarize(values) for name, values in sinks.items()},
        "first_call": {name: values[0] for name, values in first.items()},
    }


def reset_peak_memory() -> int:
    """Reset PyTorch's peak memory counter and return the bytes allocated now (0 without a GPU)."""
    if not torch.cuda.is_available():
        return 0
    torch.cuda.reset_peak_memory_stats()
    return torch.cuda.memory_allocated()


def gpu_memory(baseline: int = 0) -> dict[str, Any]:
    """PyTorch's GPU memory counters, with the peak above ``baseline`` as ``torch_peak_delta``."""
    if not torch.cuda.is_available():
        return {}
    return {
        "torch_allocated": torch.cuda.memory_allocated(),
        "torch_reserved": torch.cuda.memory_reserved(),
        "torch_peak": torch.cuda.max_memory_allocated(),
        "torch_peak_delta": torch.cuda.max_memory_allocated() - baseline,
    }


def _command(args: list[str]) -> str | None:
    try:
        return subprocess.run(args, capture_output=True, text=True, check=False, timeout=30).stdout.strip() or None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def environment() -> dict[str, Any]:
    """Hardware, package versions and git revision."""
    repo = Path(__file__).resolve().parent.parent
    info: dict[str, Any] = {
        "host": platform.node(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "git_commit": _command(["git", "-C", str(repo), "rev-parse", "HEAD"]),
        "git_dirty": bool(_command(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"])),
        "nvidia_smi": _command(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,clocks.max.sm", "--format=csv,noheader"]
        ),
    }
    return info


class ResultLog:
    """Append-only JSON Lines log of cells, so that an interrupted sweep resumes.

    Each record has a ``key`` (the cell's settings) and a ``status``: ``ok``, ``oom``, ``skipped`` or ``error``.
    """

    def __init__(self, path: str | Path) -> None:
        """Open the log and read the keys it already holds."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.done: dict[str, str] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    record = json.loads(line)
                    self.done[self.key_id(record["key"])] = record["status"]

    @staticmethod
    def key_id(key: dict[str, Any]) -> str:
        """Canonical string of a cell key."""
        return json.dumps(key, sort_keys=True)

    def status(self, key: dict[str, Any]) -> str | None:
        """Status of a finished cell, or None."""
        return self.done.get(self.key_id(key))

    def write(self, key: dict[str, Any], status: str, **fields: Any) -> None:
        """Append one cell with its status and fields."""
        record = {"key": key, "status": status, **fields}
        with open(self.path, "a") as handle:
            handle.write(json.dumps(record) + "\n")
        self.done[self.key_id(key)] = status
