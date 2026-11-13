"""Tests of the memory module profiler on the CPU."""

import json
from pathlib import Path

import pytest
import torch

from memorilla.memory import MemoryModule
from profiling import run, summarize
from profiling.cached_memory import CachedMemory, Fp8CachedMemory, tensor_bytes
from profiling.timing import ResultLog, run_reps
from profiling.timing import summarize as stats


def tiny_module(**options) -> MemoryModule:
    torch.manual_seed(0)
    return MemoryModule(embedding_dim=16, output_dim=12, num_memories=4, num_heads=2, **options).eval()


@pytest.mark.parametrize("write_chunk", [4, 64])
@pytest.mark.parametrize("splits", [1, 2, 4])
@pytest.mark.parametrize("shared", [False, True])
def test_cached_memory_matches_the_module(splits: int, shared: bool, write_chunk: int) -> None:
    module = tiny_module()
    cached = CachedMemory(module, splits=splits, write_chunk=write_chunk)
    docs, question = torch.randn(1 if shared else 3, 9, 16), torch.randn(3, 16)
    cache = cached.write(docs)
    padded = -(-9 // splits) * splits
    # layers x collections x (N keys + padded N values) x dim x float32
    assert tensor_bytes(cache) == 2 * docs.shape[0] * (9 + padded) * 16 * 4
    expected = module(docs.expand(3, -1, -1), None, question)
    torch.testing.assert_close(cached.read(cache, question), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("shared", [False, True])
def test_fp8_cache_is_close_to_the_module(shared: bool) -> None:
    module = tiny_module()
    cached = Fp8CachedMemory(module, write_chunk=5)
    docs, question = torch.randn(1 if shared else 2, 21, 16), torch.randn(2, 16)
    cache = cached.write(docs)
    keys, key_scales, values, value_scales, num_docs = cache[0]
    assert keys.dtype == values.dtype == torch.float8_e4m3fn and num_docs == 21
    assert keys.shape[2] == values.shape[3] == 32  # documents padded to a multiple of 16
    expected = module(docs.expand(2, -1, -1), None, question)
    got = cached.read(cache, question)
    assert ((got - expected).norm() / expected.norm()).item() < 0.05


@pytest.mark.parametrize("options", [{"retrieval_init": True}, {"num_self_attn_layers": 1}])
def test_cached_memory_rejects_question_dependent_document_work(options: dict) -> None:
    with pytest.raises(ValueError, match="learned slots"):
        CachedMemory(tiny_module(**options))


def test_stats_and_first_call() -> None:
    result = stats([4.0, 1.0, 3.0, 2.0])
    assert result["median_ms"] == 2.5 and result["min_ms"] == 1.0 and result["max_ms"] == 4.0
    assert result["p25_ms"] == pytest.approx(1.75) and result["raw_ms"] == [4.0, 1.0, 3.0, 2.0]

    calls = iter(range(100))
    out = run_reps(lambda sink: sink.setdefault("t", []).append(float(next(calls))), warmup=2, reps=3)
    assert out["first_call"] == {"t": 0.0}
    assert out["timings"]["t"]["raw_ms"] == [2.0, 3.0, 4.0]


def test_sweep_resumes_and_skips_larger_n_after_oom(tmp_path: Path) -> None:
    log = ResultLog(tmp_path / "part.jsonl")
    keys = run.grid({}, cache=["gpu", "kv"], mode=["eager"], shared=[0], splits=[1], batch=[1], n=[1, 2, 3])
    ran = []

    def cell(key: dict) -> tuple[str, dict]:
        ran.append((key["cache"], key["n"]))
        if key["cache"] == "gpu" and key["n"] == 2:
            raise torch.OutOfMemoryError("CUDA out of memory")
        return "ok", {}

    run.sweep(log, keys, cell, {})
    assert ran == [("gpu", 1), ("gpu", 2), ("kv", 1), ("kv", 2), ("kv", 3)]
    statuses = [json.loads(line)["status"] for line in (tmp_path / "part.jsonl").read_text().splitlines()]
    assert statuses == ["ok", "oom", "skipped", "ok", "ok", "ok"]

    ran.clear()
    run.sweep(ResultLog(tmp_path / "part.jsonl"), keys, cell, {})
    assert ran == []


def test_profile_and_summary_on_cpu(tmp_path: Path) -> None:
    run.main(
        ["--out", str(tmp_path), "--ns", "1,5", "--warmup", "1", "--reps", "2", "--device", "cpu", "--dtype", "float32"]
        + ["--embedding_dim", "16", "--output_dim", "12", "--num_memories", "4", "--batch_sizes", "1,2"]
        + ["--shared", "0,1", "--splits", "1,2", "--modes", "eager,cuda_graph"]
    )
    records = {
        tuple(r["key"][name] for name in ("cache", "mode", "shared", "splits", "batch", "n")): r
        for r in map(json.loads, (tmp_path / run.RESULTS_FILE).read_text().splitlines())
    }
    # host runs eager only, splits apply to kv only, and shared needs a batch above 1
    assert ("host", "cuda_graph", 0, 1, 1, 1) not in records and ("gpu", "eager", 0, 2, 1, 1) not in records
    assert ("gpu", "eager", 1, 1, 1, 1) not in records
    assert records[("gpu", "cuda_graph", 0, 1, 1, 1)]["status"] == "skipped"  # no GPU here
    assert all(r["status"] == "ok" for key, r in records.items() if key[1] == "eager")
    # every exact variant matches the default forward pass; float8 is caught as outside tolerance in a float32 run
    checked = {key: r for key, r in records.items() if r["status"] == "ok"}
    assert all("within_tolerance" in r for r in checked.values())
    assert all(r["within_tolerance"] for key, r in checked.items() if key[0] != "kv_fp8")
    assert records[("gpu", "eager", 0, 1, 2, 5)]["rel_diff_default"] == 0.0
    fp8 = records[("kv_fp8", "eager", 1, 1, 2, 5)]
    assert not fp8["within_tolerance"] and fp8["rel_error"] < 0.1
    kv = records[("kv", "eager", 0, 1, 1, 5)]
    assert kv["timings"]["read"]["n"] == 2 and kv["achieved_gbs"] > 0 and "write" in kv
    assert kv["state_bytes"] == 2 * 2 * 5 * 16 * 4  # layers x (keys, values) x N x dim x float32
    assert (tmp_path / "memory_module.env.json").exists()

    summarize.main([str(tmp_path)])
    summary = (tmp_path / "summary.md").read_text()
    assert "# Memory module profile" in summary and "per question (ms)" in summary
    assert "differ from the default forward pass by more than the tolerance" in summary and "cache=kv_fp8" in summary
    assert len((tmp_path / "summary.csv").read_text().splitlines()) == len(records) + 1


def test_meaningful_combinations() -> None:
    base = {"cache": "kv", "mode": "eager", "shared": 0, "splits": 1, "batch": 1}
    assert run.meaningful(base)
    assert not run.meaningful({**base, "cache": "gpu", "splits": 4})
    assert not run.meaningful({**base, "cache": "host", "mode": "compile"})
    assert not run.meaningful({**base, "shared": 1})
    assert run.meaningful({**base, "shared": 1, "batch": 8})


def test_unknown_cache_setting_is_rejected() -> None:
    with pytest.raises(SystemExit):
        run.parse_args(["--out", "x", "--cache", "none"])
    with pytest.raises(SystemExit):
        run.parse_args(["--out", "x", "--modes", "jit"])


def test_output_check_tolerance_follows_the_default_error() -> None:
    expected = torch.ones(2, 4, 3)
    default = expected * (1 + 1e-2)  # the default forward pass is 1% off float32
    within = run.output_check(expected * (1 + 1.5e-2), default, expected, factor=2.0, floor=1e-5)
    assert within["within_tolerance"] and within["default_rel_error"] == pytest.approx(1e-2, rel=1e-4)
    assert within["tolerance"] == pytest.approx(2e-2, rel=1e-4)
    assert within["rel_diff_default"] == pytest.approx(5e-3 / 1.01, rel=1e-4)  # relative to the default's norm
    outside = run.output_check(expected * 1.05, default, expected, factor=2.0, floor=1e-5)
    assert not outside["within_tolerance"]
    exact = run.output_check(expected, expected, expected, factor=2.0, floor=1e-5)
    assert exact["tolerance"] == 1e-5 and exact["within_tolerance"]


def test_chunked_fp8_write_matches_one_chunk() -> None:
    module = tiny_module()
    docs = torch.randn(2, 21, 16)
    whole = Fp8CachedMemory(module, write_chunk=64).write(docs)
    chunked = Fp8CachedMemory(module, write_chunk=4).write(docs)
    for layer_whole, layer_chunked in zip(whole, chunked, strict=True):
        keys, key_scales, values, value_scales, _ = layer_whole
        assert torch.equal(keys.float(), layer_chunked[0].float()) and torch.equal(
            values.float(), layer_chunked[2].float()
        )
        # chunked projections round differently in the last bits, so the float32 scales agree to rounding
        torch.testing.assert_close(key_scales, layer_chunked[1])
        torch.testing.assert_close(value_scales, layer_chunked[3])
