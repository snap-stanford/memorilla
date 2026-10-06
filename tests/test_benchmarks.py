"""Consistency checks for the benchmark registry."""

from dataclasses import FrozenInstanceError

import pytest

from memorilla.benchmarks import (
    BENCHMARKS,
    CHOICE,
    EXACT_MATCH,
    GENERATION,
    GENERATION_METRICS,
    METRICS_BY_SCORING,
    RETRIEVAL_CONTEXT_LENGTH,
    SCORINGS,
    TRIVIA_SUFFIX,
    YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS,
    BaselineProtocol,
    Benchmark,
    get_benchmarks,
    validate_metrics,
)

EXPECTED = ("pmv2", "pv4", "factkg", "triviaqa", "narrativeqa", "pubmedqa", "lamp4", "lamp7")


def test_registry_contents() -> None:
    """The registry holds the expected benchmarks, keyed by name and evaluated on their test split."""
    assert tuple(BENCHMARKS) == EXPECTED
    for name, benchmark in BENCHMARKS.items():
        assert benchmark.name == name
        assert benchmark.split == "test"
        assert benchmark.config


@pytest.mark.parametrize("benchmark", BENCHMARKS.values(), ids=list(BENCHMARKS))
def test_benchmark_protocol(benchmark: Benchmark) -> None:
    """Every benchmark has a consistent scoring method, budgets and a baseline protocol."""
    assert benchmark.scoring in SCORINGS
    assert benchmark.metrics and len(set(benchmark.metrics)) == len(benchmark.metrics)
    assert set(benchmark.metrics) <= set(METRICS_BY_SCORING[benchmark.scoring])
    assert 0 < benchmark.max_new_tokens < benchmark.max_length
    assert benchmark.system_prompt.strip()

    baseline = benchmark.baseline
    assert isinstance(baseline, BaselineProtocol)
    assert baseline.system_prompt.strip() and baseline.max_new_tokens > 0
    assert baseline.full_context_length > baseline.max_new_tokens
    if baseline.yarn_factor is not None:
        assert baseline.full_context_length == int(baseline.yarn_factor * YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS)
        assert baseline.full_context_length > RETRIEVAL_CONTEXT_LENGTH


def test_scoring_assignments() -> None:
    """Answer-choice benchmarks use embedding matching; TriviaQA shares its suffix with its baseline."""
    assert {name for name, benchmark in BENCHMARKS.items() if benchmark.scoring == CHOICE} == {"pmv2", "pv4", "factkg"}
    assert "accuracy_hard" in BENCHMARKS["pv4"].metrics
    assert BENCHMARKS["triviaqa"].suffix == TRIVIA_SUFFIX == BENCHMARKS["triviaqa"].baseline.suffix
    assert TRIVIA_SUFFIX.endswith("Your answer:")


def test_lookup() -> None:
    """``all`` returns every benchmark in order; unknown names raise."""
    assert [benchmark.name for benchmark in get_benchmarks("all")] == list(EXPECTED)
    assert get_benchmarks("lamp7") == [BENCHMARKS["lamp7"]]
    with pytest.raises(KeyError):
        get_benchmarks("unknown")


def test_validation_and_immutability() -> None:
    """Benchmarks reject unknown scorings and foreign metrics, and cannot be modified."""
    with pytest.raises(ValueError, match="^x: Unknown scoring"):
        Benchmark("x", "x", "test", "s", 128, 8, scoring="bogus", metrics=("accuracy",))
    with pytest.raises(ValueError, match="^x: Metrics"):
        Benchmark("x", "x", "test", "s", 128, 8, scoring=CHOICE, metrics=("bleu1",))
    validate_metrics(GENERATION, GENERATION_METRICS)
    with pytest.raises(ValueError):
        validate_metrics(GENERATION, (EXACT_MATCH,))
    with pytest.raises(FrozenInstanceError):
        BENCHMARKS["pv4"].max_length = 1
