"""Tests for the scorers, on hand-made fixtures."""

from dataclasses import replace
import json
from pathlib import Path
from typing import NoReturn

from datasets import Dataset
import numpy as np
import pytest

from memorilla import metrics
from memorilla.benchmarks import (
    ACCURACY,
    ACCURACY_HARD,
    BENCHMARKS,
    CHOICE,
    EXACT_MATCH,
    GENERATION,
    GENERATION_METRICS,
    POLARITY,
)
from memorilla.metrics import (
    METRICS_FILE,
    NLTK_RESOURCES,
    PREDICTIONS_FILE,
    bleu,
    check_scoring_inputs,
    choice_accuracy,
    choice_index,
    choice_predictions,
    clean_generation,
    compute_metrics,
    ensure_nltk_resources,
    exact_match,
    first_polarity,
    nltk_resource_available,
    normalize_answer,
    polarity_accuracy,
    rouge,
    score_predictions,
    token_f1,
    write_metrics,
    write_predictions,
)
from tests.helpers import FakeEncoder, make_rows


def test_clean_generation() -> None:
    """Chat-template tokens and surrounding whitespace are removed."""
    assert clean_generation("  Paris<|im_end|>\n") == "Paris"
    assert clean_generation("<|im_start|>a<|endoftext|> b<|im_sep|>") == "a b"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The Eiffel Tower!", "eiffel tower"),
        ("  a  Tale of  TWO cities. ", "tale of two cities"),
        ("Anthem", "anthem"),
        ("", ""),
    ],
)
def test_normalize_answer(text: str, expected: str) -> None:
    """SQuAD normalisation lower-cases and drops punctuation, articles and extra whitespace."""
    assert normalize_answer(text) == expected


def test_exact_match() -> None:
    """Exact match compares normalised strings and requires full equality."""
    predictions = ["The Pacific Ocean.", "Mark Twain wrote it", "1969"]
    references = ["pacific ocean", "Mark Twain", "1969"]
    assert exact_match(predictions, references) == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Yes. The study shows", "yes"),
        ("no", "no"),
        ("Maybe, it depends", "maybe"),
        ("- Yes.", "yes"),
        ('"No," the authors say', "no"),
        ("1. Maybe", "maybe"),
        ("* - No", "no"),
        ("Perhaps yes", None),
        ("", None),
        ("   ", None),
        ("-", None),
    ],
)
def test_first_polarity(text: str, expected: str | None) -> None:
    """The first word decides polarity after list markers, numbering and quotes are skipped."""
    assert first_polarity(text) == expected


def test_polarity_accuracy_skips_references_without_polarity() -> None:
    """References without a leading yes/no/maybe do not count."""
    predictions = ["Yes, clearly.", "No.", "Maybe", "yes"]
    references = ["yes. Because", "Maybe. Unclear", "unknown", "No"]
    assert polarity_accuracy(predictions, references) == pytest.approx(1 / 3)
    assert polarity_accuracy(["yes"], ["unclear"]) == 0.0


def test_token_f1() -> None:
    """Token F1 uses clipped counts of lower-cased whitespace tokens."""
    assert token_f1(["the cat sat"], ["The cat"]) == pytest.approx(2 * (2 / 3) * 1 / (2 / 3 + 1))
    assert token_f1(["a a b"], ["a c"]) == pytest.approx(2 * (1 / 3) * (1 / 2) / (1 / 3 + 1 / 2))
    assert token_f1(["", "x"], ["y", "z"]) == 0.0


def test_bleu_and_rouge() -> None:
    """Identical texts score 1; disjoint texts score 0 for ROUGE."""
    texts = ["the quick brown fox jumps over the lazy dog"]
    assert bleu(texts, texts) == {"bleu1": pytest.approx(1.0), "bleu4": pytest.approx(1.0)}
    assert rouge(texts, texts) == {"rouge1": pytest.approx(1.0), "rougeL": pytest.approx(1.0)}
    assert rouge(["alpha beta"], ["gamma delta"]) == {"rouge1": 0.0, "rougeL": 0.0}
    scores = rouge(["running dogs"], ["the dog runs"])
    assert scores["rouge1"] == pytest.approx(2 * 1.0 * (2 / 3) / (1.0 + 2 / 3))
    assert bleu(["the cat"], ["the cat sat on the mat"])["bleu1"] < 1.0


@pytest.mark.skipif(
    not all(nltk_resource_available(resource) for resource in NLTK_RESOURCES.values()),
    reason="the NLTK data METEOR needs is not installed",
)
def test_generation_metrics_on_identical_texts() -> None:
    """Every generation metric is maximal for an exact copy."""
    texts = ["the quick brown fox jumps over the lazy dog"]
    scores = compute_metrics(GENERATION, GENERATION_METRICS, texts, texts)
    assert list(scores) == list(GENERATION_METRICS)
    assert scores["meteor"] > 0.99 and scores["f1"] == pytest.approx(1.0)


def test_choice_predictions_never_pick_empty_choices() -> None:
    """The nearest non-empty choice wins and the index refers to the full choice list."""
    predictions = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float16)
    choice_embeddings = [np.array([[0.0, 1.0], [0.9, 0.1]], dtype=np.float16), np.array([[1.0, 1.0]], np.float16)]
    choices = [["b", "", "c"], ["", "z"]]
    assert choice_predictions(predictions, choice_embeddings, choices) == [2, 1]


def test_choice_index_ignores_whitespace() -> None:
    """The reference is located among the choices regardless of surrounding whitespace; a missing one raises."""
    assert choice_index(["paris ", "", "rome"], "rome") == 2
    assert choice_index(["paris", "rome"], " paris") == 0
    with pytest.raises(ValueError):
        choice_index(["paris", "rome"], "berlin")


def test_choice_accuracy_with_hard_subset() -> None:
    """Accuracy and hard-subset accuracy follow the nearest-choice rule."""
    table = {
        "paris": [1.0, 0.0],
        "rome": [0.0, 1.0],
        "berlin": [-1.0, 0.0],
        "it is paris": [0.9, 0.1],
        "rome i think": [0.2, 0.8],
        "no idea": [-0.8, -0.1],
    }
    encoder = FakeEncoder(dim=2, table=table)
    predictions = ["it is paris", " rome i think ", "no idea"]
    references = ["paris", "berlin", "rome"]
    choices = [["paris", "rome", ""], ["paris", "rome", "berlin"], ["rome", "", "berlin"]]
    scores = choice_accuracy(predictions, references, choices, encoder, hard=[False, True, True])
    assert scores == {ACCURACY: pytest.approx(1 / 3), ACCURACY_HARD: 0.0}
    assert "" not in encoder.calls[-1]


def test_compute_metrics_dispatch_and_validation() -> None:
    """Each scoring method returns the requested metrics in order and rejects foreign metrics."""
    scores = compute_metrics(EXACT_MATCH, (EXACT_MATCH, "bleu1", "rougeL"), ["Paris"], ["paris"])
    assert list(scores) == [EXACT_MATCH, "bleu1", "rougeL"] and scores[EXACT_MATCH] == 1.0
    assert compute_metrics(POLARITY, (ACCURACY,), ["no"], ["No."]) == {ACCURACY: 1.0}
    with pytest.raises(ValueError):
        compute_metrics(POLARITY, (EXACT_MATCH,), ["no"], ["no"])
    with pytest.raises(ValueError):
        compute_metrics("unknown", (ACCURACY,), ["no"], ["no"])
    with pytest.raises(ValueError):
        compute_metrics(CHOICE, (ACCURACY,), ["no"], ["no"])


def test_write_predictions_and_metrics(tmp_path: Path) -> None:
    """Metrics are written as percentages with two decimals next to the per-example predictions."""
    records = [{"question": "q", "answer": "a", "prediction": "a"}, {"question": "r", "answer": "b", "prediction": "c"}]
    write_predictions(tmp_path, records)
    summary = write_metrics(tmp_path, {ACCURACY: 2 / 3}, len(records))
    assert summary == {ACCURACY: 66.67, "num_examples": 2}
    assert json.loads((tmp_path / METRICS_FILE).read_text()) == summary
    lines = (tmp_path / PREDICTIONS_FILE).read_text().splitlines()
    assert [json.loads(line) for line in lines] == records


def test_predictions_survive_a_scoring_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every benchmark's predictions are on disk before scoring starts, even when scoring then fails."""
    dataset = Dataset.from_list(make_rows())
    jobs = [
        (BENCHMARKS["pubmedqa"], dataset, ["yes"] * len(dataset), tmp_path / "first"),
        (BENCHMARKS["lamp7"], dataset, ["paris"] * len(dataset), tmp_path / "second"),
    ]

    def fail(*args: object) -> NoReturn:
        """Raise like a scorer whose resources are missing.

        Args:
            *args: Ignored.

        Raises:
            LookupError: Always.
        """
        raise LookupError("missing scorer data")

    monkeypatch.setattr(metrics, "score_dataset", fail)
    with pytest.raises(LookupError):
        score_predictions(jobs)
    for _, _, predictions, output_dir in jobs:
        lines = (output_dir / PREDICTIONS_FILE).read_text().splitlines()
        assert [json.loads(line)["prediction"] for line in lines] == predictions
        assert not (output_dir / METRICS_FILE).exists()


def test_check_scoring_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Choice scoring needs a choices column holding every reference; METEOR needs its NLTK data."""
    rows = make_rows()
    check_scoring_inputs(BENCHMARKS["pv4"], Dataset.from_list(rows))
    with pytest.raises(ValueError, match="choices"):
        check_scoring_inputs(BENCHMARKS["pv4"], Dataset.from_list(rows).remove_columns("choices"))
    rows[1]["answer"] = "marlowe"
    with pytest.raises(ValueError, match="marlowe"):
        check_scoring_inputs(BENCHMARKS["pv4"], Dataset.from_list(rows))

    monkeypatch.setattr(metrics, "nltk_resource_available", lambda resource: False)
    monkeypatch.setattr(metrics.nltk, "download", lambda *args, **kwargs: False)
    with pytest.raises(LookupError, match="nltk.downloader"):
        ensure_nltk_resources()
    with pytest.raises(LookupError):
        check_scoring_inputs(BENCHMARKS["lamp7"], Dataset.from_list(make_rows()))
    check_scoring_inputs(replace(BENCHMARKS["lamp7"], metrics=("rouge1",)), Dataset.from_list(make_rows()))
