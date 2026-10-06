"""Scorers for every benchmark and the shared writer of predictions and metrics.

All metrics are returned as fractions in ``[0, 1]``; ``write_metrics`` reports them as percentages.
"""

from collections.abc import Sequence
from itertools import chain
import json
from pathlib import Path
import re
import string
from typing import TYPE_CHECKING, Any

from datasets import Dataset
import nltk
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from nltk.translate.meteor_score import single_meteor_score
import numpy as np
from rouge_score import rouge_scorer

from memorilla.benchmarks import (
    ACCURACY,
    ACCURACY_HARD,
    BLEU1,
    BLEU4,
    CHOICE,
    EXACT_MATCH,
    F1,
    GENERATION_METRICS,
    METEOR,
    POLARITY,
    ROUGE1,
    ROUGEL,
    Benchmark,
    validate_metrics,
)
from memorilla.data import ANSWER_COLUMN, CHOICES_COLUMN, HARD_COLUMN, QUESTION_COLUMN
from memorilla.embeddings import embed_texts, load_encoder
from memorilla.paths import DEFAULT_ENCODER
from memorilla.utils import write_json, write_jsonl

if TYPE_CHECKING:
    from vllm import LLM

CHAT_TOKEN_PATTERN = re.compile(r"<\|(?:im_end|im_start|endoftext|im_sep)\|>")
ARTICLES_PATTERN = re.compile(r"\b(a|an|the)\b", re.UNICODE)
PUNCTUATION = set(string.punctuation)
POLARITY_WORDS = {"yes", "no", "maybe"}
LIST_MARKERS = ("-", "*", "•")
QUOTES = ('"', "'")
NUMBERED_MARKERS = ("1.", "2.", "3.")
MAX_PREFIX_STRIPS = 3
NLTK_RESOURCES = {"wordnet": "corpora/wordnet", "punkt": "tokenizers/punkt"}
SCORER_GPU_MEMORY_UTILIZATION = 0.3
PREDICTIONS_FILE = "predictions.jsonl"
METRICS_FILE = "metrics.json"


def clean_generation(text: str) -> str:
    """Remove chat-template special tokens and surrounding whitespace.

    Args:
        text: Raw generation.

    Returns:
        The cleaned text.
    """
    return CHAT_TOKEN_PATTERN.sub("", text).strip()


def _tokenize(text: str) -> list[str]:
    """Lower-case whitespace tokenisation used by the generation metrics.

    Args:
        text: Input text.

    Returns:
        Tokens.
    """
    return text.lower().split()


def bleu(predictions: Sequence[str], references: Sequence[str]) -> dict[str, float]:
    """Corpus BLEU-1 and BLEU-4 with method-1 smoothing.

    Args:
        predictions: Generated texts.
        references: One reference per prediction.

    Returns:
        ``bleu1`` and ``bleu4``.
    """
    hypotheses = [_tokenize(prediction) for prediction in predictions]
    reference_lists = [[_tokenize(reference)] for reference in references]
    smoothing = SmoothingFunction().method1
    return {
        BLEU1: corpus_bleu(reference_lists, hypotheses, weights=(1, 0, 0, 0), smoothing_function=smoothing),
        BLEU4: corpus_bleu(reference_lists, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing),
    }


def rouge(predictions: Sequence[str], references: Sequence[str]) -> dict[str, float]:
    """Mean ROUGE-1 and ROUGE-L F-measure with Porter stemming.

    Args:
        predictions: Generated texts.
        references: One reference per prediction.

    Returns:
        ``rouge1`` and ``rougeL``.
    """
    scorer = rouge_scorer.RougeScorer([ROUGE1, ROUGEL], use_stemmer=True)
    scores = [
        scorer.score(reference, prediction.strip())
        for prediction, reference in zip(predictions, references, strict=True)
    ]
    return {
        ROUGE1: float(np.mean([score[ROUGE1].fmeasure for score in scores])),
        ROUGEL: float(np.mean([score[ROUGEL].fmeasure for score in scores])),
    }


def token_f1(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Mean word-overlap F1 on lower-cased whitespace tokens.

    Args:
        predictions: Generated texts.
        references: One reference per prediction.

    Returns:
        The mean F1.
    """
    scores = []
    for prediction, reference in zip(predictions, references, strict=True):
        predicted, gold = _tokenize(prediction), _tokenize(reference)
        if not predicted or not gold:
            scores.append(0.0)
            continue
        overlap = sum(min(predicted.count(token), gold.count(token)) for token in set(predicted) & set(gold))
        if overlap == 0:
            scores.append(0.0)
            continue
        precision, recall = overlap / len(predicted), overlap / len(gold)
        scores.append(2 * precision * recall / (precision + recall))
    return float(np.mean(scores))


def nltk_resource_available(resource: str) -> bool:
    """Whether an NLTK resource is installed, unpacked or as a zip archive.

    Args:
        resource: Resource path such as ``corpora/wordnet``.

    Returns:
        True when NLTK can find it.
    """
    for candidate in (resource, f"{resource}.zip"):
        try:
            nltk.data.find(candidate)
        except LookupError:
            continue
        return True
    return False


def ensure_nltk_resources() -> None:
    """Download the NLTK data METEOR needs when it is not installed.

    Raises:
        LookupError: If a resource is still missing afterwards, e.g. on a machine without network access.
    """
    for name, resource in NLTK_RESOURCES.items():
        if not nltk_resource_available(resource):
            nltk.download(name, quiet=True)
        if not nltk_resource_available(resource):
            raise LookupError(
                f"METEOR needs the NLTK resource {name!r}; install it with `python -m nltk.downloader {name}`."
            )


def meteor(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Mean single-reference METEOR on lower-cased whitespace tokens.

    Args:
        predictions: Generated texts.
        references: One reference per prediction.

    Returns:
        The mean METEOR score.
    """
    ensure_nltk_resources()
    scores = [
        single_meteor_score(_tokenize(reference), _tokenize(prediction))
        for prediction, reference in zip(predictions, references, strict=True)
    ]
    return float(np.mean(scores))


def generation_metrics(
    predictions: Sequence[str],
    references: Sequence[str],
    names: Sequence[str] = GENERATION_METRICS,
) -> dict[str, float]:
    """BLEU-1/4, ROUGE-1/L, token F1 and METEOR, computing only what is requested.

    Args:
        predictions: Generated texts.
        references: One reference per prediction.
        names: Metric names to compute, a subset of ``GENERATION_METRICS``.

    Returns:
        The requested metrics.
    """
    scores: dict[str, float] = {}
    if {BLEU1, BLEU4} & set(names):
        scores.update(bleu(predictions, references))
    if {ROUGE1, ROUGEL} & set(names):
        scores.update(rouge(predictions, references))
    if F1 in names:
        scores[F1] = token_f1(predictions, references)
    if METEOR in names:
        scores[METEOR] = meteor(predictions, references)
    return {name: scores[name] for name in names}


def normalize_answer(text: str) -> str:
    """SQuAD answer normalisation: lower-case, drop punctuation and articles, collapse whitespace.

    Args:
        text: Input text.

    Returns:
        The normalised text.
    """
    if not text:
        return ""
    text = "".join(character for character in text.lower() if character not in PUNCTUATION)
    return " ".join(ARTICLES_PATTERN.sub(" ", text).split())


def exact_match(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Fraction of predictions equal to their reference after ``normalize_answer``.

    Args:
        predictions: Generated texts.
        references: One reference per prediction.

    Returns:
        The exact-match rate.
    """
    matches = [
        normalize_answer(prediction) == normalize_answer(reference)
        for prediction, reference in zip(predictions, references, strict=True)
    ]
    return float(np.mean(matches))


def first_polarity(text: str) -> str | None:
    """Return the leading yes/no/maybe of a text, skipping list markers and quotes.

    Args:
        text: Input text.

    Returns:
        ``yes``, ``no`` or ``maybe``, or None when the first word is none of them.
    """
    if not text:
        return None
    text = text.strip()
    for _ in range(MAX_PREFIX_STRIPS):
        if not text:
            return None
        if text[0] in LIST_MARKERS:
            text = text[1:].lstrip()
        elif text[0] in QUOTES:
            text = text[1:]
        elif text[:2] in NUMBERED_MARKERS:
            text = text[2:].lstrip()
        else:
            break
    if not text:
        return None
    head = text.split()[0].rstrip(".,!?:;\"'").lower()
    return head if head in POLARITY_WORDS else None


def polarity_accuracy(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Fraction of predictions whose first word matches the reference's yes/no/maybe.

    References without a leading yes/no/maybe are skipped.

    Args:
        predictions: Generated texts.
        references: One reference per prediction.

    Returns:
        The accuracy (0 when no reference has a polarity).
    """
    pairs = [
        (first_polarity(prediction), gold)
        for prediction, reference in zip(predictions, references, strict=True)
        if (gold := first_polarity(reference)) is not None
    ]
    return sum(predicted == gold for predicted, gold in pairs) / len(pairs) if pairs else 0.0


def cosine_similarity(vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between a vector and each row of a matrix.

    Args:
        vector: ``[D]``.
        matrix: ``[N, D]``.

    Returns:
        ``[N]`` similarities.
    """
    vector = vector / (np.linalg.norm(vector) + 1e-8)
    matrix = matrix / (np.linalg.norm(matrix, axis=-1, keepdims=True) + 1e-8)
    return vector @ matrix.T


def choice_index(choices: Sequence[str], reference: str) -> int:
    """Return the position of the reference answer among a row's choices, ignoring surrounding whitespace.

    Args:
        choices: The row's answer choices.
        reference: The reference answer.

    Returns:
        The index of the first choice equal to ``reference``.

    Raises:
        ValueError: If no choice equals the reference.
    """
    stripped = [choice.strip() for choice in choices]
    if reference.strip() not in stripped:
        raise ValueError(f"Reference answer {reference!r} is not one of the choices {list(choices)}.")
    return stripped.index(reference.strip())


def choice_predictions(
    prediction_embeddings: np.ndarray,
    choice_embeddings: Sequence[np.ndarray],
    choices: Sequence[Sequence[str]],
) -> list[int]:
    """Pick, for every prediction, the most cosine-similar non-empty choice.

    Args:
        prediction_embeddings: ``[N, D]`` embeddings of the generations.
        choice_embeddings: Per example, ``[num_nonempty_choices, D]`` embeddings of its non-empty choices, in order.
        choices: Per example, the choice texts; empty strings are never selected.

    Returns:
        The index (into ``choices``) of the selected choice per example.
    """
    selected = []
    for embedding, candidates, texts in zip(prediction_embeddings, choice_embeddings, choices, strict=True):
        similarities = np.full(len(texts), -np.inf)
        similarities[[index for index, text in enumerate(texts) if text.strip()]] = cosine_similarity(
            embedding, candidates
        )
        selected.append(int(np.argmax(similarities)))
    return selected


def choice_accuracy(
    predictions: Sequence[str],
    references: Sequence[str],
    choices: Sequence[Sequence[str]],
    engine: "LLM",
    hard: Sequence[bool] | None = None,
) -> dict[str, float]:
    """Embedding nearest-neighbour accuracy over answer choices.

    Each generation and each non-empty choice is embedded with the encoder; the prediction is the most similar
    choice and it is correct when it is the reference's position in ``choices``.

    Args:
        predictions: Generated texts.
        references: Reference answers, each one of its row's choices.
        choices: Answer choices per example.
        engine: Embedding engine from ``memorilla.embeddings.load_encoder``.
        hard: Optional per-example flags; when given, ``accuracy_hard`` averages over the flagged rows.

    Returns:
        ``accuracy`` and, when ``hard`` is given, ``accuracy_hard``.
    """
    nonempty = [[choice for choice in row if choice.strip()] for row in choices]
    vectors = embed_texts(
        engine, [prediction.strip() for prediction in predictions] + list(chain.from_iterable(nonempty))
    )

    choice_vectors, offset = [], len(predictions)
    for row in nonempty:
        choice_vectors.append(vectors[offset : offset + len(row)])
        offset += len(row)

    selected = choice_predictions(vectors[: len(predictions)], choice_vectors, choices)
    correct = np.array(
        [
            choice_index(row, reference) == index
            for row, reference, index in zip(choices, references, selected, strict=True)
        ]
    )
    scores = {ACCURACY: float(correct.mean())}
    if hard is not None:
        mask = np.asarray(hard, dtype=bool)
        scores[ACCURACY_HARD] = float(correct[mask].mean()) if mask.any() else 0.0
    return scores


def compute_metrics(
    scoring: str,
    metrics: Sequence[str],
    predictions: Sequence[str],
    references: Sequence[str],
    choices: Sequence[Sequence[str]] | None = None,
    hard: Sequence[bool] | None = None,
    engine: "LLM | None" = None,
) -> dict[str, float]:
    """Score predictions with one scoring method.

    Args:
        scoring: One of ``SCORINGS``.
        metrics: Metric names to report; they must belong to ``scoring``.
        predictions: Generated texts (chat tokens already removed).
        references: Reference answers.
        choices: Answer choices per example (``choice`` scoring).
        hard: Per-example hard flags (``accuracy_hard``).
        engine: Embedding engine (``choice`` scoring).

    Returns:
        The requested metrics as fractions, in the order of ``metrics``.

    Raises:
        ValueError: If the scoring method is unknown, a metric does not belong to it, or ``choice`` scoring lacks
            choices or an engine.
    """
    validate_metrics(scoring, metrics)

    scores: dict[str, float] = {}
    if scoring == CHOICE:
        if choices is None or engine is None:
            raise ValueError("Choice scoring needs the answer choices and an embedding engine.")
        scores = choice_accuracy(predictions, references, choices, engine, hard if ACCURACY_HARD in metrics else None)
    elif scoring == POLARITY:
        scores[ACCURACY] = polarity_accuracy(predictions, references)
    elif scoring == EXACT_MATCH:
        scores[EXACT_MATCH] = exact_match(predictions, references)

    requested = [name for name in metrics if name in GENERATION_METRICS]
    if requested:
        scores.update(generation_metrics(predictions, references, requested))
    return {name: scores[name] for name in metrics}


def check_scoring_inputs(benchmark: Benchmark, dataset: Dataset) -> None:
    """Check, before generating, that a split can be scored with a benchmark's metrics.

    Args:
        benchmark: Benchmark whose scoring method and metrics apply.
        dataset: Split to be evaluated.

    Raises:
        ValueError: If ``choice`` scoring finds no ``choices`` column or a reference that is not among its row's
            choices.
        LookupError: If METEOR is requested and the NLTK data it needs cannot be installed.
    """
    if benchmark.scoring == CHOICE:
        if CHOICES_COLUMN not in dataset.column_names:
            raise ValueError(f"{benchmark.name}: choice scoring needs a {CHOICES_COLUMN!r} column.")
        for choices, answer in zip(dataset[CHOICES_COLUMN], dataset[ANSWER_COLUMN], strict=True):
            choice_index(choices, str(answer))
    if METEOR in benchmark.metrics:
        ensure_nltk_resources()


def evaluation_records(dataset: Dataset, predictions: Sequence[str]) -> list[dict[str, Any]]:
    """Pair every evaluated row with its prediction.

    Args:
        dataset: Evaluated split.
        predictions: One prediction per row.

    Returns:
        Records with ``question``, ``answer`` and ``prediction``, plus ``choices`` and ``hard`` when the split has them.
    """
    optional = [column for column in (CHOICES_COLUMN, HARD_COLUMN) if column in dataset.column_names]
    columns = dataset.select_columns([QUESTION_COLUMN, ANSWER_COLUMN, *optional])
    return [
        {
            QUESTION_COLUMN: row[QUESTION_COLUMN],
            ANSWER_COLUMN: row[ANSWER_COLUMN],
            "prediction": prediction,
            **{column: row[column] for column in optional},
        }
        for row, prediction in zip(columns, predictions, strict=True)
    ]


def score_dataset(
    benchmark: Benchmark, dataset: Dataset, predictions: Sequence[str], engine: "LLM | None" = None
) -> dict[str, float]:
    """Score the predictions for one benchmark split.

    Args:
        benchmark: Benchmark whose scoring method and metrics apply.
        dataset: Evaluated split.
        predictions: One prediction per row.
        engine: Embedding engine, required for ``choice`` scoring.

    Returns:
        The benchmark's metrics as fractions.
    """
    references = [str(answer).strip() for answer in dataset[ANSWER_COLUMN]]
    choices = list(dataset[CHOICES_COLUMN]) if CHOICES_COLUMN in dataset.column_names else None
    hard = list(dataset[HARD_COLUMN]) if HARD_COLUMN in dataset.column_names else None
    return compute_metrics(benchmark.scoring, benchmark.metrics, predictions, references, choices, hard, engine)


def score_predictions(
    jobs: Sequence[tuple[Benchmark, Dataset, Sequence[str], Path]],
    encoder: str = DEFAULT_ENCODER,
    cache_dir: str | None = None,
    gpu_memory_utilization: float = SCORER_GPU_MEMORY_UTILIZATION,
) -> dict[str, dict[str, Any]]:
    """Write every benchmark's predictions, then score them, write their metrics and print a summary line each.

    All predictions are on disk before any scoring starts, so a scoring failure never loses generations. The
    embedding engine is started once, and only when a benchmark uses ``choice`` scoring.

    Args:
        jobs: ``(benchmark, dataset, predictions, output_dir)`` per benchmark.
        encoder: Encoder id used for ``choice`` scoring.
        cache_dir: Model cache directory.
        gpu_memory_utilization: Fraction of GPU memory the embedding engine may use.

    Returns:
        The ``metrics.json`` content keyed by benchmark name.
    """
    for _, dataset, predictions, output_dir in jobs:
        write_predictions(output_dir, evaluation_records(dataset, predictions))

    engine = None
    if any(benchmark.scoring == CHOICE for benchmark, _, _, _ in jobs):
        engine = load_encoder(encoder, cache_dir=cache_dir, gpu_memory_utilization=gpu_memory_utilization)

    summaries = {}
    for benchmark, dataset, predictions, output_dir in jobs:
        scores = score_dataset(benchmark, dataset, predictions, engine)
        summary = write_metrics(output_dir, scores, len(predictions))
        summaries[benchmark.name] = summary
        print(f"{benchmark.name}: {json.dumps(summary)}  ->  {output_dir}")
    return summaries


def write_predictions(output_dir: str | Path, records: list[dict[str, Any]]) -> None:
    """Write ``predictions.jsonl``.

    Args:
        output_dir: Output directory.
        records: One JSON-serialisable record per example.
    """
    write_jsonl(Path(output_dir) / PREDICTIONS_FILE, records)


def write_metrics(output_dir: str | Path, scores: dict[str, float], num_examples: int) -> dict[str, Any]:
    """Write ``metrics.json`` (percentages with two decimals and the number of examples).

    Args:
        output_dir: Output directory.
        scores: Metrics as fractions.
        num_examples: Number of evaluated examples.

    Returns:
        The content written to ``metrics.json``.
    """
    summary: dict[str, Any] = {name: round(100 * value, 2) for name, value in scores.items()}
    summary["num_examples"] = num_examples
    write_json(Path(output_dir) / METRICS_FILE, summary)
    return summary
