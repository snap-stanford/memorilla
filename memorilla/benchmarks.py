"""Benchmark registry: evaluation protocol for Memorilla and for the closed-book, RAG and full-context baselines."""

from collections.abc import Sequence
from dataclasses import dataclass

CHOICE = "choice"
EXACT_MATCH = "exact_match"
POLARITY = "polarity"
GENERATION = "generation"
SCORINGS = (CHOICE, EXACT_MATCH, POLARITY, GENERATION)

ACCURACY = "accuracy"
ACCURACY_HARD = "accuracy_hard"
BLEU1 = "bleu1"
BLEU4 = "bleu4"
ROUGE1 = "rouge1"
ROUGEL = "rougeL"
F1 = "f1"
METEOR = "meteor"
GENERATION_METRICS = (BLEU1, BLEU4, ROUGE1, ROUGEL, F1, METEOR)
LAMP_METRICS = (ROUGE1, ROUGEL, F1, METEOR, BLEU1, BLEU4)
TRIVIAQA_METRICS = (EXACT_MATCH, BLEU1, BLEU4, F1, ROUGE1, ROUGEL, METEOR)
METRICS_BY_SCORING = {
    CHOICE: (ACCURACY, ACCURACY_HARD),
    EXACT_MATCH: (EXACT_MATCH, *GENERATION_METRICS),
    POLARITY: (ACCURACY,),
    GENERATION: GENERATION_METRICS,
}

RETRIEVAL_CONTEXT_LENGTH = 32_768
YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS = 32_768

PERSONA_PROMPT = (
    "You are a personalized AI assistant. Answer the question about the user based on your understanding of the user."
)
TRIVIA_PROMPT = (
    "You are a helpful assistant that answers trivia questions using a compressed memory of retrieved Wikipedia "
    "articles. Answer concisely with a short factual phrase or single word."
)
HELPFUL_PROMPT = "You are a helpful assistant."
FACT_CHECK_PROMPT = (
    "You are a fact-verification assistant. Decide whether the given claim is true or false based on your knowledge. "
    "Output exactly 'True' or 'False' with no other text."
)
BRIEF_ANSWER_PROMPT = (
    "You are a question-answering assistant. Provide a direct, brief answer in 1-2 sentences. "
    "Do not include preamble, context, or explanation."
)
PERSONALIZED_STYLE_PROMPT = (
    "You are a personalized AI assistant. Use your memory of the user's history to respond in their style and "
    "preferences."
)

TRIVIA_SUFFIX = (
    "\n\nAnswer with the exact short factual phrase that answers the question. For example:\n"
    '- "Mark Twain"\n'
    '- "1969"\n'
    '- "Mount Everest"\n'
    '- "the Pacific Ocean"\n'
    '- "carbon dioxide"\n'
    "\nYour answer:"
)
FACT_CHECK_SUFFIX = (
    '\n\nOutput ONLY "True" or "False". Do not include explanations, rephrasings, or any other text. Examples:\n'
    "\nQ: Is the following claim true or false? Claim: Albert Einstein was born in Germany.\nA: True\n"
    "\nQ: Is the following claim true or false? Claim: The Eiffel Tower is in London.\nA: False\n"
    "\nQ: Is the following claim true or false? Claim: Water boils at 100 degrees Celsius at standard atmospheric "
    "pressure.\nA: True\n"
    "\nQ: Is the following claim true or false? Claim: Mars has more moons than Jupiter.\nA: False\n"
    "\nQ: Is the following claim true or false? Claim: William Shakespeare wrote Hamlet.\nA: True\n"
    "\nA:\n"
)
POLARITY_SUFFIX = (
    '\nAnswer the research question with one of "Yes.", "No.", or "Maybe." followed by a brief justification, in '
    "the format:\n"
    '- "Yes. UDR provides an objective measurement of VUR and appears as a predictive tool of success after '
    'endoscopic injection."\n'
    '- "Yes. Serum TB level was independently associated with cardioembolic stroke."\n'
    '- "No. The results of this study cast doubt on the suggested advantage of HBO in reducing patient mortality and '
    'morbidity."\n'
    '- "No. Baseline pain intensity does not predict the outcome after an appropriate opioid titration."\n'
    '- "Maybe. The body mass index is one of the prognostic factors of stage 2 and 3a gastric cancer but not for '
    'other stages."\n'
    '- "Maybe. Limb-salvage surgery offers better gait efficiency than above-knee amputation, but does not improve '
    'perceived quality of life."\n'
    "\nYour answer:"
)


def validate_metrics(scoring: str, metrics: Sequence[str]) -> None:
    """Check that a scoring method is known and produces every requested metric.

    Args:
        scoring: One of ``SCORINGS``.
        metrics: Metric names.

    Raises:
        ValueError: If the scoring method is unknown or a metric does not belong to it.
    """
    if scoring not in SCORINGS:
        raise ValueError(f"Unknown scoring {scoring!r}; expected one of {SCORINGS}.")
    unknown = set(metrics) - set(METRICS_BY_SCORING[scoring])
    if unknown:
        raise ValueError(f"Metrics {sorted(unknown)} are not produced by {scoring!r} scoring.")


@dataclass(frozen=True)
class BaselineProtocol:
    """Prompting and context settings for the text-only baselines.

    Attributes:
        system_prompt: System prompt of the chat.
        max_new_tokens: Generation budget.
        full_context_length: vLLM context length for the full-context baseline.
        suffix: Text appended to the user turn after the question.
        yarn_factor: YaRN RoPE scaling factor applied for the full-context baseline, or None.
    """

    system_prompt: str
    max_new_tokens: int
    full_context_length: int
    suffix: str = ""
    yarn_factor: float | None = None


@dataclass(frozen=True)
class Benchmark:
    """One evaluation benchmark.

    Attributes:
        name: Registry key.
        config: Dataset config in the data repository.
        split: Evaluated split.
        system_prompt: System prompt for Memorilla evaluation.
        max_length: Prompt token limit and vLLM context length for Memorilla evaluation.
        max_new_tokens: Generation budget for Memorilla evaluation.
        scoring: One of ``SCORINGS``.
        metrics: Reported metric names, in display order.
        baseline: Protocol for the text-only baselines, or None for ad-hoc evaluation targets.
        suffix: Text appended to the user turn after the question for Memorilla evaluation.
    """

    name: str
    config: str
    split: str
    system_prompt: str
    max_length: int
    max_new_tokens: int
    scoring: str
    metrics: tuple[str, ...]
    baseline: BaselineProtocol | None = None
    suffix: str = ""

    def __post_init__(self) -> None:
        """Validate the scoring method and metric names.

        Raises:
            ValueError: If the scoring method is unknown or a metric does not belong to it.
        """
        try:
            validate_metrics(self.scoring, self.metrics)
        except ValueError as error:
            raise ValueError(f"{self.name}: {error}") from None


BENCHMARKS: dict[str, Benchmark] = {
    benchmark.name: benchmark
    for benchmark in (
        Benchmark(
            name="pmv2",
            config="pmv2",
            split="test",
            system_prompt=PERSONA_PROMPT,
            max_length=8192,
            max_new_tokens=256,
            scoring=CHOICE,
            metrics=(ACCURACY,),
            baseline=BaselineProtocol(HELPFUL_PROMPT, max_new_tokens=256, full_context_length=40_960),
        ),
        Benchmark(
            name="pv4",
            config="pv4",
            split="test",
            system_prompt=PERSONA_PROMPT,
            max_length=8192,
            max_new_tokens=256,
            scoring=CHOICE,
            metrics=(ACCURACY, ACCURACY_HARD),
            baseline=BaselineProtocol(HELPFUL_PROMPT, max_new_tokens=256, full_context_length=65_536, yarn_factor=2.0),
        ),
        Benchmark(
            name="factkg",
            config="factkg",
            split="test",
            system_prompt=PERSONA_PROMPT,
            max_length=2048,
            max_new_tokens=8,
            scoring=CHOICE,
            metrics=(ACCURACY,),
            baseline=BaselineProtocol(
                FACT_CHECK_PROMPT, max_new_tokens=8, full_context_length=8192, suffix=FACT_CHECK_SUFFIX
            ),
        ),
        Benchmark(
            name="triviaqa",
            config="triviaqa",
            split="test",
            system_prompt=TRIVIA_PROMPT,
            max_length=2048,
            max_new_tokens=64,
            scoring=EXACT_MATCH,
            metrics=TRIVIAQA_METRICS,
            baseline=BaselineProtocol(
                HELPFUL_PROMPT, max_new_tokens=64, full_context_length=131_072, suffix=TRIVIA_SUFFIX, yarn_factor=4.0
            ),
            suffix=TRIVIA_SUFFIX,
        ),
        Benchmark(
            name="narrativeqa",
            config="narrativeqa",
            split="test",
            system_prompt=PERSONA_PROMPT,
            max_length=8192,
            max_new_tokens=32,
            scoring=GENERATION,
            metrics=GENERATION_METRICS,
            baseline=BaselineProtocol(
                BRIEF_ANSWER_PROMPT, max_new_tokens=64, full_context_length=131_072, yarn_factor=4.0
            ),
        ),
        Benchmark(
            name="pubmedqa",
            config="pubmedqa",
            split="test",
            system_prompt=PERSONA_PROMPT,
            max_length=8192,
            max_new_tokens=64,
            scoring=POLARITY,
            metrics=(ACCURACY,),
            baseline=BaselineProtocol(
                HELPFUL_PROMPT, max_new_tokens=64, full_context_length=8192, suffix=POLARITY_SUFFIX
            ),
        ),
        Benchmark(
            name="lamp4",
            config="lamp4",
            split="test",
            system_prompt=PERSONA_PROMPT,
            max_length=2048,
            max_new_tokens=128,
            scoring=GENERATION,
            metrics=LAMP_METRICS,
            baseline=BaselineProtocol(PERSONALIZED_STYLE_PROMPT, max_new_tokens=128, full_context_length=32_768),
        ),
        Benchmark(
            name="lamp7",
            config="lamp7",
            split="test",
            system_prompt=PERSONA_PROMPT,
            max_length=1024,
            max_new_tokens=128,
            scoring=GENERATION,
            metrics=LAMP_METRICS,
            baseline=BaselineProtocol(PERSONALIZED_STYLE_PROMPT, max_new_tokens=128, full_context_length=8192),
        ),
    )
}


def get_benchmarks(name: str) -> list[Benchmark]:
    """Look up one benchmark, or every benchmark with ``all``.

    Args:
        name: A registry key or ``all``.

    Returns:
        The selected benchmarks in registry order.

    Raises:
        KeyError: If the name is unknown.
    """
    if name == "all":
        return list(BENCHMARKS.values())
    if name not in BENCHMARKS:
        raise KeyError(f"Unknown benchmark {name!r}; choose from {sorted(BENCHMARKS)} or 'all'.")
    return [BENCHMARKS[name]]
