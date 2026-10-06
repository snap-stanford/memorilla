"""Generate PersonalizationV4 users from persona profiles with the OpenAI Responses API.

Each user goes through five resumable steps: question categories, chat topics, two-turn chats, evaluation questions,
and the parsed question table with shuffled answer positions.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import random
import re
import time
from typing import TYPE_CHECKING

from personalizationv4 import prompts

if TYPE_CHECKING:
    from openai import OpenAI

LOGGER = logging.getLogger(__name__)

LETTERS = ("A", "B", "C", "D", "E")
CHOICE_COLUMNS = tuple(f"choice_{letter.lower()}" for letter in LETTERS)
QA_COLUMNS = ("index", "category", "question", *CHOICE_COLUMNS, "correct_choice", "rationale")
QA_PATTERNS = {
    "question": r"<QUESTION>\s*(.*?)(?=<CHOICE_A>|$)",
    "choice_a": r"<CHOICE_A>\s*(.*?)(?=<CHOICE_B>|$)",
    "choice_b": r"<CHOICE_B>\s*(.*?)(?=<CHOICE_C>|$)",
    "choice_c": r"<CHOICE_C>\s*(.*?)(?=<CHOICE_D>|$)",
    "choice_d": r"<CHOICE_D>\s*(.*?)(?=<CHOICE_E>|$)",
    "choice_e": r"<CHOICE_E>\s*(.*?)(?=<CORRECT_CHOICE>|$)",
    "correct_choice": r"<CORRECT_CHOICE>\s*([A-E])\b",
    "rationale": r"<RATIONALE[^>]*>\s*(.*?)$",
}
NUMBERED_ITEM = re.compile(r"^\d+\.\s*")
CORRECT_LETTER = re.compile(r"^[A-E]", re.IGNORECASE)
PERSONA_FILE = re.compile(r"^user_(\d+)\.txt$")
RETRY_DELAY_SECONDS = 2.0


@dataclass(frozen=True)
class GenerationConfig:
    """Models, sizes and seeds of one generation run.

    Attributes:
        model: Model for question categories, chat topics and questions.
        reasoning_effort: Reasoning effort for ``model``.
        chat_model: Model for the two-turn chats.
        chat_reasoning_effort: Reasoning effort for ``chat_model``.
        num_categories: Question categories requested per user.
        num_topics: Chat topics requested per user (one chat per topic).
        num_questions: Questions requested per category.
        max_workers: Concurrent requests while generating chats and questions.
        max_retries: Attempts per request before giving up.
        seed: Seed of the answer-position shuffle.
    """

    model: str = "gpt-5.1"
    reasoning_effort: str = "none"
    chat_model: str = "gpt-5-mini"
    chat_reasoning_effort: str = "minimal"
    num_categories: int = 10
    num_topics: int = 200
    num_questions: int = 15
    max_workers: int = 50
    max_retries: int = 3
    seed: int = 42


def create_client() -> "OpenAI":
    """Create an OpenAI client configured from the environment (``OPENAI_API_KEY``).

    Returns:
        An ``openai.OpenAI`` client.

    Raises:
        ImportError: If the ``openai`` package is not installed.
    """
    try:
        from openai import OpenAI
    except ImportError as error:
        raise ImportError("PV4 generation needs the openai package: pip install -e '.[personalizationv4]'") from error
    return OpenAI()


def complete(client: "OpenAI", prompt: str, model: str, reasoning_effort: str, max_retries: int) -> str:
    """Send one user prompt to the Responses API and return the output text.

    Args:
        client: OpenAI client.
        prompt: User message.
        model: Model name.
        reasoning_effort: Reasoning effort passed to the model.
        max_retries: Attempts before giving up; failed or empty responses are retried with exponential backoff.

    Returns:
        The non-empty output text.

    Raises:
        RuntimeError: If every attempt fails or returns an empty response.
    """
    for attempt in range(max_retries):
        try:
            response = client.responses.create(
                model=model,
                input=[{"role": "user", "content": prompt}],
                reasoning={"effort": reasoning_effort},
            )
            if response.output_text.strip():
                return response.output_text
            LOGGER.warning("Empty response from %s (attempt %d/%d)", model, attempt + 1, max_retries)
        except Exception as error:
            LOGGER.warning("Request to %s failed (attempt %d/%d): %s", model, attempt + 1, max_retries, error)
        if attempt + 1 < max_retries:
            time.sleep(RETRY_DELAY_SECONDS * 2**attempt)
    raise RuntimeError(f"No response from {model} after {max_retries} attempts")


def parse_numbered_list(text: str) -> list[str]:
    """Split a numbered list into its items, dropping blank lines and the ``N.`` prefixes.

    Args:
        text: Model output with one item per line.

    Returns:
        The items in order.
    """
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    return [NUMBERED_ITEM.sub("", line) for line in lines]


def parse_questions(text: str) -> list[dict[str, str]]:
    """Parse one category's question file into records with the tagged fields.

    Args:
        text: Model output made of ``<QUESTION>`` blocks.

    Returns:
        One record per block with a non-empty question; fields missing from a block are absent from its record.
    """
    records = []
    for block in text.strip().split("<QUESTION>"):
        if not block.strip():
            continue
        tagged = "<QUESTION>" + block
        record = {}
        for field, pattern in QA_PATTERNS.items():
            match = re.search(pattern, tagged, re.DOTALL)
            if match:
                record[field] = match.group(1).strip()
        if record.get("question"):
            records.append(record)
    return records


def shuffle_choices(records: list[dict[str, str]], rng: random.Random) -> list[dict[str, str]]:
    """Move each correct answer to a random letter and shuffle the distractors around it.

    Records without a valid correct letter or without text for the correct choice are dropped.

    Args:
        records: Parsed question records in a fixed order.
        rng: Random generator shared across all records of a user.

    Returns:
        The kept records with shuffled choices; missing distractors are empty strings.
    """
    shuffled = []
    for record in records:
        match = CORRECT_LETTER.match(record.get("correct_choice", ""))
        if match is None:
            continue
        correct = record.get(f"choice_{match.group(0).lower()}", "")
        if not correct:
            continue

        new_letter = rng.choice(LETTERS)
        choices = [record.get(column, "") for column in CHOICE_COLUMNS]
        choices.remove(correct)
        rng.shuffle(choices)
        choices.insert(LETTERS.index(new_letter), correct)

        shuffled.append({**record, **dict(zip(CHOICE_COLUMNS, choices, strict=True)), "correct_choice": new_letter})
    return shuffled


def build_question_table(qa_dir: Path, num_categories: int, seed: int) -> list[dict[str, str | int]]:
    """Parse every category file of a user and shuffle answer positions.

    Args:
        qa_dir: Directory holding ``{category}.txt`` question files.
        num_categories: Number of category files, read in category order.
        seed: Seed of the answer-position shuffle.

    Returns:
        Rows with the ``QA_COLUMNS`` fields, numbered by ``index`` in order.
    """
    records = []
    for category in range(num_categories):
        text = (qa_dir / f"{category}.txt").read_text(encoding="utf-8")
        records.extend({**record, "category": category} for record in parse_questions(text))

    rows = shuffle_choices(records, random.Random(seed))
    return [{column: row.get(column, "") for column in QA_COLUMNS} | {"index": index} for index, row in enumerate(rows)]


def write_text(path: Path, text: str) -> None:
    """Write a UTF-8 text file atomically so interrupted runs never leave partial outputs.

    Args:
        path: Destination file.
        text: File contents.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_question_table(path: Path, rows: list[dict[str, str | int]]) -> None:
    """Write the question table as CSV atomically.

    Args:
        path: Destination ``qa.csv``.
        rows: Rows with the ``QA_COLUMNS`` fields.
    """
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=QA_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


class UserGenerator:
    """Runs the five generation steps for one user, reusing any output that already exists."""

    def __init__(self, client: "OpenAI", config: GenerationConfig) -> None:
        """Initialise the generator.

        Args:
            client: OpenAI client (or any object exposing ``responses.create``).
            config: Models, sizes and seeds.
        """
        self.client = client
        self.config = config

    def run(self, persona_path: Path, user_dir: Path) -> bool:
        """Generate every output of one user under ``user_dir``.

        Args:
            persona_path: Persona profile text file.
            user_dir: Output directory of this user.

        Returns:
            True when every output exists and ``qa.csv`` is written; False when some requests failed and the user
            needs another run.
        """
        persona = persona_path.read_text(encoding="utf-8")

        try:
            categories = self.generate_list(
                user_dir / "categories.txt", prompts.categories_prompt(persona, self.config.num_categories)
            )
            topics = self.generate_list(
                user_dir / "chat_topics.txt", prompts.topics_prompt(persona, self.config.num_topics)
            )
        except RuntimeError as error:
            LOGGER.error("%s: %s", user_dir.name, error)
            return False

        chat_jobs = {
            user_dir / "chats" / f"{index}.txt": prompts.chat_prompt(persona, topic)
            for index, topic in enumerate(topics)
        }
        chats_complete = self.generate_files(
            chat_jobs, self.config.chat_model, self.config.chat_reasoning_effort, strip=True
        )

        question_jobs = {
            user_dir / "qa" / f"{index}.txt": prompts.questions_prompt(persona, category, self.config.num_questions)
            for index, category in enumerate(categories)
        }
        questions_complete = self.generate_files(
            question_jobs, self.config.model, self.config.reasoning_effort, strip=False
        )

        if not (chats_complete and questions_complete):
            LOGGER.warning("%s is incomplete; rerun to fill in the missing chats or questions", user_dir.name)
            return False

        table_path = user_dir / "qa.csv"
        if not table_path.exists():
            rows = build_question_table(user_dir / "qa", len(categories), self.config.seed)
            write_question_table(table_path, rows)
            LOGGER.info("%s: %d questions", user_dir.name, len(rows))
        return True

    def generate_list(self, path: Path, prompt: str) -> list[str]:
        """Return the numbered-list items stored at ``path``, generating them first if missing.

        Args:
            path: Output file with one item per line.
            prompt: Prompt that produces the numbered list.

        Returns:
            The list items.

        Raises:
            RuntimeError: If the list is missing and the request fails.
        """
        if not path.exists():
            text = complete(
                self.client, prompt, self.config.model, self.config.reasoning_effort, self.config.max_retries
            )
            write_text(path, "\n".join(parse_numbered_list(text)))
        return path.read_text(encoding="utf-8").split("\n")

    def generate_files(self, jobs: dict[Path, str], model: str, reasoning_effort: str, strip: bool) -> bool:
        """Generate one file per prompt concurrently, skipping files that already exist.

        Args:
            jobs: Output path to prompt.
            model: Model name.
            reasoning_effort: Reasoning effort.
            strip: Whether to strip surrounding whitespace from the response before writing.

        Returns:
            True when every output file exists afterwards.
        """
        pending = [(path, prompt) for path, prompt in jobs.items() if not path.exists()]

        def generate(job: tuple[Path, str]) -> bool:
            """Generate and write one file.

            Args:
                job: Output path and prompt.

            Returns:
                Whether the request succeeded and the file was written.
            """
            path, prompt = job
            try:
                text = complete(self.client, prompt, model, reasoning_effort, self.config.max_retries)
            except RuntimeError as error:
                LOGGER.error("%s: %s", path, error)
                return False
            write_text(path, text.strip() if strip else text)
            return True

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            results = list(executor.map(generate, pending))
        return all(results)


def find_personas(persona_dir: Path, user_start: int | None, user_end: int | None) -> list[tuple[int, Path]]:
    """List ``user_N.txt`` persona files in ascending user order, optionally restricted to an inclusive range.

    Args:
        persona_dir: Directory of persona profiles.
        user_start: Smallest user id to include, or None for no lower bound.
        user_end: Largest user id to include, or None for no upper bound.

    Returns:
        ``(user_id, path)`` pairs sorted by user id.
    """
    personas = []
    for path in persona_dir.iterdir():
        match = PERSONA_FILE.match(path.name)
        if match is None:
            continue
        user_id = int(match.group(1))
        if (user_start is None or user_id >= user_start) and (user_end is None or user_id <= user_end):
            personas.append((user_id, path))
    return sorted(personas)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed arguments.
    """
    defaults = GenerationConfig()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--persona_dir", type=Path, required=True, help="Directory of user_N.txt persona profiles.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory that receives one user_N/ per user.")
    parser.add_argument("--user_start", type=int, default=None, help="Smallest user id to generate (inclusive).")
    parser.add_argument("--user_end", type=int, default=None, help="Largest user id to generate (inclusive).")
    parser.add_argument("--model", default=defaults.model, help="Model for categories, topics and questions.")
    parser.add_argument("--reasoning_effort", default=defaults.reasoning_effort, help="Reasoning effort for --model.")
    parser.add_argument("--chat_model", default=defaults.chat_model, help="Model for the two-turn chats.")
    parser.add_argument(
        "--chat_reasoning_effort", default=defaults.chat_reasoning_effort, help="Reasoning effort for --chat_model."
    )
    parser.add_argument("--num_categories", type=int, default=defaults.num_categories, help="Categories per user.")
    parser.add_argument("--num_topics", type=int, default=defaults.num_topics, help="Chat topics (and chats) per user.")
    parser.add_argument("--num_questions", type=int, default=defaults.num_questions, help="Questions per category.")
    parser.add_argument("--max_workers", type=int, default=defaults.max_workers, help="Concurrent requests.")
    parser.add_argument("--max_retries", type=int, default=defaults.max_retries, help="Attempts per request.")
    parser.add_argument("--seed", type=int, default=defaults.seed, help="Seed of the answer-position shuffle.")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Generate every selected user and report the ones that need another run.

    Args:
        args: Parsed command-line arguments.

    Raises:
        FileNotFoundError: If no persona file is selected.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    config = GenerationConfig(
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        chat_model=args.chat_model,
        chat_reasoning_effort=args.chat_reasoning_effort,
        num_categories=args.num_categories,
        num_topics=args.num_topics,
        num_questions=args.num_questions,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
        seed=args.seed,
    )
    personas = find_personas(args.persona_dir, args.user_start, args.user_end)
    if not personas:
        raise FileNotFoundError(f"No user_N.txt persona files selected in {args.persona_dir}")

    generator = UserGenerator(create_client(), config)
    incomplete = []
    for user_id, persona_path in personas:
        LOGGER.info("Generating user_%d", user_id)
        if not generator.run(persona_path, args.output_dir / f"user_{user_id}"):
            incomplete.append(user_id)

    if incomplete:
        LOGGER.warning("Incomplete users (rerun the same command to resume): %s", incomplete)
    else:
        LOGGER.info("Generated %d users in %s", len(personas), args.output_dir)


if __name__ == "__main__":
    main(parse_args())
