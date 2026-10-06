"""CPU tests of the PersonalizationV4 pipeline: generation with a mocked OpenAI client and dataset building."""

import csv
import math
from pathlib import Path
import random
import threading
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest

from personalizationv4 import generate, prompts
from personalizationv4.build_dataset import build, parse_user_range, read_questions, split_questions
from personalizationv4.generate import CHOICE_COLUMNS, GenerationConfig, UserGenerator, parse_questions, shuffle_choices

PERSONA = "Mara is a violin teacher in Lisbon who runs before dawn and keeps a secret recipe archive."
CATEGORIES_RESPONSE = "1. Teaching Style & Students\n2. Running Routine\n"
TOPICS_RESPONSE = "\n".join(f"{index + 1}. Topic number {index}." for index in range(12))
QUESTION_BLOCK = """<QUESTION>
{stem}
<CHOICE_A>
Option alpha for {stem}
<CHOICE_B>
Option bravo for {stem}
<CHOICE_C>
Option charlie for {stem}
<CHOICE_D>
Option delta for {stem}
<CHOICE_E>
Option echo for {stem}
<CORRECT_CHOICE>
{letter}
<RATIONALE>
Because of the persona.
<WORD_COUNTS>
A: 4 | B: 4 | C: 4 | D: 4 | E: 4
"""
MALFORMED_BLOCK = "<QUESTION>\nA question without choices or an answer.\n"


class FakeResponses:
    """Stands in for ``client.responses``: answers each prompt by its template and records the calls."""

    def __init__(self) -> None:
        """Initialise an empty call log."""
        self.calls: list[tuple[str, str]] = []
        self.lock = threading.Lock()

    def create(self, model: str, input: list[dict[str, str]], reasoning: dict[str, str]) -> SimpleNamespace:
        """Return a canned response for the prompt in ``input``.

        Args:
            model: Requested model.
            input: Chat messages; the first holds the prompt.
            reasoning: Reasoning options.

        Returns:
            An object with ``output_text``.
        """
        prompt = input[0]["content"]
        with self.lock:
            self.calls.append((model, reasoning["effort"]))
        if prompt.startswith(prompts.CATEGORIES_TEMPLATE[:40]):
            return SimpleNamespace(output_text=CATEGORIES_RESPONSE)
        if prompt.startswith(prompts.TOPICS_TEMPLATE[:40]):
            return SimpleNamespace(output_text=TOPICS_RESPONSE)
        if prompt.startswith(prompts.CHAT_TEMPLATE[:40]):
            topic = prompt.split("2. **Scenario:**\n", 1)[1].split("\n", 1)[0]
            return SimpleNamespace(output_text=f"  Leo: Help with {topic}\nAssistant: Sure.  \n")
        category = prompt.split("following category:\n**", 1)[1].split("**", 1)[0]
        blocks = [QUESTION_BLOCK.format(stem=f"{category} question {index}?", letter="BD"[index]) for index in range(2)]
        return SimpleNamespace(output_text="\n".join(blocks) + MALFORMED_BLOCK)


@pytest.fixture
def config() -> GenerationConfig:
    """Build a tiny generation config.

    Returns:
        The config.
    """
    return GenerationConfig(num_categories=2, num_topics=12, num_questions=2, max_workers=4, max_retries=1)


@pytest.fixture
def persona_path(tmp_path: Path) -> Path:
    """Write one fake persona profile.

    Args:
        tmp_path: Temporary directory.

    Returns:
        The persona file.
    """
    path = tmp_path / "personas" / "user_3.txt"
    path.parent.mkdir()
    path.write_text(PERSONA, encoding="utf-8")
    return path


def read_table(path: Path) -> list[dict[str, str]]:
    """Read a generated ``qa.csv``.

    Args:
        path: The question table.

    Returns:
        One dict per row.
    """
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_generate_writes_every_step(tmp_path: Path, persona_path: Path, config: GenerationConfig) -> None:
    """All five steps write their outputs with the configured models."""
    responses = FakeResponses()
    user_dir = tmp_path / "users" / "user_3"
    assert UserGenerator(SimpleNamespace(responses=responses), config).run(persona_path, user_dir)

    categories = (user_dir / "categories.txt").read_text(encoding="utf-8").split("\n")
    topics = (user_dir / "chat_topics.txt").read_text(encoding="utf-8").split("\n")
    assert categories == ["Teaching Style & Students", "Running Routine"]
    assert topics == [f"Topic number {index}." for index in range(12)]

    for index, topic in enumerate(topics):
        assert (user_dir / "chats" / f"{index}.txt").read_text(
            encoding="utf-8"
        ) == f"Leo: Help with {topic}\nAssistant: Sure."
    assert sorted(path.name for path in (user_dir / "qa").iterdir()) == ["0.txt", "1.txt"]

    rows = read_table(user_dir / "qa.csv")
    assert len(rows) == 4
    assert [row["index"] for row in rows] == ["0", "1", "2", "3"]
    assert [row["category"] for row in rows] == ["0", "0", "1", "1"]
    for row in rows:
        stem = row["question"]
        correct = row[f"choice_{row['correct_choice'].lower()}"]
        assert correct == f"Option {'bravo' if stem.endswith('0?') else 'delta'} for {stem}"
        assert sorted(row[column] for column in CHOICE_COLUMNS) == sorted(
            f"Option {word} for {stem}" for word in ("alpha", "bravo", "charlie", "delta", "echo")
        )

    assert responses.calls.count((config.chat_model, config.chat_reasoning_effort)) == 12
    assert responses.calls.count((config.model, config.reasoning_effort)) == 4


def test_generate_is_resumable_and_deterministic(tmp_path: Path, persona_path: Path, config: GenerationConfig) -> None:
    """A rerun makes no requests, missing files are regenerated alone, and the question table is reproducible."""
    user_dir = tmp_path / "user_3"
    UserGenerator(SimpleNamespace(responses=FakeResponses()), config).run(persona_path, user_dir)
    first_table = (user_dir / "qa.csv").read_text(encoding="utf-8")

    responses = FakeResponses()
    assert UserGenerator(SimpleNamespace(responses=responses), config).run(persona_path, user_dir)
    assert responses.calls == []

    (user_dir / "chats" / "5.txt").unlink()
    (user_dir / "qa.csv").unlink()
    assert UserGenerator(SimpleNamespace(responses=responses), config).run(persona_path, user_dir)
    assert responses.calls == [(config.chat_model, config.chat_reasoning_effort)]
    assert (user_dir / "qa.csv").read_text(encoding="utf-8") == first_table


def test_generate_reports_failed_requests(tmp_path: Path, persona_path: Path, config: GenerationConfig) -> None:
    """A failing chat request leaves the user incomplete without writing the question table."""

    class FlakyResponses(FakeResponses):
        """Fails every chat request."""

        def create(self, model: str, input: list[dict[str, str]], reasoning: dict[str, str]) -> SimpleNamespace:
            """Raise for chats, answer everything else.

            Args:
                model: Requested model.
                input: Chat messages; the first holds the prompt.
                reasoning: Reasoning options.

            Returns:
                An object with ``output_text``.

            Raises:
                ConnectionError: For every chat request.
            """
            if model == config.chat_model:
                raise ConnectionError("unavailable")
            return super().create(model, input, reasoning)

    user_dir = tmp_path / "user_3"
    assert not UserGenerator(SimpleNamespace(responses=FlakyResponses()), config).run(persona_path, user_dir)
    assert not (user_dir / "qa.csv").exists()
    assert not list((user_dir / "chats").glob("*.txt"))


def test_generate_reports_failed_list_requests(tmp_path: Path, persona_path: Path, config: GenerationConfig) -> None:
    """A failing category request marks the user incomplete without raising, and a rerun completes it."""

    class NoCategories(FakeResponses):
        """Fails every category request."""

        def create(self, model: str, input: list[dict[str, str]], reasoning: dict[str, str]) -> SimpleNamespace:
            """Raise for categories, answer everything else.

            Args:
                model: Requested model.
                input: Chat messages; the first holds the prompt.
                reasoning: Reasoning options.

            Returns:
                An object with ``output_text``.

            Raises:
                ConnectionError: For every category request.
            """
            if input[0]["content"].startswith(prompts.CATEGORIES_TEMPLATE[:40]):
                raise ConnectionError("unavailable")
            return super().create(model, input, reasoning)

    user_dir = tmp_path / "user_3"
    assert not UserGenerator(SimpleNamespace(responses=NoCategories()), config).run(persona_path, user_dir)
    assert not (user_dir / "categories.txt").exists()
    assert not (user_dir / "qa.csv").exists()

    assert UserGenerator(SimpleNamespace(responses=FakeResponses()), config).run(persona_path, user_dir)
    assert (user_dir / "qa.csv").exists()


def test_parse_and_shuffle_keep_answers() -> None:
    """Blocks without a correct letter are dropped and the correct text survives the shuffle."""
    text = QUESTION_BLOCK.format(stem="Stem?", letter="C") + MALFORMED_BLOCK
    records = parse_questions(text)
    assert len(records) == 2
    assert records[0]["choice_c"] == "Option charlie for Stem?"
    assert records[0]["rationale"].startswith("Because of the persona.")

    shuffled = shuffle_choices(records, random.Random(1))
    assert len(shuffled) == 1
    assert shuffled[0][f"choice_{shuffled[0]['correct_choice'].lower()}"] == "Option charlie for Stem?"


def write_user(users_dir: Path, user_id: int, num_questions: int, num_chats: int) -> None:
    """Write one synthetic user in the generation layout.

    Args:
        users_dir: Directory of ``user_N/`` folders.
        user_id: User id.
        num_questions: Rows of the user's ``qa.csv``.
        num_chats: Chat files of the user.
    """
    user_dir = users_dir / f"user_{user_id}"
    (user_dir / "chats").mkdir(parents=True)
    for index in range(num_chats):
        (user_dir / "chats" / f"{index}.txt").write_text(
            f"Leo: chat {index} of {user_id}\nAssistant: ok", encoding="utf-8"
        )
    with (user_dir / "qa.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["index", "category", "question", *CHOICE_COLUMNS, "correct_choice", "rationale"])
        for index in range(num_questions):
            choices = [f"u{user_id} q{index} {letter}" for letter in "abcde"]
            if index == 1:
                choices[4] = ""
            writer.writerow([index, 0, f"u{user_id} q{index}?", *choices, "ABCDE"[index % 4], "why"])


def test_build_dataset_splits(tmp_path: Path) -> None:
    """Training users are split per user, evaluation users keep all rows, and hard flags land on test rows."""
    users_dir = tmp_path / "users"
    sizes = {1: 12, 2: 7, 4: 21, 5: 9, 9: 3}
    for user_id, num_questions in sizes.items():
        write_user(users_dir, user_id, num_questions, num_chats=12)
    hard_path = tmp_path / "hard.csv"
    hard_path.write_text("user_id,question_index\n4,2\n5,0\n", encoding="utf-8")

    counts = build(
        users_dir, tmp_path / "data", parse_user_range("1-3"), parse_user_range("4-5"), 0.1, 23, hard_path, "user_id"
    )
    tables = {split: pq.read_table(tmp_path / "data" / f"{split}.parquet").to_pylist() for split in counts}

    assert counts == {"train": 12 + 7 - 2 - 1, "validation": 3, "test": 21 + 9}
    assert pq.read_schema(tmp_path / "data" / "train.parquet").names == [
        "user_id",
        "question",
        "answer",
        "choices",
        "documents",
        "hard",
    ]

    for user_id in (1, 2):
        train_positions, held_out_positions = split_questions(sizes[user_id], 0.1, 23)
        assert len(held_out_positions) == math.ceil(0.1 * sizes[user_id])
        train_rows = [row for row in tables["train"] if row["user_id"] == user_id]
        held_out_rows = [row for row in tables["validation"] if row["user_id"] == user_id]
        assert [row["question"] for row in train_rows] == [f"u{user_id} q{position}?" for position in train_positions]
        assert [row["question"] for row in held_out_rows] == [
            f"u{user_id} q{position}?" for position in held_out_positions
        ]

    test_rows = tables["test"]
    assert [row["question"] for row in test_rows[:21]] == [f"u4 q{index}?" for index in range(21)]
    assert [(row["user_id"], row["question"]) for row in test_rows if row["hard"]] == [(4, "u4 q2?"), (5, "u5 q0?")]
    assert not any(row["hard"] for row in tables["train"] + tables["validation"])

    with (tmp_path / "data" / "user_splits.csv").open(encoding="utf-8", newline="") as handle:
        user_splits = list(csv.DictReader(handle))
    assert [(row["user_id"], row["role"], row["questions"]) for row in user_splits] == [
        ("1", "training", "12"),
        ("2", "training", "7"),
        ("4", "evaluation", "21"),
        ("5", "evaluation", "9"),
    ]
    assert [(row["train"], row["validation"], row["test"], row["hard"]) for row in user_splits] == [
        ("10", "2", "0", "0"),
        ("6", "1", "0", "0"),
        ("0", "0", "21", "1"),
        ("0", "0", "9", "1"),
    ]

    row = test_rows[1]
    assert row["choices"] == ["u4 q1 a", "u4 q1 b", "u4 q1 c", "u4 q1 d", ""]
    assert row["answer"] == "u4 q1 b"
    assert row["documents"] == [f"Leo: chat {index} of 4\nAssistant: ok" for index in range(12)]


def test_build_dataset_rejects_unknown_hard_questions(tmp_path: Path) -> None:
    """Hard-subset entries must point to evaluation questions."""
    write_user(tmp_path / "users", 1, 10, num_chats=2)
    write_user(tmp_path / "users", 4, 10, num_chats=2)
    hard_path = tmp_path / "hard.csv"
    hard_path.write_text("user_id,question_index\n1,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hard-subset"):
        build(tmp_path / "users", tmp_path / "data", range(1, 2), range(4, 5), 0.1, 23, hard_path, "user_id")


def test_split_rule_matches_datasets_train_test_split() -> None:
    """The per-user split equals ``datasets.Dataset.train_test_split`` with the same fraction and seed."""
    datasets = pytest.importorskip("datasets")
    for num_questions in (7, 10, 143, 158):
        dataset = datasets.Dataset.from_dict({"position": list(range(num_questions))})
        split = dataset.train_test_split(test_size=0.1, seed=23)
        assert split_questions(num_questions, 0.1, 23) == (split["train"]["position"], split["test"]["position"])


def test_generated_user_builds(tmp_path: Path, persona_path: Path, config: GenerationConfig) -> None:
    """Generation output is directly consumable by the dataset builder."""
    UserGenerator(SimpleNamespace(responses=FakeResponses()), config).run(persona_path, tmp_path / "users" / "user_3")
    questions = read_questions(tmp_path / "users" / "user_3" / "qa.csv")
    counts = build(tmp_path / "users", tmp_path / "data", range(1, 3), range(3, 4), 0.1, 23, None, "user_id")
    assert counts == {"train": 0, "validation": 0, "test": len(questions)}
    rows = pq.read_table(tmp_path / "data" / "test.parquet").to_pylist()
    assert [row["answer"] for row in rows] == [question.answer for question in questions]
    assert len(rows[0]["documents"]) == 12


def test_generate_cli_selects_users(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The command line generates only the requested user range, in ascending order, with the lazy client factory."""
    persona_dir = tmp_path / "personas"
    persona_dir.mkdir()
    for user_id in (2, 10, 11):
        (persona_dir / f"user_{user_id}.txt").write_text(PERSONA, encoding="utf-8")
    (persona_dir / "notes.txt").write_text("not a persona", encoding="utf-8")

    responses = FakeResponses()
    monkeypatch.setattr(generate, "create_client", lambda: SimpleNamespace(responses=responses))
    monkeypatch.setattr(
        "sys.argv",
        ["generate.py", "--persona_dir", str(persona_dir), "--output_dir", str(tmp_path / "users")]
        + ["--user_start", "3", "--num_categories", "2", "--num_topics", "12", "--num_questions", "2"],
    )
    generate.main(generate.parse_args())

    assert sorted(path.name for path in (tmp_path / "users").iterdir()) == ["user_10", "user_11"]
    assert all((tmp_path / "users" / name / "qa.csv").exists() for name in ("user_10", "user_11"))
    assert len(responses.calls) == 2 * (1 + 1 + 12 + 2)
