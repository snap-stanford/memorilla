"""Build the PersonalizationV4 train / validation / test parquet files from per-user generation outputs.

Training users contribute a seeded 90/10 split of their questions to train and validation; evaluation users contribute
every question to test. Each row carries the user's full chat history as its documents.
"""

import argparse
import csv
from dataclasses import dataclass
import logging
import math
from pathlib import Path
import re

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from personalizationv4.generate import CHOICE_COLUMNS, LETTERS

LOGGER = logging.getLogger(__name__)

USER_DIR = re.compile(r"^user_(\d+)$")
SPLITS = ("train", "validation", "test")
TRAIN_USERS = "6-125"
TEST_USERS = "126-155"
VALIDATION_FRACTION = 0.1
SPLIT_SEED = 23
USER_SPLITS_FILE = "user_splits.csv"
USER_SPLITS_COLUMNS = ("user_id", "role", "questions", *SPLITS, "hard")


@dataclass(frozen=True)
class Question:
    """One parsed row of a user's ``qa.csv``.

    Attributes:
        index: Position of the question in ``qa.csv``.
        question: Question text.
        choices: The five candidate answers in A-E order; a missing candidate is an empty string.
        answer: Text of the correct candidate.
    """

    index: int
    question: str
    choices: list[str]
    answer: str


def parse_user_range(spec: str) -> range:
    """Parse an inclusive user-id range such as ``"6-125"`` or a single id such as ``"7"``.

    Args:
        spec: Range specification.

    Returns:
        The ids as a ``range``.
    """
    start, _, end = spec.partition("-")
    return range(int(start), int(end or start) + 1)


def find_users(users_dir: Path) -> dict[int, Path]:
    """Map user ids to their ``user_N`` directories.

    Args:
        users_dir: Directory holding one ``user_N/`` folder per user.

    Returns:
        User id to directory.
    """
    users = {}
    for path in users_dir.iterdir():
        match = USER_DIR.match(path.name)
        if match and path.is_dir():
            users[int(match.group(1))] = path
    return users


def read_questions(path: Path) -> list[Question]:
    """Read a user's question table.

    Args:
        path: The user's ``qa.csv``.

    Returns:
        Questions ordered by their ``index`` column.

    Raises:
        ValueError: If a row has no valid correct letter or the correct candidate is empty.
    """
    questions = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            choices = [row[column] for column in CHOICE_COLUMNS]
            letter = row["correct_choice"].strip().upper()
            if letter not in LETTERS or not choices[LETTERS.index(letter)]:
                raise ValueError(f"{path}: question {row['index']} has no valid correct choice")
            questions.append(Question(int(row["index"]), row["question"], choices, choices[LETTERS.index(letter)]))
    return sorted(questions, key=lambda question: question.index)


def read_chats(chats_dir: Path) -> list[str]:
    """Read a user's chats in topic order.

    Args:
        chats_dir: Directory of ``{topic}.txt`` chat files.

    Returns:
        Chat texts ordered by topic number.
    """
    paths = sorted(chats_dir.glob("*.txt"), key=lambda path: int(path.stem))
    return [path.read_text(encoding="utf-8") for path in paths]


def split_questions(num_questions: int, validation_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Split one user's question positions into training and held-out (validation) parts.

    A seeded permutation is drawn with ``numpy.random.default_rng(seed)``; its first
    ``ceil(validation_fraction * n)`` entries are held out and the rest are kept for training, both in permutation
    order. This is the rule of ``datasets.Dataset.train_test_split(test_size=validation_fraction, seed=seed)``.

    Args:
        num_questions: Number of questions of the user.
        validation_fraction: Fraction held out, rounded up.
        seed: Permutation seed, shared by every user.

    Returns:
        ``(train_positions, held_out_positions)``.
    """
    permutation = np.random.default_rng(seed).permutation(num_questions).tolist()
    num_held_out = math.ceil(validation_fraction * num_questions)
    return permutation[num_held_out:], permutation[:num_held_out]


def read_hard_subset(path: Path | None) -> set[tuple[int, int]]:
    """Read the hard-subset membership table.

    Args:
        path: CSV with ``user_id`` and ``question_index`` columns, or None for an empty subset.

    Returns:
        ``(user_id, question_index)`` pairs.
    """
    if path is None:
        return set()
    with path.open(encoding="utf-8", newline="") as handle:
        return {(int(row["user_id"]), int(row["question_index"])) for row in csv.DictReader(handle)}


def make_schema(id_column: str) -> pa.Schema:
    """Return the Arrow schema of the released parquet files.

    Args:
        id_column: Name of the user-id column.

    Returns:
        The schema.
    """
    return pa.schema(
        [
            pa.field(id_column, pa.int64()),
            pa.field("question", pa.string()),
            pa.field("answer", pa.string()),
            pa.field("choices", pa.list_(pa.string())),
            pa.field("documents", pa.list_(pa.string())),
            pa.field("hard", pa.bool_()),
        ]
    )


def make_table(
    user_id: int,
    questions: list[Question],
    documents: list[str],
    hard: set[tuple[int, int]],
    schema: pa.Schema,
) -> pa.Table:
    """Build the rows of one user for one split.

    Args:
        user_id: User id.
        questions: Questions of this user in this split, in row order.
        documents: The user's chats, attached to every row.
        hard: Hard-subset membership.
        schema: Output schema; its first field names the user-id column.

    Returns:
        An Arrow table with one row per question.
    """
    columns = [
        [user_id] * len(questions),
        [question.question for question in questions],
        [question.answer for question in questions],
        [question.choices for question in questions],
        [documents] * len(questions),
        [(user_id, question.index) in hard for question in questions],
    ]
    return pa.Table.from_arrays(
        [pa.array(values, type=field.type) for values, field in zip(columns, schema, strict=True)], schema=schema
    )


def build(
    users_dir: Path,
    output_dir: Path,
    train_users: range,
    test_users: range,
    validation_fraction: float,
    seed: int,
    hard_subset: Path | None,
    id_column: str,
) -> dict[str, int]:
    """Write ``train.parquet``, ``validation.parquet``, ``test.parquet`` and the per-user ``user_splits.csv``.

    Users present in ``users_dir`` and in ``train_users`` are split per user into train and validation; users in
    ``test_users`` go to test with all their questions in ``qa.csv`` order. Rows are grouped by ascending user id with
    one parquet row group per user. ``user_splits.csv`` lists every released user with its role (``training`` or
    ``evaluation``) and its number of questions in each split and in the hard subset.

    Args:
        users_dir: Directory of ``user_N/`` folders, each with ``qa.csv`` and ``chats/``.
        output_dir: Destination directory.
        train_users: Ids of training users.
        test_users: Ids of evaluation users.
        validation_fraction: Fraction of each training user's questions held out for validation.
        seed: Split seed.
        hard_subset: Hard-subset membership CSV, or None.
        id_column: Name of the user-id column.

    Returns:
        Number of rows written per split.

    Raises:
        ValueError: If the user ranges overlap or a hard-subset entry does not point to a test question.
    """
    if set(train_users) & set(test_users):
        raise ValueError("Training and evaluation user ranges overlap")

    users = find_users(users_dir)
    hard = read_hard_subset(hard_subset)
    schema = make_schema(id_column)
    output_dir.mkdir(parents=True, exist_ok=True)
    writers = {split: pq.ParquetWriter(output_dir / f"{split}.parquet", schema) for split in SPLITS}
    counts = dict.fromkeys(SPLITS, 0)
    test_keys = set()
    user_rows = []

    try:
        for user_id in sorted(users):
            if user_id not in train_users and user_id not in test_users:
                continue
            questions = read_questions(users[user_id] / "qa.csv")
            documents = read_chats(users[user_id] / "chats")

            if user_id in train_users:
                train_positions, held_out_positions = split_questions(len(questions), validation_fraction, seed)
                parts = {
                    "train": [questions[position] for position in train_positions],
                    "validation": [questions[position] for position in held_out_positions],
                }
            else:
                parts = {"test": questions}
                test_keys.update((user_id, question.index) for question in questions)

            for split, selected in parts.items():
                writers[split].write_table(make_table(user_id, selected, documents, hard, schema))
                counts[split] += len(selected)
            user_rows.append(
                {
                    "user_id": user_id,
                    "role": "training" if user_id in train_users else "evaluation",
                    "questions": len(questions),
                    **{split: len(parts.get(split, [])) for split in SPLITS},
                    "hard": sum((user_id, question.index) in hard for question in questions),
                }
            )
    finally:
        for writer in writers.values():
            writer.close()

    with (output_dir / USER_SPLITS_FILE).open("w", encoding="utf-8", newline="") as handle:
        splits_writer = csv.DictWriter(handle, fieldnames=USER_SPLITS_COLUMNS, lineterminator="\n")
        splits_writer.writeheader()
        splits_writer.writerows(user_rows)

    if hard - test_keys:
        raise ValueError(f"{len(hard - test_keys)} hard-subset entries do not match a test question")
    return counts


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--users_dir", type=Path, required=True, help="Directory of user_N/ generation outputs.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory for {train,validation,test}.parquet and user_splits.csv.",
    )
    parser.add_argument(
        "--hard_subset", type=Path, default=None, help="CSV of (user_id, question_index) hard questions."
    )
    parser.add_argument("--train_users", default=TRAIN_USERS, help="Inclusive id range of training users.")
    parser.add_argument("--test_users", default=TEST_USERS, help="Inclusive id range of evaluation users.")
    parser.add_argument(
        "--validation_fraction", type=float, default=VALIDATION_FRACTION, help="Validation share per training user."
    )
    parser.add_argument("--seed", type=int, default=SPLIT_SEED, help="Seed of the per-user split.")
    parser.add_argument("--id_column", default="user_id", help="Name of the user-id column.")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Build the splits and log their sizes.

    Args:
        args: Parsed command-line arguments.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    counts = build(
        users_dir=args.users_dir,
        output_dir=args.output_dir,
        train_users=parse_user_range(args.train_users),
        test_users=parse_user_range(args.test_users),
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        hard_subset=args.hard_subset,
        id_column=args.id_column,
    )
    for split, count in counts.items():
        LOGGER.info("%s: %d rows -> %s", split, count, args.output_dir / f"{split}.parquet")


if __name__ == "__main__":
    main(parse_args())
