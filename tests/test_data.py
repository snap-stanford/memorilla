"""Tests for data access, chat formatting and collation."""

from pathlib import Path

import pytest
import torch
from transformers import PreTrainedTokenizerFast

from memorilla.data import (
    IGNORE_INDEX,
    MemoryCollator,
    MemoryDataset,
    build_messages,
    ensure_data,
    load_split,
    memory_placeholder,
    split_complete,
    split_files,
)
from memorilla.embeddings import EmbeddingStore
from memorilla.utils import MEMORY_TOKEN, prepare_tokenizer
from tests.helpers import make_rows, write_split

NUM_MEMORIES = 3
SYSTEM_PROMPT = "You are a helpful assistant"
SUFFIX = " please answer briefly"


def make_collator(
    tokenizer: PreTrainedTokenizerFast, data_repo: Path, train: bool, suffix: str = ""
) -> tuple[MemoryCollator, MemoryDataset, int]:
    """Build a collator and dataset over the ``toy/train`` split of the synthetic repository.

    Args:
        tokenizer: Tokenizer fixture; the memory token is registered on it.
        data_repo: Synthetic data repository.
        train: Whether the collator builds training batches.
        suffix: Text appended to every user turn.

    Returns:
        The collator, the dataset and the id of the memory token.
    """
    memory_token_id = prepare_tokenizer(tokenizer)
    store = EmbeddingStore("toy", "train", root=data_repo)
    dataset = MemoryDataset([("toy", "train", load_split("toy", "train", data_repo))])
    collator = MemoryCollator(
        tokenizer,
        {("toy", "train"): store},
        num_memories=NUM_MEMORIES,
        max_length=128,
        system_prompt=SYSTEM_PROMPT,
        suffix=suffix,
        train=train,
    )
    return collator, dataset, memory_token_id


def test_messages_and_placeholder() -> None:
    """The system turn carries the placeholder block; the question is stripped and the suffix appended."""
    assert memory_placeholder(2) == f"\nMemory{MEMORY_TOKEN}{MEMORY_TOKEN}Memory"
    messages = build_messages("  who wrote hamlet ", "System", 2, suffix=" ok", answer=" shakespeare ")
    assert messages == [
        {"role": "system", "content": "System" + memory_placeholder(2)},
        {"role": "user", "content": "who wrote hamlet ok"},
        {"role": "assistant", "content": "shakespeare"},
    ]
    assert len(build_messages("q", "s", 1)) == 2


def test_local_repository_access(data_repo: Path) -> None:
    """A local repository is used in place; requested columns the split lacks are skipped."""
    assert ensure_data("toy", ["train", "validation"], repo=str(data_repo)) == data_repo
    assert split_files("toy", "validation", data_repo) == []
    dataset = load_split("toy", "train", data_repo, ["collection_id", "question", "missing"])
    assert dataset.column_names == ["collection_id", "question"]
    assert dataset["collection_id"] == [row["collection_id"] for row in make_rows()]
    with pytest.raises(FileNotFoundError):
        load_split("toy", "validation", data_repo)


def test_split_completeness(tmp_path: Path) -> None:
    """A sharded split counts as present only when all of its shards are; missing shards are not loaded silently."""
    rows = make_rows()
    write_split(tmp_path, "toy", "train", rows)
    assert split_complete("toy", "train", tmp_path)
    assert not split_complete("toy", "validation", tmp_path)

    directory = tmp_path / "data" / "toy"
    (directory / "train-00000-of-00001.parquet").rename(directory / "train-00000-of-00002.parquet")
    assert not split_complete("toy", "train", tmp_path)
    assert ensure_data("toy", ["train"], repo=str(tmp_path)) == tmp_path
    with pytest.raises(FileNotFoundError, match="missing"):
        load_split("toy", "train", tmp_path)

    (directory / "train-00000-of-00002.parquet").rename(directory / "train-part.parquet")
    assert split_complete("toy", "train", tmp_path)
    assert len(load_split("toy", "train", tmp_path)) == len(rows)


def test_memory_dataset_tracks_sources(data_repo: Path) -> None:
    """Items of concatenated parts carry their config, split and local row."""
    train = load_split("toy", "train", data_repo)
    test = load_split("toy", "test", data_repo)
    dataset = MemoryDataset([("toy", "train", train), ("toy", "test", test)])
    assert len(dataset) == len(train) + len(test)
    item = dataset[len(train) + 1]
    assert (item["config"], item["split"], item["row"]) == ("toy", "test", 1)
    assert item["question"] == test[1]["question"]


def test_training_batch(tokenizer: PreTrainedTokenizerFast, data_repo: Path) -> None:
    """Placeholders, left padding, labels on answer tokens only and embeddings from the right rows."""
    collator, dataset, memory_token_id = make_collator(tokenizer, data_repo, train=True)
    items = [dataset[index] for index in range(len(dataset))]
    batch = collator(items)

    input_ids, attention_mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
    assert ((input_ids == memory_token_id).sum(dim=1) == NUM_MEMORIES).all()

    lengths = attention_mask.sum(dim=1)
    assert lengths.max() == input_ids.shape[1] and lengths.min() < input_ids.shape[1]
    for row in range(len(items)):
        padding = input_ids.shape[1] - int(lengths[row])
        assert attention_mask[row, :padding].sum() == 0 and attention_mask[row, padding:].all()
        assert (input_ids[row, :padding] == tokenizer.pad_token_id).all()

        supervised = labels[row][labels[row] != IGNORE_INDEX].tolist()
        assert supervised == tokenizer.encode(items[row]["answer"] + "<|im_end|>", add_special_tokens=False)
        assert (labels[row, :padding] == IGNORE_INDEX).all()

    store = collator.stores[("toy", "train")]
    assert batch["question_embeds"].dtype == torch.bfloat16
    assert batch["doc_embeds"].dtype == torch.bfloat16
    for row, item in enumerate(items):
        assert torch.equal(batch["question_embeds"][row], store.question(row).to(torch.bfloat16))
        documents = store.documents(item["collection_id"])
        count = documents.shape[0]
        assert torch.equal(batch["doc_embeds"][row, :count], documents.to(torch.bfloat16))
        assert not batch["doc_padding_mask"][row, :count].any()
        assert batch["doc_padding_mask"][row, count:].all()
        assert (batch["doc_embeds"][row, count:] == 0).all()


def test_generation_batch_with_suffix(tokenizer: PreTrainedTokenizerFast, data_repo: Path) -> None:
    """Generation prompts end with the assistant header, carry the suffix and have no labels."""
    collator, dataset, _ = make_collator(tokenizer, data_repo, train=False, suffix=SUFFIX)
    plain, _, _ = make_collator(tokenizer, data_repo, train=False)
    items = [dataset[index] for index in range(len(dataset))]
    batch, reference = collator(items), plain(items)

    assert "labels" not in batch
    header = tokenizer.encode("<|im_start|> assistant", add_special_tokens=False)
    for row in range(len(items)):
        ids = batch["input_ids"][row][batch["attention_mask"][row].bool()].tolist()
        assert ids[-len(header) :] == header
        assert tokenizer.decode(ids).count("please answer briefly") == 1
    assert torch.equal(batch["question_embeds"], reference["question_embeds"])


def test_truncation_respects_max_length(tokenizer: PreTrainedTokenizerFast, data_repo: Path) -> None:
    """Tokenised examples never exceed ``max_length``."""
    collator, dataset, _ = make_collator(tokenizer, data_repo, train=True)
    collator.max_length = 12
    batch = collator([dataset[0], dataset[1]])
    assert batch["input_ids"].shape[1] == 12
