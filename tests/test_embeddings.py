"""Tests for the embedding store: on-disk layout, reader, writer, builder and provisioning."""

import os
from pathlib import Path
import pickle
from typing import NoReturn

import numpy as np
import pytest
from safetensors.torch import load_file
import torch

from memorilla import embeddings
from memorilla.data import load_split
from memorilla.embeddings import (
    INDEX_FILE,
    QUESTIONS_FILE,
    SHARD_TEMPLATE,
    TENSOR_NAME,
    EmbeddingStore,
    EmbeddingWriter,
    build_embeddings,
    embedding_dir,
    ensure_embeddings,
    load_split_with_embeddings,
    missing_files,
    run_isolated,
)
from tests.helpers import EMBEDDING_DIM, FakeEncoder, make_rows, write_split

DIM = 4
BYTES_PER_ROW = DIM * 2


def make_collections() -> dict[int, torch.Tensor]:
    """Draw collections of varying size, including an empty one and one larger than the shard limit.

    Returns:
        Float16 document embeddings keyed by collection id.
    """
    generator = torch.Generator().manual_seed(0)
    sizes = {2: 3, 5: 1, 9: 0, 11: 4, 20: 12, 21: 2}
    return {collection_id: torch.randn(size, DIM, generator=generator).half() for collection_id, size in sizes.items()}


def write_store(directory: Path, collections: dict[int, torch.Tensor], max_rows: int) -> torch.Tensor:
    """Write collections and random questions with a shard limit of ``max_rows`` document rows.

    Args:
        directory: Split directory.
        collections: Document embeddings keyed by collection id.
        max_rows: Shard limit in document rows.

    Returns:
        The written question embeddings.
    """
    questions = torch.randn(5, DIM).half()
    with EmbeddingWriter(directory, max_shard_bytes=max_rows * BYTES_PER_ROW) as writer:
        for collection_id in sorted(collections):
            writer.add(collection_id, collections[collection_id])
        writer.write_questions(questions)
    return questions


def test_round_trip_across_shards(tmp_path: Path) -> None:
    """Every collection and question reads back bit-identically when the data spans several shards."""
    collections = make_collections()
    directory = embedding_dir("toy", "train", tmp_path)
    questions = write_store(directory, collections, max_rows=5)
    store = EmbeddingStore("toy", "train", root=tmp_path)

    assert len(sorted(directory.glob("documents-*.safetensors"))) > 2
    assert len(store) == len(collections) and store.num_questions == questions.shape[0]
    for collection_id, expected in collections.items():
        assert collection_id in store
        actual = store.documents(collection_id)
        assert actual.dtype == torch.float16 and actual.shape == expected.shape
        assert torch.equal(actual, expected)
    assert torch.equal(store.questions(), questions)
    for row in range(questions.shape[0]):
        assert torch.equal(store.question(row), questions[row])


def test_collections_stay_within_one_shard(tmp_path: Path) -> None:
    """Index entries never cross a shard boundary; an oversized collection gets a shard of its own."""
    collections = make_collections()
    write_store(tmp_path, collections, max_rows=5)
    index = load_file(str(tmp_path / INDEX_FILE))

    assert index["collection_ids"].dtype == torch.int64
    assert index["shard"].dtype == torch.int32
    assert index["start"].dtype == torch.int64 and index["length"].dtype == torch.int64
    assert torch.equal(index["collection_ids"], torch.tensor(sorted(collections)))

    for shard in index["shard"].unique().tolist():
        rows = load_file(str(tmp_path / SHARD_TEMPLATE.format(shard)))[TENSOR_NAME].shape[0]
        members = index["shard"] == shard
        ends = index["start"][members] + index["length"][members]
        assert int(ends.max()) == rows
        if int(members.sum()) > 1:
            assert rows <= 5

    oversized = int((index["collection_ids"] == 20).nonzero())
    assert int((index["shard"] == index["shard"][oversized]).sum()) == 1


def test_empty_collections_without_dimension(tmp_path: Path) -> None:
    """Collections without documents are indexed even when their array carries no embedding width."""
    with EmbeddingWriter(embedding_dir("toy", "train", tmp_path), max_shard_bytes=1024) as writer:
        writer.add(1, np.zeros((0, 0), dtype=np.float16))
        writer.add(2, torch.ones(2, DIM))
        writer.write_questions(torch.zeros(1, DIM))
    store = EmbeddingStore("toy", "train", root=tmp_path)
    assert store.documents(1).shape == (0, DIM)
    assert torch.equal(store.documents(2), torch.ones(2, DIM, dtype=torch.float16))


def test_empty_collection_after_full_shard(tmp_path: Path) -> None:
    """An empty collection that follows a full shard is indexed in a shard file that exists."""
    with EmbeddingWriter(embedding_dir("toy", "train", tmp_path), max_shard_bytes=2 * BYTES_PER_ROW) as writer:
        writer.add(1, torch.ones(5, DIM))
        writer.add(2, torch.zeros(0, DIM))
        writer.write_questions(torch.zeros(1, DIM))
    store = EmbeddingStore("toy", "train", root=tmp_path)
    assert store.documents(2).shape == (0, DIM)
    assert torch.equal(store.documents(1), torch.ones(5, DIM, dtype=torch.float16))


def test_writer_requires_ascending_ids(tmp_path: Path) -> None:
    """Out-of-order collection ids are rejected."""
    writer = EmbeddingWriter(tmp_path)
    writer.add(5, torch.zeros(1, DIM))
    with pytest.raises(ValueError):
        writer.add(5, torch.zeros(1, DIM))
    with pytest.raises(ValueError):
        writer.add(3, torch.zeros(1, DIM))


def test_store_errors_and_pickling(tmp_path: Path) -> None:
    """Missing splits and collections raise; a store with open handles pickles and keeps working."""
    with pytest.raises(FileNotFoundError):
        EmbeddingStore("toy", "missing", root=tmp_path)

    directory = embedding_dir("toy", "train", tmp_path)
    collections = make_collections()
    write_store(directory, collections, max_rows=100)
    store = EmbeddingStore("toy", "train", root=tmp_path)
    store.documents(2)
    with pytest.raises(KeyError):
        store.documents(4)
    assert 4 not in store

    restored = pickle.loads(pickle.dumps(store))
    assert torch.equal(restored.documents(11), collections[11])


def test_build_embeddings_matches_layout(tmp_path: Path, fake_encoder: FakeEncoder) -> None:
    """Built embeddings take each collection's documents once and embed stripped questions row by row."""
    rows = make_rows()
    dataset = write_split(tmp_path, "toy", "train", rows)
    out_dir = embedding_dir("toy", "train", tmp_path)
    build_embeddings(dataset, out_dir, fake_encoder, chunk_size=3)
    store = EmbeddingStore("toy", "train", root=tmp_path)

    assert store.collection_ids.tolist() == [3, 7]
    for collection_id in (3, 7):
        documents = next(row["documents"] for row in rows if row["collection_id"] == collection_id)
        expected = torch.from_numpy(np.stack([fake_encoder.vector(doc) for doc in documents])).half()
        assert torch.equal(store.documents(collection_id), expected)

    questions = store.questions()
    assert questions.shape == (len(rows), EMBEDDING_DIM)
    for row, item in enumerate(rows):
        expected = torch.from_numpy(fake_encoder.vector(item["question"].strip())).half()
        assert torch.equal(questions[row], expected)
    assert (out_dir / QUESTIONS_FILE).exists()


def test_ensure_embeddings_builds_missing_splits(
    tmp_path: Path, fake_encoder: FakeEncoder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A local repository without embeddings gets them computed once; existing splits are left alone."""
    write_split(tmp_path, "toy", "train", make_rows())
    write_split(tmp_path, "toy", "test", make_rows()[:1], fake_encoder)
    test_index = embedding_dir("toy", "test", tmp_path) / INDEX_FILE
    before = test_index.stat().st_mtime_ns

    monkeypatch.setattr(embeddings, "load_encoder", lambda *args, **kwargs: fake_encoder)
    monkeypatch.setattr(embeddings, "release_gpu_memory", lambda: None)
    monkeypatch.setattr(embeddings, "run_isolated", lambda function, *args: function(*args))
    root = ensure_embeddings("toy", ["train", "test"], repo=str(tmp_path))

    assert root == tmp_path
    assert test_index.stat().st_mtime_ns == before
    store = EmbeddingStore("toy", "train", root=tmp_path)
    assert len(store) == 2
    assert store.questions().shape[0] == len(load_split("toy", "train", tmp_path))

    calls = len(fake_encoder.calls)
    ensure_embeddings("toy", ["train", "test"], repo=str(tmp_path))
    assert len(fake_encoder.calls) == calls


def test_ensure_embeddings_rejects_absent_splits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A requested split without data files is reported before any encoder starts."""
    write_split(tmp_path, "toy", "train", make_rows())

    def compute(*args: object) -> NoReturn:
        """Stand in for the encoder process, which must not start.

        Args:
            *args: Ignored.

        Raises:
            AssertionError: Always.
        """
        raise AssertionError("the encoder was started")

    monkeypatch.setattr(embeddings, "run_isolated", compute)
    with pytest.raises(FileNotFoundError, match=r"toy splits \['validation'\]"):
        ensure_embeddings("toy", ["train", "validation"], repo=str(tmp_path))


def test_build_embeddings_rejects_empty_collections(tmp_path: Path, fake_encoder: FakeEncoder) -> None:
    """A collection without documents is an error, raised before anything is embedded."""
    rows = make_rows()
    rows[1]["documents"] = []
    dataset = write_split(tmp_path, "toy", "train", rows)
    with pytest.raises(ValueError, match=r"\[3\]"):
        build_embeddings(dataset, embedding_dir("toy", "train", tmp_path), fake_encoder)
    assert fake_encoder.calls == []


def test_missing_files_are_detected(tmp_path: Path) -> None:
    """A split whose index refers to an absent shard is incomplete and cannot be opened."""
    directory = embedding_dir("toy", "train", tmp_path)
    assert missing_files(directory) == [INDEX_FILE]
    write_store(directory, make_collections(), max_rows=5)
    assert missing_files(directory) == []

    (directory / SHARD_TEMPLATE.format(1)).unlink()
    assert missing_files(directory) == [SHARD_TEMPLATE.format(1)]
    with pytest.raises(FileNotFoundError, match=SHARD_TEMPLATE.format(1)):
        EmbeddingStore("toy", "train", root=tmp_path)


def test_split_rows_must_match_question_embeddings(data_repo: Path, fake_encoder: FakeEncoder) -> None:
    """Loading a split together with embeddings for a different number of rows raises."""
    dataset, store = load_split_with_embeddings("toy", "train", data_repo)
    assert len(dataset) == store.num_questions == len(make_rows())

    write_split(data_repo, "toy", "train", make_rows()[:2])
    with pytest.raises(ValueError, match="2 rows but 3 question embeddings"):
        load_split_with_embeddings("toy", "train", data_repo)


def test_run_isolated(tmp_path: Path) -> None:
    """The function runs in a separate process; a failure there raises in the caller."""
    run_isolated(os.mkdir, str(tmp_path / "made"))
    assert (tmp_path / "made").is_dir()
    with pytest.raises(RuntimeError, match="exit code"):
        run_isolated(os.rmdir, str(tmp_path / "absent"))
