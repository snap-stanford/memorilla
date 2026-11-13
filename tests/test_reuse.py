"""Reusable document collections with real CPU memory computation and fake serving engines."""

import importlib
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from memorilla.embeddings import EmbeddingStore
from tests.helpers import EMBEDDING_DIM, FakeEncoder
from tests.test_vllm import DOCUMENTS, QUESTION, make_model


@pytest.fixture
def model(fake_encoder: FakeEncoder, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Build a CPU model without importing or starting the optional serving runtime."""
    module = importlib.import_module("memorilla.vllm")
    monkeypatch.setattr(module, "_require_vllm", lambda: SimpleNamespace(SamplingParams=SimpleNamespace))
    return make_model(module, fake_encoder)


def test_encode_documents_returns_reusable_cpu_vectors(model: Any, fake_encoder: FakeEncoder) -> None:
    """Encoding uses the existing helper, without a separate first-document probe."""
    vectors = model.encode_documents(DOCUMENTS)
    expected = torch.from_numpy(np.stack([fake_encoder.vector(text) for text in DOCUMENTS])).to(torch.float16)
    assert vectors.shape == (len(DOCUMENTS), EMBEDDING_DIM)
    assert vectors.device.type == "cpu" and vectors.dtype == torch.float16
    assert torch.equal(vectors, expected)
    assert fake_encoder.calls == [DOCUMENTS]


def test_questions_reuse_documents_and_recompute_memory(model: Any, fake_encoder: FakeEncoder) -> None:
    """Only fresh questions reach the encoder after the document collection is encoded."""
    vectors = model.encode_documents(DOCUMENTS)
    original = vectors.clone()
    fake_encoder.calls.clear()
    seen = []
    hook = model.memory.register_forward_pre_hook(lambda _module, args: seen.append(args))
    questions = [QUESTION, "who wrote hamlet"]
    try:
        for question in questions:
            assert model.answer_from_embeddings(question, vectors, max_new_tokens=9) == "paris"
    finally:
        hook.remove()

    assert fake_encoder.calls == [[questions[0]], [questions[1]]]
    assert len(seen) == len(questions)
    for (documents, padding, query), question in zip(seen, questions, strict=True):
        assert torch.equal(documents, vectors[None].to(torch.bfloat16))
        assert not padding.any()
        expected_query = torch.from_numpy(fake_encoder.vector(question)).to(torch.float16).to(torch.bfloat16)
        assert torch.equal(query, expected_query[None])
    assert torch.equal(vectors, original)
    assert model.engine.sampling.max_tokens == 9


def test_current_collection_reaches_memory_after_updates(model: Any, fake_encoder: FakeEncoder) -> None:
    """Append, replacement and removal use supplied rows, without re-encoding old documents."""
    original = model.encode_documents(DOCUMENTS)
    added = model.encode_documents(["shakespeare wrote hamlet"])
    assert fake_encoder.calls == [DOCUMENTS, ["shakespeare wrote hamlet"]]
    appended = torch.cat([original, added])
    replaced = torch.cat([added, original[1:]])
    removed = appended[1:]
    fake_encoder.calls.clear()
    seen = []
    hook = model.memory.register_forward_pre_hook(lambda _module, args: seen.append(args[0].clone()))
    try:
        for collection in (appended, replaced, removed, original):
            model.answer_from_embeddings(QUESTION, collection)
    finally:
        hook.remove()
    for actual, expected in zip(seen, (appended, replaced, removed, original), strict=True):
        assert torch.equal(actual, expected[None].to(torch.bfloat16))
    assert fake_encoder.calls == [[QUESTION]] * 4


def test_embedding_store_collection_is_accepted(model: Any, data_repo: Any) -> None:
    """An existing store's float16 collection can be passed directly without a new storage layer."""
    store = EmbeddingStore("toy", "test", root=data_repo)
    assert model.answer_from_embeddings(QUESTION, store.documents(7)) == "paris"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_floating_collections_are_converted_without_mutation(model: Any, dtype: torch.dtype) -> None:
    """The existing generation path converts external vectors to the memory module's dtype."""
    vectors = torch.arange(2 * EMBEDDING_DIM, dtype=dtype).reshape(2, EMBEDDING_DIM)
    original = vectors.clone()
    seen = []
    hook = model.memory.register_forward_pre_hook(lambda _module, args: seen.append(args[0]))
    try:
        assert model.answer_from_embeddings(QUESTION, vectors) == "paris"
    finally:
        hook.remove()
    assert seen[0].device == model.device and seen[0].dtype == torch.bfloat16
    assert vectors.dtype == dtype and torch.equal(vectors, original)


@pytest.mark.parametrize(
    ("vectors", "error"),
    [
        ([], TypeError),
        (torch.ones(2, EMBEDDING_DIM, dtype=torch.int64), TypeError),
        (torch.ones(EMBEDDING_DIM), ValueError),
        (torch.ones(1, 2, EMBEDDING_DIM), ValueError),
        (torch.ones(0, EMBEDDING_DIM), ValueError),
        (torch.ones(2, EMBEDDING_DIM + 1), ValueError),
    ],
)
def test_invalid_vectors_fail_before_encoding(
    model: Any, fake_encoder: FakeEncoder, vectors: Any, error: type[Exception]
) -> None:
    """Bad collections fail before any encoder or decoder work."""
    with pytest.raises(error, match="embedding|Tensor"):
        model.answer_from_embeddings(QUESTION, vectors)
    assert not fake_encoder.calls and not model.engine.requests


@pytest.mark.parametrize(("documents", "error"), [([], ValueError), ("france", TypeError), ([None], TypeError)])
def test_invalid_documents_fail_before_encoding(
    model: Any, fake_encoder: FakeEncoder, documents: Any, error: type[Exception]
) -> None:
    """Empty collections and malformed document lists do not reach the encoder."""
    with pytest.raises(error):
        model.encode_documents(documents)
    assert not fake_encoder.calls


@pytest.mark.parametrize("method", ["encode_documents", "answer_from_embeddings"])
def test_reuse_methods_need_an_encoder(model: Any, method: str) -> None:
    """Reusing documents still requires an encoder for each new question."""
    model.encoder = None
    args = (DOCUMENTS,) if method == "encode_documents" else (QUESTION, torch.ones(2, EMBEDDING_DIM))
    with pytest.raises(ValueError, match="encoder"):
        getattr(model, method)(*args)


def test_incompatible_encoder_width_is_reported(model: Any) -> None:
    """An incorrectly configured document or question encoder gives a useful error."""
    model.encoder = FakeEncoder(dim=EMBEDDING_DIM + 1)
    with pytest.raises(ValueError, match="width"):
        model.encode_documents(DOCUMENTS)
    with pytest.raises(ValueError, match="width"):
        model.answer_from_embeddings(QUESTION, torch.ones(2, EMBEDDING_DIM))
