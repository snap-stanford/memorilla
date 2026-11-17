"""Embedding batches infer their width without submitting a separate probe."""

import numpy as np
import pytest

from memorilla.embeddings import embed_texts
from tests.helpers import FakeEncoder


@pytest.mark.parametrize(("count", "chunk_size"), [(0, 3), (1, 3), (3, 3), (4, 3), (7, 3)])
def test_embed_texts_submits_each_text_once(count: int, chunk_size: int) -> None:
    encoder = FakeEncoder()
    texts = [f"text {index}" for index in range(count)]
    actual = embed_texts(encoder, texts, chunk_size=chunk_size)
    assert encoder.calls == [texts[start : start + chunk_size] for start in range(0, count, chunk_size)]
    expected = (
        np.stack([encoder.vector(text) for text in texts]).astype(np.float16)
        if count
        else np.zeros((0, 0), dtype=np.float16)
    )
    assert actual.dtype == np.float16
    assert np.array_equal(actual, expected)
