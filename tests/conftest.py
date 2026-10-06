"""Shared fixtures: a tiny chat tokenizer, a synthetic data repository and a deterministic fake encoder."""

from pathlib import Path

import pytest
from transformers import PreTrainedTokenizerFast

from tests.helpers import FakeEncoder, build_tokenizer, make_rows, write_split


@pytest.fixture
def tokenizer() -> PreTrainedTokenizerFast:
    """Build a word-level tokenizer with a ChatML-style template.

    Returns:
        The tokenizer.
    """
    return build_tokenizer()


@pytest.fixture
def fake_encoder() -> FakeEncoder:
    """Build a deterministic fake embedding engine.

    Returns:
        The engine.
    """
    return FakeEncoder()


@pytest.fixture
def data_repo(tmp_path: Path, fake_encoder: FakeEncoder) -> Path:
    """Write a local data repository with one config (``toy``) holding train and test splits with embeddings.

    Args:
        tmp_path: Temporary directory that becomes the repository.
        fake_encoder: Encoder for the embeddings.

    Returns:
        The repository directory.
    """
    rows = make_rows()
    write_split(tmp_path, "toy", "train", rows, fake_encoder)
    write_split(tmp_path, "toy", "test", rows[:2], fake_encoder)
    return tmp_path
