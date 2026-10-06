"""Tests for the memory module: shapes, masking, architecture options and checkpoint I/O."""

from collections.abc import Callable
import json
import os
from pathlib import Path
from typing import Any

import pytest
import torch

from memorilla.memory import DEFAULT_NUM_MEMORIES, MemoryConfig, MemoryModule
from memorilla.utils import CHECKPOINT_FILE, CONFIG_FILE, load_memory_state

EMBEDDING_DIM = 32
OUTPUT_DIM = 48
NUM_MEMORIES = 4
CHECKPOINTS_ENV = "MEMORILLA_TEST_CHECKPOINTS"


def make_memory(**overrides: int | float | bool) -> MemoryModule:
    """Build a small module in evaluation mode.

    Args:
        **overrides: Constructor arguments that replace the small defaults.

    Returns:
        The module, seeded for reproducible weights.
    """
    options = {
        "embedding_dim": EMBEDDING_DIM,
        "output_dim": OUTPUT_DIM,
        "num_memories": NUM_MEMORIES,
        "num_heads": 4,
        "dropout": 0.0,
        **overrides,
    }
    torch.manual_seed(0)
    return MemoryModule(**options).eval()


def random_inputs(batch: int = 2, num_docs: int = 6) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draw random documents, a padding mask with padding in the second row, and questions.

    Args:
        batch: Batch size.
        num_docs: Documents per example.

    Returns:
        Documents ``[batch, num_docs, D]``, padding mask ``[batch, num_docs]`` and questions ``[batch, D]``.
    """
    generator = torch.Generator().manual_seed(1)
    docs = torch.randn(batch, num_docs, EMBEDDING_DIM, generator=generator)
    mask = torch.zeros(batch, num_docs, dtype=torch.bool)
    mask[1, num_docs // 2 :] = True
    question = torch.randn(batch, EMBEDDING_DIM, generator=generator)
    return docs, mask, question


@pytest.mark.parametrize("num_self_attn_layers", [0, 1, 2])
@pytest.mark.parametrize("retrieval_init", [False, True])
def test_output_shape(num_self_attn_layers: int, retrieval_init: bool) -> None:
    """The module returns ``[batch, num_memories, output_dim]`` for every architecture option."""
    memory = make_memory(num_self_attn_layers=num_self_attn_layers, retrieval_init=retrieval_init)
    docs, mask, question = random_inputs()
    assert memory(docs, mask, question).shape == (2, NUM_MEMORIES, OUTPUT_DIM)
    assert memory(docs, None, question).shape == (2, NUM_MEMORIES, OUTPUT_DIM)


@pytest.mark.parametrize("num_self_attn_layers", [0, 1])
def test_padding_positions_are_ignored(num_self_attn_layers: int) -> None:
    """Changing the content of padded documents does not change the output."""
    memory = make_memory(num_self_attn_layers=num_self_attn_layers)
    docs, mask, question = random_inputs()
    reference = memory(docs, mask, question)

    perturbed = docs.clone()
    perturbed[mask] = 100.0
    torch.testing.assert_close(memory(perturbed, mask, question), reference)

    truncated = memory(docs[1:, : docs.shape[1] // 2], None, question[1:])
    torch.testing.assert_close(truncated, reference[1:], rtol=1e-4, atol=1e-5)


def test_self_attention_toggle() -> None:
    """The document encoder exists exactly when self-attention layers are requested."""
    assert not hasattr(make_memory(num_self_attn_layers=0), "encoder")
    encoder = make_memory(num_self_attn_layers=2).encoder
    assert len(encoder.layers) == 2


def test_retrieval_init_uses_most_similar_documents() -> None:
    """With ``retrieval_init`` there are no learned slots and the slots are the top documents, zero padded."""
    memory = make_memory(retrieval_init=True)
    assert not hasattr(memory, "memory_queries")

    question = torch.zeros(1, EMBEDDING_DIM)
    question[0, 0] = 1.0
    docs = torch.zeros(1, 3, EMBEDDING_DIM)
    docs[0, 0, 0], docs[0, 0, 1] = 0.1, 1.0
    docs[0, 1, 0] = 5.0
    docs[0, 2, 0], docs[0, 2, 1] = 1.0, 1.0
    mask = torch.tensor([[False, False, True]])

    slots = memory._initial_slots(docs, mask, question)
    assert slots.shape == (1, NUM_MEMORIES, EMBEDDING_DIM)
    torch.testing.assert_close(slots[0, 0], docs[0, 1])
    torch.testing.assert_close(slots[0, 1], docs[0, 0])
    assert torch.equal(slots[0, 3:], torch.zeros(NUM_MEMORIES - 3, EMBEDDING_DIM))


def test_question_conditions_the_output() -> None:
    """Different questions over the same documents give different memory tokens."""
    memory = make_memory()
    docs, mask, question = random_inputs()
    assert not torch.allclose(memory(docs, mask, question), memory(docs, mask, -question))


def test_save_and_from_pretrained_round_trip(tmp_path: Path) -> None:
    """``save`` writes weights and config; ``from_pretrained`` restores an identical module from either path."""
    memory = make_memory(num_self_attn_layers=1, retrieval_init=True, num_heads=2)
    memory.save(tmp_path)
    assert (tmp_path / CHECKPOINT_FILE).exists()
    with open(tmp_path / CONFIG_FILE) as handle:
        assert json.load(handle)["num_heads"] == 2

    docs, mask, question = random_inputs()
    expected = memory(docs, mask, question)
    for path in (tmp_path, tmp_path / CHECKPOINT_FILE):
        restored = MemoryModule.from_pretrained(path)
        assert restored.config == memory.config
        torch.testing.assert_close(restored(docs, mask, question), expected)


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"num_self_attn_layers": 2, "num_cross_attn_layers": 1},
        {"retrieval_init": True, "num_self_attn_layers": 1, "num_memories": DEFAULT_NUM_MEMORIES},
        {"num_memories": 7, "num_cross_attn_layers": 3},
    ],
)
def test_config_inferred_from_state_dict(tmp_path: Path, options: dict[str, Any]) -> None:
    """Without ``config.json`` the architecture is recovered from the weights.

    Heads, dropout and the slot count of a ``retrieval_init`` module leave no trace in the weights and take defaults.
    """
    memory = make_memory(num_heads=8, dropout=0.1, **options)
    torch.save(memory.state_dict(), tmp_path / CHECKPOINT_FILE)
    assert MemoryConfig.from_state_dict(memory.state_dict()) == memory.config
    restored = MemoryModule.from_pretrained(tmp_path)
    assert restored.config == memory.config


def test_from_pretrained_keeps_stored_dtype(tmp_path: Path) -> None:
    """Weights saved in bfloat16 load into a bfloat16 module."""
    memory = make_memory().to(torch.bfloat16)
    memory.save(tmp_path)
    assert next(MemoryModule.from_pretrained(tmp_path).parameters()).dtype == torch.bfloat16


@pytest.mark.parametrize(
    "wrap",
    [
        lambda state: state,
        lambda state: {f"memory.{key}": value for key, value in state.items()},
        lambda state: {f"module.{key}": value for key, value in state.items()},
        lambda state: {"state_dict": {**{f"memory.{k}": v for k, v in state.items()}, "llm.weight": torch.ones(1)}},
    ],
)
def test_load_memory_state_unwraps_checkpoints(
    tmp_path: Path, wrap: Callable[[dict[str, torch.Tensor]], dict[str, Any]]
) -> None:
    """Prefixed and nested checkpoints normalise to the plain module state dict."""
    state = make_memory().state_dict()
    torch.save(wrap(state), tmp_path / "weights.pt")
    loaded = load_memory_state(tmp_path / "weights.pt")
    assert loaded.keys() == state.keys()
    for key, value in state.items():
        assert torch.equal(loaded[key], value)


def checkpoints_under_env() -> list[Path]:
    """List the checkpoints of the directory named by ``MEMORILLA_TEST_CHECKPOINTS``.

    Returns:
        Every ``memory.pt`` below that directory, or an empty list when the variable is unset.
    """
    root = os.environ.get(CHECKPOINTS_ENV)
    return sorted(Path(root).rglob(CHECKPOINT_FILE)) if root else []


@pytest.mark.skipif(not checkpoints_under_env(), reason=f"set {CHECKPOINTS_ENV} to a directory of checkpoints")
@pytest.mark.parametrize("path", checkpoints_under_env(), ids=lambda path: path.parent.name)
def test_checkpoint_loads_strictly(path: Path) -> None:
    """Trained checkpoints load with ``strict=True`` and run a forward pass."""
    memory = MemoryModule.from_pretrained(path)
    state = load_memory_state(path)
    assert memory.state_dict().keys() == state.keys()

    config = memory.config
    dtype = next(memory.parameters()).dtype
    docs = torch.randn(1, 5, config.embedding_dim, dtype=dtype)
    question = torch.randn(1, config.embedding_dim, dtype=dtype)
    with torch.inference_mode():
        output = memory(docs, None, question)
    assert output.shape == (1, config.num_memories, config.output_dim)
    assert torch.isfinite(output.float()).all()
