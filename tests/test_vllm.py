"""Tests for ``MemoryVLLM.from_pretrained`` and ``answer`` with fake engines; they run wherever vLLM is installed."""

import importlib
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn

from memorilla.data import DEFAULT_SYSTEM_PROMPT, apply_chat_template, build_messages
from memorilla.memory import MemoryModule
from memorilla.paths import DEFAULT_DECODER, DEFAULT_ENCODER
from memorilla.utils import prepare_tokenizer
from tests.helpers import EMBEDDING_DIM, FakeEncoder, build_tokenizer

HIDDEN_SIZE = 16
NUM_MEMORIES = 2
QUESTION = "what is the capital of france"
DOCUMENTS = ["paris is the capital", "france"]


class FakeEngine:
    """Stand-in for a vLLM decoder engine: records each request and answers it with a fixed text."""

    def __init__(self, text: str) -> None:
        """Store the answer.

        Args:
            text: Text returned for every request.
        """
        self.text = text
        self.requests: list[dict[str, torch.Tensor]] = []
        self.sampling: Any = None

    def generate(self, requests: list[dict[str, torch.Tensor]], sampling_params: Any, use_tqdm: bool) -> list[Any]:
        """Record the requests.

        Args:
            requests: Prompt-embedding requests.
            sampling_params: Sampling parameters.
            use_tqdm: Ignored.

        Returns:
            One output per request.
        """
        self.requests.extend(requests)
        self.sampling = sampling_params
        return [SimpleNamespace(outputs=[SimpleNamespace(text=self.text)]) for _ in requests]


@pytest.fixture
def vllm_module() -> ModuleType:
    """Import ``memorilla.vllm``, skipping the test where vLLM is not installed.

    Returns:
        The module.
    """
    pytest.importorskip("vllm")
    return importlib.import_module("memorilla.vllm")


def make_memory() -> MemoryModule:
    """Build a small module in evaluation mode.

    Returns:
        The module, seeded for reproducible weights.
    """
    torch.manual_seed(0)
    return MemoryModule(EMBEDDING_DIM, HIDDEN_SIZE, num_memories=NUM_MEMORIES, num_heads=2, dropout=0.0).eval()


def make_model(vllm_module: ModuleType, encoder: FakeEncoder | None) -> Any:
    """Assemble a ``MemoryVLLM`` on CPU around fake engines, without starting vLLM.

    Args:
        vllm_module: The ``memorilla.vllm`` module.
        encoder: Embedding engine, or None.

    Returns:
        The model.
    """
    model = vllm_module.MemoryVLLM.__new__(vllm_module.MemoryVLLM)
    model.device = torch.device("cpu")
    model.memory = make_memory().to(torch.bfloat16)
    model.tokenizer = build_tokenizer()
    model.memory_token_id = prepare_tokenizer(model.tokenizer)
    # Zero token embeddings make the scattered memory tokens exact: 0 + (memory - 0) == memory.
    model.embed_tokens = nn.Embedding.from_pretrained(
        torch.zeros(len(model.tokenizer), HIDDEN_SIZE, dtype=torch.bfloat16)
    )
    model.engine = FakeEngine(" paris \n")
    model.encoder = encoder
    return model


def test_answer_fills_the_memory_positions(vllm_module: ModuleType, fake_encoder: FakeEncoder) -> None:
    """``answer`` embeds the documents and the question, and sends the memory tokens in the prompt."""
    model = make_model(vllm_module, fake_encoder)
    assert model.answer(QUESTION, DOCUMENTS, max_new_tokens=8) == "paris"
    assert fake_encoder.calls[-1] == [*DOCUMENTS, QUESTION]

    messages = build_messages(QUESTION, DEFAULT_SYSTEM_PROMPT, num_memories=NUM_MEMORIES)
    prompt = apply_chat_template(model.tokenizer, messages, add_generation_prompt=True)
    input_ids = model.tokenizer(prompt, return_tensors="pt")["input_ids"][0]
    vectors = torch.from_numpy(np.stack([fake_encoder.vector(text) for text in [*DOCUMENTS, QUESTION]]))
    vectors = vectors.to(torch.float16).to(torch.bfloat16)  # the encoder's float16 output, cast like ``generate``
    with torch.inference_mode():
        memory_tokens = model.memory(vectors[None, :-1], torch.zeros(1, len(DOCUMENTS), dtype=torch.bool), vectors[-1:])

    (request,) = model.engine.requests
    assert request["prompt_embeds"].shape == (len(input_ids), HIDDEN_SIZE)
    assert torch.equal(request["prompt_embeds"][input_ids == model.memory_token_id], memory_tokens[0])
    assert model.engine.sampling.max_tokens == 8 and model.engine.sampling.temperature == 0.0


def test_answer_needs_an_encoder(vllm_module: ModuleType) -> None:
    """A model built without an encoder cannot answer from document texts."""
    with pytest.raises(ValueError, match="encoder"):
        make_model(vllm_module, None).answer(QUESTION, DOCUMENTS)


def test_from_pretrained_loads_a_local_subfolder(
    vllm_module: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A local checkpoint folder is loaded without a download and handed to the decoder with the encoder."""
    make_memory().save(tmp_path / "pmv2")
    built: dict[str, Any] = {}
    monkeypatch.setattr(vllm_module, "load_encoder", lambda encoder, **kwargs: f"engine for {encoder}")
    monkeypatch.setattr(vllm_module, "snapshot_download", lambda **kwargs: pytest.fail("unexpected download"))
    monkeypatch.setattr(
        vllm_module.MemoryVLLM, "__init__", lambda self, *args, **kwargs: built.update(args=args, **kwargs)
    )

    vllm_module.MemoryVLLM.from_pretrained(tmp_path, "pmv2", max_model_len=4096)
    decoder, memory = built["args"]
    assert decoder == DEFAULT_DECODER and memory.num_memories == NUM_MEMORIES
    assert built["encoder"] == f"engine for {DEFAULT_ENCODER}" and built["max_model_len"] == 4096
