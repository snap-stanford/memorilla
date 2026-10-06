"""Seeding, the memory special token, checkpoint I/O, model resolution and small file helpers."""

import gc
import json
from pathlib import Path
import random
from typing import Any

from huggingface_hub import snapshot_download
import numpy as np
import torch
from transformers import PreTrainedTokenizerBase
import yaml

MEMORY_TOKEN = "<|memory|>"
CHECKPOINT_FILE = "memory.pt"
CONFIG_FILE = "config.json"
WRAPPER_PREFIXES = ("module.", "_orig_mod.")
MEMORY_PREFIX = "memory."
DEFAULT_SEED = 23


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch random number generators.

    Args:
        seed: The seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def str2bool(value: str) -> bool:
    """Parse a command-line boolean.

    Args:
        value: A string such as ``true``/``false``/``1``/``0``.

    Returns:
        The parsed boolean.

    Raises:
        ValueError: If the string is not a recognised boolean.
    """
    lowered = value.strip().lower()
    if lowered in ("true", "t", "1", "yes", "y"):
        return True
    if lowered in ("false", "f", "0", "no", "n"):
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def prepare_tokenizer(tokenizer: PreTrainedTokenizerBase) -> int:
    """Register the memory placeholder token and make sure a padding token exists.

    Args:
        tokenizer: The decoder tokenizer, modified in place.

    Returns:
        The id of the memory placeholder token.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": [MEMORY_TOKEN]})
    return tokenizer.convert_tokens_to_ids(MEMORY_TOKEN)


def checkpoint_file(path: str | Path) -> Path:
    """Return the ``memory.pt`` file for a checkpoint directory or file path.

    Args:
        path: A checkpoint directory or a path to a weights file.

    Returns:
        The weights file path.
    """
    path = Path(path).expanduser()
    return path / CHECKPOINT_FILE if path.is_dir() else path


def load_memory_state(path: str | Path) -> dict[str, torch.Tensor]:
    """Load memory-module weights from a checkpoint, normalising common wrappers.

    Accepts a plain state dict, a dict with a ``state_dict`` entry (training checkpoints), keys prefixed by the
    owning module (``memory.``) and keys prefixed by distributed or compiled wrappers.

    Args:
        path: A checkpoint directory or weights file.

    Returns:
        The memory module state dict on CPU.
    """
    state = torch.load(checkpoint_file(path), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]

    for prefix in WRAPPER_PREFIXES:
        if state and all(key.startswith(prefix) for key in state):
            state = {key[len(prefix) :]: value for key, value in state.items()}

    if any(key.startswith(MEMORY_PREFIX) for key in state):
        state = {key[len(MEMORY_PREFIX) :]: value for key, value in state.items() if key.startswith(MEMORY_PREFIX)}
    return dict(state)


def read_recipe(path: str | Path) -> dict[str, Any]:
    """Read a YAML training recipe.

    Args:
        path: Recipe file, e.g. ``configs/stage3_multitask.yaml``.

    Returns:
        The recipe's options (empty for an empty file).
    """
    with open(path) as handle:
        return yaml.safe_load(handle) or {}


def local_model_path(model: str, cache_dir: str | None) -> str:
    """Return the model location to hand to vLLM so that it reads every model file from ``cache_dir``.

    vLLM downloads weights into its ``download_dir`` but looks up the config and tokenizer in the default Hugging Face
    cache; resolving the model to a local snapshot of ``cache_dir`` first keeps all of its files there.

    Args:
        model: Hugging Face model id or local path.
        cache_dir: Model cache directory, or None for the Hugging Face default.

    Returns:
        The snapshot directory inside ``cache_dir``, or ``model`` unchanged when it is a local path or no cache
        directory is set.
    """
    if cache_dir is None or Path(model).expanduser().exists():
        return model
    return snapshot_download(repo_id=model, cache_dir=cache_dir)


def write_json(path: str | Path, payload: Any) -> None:
    """Write a JSON file, creating parent directories.

    Args:
        path: Output path.
        payload: JSON-serialisable object.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write one JSON object per line, creating parent directories.

    Args:
        path: Output path.
        rows: JSON-serialisable rows.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def release_gpu_memory() -> None:
    """Tear down vLLM's distributed state and return cached GPU memory.

    Call it after the last reference to a ``vllm.LLM`` engine has been dropped, before starting another engine.
    """
    from vllm.distributed import cleanup_dist_env_and_memory

    gc.collect()
    cleanup_dist_env_and_memory()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
