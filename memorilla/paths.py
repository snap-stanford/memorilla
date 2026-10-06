"""Default locations and Hub repository ids.

The data, results and model-cache locations and the data repository can be set with the ``MEMORILLA_*`` environment
variables; every script also accepts explicit arguments.
"""

import os
from pathlib import Path

HOME = Path(os.environ.get("MEMORILLA_HOME", ".")).expanduser()
DATA_DIR = Path(os.environ.get("MEMORILLA_DATA_DIR", str(HOME / "data"))).expanduser()
RESULTS_DIR = HOME / "results"
MODEL_CACHE_DIR = os.environ.get("MEMORILLA_MODEL_CACHE") or None

DATA_REPO = os.environ.get("MEMORILLA_DATA_REPO", "memorilla/Memorilla-Data")
DEFAULT_DECODER = "Qwen/Qwen3-8B"
DEFAULT_ENCODER = "Qwen/Qwen3-Embedding-4B"


def resolve(path: str | Path) -> Path:
    """Resolve a path against ``MEMORILLA_HOME`` unless it is already absolute.

    Args:
        path: Absolute path, or a path relative to ``MEMORILLA_HOME``.

    Returns:
        The resolved path.
    """
    path = Path(path).expanduser()
    return path if path.is_absolute() else HOME / path


def local_repo(repo: str) -> Path | None:
    """Return the local directory a data repository refers to, if any.

    Args:
        repo: Hub dataset repo id, or a local directory (absolute or relative to ``MEMORILLA_HOME``).

    Returns:
        The directory when ``repo`` is a local directory, otherwise None.
    """
    path = resolve(repo)
    return path if path.is_dir() else None


def data_root(repo: str = DATA_REPO, data_dir: str | Path = DATA_DIR) -> Path:
    """Return the local directory that holds a data repository's ``data/`` and ``embeddings/`` folders.

    Args:
        repo: Hub dataset repo id, or a local directory with the same layout.
        data_dir: Local mirror used when ``repo`` is a Hub repo id.

    Returns:
        ``repo`` itself when it is a local directory, otherwise ``data_dir``.
    """
    return local_repo(repo) or resolve(data_dir)


def encoder_slug(encoder: str) -> str:
    """Return the directory name used for an encoder's embeddings.

    Args:
        encoder: Hugging Face model id or local path of the encoder.

    Returns:
        The lower-cased last path component, e.g. ``qwen3-embedding-4b``.
    """
    return encoder.rstrip("/").split("/")[-1].lower()
