"""Precomputed document and question embeddings: on-disk layout, reader, writer and builder.

Each ``(config, split)`` lives in ``embeddings/<encoder>/<config>/<split>/`` under the data root, in the file layout
described by ``EmbeddingWriter``.
"""

from collections.abc import Callable, Iterable
import multiprocessing
from pathlib import Path
from typing import TYPE_CHECKING, Any

from datasets import Dataset
from filelock import FileLock
from huggingface_hub import constants as hub_constants
from huggingface_hub import snapshot_download
import numpy as np
import pyarrow.compute as pc
from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch

from memorilla.data import COLLECTION_COLUMN, DOCUMENTS_COLUMN, QUESTION_COLUMN, ensure_data, load_split, split_complete
from memorilla.paths import DATA_DIR, DATA_REPO, DEFAULT_ENCODER, data_root, encoder_slug, local_repo
from memorilla.utils import local_model_path, release_gpu_memory

if TYPE_CHECKING:
    from vllm import LLM

INDEX_FILE = "index.safetensors"
QUESTIONS_FILE = "questions.safetensors"
SHARD_TEMPLATE = "documents-{:05d}.safetensors"
TENSOR_NAME = "embeddings"
MAX_SHARD_BYTES = 4 * 1024**3
EMBED_CHUNK_SIZE = 200_000
ENCODER_GPU_MEMORY_UTILIZATION = 0.9
STORAGE_DTYPE = torch.float16
LOCK_FILE = ".memorilla.lock"


def embedding_dir(config: str, split: str, root: str | Path, encoder: str = DEFAULT_ENCODER) -> Path:
    """Return the directory holding one split's embeddings.

    Args:
        config: Dataset config name.
        split: Split name.
        root: Data root (contains ``embeddings/``).
        encoder: Encoder id; its last path component names the embedding folder.

    Returns:
        ``root/embeddings/<encoder>/<config>/<split>``.
    """
    return Path(root) / "embeddings" / encoder_slug(encoder) / config / split


def missing_files(directory: str | Path) -> list[str]:
    """List the files of one split's embeddings that are absent.

    Args:
        directory: Split directory written by ``EmbeddingWriter``.

    Returns:
        The missing file names: the index, the question file and every document shard the index refers to.
    """
    directory = Path(directory)
    if not (directory / INDEX_FILE).exists():
        return [INDEX_FILE]
    shards = load_file(str(directory / INDEX_FILE))["shard"].unique().tolist()
    names = [QUESTIONS_FILE] + [SHARD_TEMPLATE.format(shard) for shard in shards]
    return [name for name in names if not (directory / name).exists()]


class EmbeddingStore:
    """Read-only, memory-mapped access to one split's document and question embeddings.

    Shards are opened lazily with ``safetensors.safe_open`` and sliced on demand, so resident memory stays small and
    the operating system's page cache is shared between data-loader workers and distributed ranks.
    """

    def __init__(self, config: str, split: str, root: str | Path | None = None, encoder: str = DEFAULT_ENCODER) -> None:
        """Open the index of a split.

        Args:
            config: Dataset config name.
            split: Split name.
            root: Data root; defaults to the root of the default data repository.
            encoder: Encoder id that produced the embeddings.

        Raises:
            FileNotFoundError: If the split has no embeddings under ``root`` or some of their files are missing.
        """
        self.config = config
        self.split = split
        self.directory = embedding_dir(config, split, root if root is not None else data_root(), encoder)
        missing = missing_files(self.directory)
        if missing:
            raise FileNotFoundError(f"Embeddings of {config}/{split} in {self.directory} lack {missing}.")

        index = load_file(str(self.directory / INDEX_FILE))
        self.collection_ids = index["collection_ids"]
        self._shard = index["shard"]
        self._start = index["start"]
        self._length = index["length"]
        self._handles: dict[str, Any] = {}

    def __len__(self) -> int:
        """Count the collections of the split.

        Returns:
            The number of indexed collections.
        """
        return int(self.collection_ids.numel())

    def __contains__(self, collection_id: int) -> bool:
        """Whether the split has documents for a collection.

        Args:
            collection_id: Collection id.

        Returns:
            True when the collection is indexed.
        """
        return self._position(collection_id) is not None

    @property
    def num_questions(self) -> int:
        """Number of question embeddings, one per row of the split."""
        return int(self._open(QUESTIONS_FILE).get_slice(TENSOR_NAME).get_shape()[0])

    def __getstate__(self) -> dict[str, Any]:
        """Pickle without open file handles (they are reopened lazily).

        Returns:
            The instance state without file handles.
        """
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state

    def _position(self, collection_id: int) -> int | None:
        """Return the index position of a collection, or None if it is absent.

        Args:
            collection_id: Collection id.

        Returns:
            Its position in ``collection_ids`` or None.
        """
        position = int(torch.searchsorted(self.collection_ids, torch.tensor(int(collection_id))))
        if position < len(self) and int(self.collection_ids[position]) == int(collection_id):
            return position
        return None

    def _open(self, filename: str) -> Any:
        """Return a cached ``safe_open`` handle.

        Args:
            filename: File name inside the split directory.

        Returns:
            The open handle.
        """
        if filename not in self._handles:
            self._handles[filename] = safe_open(str(self.directory / filename), framework="pt")
        return self._handles[filename]

    def documents(self, collection_id: int) -> torch.Tensor:
        """Return a collection's document embeddings.

        Args:
            collection_id: Collection id.

        Returns:
            Float16 tensor ``[num_docs, D]``.

        Raises:
            KeyError: If the collection is not in this split.
        """
        position = self._position(collection_id)
        if position is None:
            raise KeyError(f"Collection {collection_id} not found in {self.config}/{self.split}.")
        shard = int(self._shard[position])
        start = int(self._start[position])
        length = int(self._length[position])
        handle = self._open(SHARD_TEMPLATE.format(shard))
        return handle.get_slice(TENSOR_NAME)[start : start + length]

    def question(self, row: int) -> torch.Tensor:
        """Return the question embedding of one row of the split.

        Args:
            row: Row index in ``data/<config>/<split>``.

        Returns:
            Float16 tensor ``[D]``.
        """
        handle = self._open(QUESTIONS_FILE)
        return handle.get_slice(TENSOR_NAME)[row : row + 1][0]

    def questions(self) -> torch.Tensor:
        """Return every question embedding of the split.

        Returns:
            Float16 tensor ``[R, D]``.
        """
        return self._open(QUESTIONS_FILE).get_tensor(TENSOR_NAME)


class EmbeddingWriter:
    """Write one split's embeddings in the layout read by ``EmbeddingStore``.

    The split directory holds three kinds of files:

    - ``index.safetensors``: ``collection_ids`` int64 ``[C]`` (ascending), ``shard`` int32 ``[C]``, ``start`` int64
      ``[C]`` and ``length`` int64 ``[C]`` locating every collection's document rows.
    - ``documents-XXXXX.safetensors``: tensor ``embeddings`` float16 ``[n_i, D]``; collections never span two shards.
    - ``questions.safetensors``: tensor ``embeddings`` float16 ``[R, D]``, row-aligned with ``data/<config>/<split>``.

    Collections must be added in strictly ascending id order. A new shard starts whenever the next non-empty collection
    would push the current one past ``max_shard_bytes``; a collection larger than the limit gets a shard of its own.
    Collections without documents are indexed in the current shard, so every index entry points at a written file.
    """

    def __init__(self, directory: str | Path, max_shard_bytes: int = MAX_SHARD_BYTES) -> None:
        """Create the output directory.

        Args:
            directory: Output directory for the split.
            max_shard_bytes: Target maximum size of a document shard.
        """
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_shard_bytes = max_shard_bytes
        self._ids: list[int] = []
        self._shards: list[int] = []
        self._starts: list[int] = []
        self._lengths: list[int] = []
        self._buffer: list[torch.Tensor] = []
        self._buffer_rows = 0
        self._buffer_bytes = 0
        self._shard_index = 0

    def __enter__(self) -> "EmbeddingWriter":
        """Enter a ``with`` block.

        Returns:
            The writer.
        """
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Finish writing when the ``with`` block exits without an error.

        Args:
            *exc_info: Exception type, value and traceback, all None when the block succeeded.
        """
        if exc_info[0] is None:
            self.close()

    def add(self, collection_id: int, embeddings: torch.Tensor | np.ndarray) -> None:
        """Append one collection's document embeddings.

        Args:
            collection_id: Collection id, larger than every id added before.
            embeddings: ``[num_docs, D]`` embeddings, stored as float16.

        Raises:
            ValueError: If ids are not strictly ascending.
        """
        if self._ids and collection_id <= self._ids[-1]:
            raise ValueError(f"Collection ids must be strictly ascending: {collection_id} after {self._ids[-1]}.")

        tensor = torch.as_tensor(embeddings).to(STORAGE_DTYPE).contiguous()
        size = tensor.numel() * tensor.element_size()
        if size > 0 and self._buffer and self._buffer_bytes + size > self.max_shard_bytes:
            self._flush()

        self._ids.append(int(collection_id))
        self._shards.append(self._shard_index)
        self._starts.append(self._buffer_rows)
        self._lengths.append(int(tensor.shape[0]))
        if tensor.shape[0] > 0:
            self._buffer.append(tensor)
            self._buffer_rows += int(tensor.shape[0])
            self._buffer_bytes += size

    def write_questions(self, embeddings: torch.Tensor | np.ndarray) -> None:
        """Write the row-aligned question embeddings.

        Args:
            embeddings: ``[R, D]`` embeddings, stored as float16.
        """
        tensor = torch.as_tensor(embeddings).to(STORAGE_DTYPE).contiguous()
        save_file({TENSOR_NAME: tensor}, str(self.directory / QUESTIONS_FILE))

    def _flush(self) -> None:
        """Write the buffered collections as the next shard."""
        if not self._buffer:
            return
        shard = torch.cat(self._buffer, dim=0)
        save_file({TENSOR_NAME: shard}, str(self.directory / SHARD_TEMPLATE.format(self._shard_index)))
        self._buffer = []
        self._buffer_rows = 0
        self._buffer_bytes = 0
        self._shard_index += 1

    def close(self) -> None:
        """Write the last shard and the index."""
        self._flush()
        index = {
            "collection_ids": torch.tensor(self._ids, dtype=torch.int64),
            "shard": torch.tensor(self._shards, dtype=torch.int32),
            "start": torch.tensor(self._starts, dtype=torch.int64),
            "length": torch.tensor(self._lengths, dtype=torch.int64),
        }
        save_file(index, str(self.directory / INDEX_FILE))


def load_encoder(
    encoder: str = DEFAULT_ENCODER,
    cache_dir: str | None = None,
    gpu_memory_utilization: float = ENCODER_GPU_MEMORY_UTILIZATION,
) -> "LLM":
    """Start a vLLM embedding engine.

    Args:
        encoder: Hugging Face model id or local path of the encoder.
        cache_dir: Model cache directory (None uses the Hugging Face default).
        gpu_memory_utilization: Fraction of GPU memory the engine may use.

    Returns:
        A ``vllm.LLM`` running the encoder in bfloat16 with the ``embed`` task.
    """
    from vllm import LLM

    return LLM(
        model=local_model_path(encoder, cache_dir),
        task="embed",
        dtype="bfloat16",
        download_dir=cache_dir,
        gpu_memory_utilization=gpu_memory_utilization,
    )


def embed_texts(engine: "LLM", texts: list[str], chunk_size: int = EMBED_CHUNK_SIZE) -> np.ndarray:
    """Embed texts with a vLLM embedding engine (pooled vectors, no instruction prefix).

    Args:
        engine: Engine from ``load_encoder``.
        texts: Texts to embed.
        chunk_size: Number of texts handed to vLLM per call; bounds host memory.

    Returns:
        Float16 array ``[len(texts), D]``.
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float16)

    probe = engine.embed([texts[0]], use_tqdm=False)
    embeddings = np.empty((len(texts), len(probe[0].outputs.embedding)), dtype=np.float16)
    for start in range(0, len(texts), chunk_size):
        outputs = engine.embed(texts[start : start + chunk_size])
        for offset, output in enumerate(outputs):
            embeddings[start + offset] = output.outputs.embedding
    return embeddings


def _chunk_collections(lengths: np.ndarray, chunk_size: int) -> Iterable[tuple[int, int]]:
    """Group consecutive collections so that each group holds about ``chunk_size`` documents.

    Args:
        lengths: Number of documents per collection.
        chunk_size: Target number of documents per group.

    Yields:
        ``(begin, end)`` collection index ranges.
    """
    begin, total = 0, 0
    for index, length in enumerate(lengths):
        total += int(length)
        if total >= chunk_size:
            yield begin, index + 1
            begin, total = index + 1, 0
    if begin < len(lengths):
        yield begin, len(lengths)


def build_embeddings(
    dataset: Dataset,
    out_dir: str | Path,
    engine: "LLM",
    chunk_size: int = EMBED_CHUNK_SIZE,
    max_shard_bytes: int = MAX_SHARD_BYTES,
) -> None:
    """Embed one split's documents and questions and write them in the store layout.

    Each collection is embedded once, from the first row that carries it. Questions are embedded as
    ``str(question).strip()``, one row per dataset row.

    Args:
        dataset: Split with ``collection_id``, ``documents`` and ``question`` columns.
        out_dir: Output directory for the split.
        engine: Embedding engine from ``load_encoder``.
        chunk_size: Number of documents embedded per vLLM call.
        max_shard_bytes: Target maximum size of a document shard.

    Raises:
        ValueError: If a collection has no documents (the memory module needs at least one per example).
    """
    ids = np.asarray(dataset.with_format("numpy")[COLLECTION_COLUMN], dtype=np.int64)
    collection_ids, first_rows = np.unique(ids, return_index=True)
    documents = dataset.select_columns([DOCUMENTS_COLUMN])
    row_lengths = pc.list_value_length(documents.with_format("arrow")[DOCUMENTS_COLUMN]).fill_null(0).to_numpy()
    lengths = row_lengths[first_rows].astype(np.int64)
    empty = collection_ids[lengths == 0]
    if empty.size:
        raise ValueError(f"Collections without documents: {empty[:10].tolist()} ({empty.size} in total).")

    with EmbeddingWriter(out_dir, max_shard_bytes=max_shard_bytes) as writer:
        for begin, end in _chunk_collections(lengths, chunk_size):
            rows = documents.select(first_rows[begin:end].tolist())[DOCUMENTS_COLUMN]
            vectors = embed_texts(engine, [str(doc) for docs in rows for doc in docs], chunk_size)
            offset = 0
            for collection_id, length in zip(collection_ids[begin:end], lengths[begin:end], strict=True):
                writer.add(int(collection_id), vectors[offset : offset + length])
                offset += int(length)

        questions = [str(question).strip() for question in dataset[QUESTION_COLUMN]]
        unique = list(dict.fromkeys(questions))
        position = {text: index for index, text in enumerate(unique)}
        vectors = embed_texts(engine, unique, chunk_size)
        writer.write_questions(vectors[[position[text] for text in questions]])


def compute_embeddings(config: str, splits: list[str], root: Path, encoder: str, cache_dir: str | None) -> None:
    """Embed the given splits of a config with a vLLM engine started for this call.

    Args:
        config: Dataset config name.
        splits: Split names; their parquet files must be under ``root``.
        root: Data root.
        encoder: Encoder id.
        cache_dir: Model cache directory for the encoder.
    """
    engine = load_encoder(encoder, cache_dir=cache_dir)
    for split in splits:
        dataset = load_split(config, split, root, [COLLECTION_COLUMN, DOCUMENTS_COLUMN, QUESTION_COLUMN])
        build_embeddings(dataset, embedding_dir(config, split, root, encoder), engine)
    del engine
    release_gpu_memory()


def run_isolated(function: Callable[..., None], *args: Any) -> None:
    """Run a function in a fresh process started with ``spawn`` and wait for it.

    vLLM changes process-wide state when it is imported and run (environment variables such as
    ``NCCL_CUMEM_ENABLE``, CUDA and ``torch.distributed`` state, GPU memory); in a separate process none of it
    reaches the caller, e.g. one rank of a distributed training job.

    Args:
        function: A module-level function.
        *args: Its arguments (picklable).

    Raises:
        RuntimeError: If the process exits with an error.
    """
    process = multiprocessing.get_context("spawn").Process(target=function, args=args)
    process.start()
    process.join()
    if process.exitcode != 0:
        raise RuntimeError(f"{function.__name__} failed in a subprocess (exit code {process.exitcode}).")


def ensure_embeddings(
    config: str,
    splits: Iterable[str],
    repo: str = DATA_REPO,
    data_dir: str | Path = DATA_DIR,
    encoder: str = DEFAULT_ENCODER,
    cache_dir: str | None = None,
) -> Path:
    """Make the embeddings of the given splits available locally.

    Missing or incomplete splits are downloaded from ``repo`` when it publishes them; splits it does not publish
    (custom data) are computed with ``build_embeddings`` in a separate process. Concurrent processes coordinate
    through a lock file in the data root.

    Args:
        config: Dataset config name.
        splits: Split names.
        repo: Hub dataset repo id, or a local directory with the same layout.
        data_dir: Local mirror used when ``repo`` is a Hub repo id.
        encoder: Encoder id.
        cache_dir: Model cache directory for the encoder.

    Returns:
        The data root.

    Raises:
        FileNotFoundError: If a split without embeddings has no complete parquet files to compute them from.
        RuntimeError: If some embeddings are still missing after computing them.
    """
    root = data_root(repo, data_dir)
    root.mkdir(parents=True, exist_ok=True)
    splits = list(splits)

    def missing() -> list[str]:
        """List the requested splits whose embeddings are absent or incomplete.

        Returns:
            Split names.
        """
        return [split for split in splits if missing_files(embedding_dir(config, split, root, encoder))]

    if not missing():
        return root
    with FileLock(str(root / LOCK_FILE)):
        if missing() and local_repo(repo) is None and not hub_constants.HF_HUB_OFFLINE:
            patterns = [f"embeddings/{encoder_slug(encoder)}/{config}/{split}/*" for split in missing()]
            snapshot_download(repo_id=repo, repo_type="dataset", allow_patterns=patterns, local_dir=root)

        if missing():
            ensure_data(config, missing(), repo=repo, data_dir=data_dir)
            absent = [split for split in missing() if not split_complete(config, split, root)]
            if absent:
                raise FileNotFoundError(
                    f"No complete parquet files for {config} splits {absent} under {root / 'data' / config}."
                )
            run_isolated(compute_embeddings, config, missing(), root, encoder, cache_dir)
        if missing():
            raise RuntimeError(f"Embeddings of {config} {missing()} are still incomplete under {root}.")
    return root


def load_split_with_embeddings(
    config: str,
    split: str,
    root: str | Path,
    columns: list[str] | None = None,
    encoder: str = DEFAULT_ENCODER,
) -> tuple[Dataset, EmbeddingStore]:
    """Load a split and its embedding store and check that they line up row by row.

    Args:
        config: Dataset config name.
        split: Split name.
        root: Data root.
        columns: Columns to load (see ``load_split``).
        encoder: Encoder id.

    Returns:
        The split and its embedding store.

    Raises:
        ValueError: If the split's row count differs from its number of question embeddings.
    """
    dataset = load_split(config, split, root, columns)
    store = EmbeddingStore(config, split, root, encoder)
    if len(dataset) != store.num_questions:
        raise ValueError(
            f"{config}/{split} has {len(dataset)} rows but {store.num_questions} question embeddings; "
            f"remove {store.directory} to recompute them, or restore the missing data files."
        )
    return dataset, store
