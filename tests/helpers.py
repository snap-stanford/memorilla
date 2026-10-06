"""Test helpers: a tiny chat tokenizer, a synthetic data repository writer and a deterministic fake encoder."""

from pathlib import Path
from typing import Any
import zlib

from datasets import Dataset
import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from memorilla.embeddings import EmbeddingWriter, embedding_dir
from memorilla.paths import DEFAULT_ENCODER

CHAT_TEMPLATE = (
    "{% for message in messages %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
SPECIAL_TOKENS = ["[UNK]", "<|endoftext|>", "<|im_start|>", "<|im_end|>"]
WORDS = (
    "system user assistant Memory You are a helpful what is the capital of france paris who wrote hamlet "
    "shakespeare answer briefly please suffix ok"
).split()
EMBEDDING_DIM = 8


class FakeOutput:
    """Mimics the ``outputs`` field of a vLLM embedding result."""

    def __init__(self, embedding: list[float]) -> None:
        """Store the embedding.

        Args:
            embedding: The vector.
        """
        self.embedding = embedding


class FakeResult:
    """Mimics one vLLM embedding result."""

    def __init__(self, embedding: list[float]) -> None:
        """Wrap the embedding.

        Args:
            embedding: The vector.
        """
        self.outputs = FakeOutput(embedding)


class FakeEncoder:
    """Deterministic stand-in for a vLLM embedding engine: each text maps to a fixed pseudo-random vector."""

    def __init__(self, dim: int = EMBEDDING_DIM, table: dict[str, list[float]] | None = None) -> None:
        """Configure the encoder.

        Args:
            dim: Embedding width.
            table: Optional explicit vectors for specific texts.
        """
        self.dim = dim
        self.table = table or {}
        self.calls: list[list[str]] = []

    def vector(self, text: str) -> np.ndarray:
        """Return the vector of a text.

        Args:
            text: Input text.

        Returns:
            A float32 vector.
        """
        if text in self.table:
            return np.asarray(self.table[text], dtype=np.float32)
        return np.random.default_rng(zlib.crc32(text.encode())).standard_normal(self.dim).astype(np.float32)

    def embed(self, texts: list[str], use_tqdm: bool = True) -> list[FakeResult]:
        """Embed texts.

        Args:
            texts: Input texts.
            use_tqdm: Ignored.

        Returns:
            One result per text.
        """
        self.calls.append(list(texts))
        return [FakeResult(self.vector(text).tolist()) for text in texts]


def build_tokenizer() -> PreTrainedTokenizerFast:
    """Build a word-level tokenizer with a ChatML-style template.

    Returns:
        The tokenizer.
    """
    vocab = {token: index for index, token in enumerate(SPECIAL_TOKENS + WORDS)}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="<|endoftext|>",
        eos_token="<|im_end|>",
        additional_special_tokens=["<|im_start|>", "<|im_end|>"],
    )
    tokenizer.chat_template = CHAT_TEMPLATE
    return tokenizer


def make_rows() -> list[dict[str, Any]]:
    """Return the rows of a small synthetic config: two collections, one shared by two rows.

    Returns:
        Three rows with every column a config can have.
    """
    return [
        {
            "collection_id": 7,
            "question": " what is the capital of france ",
            "answer": "paris",
            "documents": ["paris is the capital", "france"],
            "choices": ["paris", "", "hamlet"],
            "hard": True,
        },
        {
            "collection_id": 3,
            "question": "who wrote hamlet",
            "answer": "shakespeare",
            "documents": ["hamlet", "shakespeare wrote hamlet", "who", "ok"],
            "choices": ["paris", "shakespeare", ""],
            "hard": False,
        },
        {
            "collection_id": 7,
            "question": "what is the capital of france",
            "answer": "paris",
            "documents": ["paris is the capital", "france"],
            "choices": ["hamlet", "paris", ""],
            "hard": False,
        },
    ]


def write_split(
    root: Path, config: str, split: str, rows: list[dict[str, Any]], encoder: FakeEncoder | None = None
) -> Dataset:
    """Write a split's parquet file and, when ``encoder`` is given, its embeddings.

    Args:
        root: Data root.
        config: Config name.
        split: Split name.
        rows: Rows to write.
        encoder: Encoder for the embeddings, or None to write data only.

    Returns:
        The split as a dataset.
    """
    dataset = Dataset.from_list(rows)
    path = root / "data" / config / f"{split}-00000-of-00001.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(str(path))
    if encoder is not None:
        first = {}
        for row in rows:
            first.setdefault(row["collection_id"], row["documents"])
        with EmbeddingWriter(embedding_dir(config, split, root, DEFAULT_ENCODER)) as writer:
            for collection_id in sorted(first):
                writer.add(collection_id, np.stack([encoder.vector(doc) for doc in first[collection_id]]))
            writer.write_questions(np.stack([encoder.vector(str(row["question"]).strip()) for row in rows]))
    return dataset
