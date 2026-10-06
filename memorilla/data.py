"""Dataset access, chat formatting and batch collation.

Every config of the data repository stores ``data/<config>/<split>-XXXXX-of-YYYYY.parquet`` with the columns
``collection_id``, ``question``, ``answer`` and ``documents`` (plus ``choices`` and ``hard`` where they apply).
"""

from bisect import bisect_right
from collections.abc import Iterable, Mapping
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any

from datasets import Dataset, load_dataset
from huggingface_hub import constants as hub_constants
from huggingface_hub import snapshot_download
import pyarrow.parquet as pq
import torch
from transformers import PreTrainedTokenizerBase

from memorilla.benchmarks import HELPFUL_PROMPT
from memorilla.paths import DATA_DIR, DATA_REPO, data_root, local_repo
from memorilla.utils import MEMORY_TOKEN

if TYPE_CHECKING:
    from memorilla.embeddings import EmbeddingStore

COLLECTION_COLUMN = "collection_id"
QUESTION_COLUMN = "question"
ANSWER_COLUMN = "answer"
DOCUMENTS_COLUMN = "documents"
CHOICES_COLUMN = "choices"
HARD_COLUMN = "hard"
EVAL_COLUMNS = [COLLECTION_COLUMN, QUESTION_COLUMN, ANSWER_COLUMN, CHOICES_COLUMN, HARD_COLUMN]
TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "validation"
TEST_SPLIT = "test"
SHARD_NAME = re.compile(r"-(\d+)-of-(\d+)\.parquet$")
DEFAULT_SYSTEM_PROMPT = HELPFUL_PROMPT
IGNORE_INDEX = -100


def split_files(config: str, split: str, root: str | Path) -> list[Path]:
    """List the parquet files of one split.

    Args:
        config: Dataset config name.
        split: Split name.
        root: Data root (contains ``data/``).

    Returns:
        The sorted parquet paths; empty when the split does not exist locally.
    """
    return sorted((Path(root) / "data" / config).glob(f"{split}-*.parquet"))


def split_complete(config: str, split: str, root: str | Path) -> bool:
    """Whether every parquet file of one split is present locally.

    Files named ``<split>-XXXXX-of-YYYYY.parquet`` must cover all ``YYYYY`` shards; a split stored under other file
    names is complete as soon as one file exists.

    Args:
        config: Dataset config name.
        split: Split name.
        root: Data root (contains ``data/``).

    Returns:
        True when the split can be loaded in full.
    """
    files = split_files(config, split, root)
    shards = [match for path in files if (match := SHARD_NAME.search(path.name))]
    if not shards:
        return bool(files)
    totals = {int(match.group(2)) for match in shards}
    return len(totals) == 1 and {int(match.group(1)) for match in shards} == set(range(totals.pop()))


def ensure_data(
    config: str,
    splits: Iterable[str],
    repo: str = DATA_REPO,
    data_dir: str | Path = DATA_DIR,
) -> Path:
    """Make the parquet files of the given splits available locally, downloading only what is missing.

    Splits the repository does not publish are left absent; check them with ``split_files``. A split with some of
    its shards missing is downloaded again (files already present are kept).

    Args:
        config: Dataset config name.
        splits: Split names.
        repo: Hub dataset repo id, or a local directory with the same layout.
        data_dir: Local mirror used when ``repo`` is a Hub repo id.

    Returns:
        The data root.
    """
    root = data_root(repo, data_dir)
    missing = [split for split in splits if not split_complete(config, split, root)]
    if missing and local_repo(repo) is None and not hub_constants.HF_HUB_OFFLINE:
        patterns = [f"data/{config}/{split}-*.parquet" for split in missing]
        snapshot_download(repo_id=repo, repo_type="dataset", allow_patterns=patterns, local_dir=root)
    return root


def load_split(config: str, split: str, root: str | Path, columns: list[str] | None = None) -> Dataset:
    """Load one split from local parquet files.

    Args:
        config: Dataset config name.
        split: Split name.
        root: Data root (contains ``data/``).
        columns: Columns to load; names the split does not have are skipped. None loads every column.

    Returns:
        The split as a ``datasets.Dataset`` in its stored row order.

    Raises:
        FileNotFoundError: If the split has no parquet files under ``root`` or some of its shards are missing.
    """
    files = split_files(config, split, root)
    if not files:
        raise FileNotFoundError(f"No parquet files for {config}/{split} under {Path(root) / 'data' / config}.")
    if not split_complete(config, split, root):
        raise FileNotFoundError(f"Some parquet shards of {config}/{split} are missing under {files[0].parent}.")
    if columns is not None:
        available = set(pq.read_schema(files[0]).names)
        columns = [column for column in columns if column in available]
    return load_dataset("parquet", data_files={split: [str(f) for f in files]}, split=split, columns=columns)


def memory_placeholder(num_memories: int) -> str:
    """Return the text appended to the system prompt that reserves the memory positions.

    Args:
        num_memories: Number of memory tokens.

    Returns:
        A newline, ``Memory``, ``<|memory|>`` repeated ``num_memories`` times, then ``Memory`` again.
    """
    return f"\nMemory{MEMORY_TOKEN * num_memories}Memory"


def build_messages(
    question: str,
    system_prompt: str,
    num_memories: int,
    suffix: str = "",
    answer: str | None = None,
) -> list[dict[str, str]]:
    """Build the chat for one example.

    Args:
        question: The question; surrounding whitespace is removed.
        system_prompt: System prompt; the memory placeholder is appended to it.
        num_memories: Number of memory tokens.
        suffix: Text appended to the user turn after the question.
        answer: Assistant turn, or None for a generation prompt.

    Returns:
        The chat messages.
    """
    messages = [
        {"role": "system", "content": system_prompt + memory_placeholder(num_memories)},
        {"role": "user", "content": str(question).strip() + suffix},
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": str(answer).strip()})
    return messages


def apply_chat_template(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
    add_generation_prompt: bool,
) -> str:
    """Render a chat with the decoder's template, with thinking disabled.

    Args:
        tokenizer: Decoder tokenizer.
        messages: Chat messages.
        add_generation_prompt: Whether to append the assistant header.

    Returns:
        The rendered text.
    """
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=False
    )


class MemoryDataset(torch.utils.data.Dataset):
    """Concatenation of dataset splits whose items remember their source.

    Every item carries ``config``, ``split`` and ``row`` so the collator can look up the matching embeddings.
    """

    def __init__(self, parts: list[tuple[str, str, Dataset]]) -> None:
        """Index the parts.

        Args:
            parts: ``(config, split, dataset)`` triples, concatenated in order.
        """
        self.parts = parts
        self.offsets = [0]
        for _, _, dataset in parts:
            self.offsets.append(self.offsets[-1] + len(dataset))

    def __len__(self) -> int:
        """Count the rows of all parts.

        Returns:
            The total number of rows.
        """
        return self.offsets[-1]

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one row with its source fields.

        Args:
            index: Global row index.

        Returns:
            The row's columns plus ``config``, ``split`` and ``row``.
        """
        part = bisect_right(self.offsets, index) - 1
        config, split, dataset = self.parts[part]
        row = index - self.offsets[part]
        return {**dataset[row], "config": config, "split": split, "row": row}


class MemoryCollator:
    """Turn rows into decoder inputs, document embeddings and question embeddings.

    Prompts are rendered with the chat template and left padded. In training mode the assistant answer is appended
    and ``labels`` supervise only the answer tokens; otherwise the prompt ends with the assistant header.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        stores: Mapping[tuple[str, str], "EmbeddingStore"],
        num_memories: int,
        max_length: int,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        suffix: str = "",
        train: bool = True,
    ) -> None:
        """Configure the collator.

        Args:
            tokenizer: Decoder tokenizer with the memory token registered.
            stores: Embedding stores keyed by ``(config, split)``.
            num_memories: Number of memory placeholders per prompt.
            max_length: Maximum tokenised length (longer texts are truncated).
            system_prompt: System prompt.
            suffix: Text appended to every user turn.
            train: Whether to append answers and build labels.
        """
        self.tokenizer = tokenizer
        self.stores = stores
        self.num_memories = num_memories
        self.max_length = max_length
        self.system_prompt = system_prompt
        self.suffix = suffix
        self.train = train

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Collate a batch.

        Args:
            batch: Items from ``MemoryDataset``.

        Returns:
            ``input_ids``, ``attention_mask``, ``doc_embeds`` (bfloat16), ``doc_padding_mask``, ``question_embeds``
            (bfloat16) and, in training mode, ``labels``.
        """
        texts, prompt_lengths = [], []
        for item in batch:
            answer = item[ANSWER_COLUMN] if self.train else None
            messages = build_messages(item[QUESTION_COLUMN], self.system_prompt, self.num_memories, self.suffix, answer)
            prompt = apply_chat_template(self.tokenizer, messages[:2], add_generation_prompt=True)
            if self.train:
                texts.append(apply_chat_template(self.tokenizer, messages, add_generation_prompt=False))
                prompt_lengths.append(len(self.tokenizer.encode(prompt, add_special_tokens=False)))
            else:
                texts.append(prompt)

        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            padding_side="left",
            return_tensors="pt",
        )
        doc_embeds, doc_padding_mask = self._documents(batch)
        stores = [self.stores[(item["config"], item["split"])] for item in batch]
        question_embeds = torch.stack([store.question(item["row"]) for store, item in zip(stores, batch, strict=True)])

        output = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "doc_embeds": doc_embeds.to(torch.bfloat16),
            "doc_padding_mask": doc_padding_mask,
            "question_embeds": question_embeds.to(torch.bfloat16),
        }
        if self.train:
            output["labels"] = self._labels(encoded["input_ids"], encoded["attention_mask"], prompt_lengths)
        return output

    def _documents(self, batch: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather and pad the document embeddings of a batch.

        Args:
            batch: Items from ``MemoryDataset``.

        Returns:
            Embeddings ``[batch, max_docs, D]`` and a padding mask ``[batch, max_docs]`` (True = padding).
        """
        rows = [self.stores[(item["config"], item["split"])].documents(item[COLLECTION_COLUMN]) for item in batch]
        max_docs = max(row.shape[0] for row in rows)
        embeds = torch.zeros(len(rows), max_docs, rows[0].shape[-1], dtype=rows[0].dtype)
        padding_mask = torch.ones(len(rows), max_docs, dtype=torch.bool)
        for index, row in enumerate(rows):
            embeds[index, : row.shape[0]] = row
            padding_mask[index, : row.shape[0]] = False
        return embeds, padding_mask

    @staticmethod
    def _labels(input_ids: torch.Tensor, attention_mask: torch.Tensor, prompt_lengths: list[int]) -> torch.Tensor:
        """Build labels that supervise only the tokens after each prompt.

        Args:
            input_ids: Left-padded ids ``[batch, seq_len]``.
            attention_mask: ``[batch, seq_len]`` with 1 for real tokens.
            prompt_lengths: Token length of each prompt.

        Returns:
            ``input_ids`` with prompt and padding positions set to ``IGNORE_INDEX``.
        """
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        for index, prompt_length in enumerate(prompt_lengths):
            start = int((attention_mask[index] == 0).sum()) + prompt_length
            labels[index, start:] = input_ids[index, start:]
        labels[attention_mask == 0] = IGNORE_INDEX
        return labels
