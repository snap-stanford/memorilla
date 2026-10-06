"""The Memorilla memory module.

``MemoryModule`` compresses a variable number of document embeddings into a fixed set of memory vectors, conditioned
on the question, and projects them into the decoder's input embedding space.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from memorilla.utils import CHECKPOINT_FILE, CONFIG_FILE, checkpoint_file, load_memory_state

DEFAULT_NUM_MEMORIES = 16
DEFAULT_NUM_HEADS = 8
DEFAULT_NUM_SELF_ATTN_LAYERS = 0
DEFAULT_NUM_CROSS_ATTN_LAYERS = 2
DEFAULT_DROPOUT = 0.1
FFN_EXPANSION = 4


@dataclass(frozen=True)
class MemoryConfig:
    """Constructor arguments of a ``MemoryModule``.

    Attributes:
        embedding_dim: Width of the document and question embeddings (encoder hidden size).
        output_dim: Width of the memory tokens (decoder hidden size).
        num_memories: Number of memory tokens produced per example.
        num_heads: Attention heads in every attention layer.
        num_self_attn_layers: Transformer encoder layers applied to the documents (0 disables them).
        num_cross_attn_layers: Number of cross-attention + feed-forward refinement blocks.
        dropout: Dropout probability.
        retrieval_init: Whether the slots start from the most question-similar documents instead of learned vectors.
    """

    embedding_dim: int
    output_dim: int
    num_memories: int = DEFAULT_NUM_MEMORIES
    num_heads: int = DEFAULT_NUM_HEADS
    num_self_attn_layers: int = DEFAULT_NUM_SELF_ATTN_LAYERS
    num_cross_attn_layers: int = DEFAULT_NUM_CROSS_ATTN_LAYERS
    dropout: float = DEFAULT_DROPOUT
    retrieval_init: bool = False

    @classmethod
    def from_state_dict(cls, state: dict[str, torch.Tensor]) -> "MemoryConfig":
        """Infer the architecture from a state dict.

        The number of attention heads and the dropout rate leave no trace in the weights and take their defaults, as
        does the number of memory slots of a ``retrieval_init`` module.

        Args:
            state: A ``MemoryModule`` state dict.

        Returns:
            The inferred configuration.
        """
        retrieval_init = "memory_queries" not in state
        return cls(
            embedding_dim=state["query_cond_norm.weight"].shape[0],
            output_dim=state["memory_projection.weight"].shape[0],
            num_memories=DEFAULT_NUM_MEMORIES if retrieval_init else state["memory_queries"].shape[1],
            num_self_attn_layers=_count_layers(state, r"encoder\.layers\.(\d+)\."),
            num_cross_attn_layers=_count_layers(state, r"cross_attn_layers\.(\d+)\."),
            retrieval_init=retrieval_init,
        )


def _count_layers(state: dict[str, torch.Tensor], pattern: str) -> int:
    """Count the distinct layer indices matched by ``pattern`` in the state dict keys.

    Args:
        state: A state dict.
        pattern: Regular expression with one group capturing the layer index.

    Returns:
        The number of distinct layer indices.
    """
    indices = {match.group(1) for key in state if (match := re.match(pattern, key))}
    return len(indices)


def _multihead_attention(embedding_dim: int, num_heads: int, dropout: float) -> nn.MultiheadAttention:
    """Build a batch-first multi-head attention layer.

    Args:
        embedding_dim: Model width.
        num_heads: Number of heads.
        dropout: Attention dropout.

    Returns:
        The attention layer.
    """
    return nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=dropout, batch_first=True)


class MemoryModule(nn.Module):
    """Question-conditioned compression of document embeddings into ``num_memories`` memory tokens.

    The forward pass:

    1. Initial slots: ``num_memories`` learned vectors, or with ``retrieval_init`` the documents most similar to the
       question (cosine similarity, zero-padded when there are fewer documents than slots).
    2. Question conditioning: slots cross-attend to the question embedding (residual + LayerNorm).
    3. Optional document self-attention encoder with ``num_self_attn_layers`` Transformer layers.
    4. ``num_cross_attn_layers`` refinement blocks: pre-norm cross-attention from the slots to the normalised
       documents, then a pre-norm feed-forward network, both residual.
    5. Final LayerNorm and a linear projection into the decoder embedding space.
    """

    def __init__(
        self,
        embedding_dim: int,
        output_dim: int,
        num_memories: int = DEFAULT_NUM_MEMORIES,
        num_heads: int = DEFAULT_NUM_HEADS,
        num_self_attn_layers: int = DEFAULT_NUM_SELF_ATTN_LAYERS,
        num_cross_attn_layers: int = DEFAULT_NUM_CROSS_ATTN_LAYERS,
        dropout: float = DEFAULT_DROPOUT,
        retrieval_init: bool = False,
    ) -> None:
        """Build the module.

        Args:
            embedding_dim: Width of the document and question embeddings (encoder hidden size).
            output_dim: Width of the memory tokens (decoder hidden size).
            num_memories: Number of memory tokens produced per example.
            num_heads: Attention heads in every attention layer.
            num_self_attn_layers: Transformer encoder layers applied to the documents (0 disables them).
            num_cross_attn_layers: Number of cross-attention + feed-forward refinement blocks.
            dropout: Dropout probability.
            retrieval_init: Initialise the slots from the most question-similar documents instead of learned vectors.
        """
        super().__init__()
        self.config = MemoryConfig(
            embedding_dim=embedding_dim,
            output_dim=output_dim,
            num_memories=num_memories,
            num_heads=num_heads,
            num_self_attn_layers=num_self_attn_layers,
            num_cross_attn_layers=num_cross_attn_layers,
            dropout=dropout,
            retrieval_init=retrieval_init,
        )

        if not retrieval_init:
            self.memory_queries = nn.Parameter(torch.randn(1, num_memories, embedding_dim))
            nn.init.xavier_uniform_(self.memory_queries)

        self.query_cond_attn = _multihead_attention(embedding_dim, num_heads, dropout)
        self.query_cond_norm = nn.LayerNorm(embedding_dim)

        if num_self_attn_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=embedding_dim * FFN_EXPANSION,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_self_attn_layers)
        self.doc_norm = nn.LayerNorm(embedding_dim)

        self.cross_attn_norms = nn.ModuleList([nn.LayerNorm(embedding_dim) for _ in range(num_cross_attn_layers)])
        self.cross_attn_layers = nn.ModuleList(
            [_multihead_attention(embedding_dim, num_heads, dropout) for _ in range(num_cross_attn_layers)]
        )
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(embedding_dim) for _ in range(num_cross_attn_layers)])
        self.ffn_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embedding_dim, embedding_dim * FFN_EXPANSION),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(embedding_dim * FFN_EXPANSION, embedding_dim),
                    nn.Dropout(dropout),
                )
                for _ in range(num_cross_attn_layers)
            ]
        )
        self.attn_dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.memory_projection = nn.Linear(embedding_dim, output_dim)
        nn.init.xavier_uniform_(self.memory_projection.weight)

    @property
    def num_memories(self) -> int:
        """Number of memory tokens produced per example."""
        return self.config.num_memories

    def forward(
        self,
        doc_embeds: torch.Tensor,
        doc_padding_mask: torch.Tensor | None,
        question_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Compress documents into memory tokens.

        Args:
            doc_embeds: Document embeddings ``[batch, num_docs, embedding_dim]``.
            doc_padding_mask: ``[batch, num_docs]`` with True at padding positions, or None when nothing is padded.
            question_embeds: Question embeddings ``[batch, embedding_dim]``.

        Returns:
            Memory tokens ``[batch, num_memories, output_dim]``.
        """
        question = question_embeds.unsqueeze(1)
        slots = self._initial_slots(doc_embeds, doc_padding_mask, question_embeds)

        conditioned, _ = self.query_cond_attn(slots, question, question)
        slots = self.query_cond_norm(slots + conditioned)

        if self.config.num_self_attn_layers > 0:
            doc_embeds = self.encoder(doc_embeds, src_key_padding_mask=doc_padding_mask)
        docs = self.doc_norm(doc_embeds)

        for cross_attn_norm, cross_attn, ffn_norm, ffn in zip(
            self.cross_attn_norms, self.cross_attn_layers, self.ffn_norms, self.ffn_layers, strict=True
        ):
            attended, _ = cross_attn(cross_attn_norm(slots), docs, docs, key_padding_mask=doc_padding_mask)
            slots = slots + self.attn_dropout(attended)
            slots = slots + ffn(ffn_norm(slots))

        return self.memory_projection(self.layer_norm(slots))

    def _initial_slots(
        self,
        doc_embeds: torch.Tensor,
        doc_padding_mask: torch.Tensor | None,
        question_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Return the slots before question conditioning.

        Args:
            doc_embeds: Document embeddings ``[batch, num_docs, embedding_dim]``.
            doc_padding_mask: ``[batch, num_docs]`` padding mask or None.
            question_embeds: Question embeddings ``[batch, embedding_dim]``.

        Returns:
            Initial slots ``[batch, num_memories, embedding_dim]``.
        """
        batch_size = doc_embeds.size(0)
        if not self.config.retrieval_init:
            return self.memory_queries.expand(batch_size, -1, -1)

        docs = F.normalize(doc_embeds, p=2, dim=-1)
        question = F.normalize(question_embeds, p=2, dim=-1)
        scores = torch.bmm(docs, question.unsqueeze(-1)).squeeze(-1)
        if doc_padding_mask is not None:
            scores = scores.masked_fill(doc_padding_mask, float("-inf"))

        k = min(self.num_memories, scores.size(1))
        top_indices = scores.topk(k, dim=-1).indices
        slots = torch.gather(doc_embeds, 1, top_indices.unsqueeze(-1).expand(-1, -1, doc_embeds.size(-1)))
        if k < self.num_memories:
            padding = slots.new_zeros(batch_size, self.num_memories - k, doc_embeds.size(-1))
            slots = torch.cat([slots, padding], dim=1)
        return slots

    def save(self, directory: str | Path) -> Path:
        """Write ``memory.pt`` (state dict) and ``config.json`` (constructor arguments).

        Args:
            directory: Output directory, created if needed.

        Returns:
            The output directory.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), directory / CHECKPOINT_FILE)
        with open(directory / CONFIG_FILE, "w") as handle:
            json.dump(asdict(self.config), handle, indent=2)
            handle.write("\n")
        return directory

    @classmethod
    def from_pretrained(cls, path: str | Path, **overrides: int | float | bool) -> "MemoryModule":
        """Load a module from a checkpoint directory or ``memory.pt`` file.

        The architecture comes from ``config.json`` next to the weights when present and is otherwise inferred from
        the state dict. Weights are loaded strictly.

        Args:
            path: Checkpoint directory or weights file.
            **overrides: Constructor arguments that replace the stored or inferred ones (e.g. ``num_heads``).

        Returns:
            The loaded module, in evaluation mode, with the dtype of the stored weights.
        """
        weights = checkpoint_file(path)
        state = load_memory_state(weights)
        config_path = weights.parent / CONFIG_FILE
        if config_path.exists():
            with open(config_path) as handle:
                config = MemoryConfig(**json.load(handle))
        else:
            config = MemoryConfig.from_state_dict(state)

        module = cls(**{**asdict(config), **overrides})
        dtype = next(iter(state.values())).dtype
        module.to(dtype=dtype).load_state_dict(state, strict=True)
        return module.eval()
