"""Profiling-only rewrites of the memory module's forward pass with the document-side work cached at write time.

The key and value projections of the cross-attention layers (and the ``doc_norm`` before them) do not depend on the
question, so a server can compute them once when documents are written and keep them next to the embeddings. Per
query only the memory slots attend over the cached keys and values. These classes measure what that would save, and
what further changes to the cached read would save. They apply to modules with learned slots and no document
self-attention (the released configuration). None of them is used by training, evaluation or ``MemoryVLLM``.

- ``CachedMemory`` is an exact rewrite of ``MemoryModule.forward``. Attention is written as two batched matrix
  products, because the head width (2560 / 8 = 320) is above what the fused SDPA kernels support and their fallback
  is slow. Two options keep it exact. When one collection serves a batch of questions (a cache written with batch
  size 1), the slots of every question attend in one product. With ``splits > 1`` the weighted sum over the N
  documents is computed in ``splits`` chunks that are added up, which gives the GPU more parallel work when N is
  large and the number of slots is small.
- ``Fp8CachedMemory`` stores the keys and values in float8 (e4m3) with one scale per document for the keys and one
  per channel for the values, and multiplies with ``torch._scaled_mm``. It is approximate; ``profiling.run`` reports
  its error. Without a GPU the products are emulated in float32, which is only good for testing.
"""

from collections.abc import Iterator
from typing import Any

import torch
import torch.nn.functional as F

from memorilla.memory import MemoryModule

FP8 = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8).max
FP8_ALIGN = 16  # torch._scaled_mm needs every dimension to be a multiple of 16
WRITE_CHUNK = 65_536  # documents projected at a time when a cache is written


def tensor_bytes(value: Any) -> int:
    """Bytes held by the tensors in a nested list or tuple."""
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, list | tuple):
        return sum(tensor_bytes(item) for item in value)
    return 0


def _round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


class CachedMemory:
    """Splits a ``MemoryModule`` into a write step (cache keys and values) and a read step (attend from the slots)."""

    def __init__(self, module: MemoryModule, splits: int = 1, write_chunk: int = WRITE_CHUNK) -> None:
        """Wrap a module.

        Args:
            module: The memory module, in evaluation mode.
            splits: Number of chunks of the weighted sum over the documents (1 computes it in one product).
            write_chunk: Documents projected at a time by ``write``, which bounds its working memory to the cache
                itself plus one chunk.

        Raises:
            ValueError: If the module initialises its slots from the documents or self-attends over the documents,
                since then the document-side work depends on the question or is not a projection.
        """
        if module.config.retrieval_init or module.config.num_self_attn_layers > 0:
            raise ValueError("CachedMemory needs learned slots and no document self-attention.")
        if splits < 1:
            raise ValueError("splits must be at least 1.")
        self.module = module
        self.heads = module.config.num_heads
        self.splits = splits
        self.write_chunk = write_chunk

    def _project(self, doc_embeds: torch.Tensor) -> Iterator[tuple[slice, list[tuple[torch.Tensor, torch.Tensor]]]]:
        """Yield the keys and values ``[collections, heads, chunk, head_dim]`` of every layer, chunk by chunk.

        Only one chunk of normalised documents and projections exists at a time, so writing a cache needs little
        memory beyond the cache itself.
        """
        collections, num_docs, dim = doc_embeds.shape
        head_dim = dim // self.heads
        for start in range(0, num_docs, self.write_chunk):
            docs = self.module.doc_norm(doc_embeds[:, start : start + self.write_chunk])
            shape = (collections, docs.shape[1], self.heads, head_dim)
            layers = []
            for attn in self.module.cross_attn_layers:
                keys = F.linear(docs, attn.in_proj_weight[dim : 2 * dim], attn.in_proj_bias[dim : 2 * dim])
                values = F.linear(docs, attn.in_proj_weight[2 * dim :], attn.in_proj_bias[2 * dim :])
                layers.append((keys.view(shape).transpose(1, 2), values.view(shape).transpose(1, 2)))
            yield slice(start, start + docs.shape[1]), layers

    @torch.inference_mode()
    def write(self, doc_embeds: torch.Tensor) -> list[Any]:
        """Compute the keys and values of every cross-attention layer.

        Args:
            doc_embeds: Document embeddings ``[collections, num_docs, embedding_dim]`` (no padding). A single
                collection can serve a batch of questions.

        Returns:
            Per layer, keys ``[collections, heads, head_dim, num_docs]`` and values
            ``[collections, heads, padded_docs, head_dim]``, where the values are zero-padded to a multiple of
            ``splits`` documents.
        """
        collections, num_docs, dim = doc_embeds.shape
        head_dim = dim // self.heads
        options = {"dtype": doc_embeds.dtype, "device": doc_embeds.device}
        padded = _round_up(num_docs, self.splits)
        cache = [
            (
                torch.empty(collections, self.heads, head_dim, num_docs, **options),
                torch.zeros(collections, self.heads, padded, head_dim, **options),
            )
            for _ in self.module.cross_attn_layers
        ]
        for rows, layers in self._project(doc_embeds):
            for (keys, values), (chunk_keys, chunk_values) in zip(cache, layers, strict=True):
                keys[..., rows] = chunk_keys.transpose(2, 3)
                values[:, :, rows] = chunk_values
        return cache

    def _group(self, queries: torch.Tensor, collections: int) -> torch.Tensor:
        """Arrange the slot queries ``[batch, heads, slots, head_dim]`` as ``[groups, rows, head_dim]``.

        With one collection for the whole batch, a group is a head and holds the slots of every question; otherwise a
        group is one (question, head) pair.
        """
        batch, heads, slots, head_dim = queries.shape
        if collections == 1 and batch > 1:
            return queries.transpose(0, 1).reshape(heads, batch * slots, head_dim)
        return queries.reshape(batch * heads, slots, head_dim)

    def _ungroup(self, attended: torch.Tensor, batch: int, slots: int) -> torch.Tensor:
        """Undo ``_group``: return ``[batch, heads, slots, head_dim]``."""
        if attended.shape[0] == self.heads and batch > 1:
            return attended.view(self.heads, batch, slots, -1).transpose(0, 1)
        return attended.view(batch, self.heads, slots, -1)

    def _attend(self, layer: Any, queries: torch.Tensor) -> torch.Tensor:
        """Attention of the slot queries ``[batch, heads, slots, head_dim]`` over one layer's cache."""
        keys, values = layer
        batch, _, slots, head_dim = queries.shape
        groups = self._group(queries, keys.shape[0])
        keys = keys.reshape(-1, head_dim, keys.shape[-1])
        values = values.reshape(-1, values.shape[-2], head_dim)
        weights = torch.softmax(torch.bmm(groups, keys).float() * head_dim**-0.5, dim=-1).to(values.dtype)
        if self.splits == 1:
            attended = torch.bmm(weights, values)
        else:
            count, rows, num_docs = weights.shape
            chunk = values.shape[1] // self.splits
            weights = F.pad(weights, (0, values.shape[1] - num_docs)).view(count, rows, self.splits, chunk)
            weights = weights.transpose(1, 2).reshape(count * self.splits, rows, chunk)
            partial = torch.bmm(weights, values.view(count * self.splits, chunk, head_dim))
            attended = partial.view(count, self.splits, rows, head_dim).sum(dim=1)
        return self._ungroup(attended, batch, slots)

    @torch.inference_mode()
    def read(self, cache: list[Any], question_embeds: torch.Tensor) -> torch.Tensor:
        """Produce the memory tokens from the cache.

        Args:
            cache: Output of ``write``.
            question_embeds: Question embeddings ``[batch, embedding_dim]``; the cache holds one collection per
                question or one collection for all of them.

        Returns:
            Memory tokens ``[batch, num_memories, output_dim]``, as ``MemoryModule.forward`` returns them.
        """
        module = self.module
        batch = question_embeds.shape[0]
        question = question_embeds.unsqueeze(1)
        slots = module.memory_queries.expand(batch, -1, -1)
        conditioned, _ = module.query_cond_attn(slots, question, question)
        slots = module.query_cond_norm(slots + conditioned)

        num_slots, dim = slots.shape[1:]
        head_dim = dim // self.heads
        for layer, norm, attn, ffn_norm, ffn in zip(
            cache, module.cross_attn_norms, module.cross_attn_layers, module.ffn_norms, module.ffn_layers, strict=True
        ):
            queries = F.linear(norm(slots), attn.in_proj_weight[:dim], attn.in_proj_bias[:dim])
            queries = queries.view(batch, num_slots, self.heads, head_dim).transpose(1, 2)
            attended = self._attend(layer, queries).transpose(1, 2).reshape(batch, num_slots, dim)
            slots = slots + attn.out_proj(attended)
            slots = slots + ffn(ffn_norm(slots))
        return module.memory_projection(module.layer_norm(slots))


def quantize_rows(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Float8 values with one float32 scale per row (the last dimension is quantised together).

    Args:
        values: ``[..., columns]``.

    Returns:
        The float8 tensor and the scales ``[..., 1]``, with ``values ~= fp8.float() * scales``.
    """
    scales = values.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / FP8_MAX
    return (values.float() / scales).to(FP8), scales


def scaled_mm(a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor) -> torch.Tensor:
    """``(a * scale_a) @ (b * scale_b.T).T`` in bfloat16 for float8 ``a`` ``[M, K]`` and ``b`` ``[N, K]``.

    Args:
        a: Row-major float8 ``[M, K]``.
        b: Row-major float8 ``[N, K]`` (passed to ``torch._scaled_mm`` as the column-major ``[K, N]``).
        scale_a: Float32 ``[M, 1]``.
        scale_b: Float32 ``[1, N]``.

    Returns:
        The bfloat16 product ``[M, N]``.
    """
    if a.is_cuda:
        return torch._scaled_mm(a, b.t(), scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16)
    return ((a.float() * scale_a) @ (b.float() * scale_b.t()).t()).to(torch.bfloat16)


class Fp8CachedMemory(CachedMemory):
    """``CachedMemory`` with float8 keys and values multiplied by ``torch._scaled_mm`` (approximate)."""

    def __init__(self, module: MemoryModule, write_chunk: int = WRITE_CHUNK) -> None:
        """Wrap a module; see ``CachedMemory``. Float8 zero is the all-zero bit pattern, so padding is zeroed bytes."""
        super().__init__(module, splits=1, write_chunk=write_chunk)

    @torch.inference_mode()
    def write(self, doc_embeds: torch.Tensor) -> list[Any]:
        """Compute and quantise the keys and values of every cross-attention layer.

        Args:
            doc_embeds: Document embeddings ``[collections, num_docs, embedding_dim]`` (no padding).

        Returns:
            Per layer, ``(keys, key_scales, values, value_scales, num_docs)``: float8 keys
            ``[collections, heads, padded_docs, head_dim]`` with one scale per document ``[..., 1, padded_docs]``,
            and float8 values stored transposed, ``[collections, heads, head_dim, padded_docs]``, with one scale per
            channel ``[..., 1, head_dim]``. Documents are zero-padded to a multiple of 16.
        """
        collections, num_docs, dim = doc_embeds.shape
        head_dim = dim // self.heads
        device = doc_embeds.device
        padded = _round_up(num_docs, FP8_ALIGN)
        layers = len(self.module.cross_attn_layers)
        # The value scales are per channel over all documents, so a first pass finds each channel's largest value.
        value_max = [torch.zeros(collections, self.heads, head_dim, device=device) for _ in range(layers)]
        for _, chunk in self._project(doc_embeds):
            for index, (_, values) in enumerate(chunk):
                value_max[index] = torch.maximum(value_max[index], values.float().abs().amax(dim=2))
        value_scales = [(peak.clamp(min=1e-12) / FP8_MAX).unsqueeze(2) for peak in value_max]

        cache = []
        for index in range(layers):
            cache.append(
                [
                    torch.zeros(collections, self.heads, padded, head_dim, device=device, dtype=torch.uint8).view(FP8),
                    torch.ones(collections, self.heads, 1, padded, device=device),
                    torch.zeros(collections, self.heads, head_dim, padded, device=device, dtype=torch.uint8).view(FP8),
                    value_scales[index].contiguous(),
                    num_docs,
                ]
            )
        for rows, chunk in self._project(doc_embeds):
            for (keys8, key_scales, values8, scales, _), (keys, values) in zip(cache, chunk, strict=True):
                quantized, row_scales = quantize_rows(keys)
                keys8[:, :, rows] = quantized
                key_scales[:, :, 0, rows] = row_scales.squeeze(-1)
                values8[..., rows] = (values.float() / scales).to(FP8).transpose(2, 3)
        return [tuple(layer) for layer in cache]

    def _attend(self, layer: Any, queries: torch.Tensor) -> torch.Tensor:
        """Attention of the slot queries over one layer's float8 cache, one product per (group, head)."""
        keys, key_scales, values, value_scales, num_docs = layer
        batch, heads, slots, head_dim = queries.shape
        collections = keys.shape[0]
        shared = collections == 1 and batch > 1
        groups = self._group(queries, collections)
        rows = groups.shape[1]
        groups = F.pad(groups, (0, 0, 0, _round_up(rows, FP8_ALIGN) - rows))
        outputs = []
        for index in range(groups.shape[0]):
            collection, head = (0, index) if shared else divmod(index, heads)
            query8, query_scales = quantize_rows(groups[index])
            scores = scaled_mm(query8, keys[collection, head], query_scales, key_scales[collection, head])
            scores = scores[:, :num_docs].float() * head_dim**-0.5
            weights = F.pad(torch.softmax(scores, dim=-1), (0, keys.shape[2] - num_docs))
            weights8, weight_scales = quantize_rows(weights)
            outputs.append(scaled_mm(weights8, values[collection, head], weight_scales, value_scales[collection, head]))
        attended = torch.stack(outputs)[:, :rows].to(queries.dtype)
        return self._ungroup(attended, batch, slots)
