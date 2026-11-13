"""Profiling-only rewrites of the memory module that cache the question-independent document work at write time.

``CachedMemory`` is an exact rewrite of ``MemoryModule.forward``; ``Fp8CachedMemory`` stores its cache in float8 and
is approximate. Neither is used by training, evaluation or ``MemoryVLLM``.
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
    """A ``MemoryModule`` split into ``write`` (cache keys and values) and ``read`` (attend from the slots).

    Attention is two batched matrix products, since the head width (320) is above what fused SDPA kernels support.
    """

    def __init__(self, module: MemoryModule, splits: int = 1, write_chunk: int = WRITE_CHUNK) -> None:
        """Wrap a module.

        Args:
            module: The memory module, in evaluation mode.
            splits: Chunks of the weighted sum over the documents.
            write_chunk: Documents projected at a time by ``write``.

        Raises:
            ValueError: If the module uses retrieval-initialised slots or document self-attention.
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
        """Yield each chunk's keys and values ``[collections, heads, chunk, head_dim]`` for every layer."""
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
            doc_embeds: ``[collections, num_docs, embedding_dim]``; one collection can serve a batch of questions.

        Returns:
            Per layer, keys ``[collections, heads, head_dim, num_docs]`` and values zero-padded to a multiple of
            ``splits`` documents, ``[collections, heads, padded_docs, head_dim]``.
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
        """Reshape ``[batch, heads, slots, head_dim]`` queries into ``[groups, rows, head_dim]``.

        A group is a head over every question when the batch shares one collection, else a (question, head) pair.
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
        """Attend from ``[batch, heads, slots, head_dim]`` queries over one layer's cache."""
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
            cache: Output of ``write``, with one collection per question or one for all of them.
            question_embeds: ``[batch, embedding_dim]``.

        Returns:
            Memory tokens ``[batch, num_memories, output_dim]``.
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
    """Quantise to float8 with one scale per row; returns the values and the scales ``[..., 1]``."""
    scales = values.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / FP8_MAX
    return (values.float() / scales).to(FP8), scales


def scaled_mm(a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor) -> torch.Tensor:
    """``(a * scale_a) @ (b * scale_b.T).T`` in bfloat16 for float8 ``a [M, K]`` and ``b [N, K]``.

    Uses ``torch._scaled_mm`` on a GPU and emulates it in float32 otherwise (for tests).
    """
    if a.is_cuda:
        return torch._scaled_mm(a, b.t(), scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16)
    return ((a.float() * scale_a) @ (b.float() * scale_b.t()).t()).to(torch.bfloat16)


class Fp8CachedMemory(CachedMemory):
    """``CachedMemory`` with float8 keys (scaled per document) and values (scaled per channel); approximate."""

    def __init__(self, module: MemoryModule, write_chunk: int = WRITE_CHUNK) -> None:
        """Wrap a module; see ``CachedMemory``."""
        super().__init__(module, splits=1, write_chunk=write_chunk)

    @torch.inference_mode()
    def write(self, doc_embeds: torch.Tensor) -> list[Any]:
        """Compute and quantise the keys and values of every cross-attention layer.

        Returns:
            Per layer, ``(keys, key_scales, values, value_scales, num_docs)``, with documents zero-padded to a
            multiple of 16 and the values stored transposed, ``[collections, heads, head_dim, padded_docs]``.
        """
        collections, num_docs, dim = doc_embeds.shape
        head_dim = dim // self.heads
        device = doc_embeds.device
        padded = _round_up(num_docs, FP8_ALIGN)
        layers = len(self.module.cross_attn_layers)
        # First pass: each value channel's largest magnitude over all documents.
        value_max = [torch.zeros(collections, self.heads, head_dim, device=device) for _ in range(layers)]
        for _, chunk in self._project(doc_embeds):
            for index, (_, values) in enumerate(chunk):
                value_max[index] = torch.maximum(value_max[index], values.float().abs().amax(dim=2))
        value_scales = [(peak.clamp(min=1e-12) / FP8_MAX).unsqueeze(2) for peak in value_max]

        cache = []
        for index in range(layers):
            cache.append(
                [
                    # float8 zero is the all-zero bit pattern
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
        """Attend over one layer's float8 cache, one product per group."""
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
