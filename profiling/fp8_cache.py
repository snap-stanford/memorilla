"""An experimental float8 version of the memory cache, measured by ``profiling.run --cache kv_fp8``.

Keys are scaled per document and values per channel, and the products run through ``torch._scaled_mm``. It is
approximate and, as implemented, slower than the bfloat16 cache of ``MemoryModule.write_cache``, so it is not part of
the library.
"""

from typing import Any

import torch
import torch.nn.functional as F

from memorilla.memory import CACHE_CHUNK_SIZE, MemoryModule

FP8 = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8).max
FP8_ALIGN = 16  # torch._scaled_mm needs every dimension to be a multiple of 16


def _round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def quantize_rows(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantise to float8 with one scale per row; returns the values and the scales ``[..., 1]``."""
    scales = values.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / FP8_MAX
    return (values.float() / scales).to(FP8), scales


def scaled_mm(a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor) -> torch.Tensor:
    """``(a * scale_a) @ (b * scale_b.T).T`` in bfloat16 for float8 ``a [M, K]`` and ``b [N, K]``.

    Uses ``torch._scaled_mm`` on a GPU and emulates it in float32 otherwise (for CPU runs).
    """
    if a.is_cuda:
        return torch._scaled_mm(a, b.t(), scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16)
    return ((a.float() * scale_a) @ (b.float() * scale_b.t()).t()).to(torch.bfloat16)


class Fp8Cache:
    """Writes and reads a float8 memory cache for a ``MemoryModule`` that supports caching."""

    def __init__(self, module: MemoryModule, chunk_size: int = CACHE_CHUNK_SIZE) -> None:
        """Wrap a module.

        Raises:
            ValueError: If the module does not support caching.
        """
        if not module.supports_cache:
            raise ValueError("Caching needs learned memory slots and no document self-attention.")
        self.module = module
        self.heads = module.config.num_heads
        self.chunk_size = chunk_size

    @torch.no_grad()
    def write(self, doc_embeds: torch.Tensor) -> list[tuple[Any, ...]]:
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
        for _, chunk in self.module.project_documents(doc_embeds, self.chunk_size):
            for index, (_, values) in enumerate(chunk):
                value_max[index] = torch.maximum(value_max[index], values.float().abs().amax(dim=2))
        value_scales = [(peak.clamp(min=1e-12) / FP8_MAX).unsqueeze(2) for peak in value_max]

        def zeros(*shape: int) -> torch.Tensor:  # float8 zero is the all-zero bit pattern
            return torch.zeros(*shape, device=device, dtype=torch.uint8).view(FP8)

        cache = [
            [
                zeros(collections, self.heads, padded, head_dim),
                torch.ones(collections, self.heads, 1, padded, device=device),
                zeros(collections, self.heads, head_dim, padded),
                scales.contiguous(),
                num_docs,
            ]
            for scales in value_scales
        ]
        for rows, chunk in self.module.project_documents(doc_embeds, self.chunk_size):
            for (keys8, key_scales, values8, scales, _), (keys, values) in zip(cache, chunk, strict=True):
                quantized, row_scales = quantize_rows(keys)
                keys8[:, :, rows] = quantized
                key_scales[:, :, 0, rows] = row_scales.squeeze(-1)
                values8[..., rows] = (values.float() / scales).to(FP8).transpose(2, 3)
        return [tuple(layer) for layer in cache]

    @torch.no_grad()
    def read(self, cache: list[tuple[Any, ...]], question_embeds: torch.Tensor) -> torch.Tensor:
        """Produce memory tokens from the cache, as ``MemoryModule.read_cache`` does."""
        module = self.module
        batch = question_embeds.size(0)
        question = question_embeds.unsqueeze(1)
        slots = module.memory_queries.expand(batch, -1, -1)
        conditioned, _ = module.query_cond_attn(slots, question, question)
        slots = module.query_cond_norm(slots + conditioned)
        for layer, norm, attn, ffn_norm, ffn in zip(
            cache, module.cross_attn_norms, module.cross_attn_layers, module.ffn_norms, module.ffn_layers, strict=True
        ):
            slots = slots + attn.out_proj(self._attend(layer, attn, norm(slots)))
            slots = slots + ffn(ffn_norm(slots))
        return module.memory_projection(module.layer_norm(slots))

    def _attend(self, layer: tuple[Any, ...], attn: torch.nn.MultiheadAttention, slots: torch.Tensor) -> torch.Tensor:
        """Attend from the slots over one layer's float8 cache, one product per (collection, head)."""
        keys, key_scales, values, value_scales, num_docs = layer
        batch, num_slots, dim = slots.shape
        head_dim = dim // self.heads
        queries = F.linear(slots, attn.in_proj_weight[:dim], attn.in_proj_bias[:dim])
        queries = queries.view(batch, num_slots, self.heads, head_dim).transpose(1, 2)
        shared = keys.size(0) == 1 and batch > 1
        if shared:  # every question's slots attend over the one collection together
            queries = queries.transpose(0, 1).reshape(1, self.heads, batch * num_slots, head_dim)
        rows = queries.size(2)
        queries = F.pad(queries, (0, 0, 0, _round_up(rows, FP8_ALIGN) - rows))
        outputs = torch.empty(queries.shape, dtype=torch.bfloat16, device=queries.device)
        for collection in range(queries.size(0)):
            for head in range(self.heads):
                query8, query_scales = quantize_rows(queries[collection, head])
                scores = scaled_mm(query8, keys[collection, head], query_scales, key_scales[collection, head])
                weights = torch.softmax(scores[:, :num_docs].float() * head_dim**-0.5, dim=-1)
                weights8, weight_scales = quantize_rows(F.pad(weights, (0, keys.size(2) - num_docs)))
                outputs[collection, head] = scaled_mm(
                    weights8, values[collection, head], weight_scales, value_scales[collection, head]
                )
        attended = outputs[:, :, :rows]
        if shared:
            attended = attended.view(self.heads, batch, num_slots, head_dim).transpose(0, 1)
        return attended.transpose(1, 2).reshape(batch, num_slots, dim).to(slots.dtype)
