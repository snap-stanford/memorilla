"""Training-time wrapper: a frozen Hugging Face decoder whose ``<|memory|>`` positions carry memory tokens."""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from memorilla.memory import MemoryModule
from memorilla.utils import prepare_tokenizer


def inject_memory(
    inputs_embeds: torch.Tensor,
    memory_tokens: torch.Tensor,
    input_ids: torch.Tensor,
    memory_token_id: int,
) -> torch.Tensor:
    """Replace the embeddings at ``<|memory|>`` positions with memory tokens, row by row in order.

    Args:
        inputs_embeds: Token embeddings ``[batch, seq_len, hidden]``.
        memory_tokens: Memory tokens ``[batch, num_memories, hidden]``; each row must contain exactly
            ``num_memories`` placeholders.
        input_ids: Token ids ``[batch, seq_len]`` used to locate the placeholders.
        memory_token_id: Id of the ``<|memory|>`` token.

    Returns:
        A new embedding tensor with the memory tokens scattered in.
    """
    hidden_size = inputs_embeds.size(-1)
    flat_embeds = inputs_embeds.reshape(-1, hidden_size)
    flat_memory = memory_tokens.reshape(-1, hidden_size).to(dtype=inputs_embeds.dtype)
    positions = (input_ids.reshape(-1) == memory_token_id).nonzero(as_tuple=False).squeeze(1)

    delta = flat_memory - flat_embeds.index_select(0, positions)
    flat_embeds = flat_embeds.scatter_add(0, positions.unsqueeze(1).expand(-1, hidden_size), delta)
    return flat_embeds.view_as(inputs_embeds)


class MemoryLLM(nn.Module):
    """A frozen causal language model conditioned on memory tokens.

    The decoder runs in bfloat16 with a ``<|memory|>`` special token added to its vocabulary. The memory module's
    outputs replace the embeddings at the placeholder positions; the loss is the decoder's next-token loss over the
    positions that carry labels (the answer tokens).
    """

    def __init__(
        self,
        decoder: str,
        memory: MemoryModule,
        attn_implementation: str = "flash_attention_2",
        cache_dir: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """Load the decoder and tokenizer.

        Args:
            decoder: Hugging Face model id or local path of the decoder.
            memory: The memory module.
            attn_implementation: Decoder attention implementation, e.g. ``flash_attention_2`` or ``sdpa``.
            cache_dir: Model cache directory (None uses the Hugging Face default).
            dtype: Decoder dtype.
        """
        super().__init__()
        self.memory = memory
        self.text_model = AutoModelForCausalLM.from_pretrained(
            decoder,
            cache_dir=cache_dir,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(decoder, cache_dir=cache_dir, trust_remote_code=True)
        self.memory_token_id = prepare_tokenizer(self.tokenizer)
        self.text_model.resize_token_embeddings(len(self.tokenizer))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        doc_embeds: torch.Tensor,
        doc_padding_mask: torch.Tensor,
        question_embeds: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> CausalLMOutputWithPast:
        """Run the memory module and the decoder.

        Args:
            input_ids: Token ids ``[batch, seq_len]`` (left padded) containing the memory placeholders.
            attention_mask: ``[batch, seq_len]`` with 1 for real tokens.
            doc_embeds: Document embeddings ``[batch, num_docs, embedding_dim]``.
            doc_padding_mask: ``[batch, num_docs]`` with True at padding positions.
            question_embeds: Question embeddings ``[batch, embedding_dim]``.
            labels: Target ids with -100 at ignored positions, or None.

        Returns:
            The decoder output; ``loss`` is set when ``labels`` is given.
        """
        memory_tokens = self.memory(doc_embeds, doc_padding_mask, question_embeds)
        inputs_embeds = self.text_model.get_input_embeddings()(input_ids)
        inputs_embeds = inject_memory(inputs_embeds, memory_tokens, input_ids, self.memory_token_id)
        return self.text_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
