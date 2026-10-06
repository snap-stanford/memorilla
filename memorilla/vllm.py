"""Inference-time wrapper: a frozen vLLM decoder fed prompt embeddings that carry memory tokens."""

from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors import safe_open
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from memorilla.llm import inject_memory
from memorilla.memory import MemoryModule
from memorilla.utils import local_model_path, prepare_tokenizer

EMBEDDING_WEIGHT_KEYS = ("model.embed_tokens.weight", "embed_tokens.weight", "transformer.wte.weight")
DECODER_GPU_MEMORY_UTILIZATION = 0.6


class MemoryVLLM:
    """Greedy or sampled generation from a frozen decoder conditioned on memory tokens.

    vLLM serves the decoder with prompt embeddings enabled. A standalone copy of the decoder's input embedding matrix
    turns token ids into embeddings, the memory tokens are scattered into the ``<|memory|>`` positions, and each
    row's left padding is removed before the request is sent.
    """

    def __init__(
        self,
        decoder: str,
        memory: MemoryModule,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = DECODER_GPU_MEMORY_UTILIZATION,
        cache_dir: str | None = None,
        tensor_parallel_size: int = 1,
        device: str = "cuda",
    ) -> None:
        """Start the vLLM engine and load the embedding matrix.

        Args:
            decoder: Hugging Face model id or local path of the decoder.
            memory: The memory module; it is moved to ``device`` in bfloat16 and set to evaluation mode.
            max_model_len: vLLM context length.
            gpu_memory_utilization: Fraction of GPU memory vLLM may use.
            cache_dir: Model cache directory (None uses the Hugging Face default).
            tensor_parallel_size: Number of GPUs for tensor parallelism.
            device: Device for the memory module and the embedding matrix.
        """
        self.decoder = decoder
        self.cache_dir = cache_dir
        self.model_path = local_model_path(decoder, cache_dir)
        self.device = torch.device(device)
        self.memory = memory.to(device=self.device, dtype=torch.bfloat16).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(decoder, cache_dir=cache_dir, trust_remote_code=True)
        self.memory_token_id = prepare_tokenizer(self.tokenizer)

        self.engine = LLM(
            model=self.model_path,
            download_dir=cache_dir,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prompt_embeds=True,
            trust_remote_code=True,
            max_model_len=max_model_len,
            dtype="bfloat16",
            tensor_parallel_size=tensor_parallel_size,
        )
        self.embed_tokens = self._load_embedding_matrix()

    def _load_embedding_matrix(self) -> nn.Embedding:
        """Load the decoder's input embedding matrix from its safetensors shards.

        Returns:
            A frozen bfloat16 embedding layer on ``self.device``.

        Raises:
            FileNotFoundError: If no shard contains an input embedding matrix.
        """
        model_path = Path(self.model_path)
        if not model_path.exists():
            model_path = Path(
                snapshot_download(
                    repo_id=self.decoder,
                    allow_patterns=["*.safetensors", "*.json"],
                    cache_dir=self.cache_dir,
                )
            )

        for shard in sorted(model_path.glob("*.safetensors")):
            with safe_open(str(shard), framework="pt", device=str(self.device)) as handle:
                available = set(handle.keys())
                key = next((key for key in EMBEDDING_WEIGHT_KEYS if key in available), None)
                if key is not None:
                    weight = handle.get_tensor(key).to(dtype=torch.bfloat16)
                    embed_tokens = nn.Embedding.from_pretrained(weight, freeze=True)
                    return embed_tokens.eval()
        raise FileNotFoundError(f"No input embedding matrix found in {model_path} (tried {EMBEDDING_WEIGHT_KEYS}).")

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        doc_embeds: torch.Tensor,
        doc_padding_mask: torch.Tensor,
        question_embeds: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> list[str]:
        """Generate one completion per row.

        Args:
            input_ids: Prompt ids ``[batch, seq_len]`` (left padded) containing the memory placeholders.
            attention_mask: ``[batch, seq_len]`` with 1 for real tokens.
            doc_embeds: Document embeddings ``[batch, num_docs, embedding_dim]``.
            doc_padding_mask: ``[batch, num_docs]`` with True at padding positions.
            question_embeds: Question embeddings ``[batch, embedding_dim]``.
            max_new_tokens: Maximum number of generated tokens.
            temperature: Sampling temperature (0 is greedy).
            top_p: Nucleus sampling threshold.

        Returns:
            The generated texts.
        """
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device).bool()
        memory_tokens = self.memory(
            doc_embeds.to(self.device, dtype=torch.bfloat16),
            doc_padding_mask.to(self.device),
            question_embeds.to(self.device, dtype=torch.bfloat16),
        )
        inputs_embeds = inject_memory(self.embed_tokens(input_ids), memory_tokens, input_ids, self.memory_token_id)

        requests = [{"prompt_embeds": row[mask]} for row, mask in zip(inputs_embeds, attention_mask, strict=True)]
        sampling = SamplingParams(max_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
        outputs = self.engine.generate(requests, sampling_params=sampling, use_tqdm=False)
        return [output.outputs[0].text for output in outputs]
