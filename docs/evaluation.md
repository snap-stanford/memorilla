# Evaluation

```bash
python evaluate.py --benchmark <name|all> --checkpoint <dir|memory.pt> [--split test] [--output_dir results]
```

`evaluate.py` loads the memory module from the checkpoint, feeds its memory tokens to the frozen decoder (vLLM) as
prompt embeddings, generates greedily for every row, frees the decoder and scores. The decoder must be the one the
checkpoint was trained with (`--decoder`, default `Qwen/Qwen3-8B`).

| Benchmark | Split | Metrics | Max new tokens |
| --- | --- | --- | --- |
| `pmv2` | test (5,000) | accuracy | 256 |
| `pv4` | test (4,237) | accuracy, accuracy on the 629 hard questions | 256 |
| `factkg` | test (9,041) | accuracy | 8 |
| `triviaqa` | test (500) | exact match, F1, BLEU-1/4, ROUGE-1/L, METEOR | 64 |
| `narrativeqa` | test (10,557) | BLEU-1/4, ROUGE-1/L, F1, METEOR | 32 |
| `pubmedqa` | test (100) | accuracy | 64 |
| `lamp4` | test (1,925) | ROUGE-1/L, F1, METEOR, BLEU-1/4 | 128 |
| `lamp7` | test (1,500) | ROUGE-1/L, F1, METEOR, BLEU-1/4 | 128 |

- **Answer-choice accuracy** (`pmv2`, `pv4`, `factkg`): the model answers in free text without seeing the choices.
  The generation and every non-empty choice are embedded with Qwen3-Embedding-4B, and the answer is correct when the
  most cosine-similar choice is the reference.
- **Exact match** (`triviaqa`): SQuAD answer normalisation (lower case, no punctuation or articles) and strict equality
  with the reference answer.
- **PubMedQA accuracy**: the first word of the generation (yes, no or maybe) must match the reference's.
- **Generation metrics**: corpus BLEU-1/4 with smoothing, token F1 and METEOR on lower-cased whitespace tokens;
  ROUGE-1/L F-measure with `rouge_score`'s tokenizer and Porter stemming.

Each benchmark writes `results/<benchmark>/predictions.jsonl` (question, answer, prediction and, where present, choices
and hard) and `results/<benchmark>/metrics.json` (scores in percent and `num_examples`). With a `--split` other than
`test`, the folder is `results/<benchmark>-<split>/`; a rerun overwrites it unless `--output_dir` differs. The prompt
settings of every benchmark (system prompt, answer-format suffix, prompt and generation limits) are registered in
`memorilla/benchmarks.py`; `--system_prompt` (or `--system_prompt_from <recipe.yaml>`), `--max_length` and
`--max_new_tokens` override them.

Evaluation runs on one GPU, and all eight benchmarks take about 15 minutes on an H100 once the data is downloaded. The
memory fractions (`--gpu_memory_utilization` for the decoder, `--scorer_gpu_memory_utilization` for the scoring
encoder) are set for an 80 GB card; raise them on smaller cards. bfloat16 numerics and vLLM batching make scores vary by
a few tenths of a point between runs, and by up to about one point for long generations.

## Baselines

```bash
python evaluate_baselines.py --benchmark <name|all> --method closed_book|rag|full_context [--top_k 5]
```

The baselines give the same frozen decoder the documents as text in the user turn, followed by the question:

- **closed_book**: no documents.
- **rag**: the `top_k` documents most cosine-similar to the question (same embeddings as Memorilla), most similar
  first, separated by blank lines.
- **full_context**: every document in stored order.

Their prompts (system prompt, answer-format suffix and generation budget) are set per benchmark by `BaselineProtocol`
in `memorilla/benchmarks.py` and differ from Memorilla's evaluation prompts. Closed-book and RAG run with a
32,768-token context. Full context uses 40,960 tokens for `pmv2`, 65,536 for `pv4` (YaRN x2), 131,072 for
`narrativeqa` and `triviaqa` (YaRN x4), 32,768 for `lamp4` and 8,192 for `factkg`, `pubmedqa` and `lamp7`; the
131,072-token runs need an 80 GB card. Prompts that do not fit are truncated from the front of the whole chat prompt
(the system prompt and chat header go first), so the question is always kept, and `context.json` reports how many fit;
if a long-context run runs out of GPU memory, pass `--max_num_seqs <n>`. Outputs go to
`results/baselines/<closed_book|rag_top<k>|full_context>/<benchmark>/` and are scored exactly like Memorilla's.
