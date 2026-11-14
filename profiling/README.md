# Profiling

This folder measures how the memory module scales with the number of documents N. It times one call of the module
(document and question embeddings in, 16 memory tokens out) and reports its GPU memory, with and without the memory
cache that `MemoryVLLM` uses when `memory_cache` is on. The encoder and the decoder are not timed, and no data is
needed, because random embeddings cost the same as real ones.

## Running

```bash
$ python -m profiling.run --out results/profiling --checkpoint checkpoints/stage3
$ python -m profiling.summarize results/profiling                     # summary.md and summary.csv
$ python -m profiling.plot results/profiling assets/efficiency.png    # the figure in the main README (needs matplotlib)
```

Without `--checkpoint`, the module is randomly initialised with the released architecture, which gives the same
timings. A rerun skips the cells already in `results/profiling/memory_module.jsonl`.

## Options

Each option takes a comma-separated list, and the sweep runs every combination.

| Option | Values | Meaning |
|---|---|---|
| `--ns` | `1,512,...,1048576` | Documents per collection |
| `--cache` | `gpu`, `host`, `kv`, `kv_fp8` | Embeddings on the GPU, embeddings copied from host memory, the memory cache, or an experimental float8 cache |
| `--modes` | `eager`, `cuda_graph`, `compile` | How each call runs |
| `--batch_sizes`, `--shared` | e.g. `8,64` and `0,1` | Questions per call, and whether they share one collection |

## Results

On one B200, with the Stage 3 checkpoint and one question per call, the median time per question in milliseconds is

| N | Default | Cache | Cache + CUDA graph | Cache + `torch.compile` |
|---|---|---|---|---|
| 1 | 0.45 | 0.36 | 0.21 | 0.27 |
| 4,096 | 0.62 | 0.36 | 0.25 | 0.26 |
| 262,144 | 18.6 | 2.28 | 2.17 | 1.48 |
| 1,048,576 | 73.8 | 8.06 | 7.92 | 5.04 |
| 4,194,304 | 282.7 | 32.8 | | |

- The cache makes a question about 9 times faster at one and four million documents. It holds four times the bytes
  of the embeddings, 80 GiB for four million documents.
- `torch.compile` speeds up the cache a further 1.6 times at large N, while CUDA graphs only help at small N.
- When 64 questions share one collection, the cache costs 0.71 ms per question at 262,144 documents, three times
  less than one question at a time.
- An A100 shows the same picture. At one million documents a question takes 296 ms by default, 25 ms with the cache
  and 17 ms with `torch.compile`.
- The float8 cache halves the memory but is about 5 times slower than the cache and less accurate, so it stays an
  experiment in `profiling/fp8_cache.py`.

The cache leaves the outputs essentially unchanged. In all 55 configurations checked (up to 262,144 documents), its
memory tokens differ from the default forward pass by no more than bfloat16 rounding. On FactKG, `evaluate.py` gives
83.61% with `--memory_cache` against 83.55% by default, and 99.2% of the predictions are identical.

## Method

- Each call is timed with CUDA events after flushing the L2 cache, over 10 warmup and 50 timed calls. The summary
  reports the median and the interquartile range.
- Each cell compares its output with the default forward pass. It passes when the difference is at most twice the
  default's own rounding error against a float32 forward pass (`--tolerance_factor`).
- Memory is the size of the stored embeddings or cache, plus PyTorch's peak allocation during the calls.
