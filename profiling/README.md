# Profiling

This folder measures the latency and GPU memory of the memory module alone: N document embeddings and a question
embedding go in, and the memory tokens (16 by default) come out. The encoder and the decoder are not part of the
measurement. No data is needed, since the cost does not depend on the embedding values. Random unit-norm vectors
stand in for encoder outputs, and the module is randomly initialised with the released architecture unless
`--checkpoint` is given.

```bash
$ python -m profiling.run --out results/profiling          # the default sweep, N from 1 to 1,048,576
$ python -m profiling.summarize results/profiling          # summary.md and summary.csv
```

## Settings

Each option takes a comma-separated list, and every combination is a cell.

| Option | Values | What it changes |
|---|---|---|
| `--cache` | `gpu` | Document embeddings on the GPU; a query runs `MemoryModule.forward` |
| | `host` | Document embeddings in pinned host memory; a query copies them to the GPU first |
| | `kv` | Keys and values of every cross-attention layer computed when the documents are written; a query attends from the slots over them |
| | `kv_fp8` | The `kv` cache in float8 with per-document and per-channel scales, multiplied by `torch._scaled_mm` (approximate, Hopper or newer) |
| `--modes` | `eager`, `cuda_graph`, `compile` | How a query runs: launched op by op, replayed from a captured CUDA graph, or compiled by `torch.compile` |
| `--shared` | `0`, `1` | Whether a batch of questions shares one collection (1) or has one collection each (0) |
| `--splits` | `1`, `2`, ... | Chunks of the weighted sum over the documents, added up afterwards (`kv` only) |
| `--batch_sizes`, `--ns` | | Questions per call and documents per collection |

The defaults time `gpu`, `host`, `kv` and `kv_fp8` in eager mode at batch size 1. Combinations that would duplicate
another cell or are not supported are left out: `--splits` above 1 applies to `kv` only, `host` runs eager only, and
`--shared 1` needs a batch above 1. `--dtype`, `--num_memories` and `--num_cross_attn_layers` change the module, and
`--help` lists every option.

The cached variants live in `profiling/cached_memory.py` and are used only here. `kv`, with or without a shared
collection or splits, is an exact rewrite of the forward pass (the tests check it in float32). `kv_fp8` is not
exact. Every cell checks its output against the default forward pass, as described below.

## Output check

An optimization must not change what the module returns. For every cell with batch x N up to `--check_max`
(262,144 by default), the output that the timed call returns, after compilation or graph capture, is compared with
the default forward pass. That is `MemoryModule.forward` in the run's dtype, run eager, as `MemoryVLLM` serves it.

In bfloat16 any change in the order of operations changes the last bits, so the tolerance is set by the default
forward pass's own rounding error. Both are measured against a float32 copy of the module on the same inputs. A
cell is within tolerance when its relative difference from the default forward pass is at most
`--tolerance_factor` (2 by default) times that error, and at least `--tolerance_floor` (1e-5) for float32 runs. Two
computations that each round like the default differ by about 1.4 times its error, so an exact rewrite passes
and an approximation that loses more than the default's rounding fails.

Each record stores `rel_diff_default`, `max_abs_diff_default`, `default_rel_error`, `rel_error` (against
float32), `tolerance` and `within_tolerance`. The summary lists every checked cell outside tolerance, and
`profiling.summarize` prints how many checked cells are within it.

## Running the sweep

The results below come from these commands, run in one job on one GPU with the released Stage 3 checkpoint. Any
checkpoint with the same architecture gives the same timings, and leaving out `--checkpoint` uses a random module.

```bash
$ RUN="python -m profiling.run --out results/profiling --checkpoint checkpoints/stage3"
$ $RUN --ns 1,512,4096,32768,262144,1048576,4194304                                            # caching settings
$ $RUN --ns 1,512,4096,32768,262144,1048576 --cache gpu,kv,kv_fp8 --modes cuda_graph,compile  # execution modes
$ $RUN --ns 512,4096,32768,262144 --cache gpu,kv,kv_fp8 --batch_sizes 8,64 --shared 0,1      # batches of questions
$ $RUN --ns 32768,262144,1048576,4194304 --cache kv --splits 1,4,16,64                       # splits over N
$ python -m profiling.summarize results/profiling
```

All runs append to the same `memory_module.jsonl`, and the summary puts every cell in one table. On one B200 the
four sweeps take about 5 minutes, and on one A100 about 8 minutes.

## Results

Measured on one NVIDIA B200 and one NVIDIA A100-SXM4-80GB with torch 2.7.0 (CUDA 12.8), bfloat16, the Stage 3
checkpoint (16 memory tokens, 2 cross-attention layers) and 2560-dimensional embeddings. Each number is the median
time per call in milliseconds over 50 calls.

**One question, B200.**

| N | `gpu` | `gpu` + compile | `kv` | `kv` + CUDA graph | `kv` + compile | `kv_fp8` |
|---|---|---|---|---|---|---|
| 1 | 0.46 | 0.30 | 0.37 | 0.21 | 0.25 | 2.04 |
| 512 | 0.47 | 0.31 | 0.36 | 0.24 | 0.25 | 1.95 |
| 4,096 | 0.62 | 0.48 | 0.37 | 0.25 | 0.26 | 1.98 |
| 32,768 | 2.55 | 1.96 | 0.55 | 0.44 | 0.40 | 2.44 |
| 262,144 | 18.6 | 14.7 | 2.27 | 2.17 | 1.48 | 10.8 |
| 1,048,576 | 73.5 | 60.0 | 8.06 | 7.92 | 5.04 | 40.4 |
| 4,194,304 | 280.5 | | 32.7 | | | 162.9 |

**One question, A100.**

| N | `gpu` | `gpu` + compile | `kv` | `kv` + CUDA graph | `kv` + compile |
|---|---|---|---|---|---|
| 1 | 1.23 | 0.75 | 0.96 | 0.43 | 0.61 |
| 4,096 | 2.15 | 1.85 | 0.96 | 0.54 | 0.70 |
| 262,144 | 74.2 | 68.4 | 6.83 | 6.51 | 4.76 |
| 1,048,576 | 295.7 | 274.0 | 25.3 | 25.0 | 17.2 |

At 4,194,304 documents the A100 runs out of memory with `gpu` and `kv`. Float8 products with per-row scales need a
Hopper or newer GPU, so the A100 has no `kv_fp8` cells.

**64 questions per call, time per question on the B200.**

| N | `gpu`, own collections | `gpu`, one collection | `kv`, own collections | `kv`, one collection |
|---|---|---|---|---|
| 512 | 0.062 | 0.062 | 0.012 | 0.010 |
| 4,096 | 0.43 | 0.42 | 0.033 | 0.021 |
| 32,768 | 3.53 | 3.39 | 0.18 | 0.097 |
| 262,144 | out of memory | out of memory | out of memory | 0.71 |

The main observations are as follows.

- The `kv` cache makes a query 9 times faster at one million documents (8.06 against 73.5 ms on the B200) and 8.6
  times faster at four million (32.7 against 280.5 ms). It stores four times the bytes of the embeddings (20 GiB
  against 5 GiB at one million documents), and writing it is a one-time cost when documents are added.
- `torch.compile` on top of `kv` gives a further 1.6 times at large N (5.04 ms at one million documents on the B200,
  where the read reaches 53% of the peak memory bandwidth). CUDA graphs help only at small N, where kernel launches
  dominate, and bring a query at N = 1 from 0.37 to 0.21 ms.
- When a batch of questions shares one collection, the `kv` cost per question falls by up to 3 times against one
  question per call (0.71 against 2.27 ms at 262,144 documents). The forward pass gains nothing from sharing,
  because it projects the documents again for every question.
- Splitting the weighted sum over N into 4, 16 or 64 chunks does not make it faster (it is up to 8% slower), since the
  matrix products already use the whole GPU.
- `kv_fp8` halves the cache but is 4 to 6 times slower than `kv`, since it runs one float8 product per head and
  quantises the attention weights on every query. It is also outside tolerance in 20 of 25 checked cells (up to
  1.6 times the tolerance). A fused kernel would be needed for it to pay off.
- Every exact variant (`gpu`, `host` and `kv`, in every mode, with or without sharing and splits) is within
  tolerance of the default forward pass in all 61 checked cells on each GPU, using at most 96% of the tolerance.

The `host` setting depends on the node's PCIe topology and load: two A100 runs on different nodes differed by up
to 4.7 times at 4,096 documents. On the B200, two runs on the same node agreed within 5% in every cell.

## Method

- Each call is timed with CUDA events. The L2 cache is overwritten before every call (`--no_flush_l2` turns this
  off), so that a small collection is not served from a cache that a real query would not find.
- A cell runs `--warmup` untimed calls, then `--reps` timed ones. `setup_ms` is the time to compile or capture a
  query plus its first call, and the first warmup call is kept as `first_call`.
- Each cell reports the median, mean, standard deviation, minimum, maximum, the 10th, 25th, 75th and 90th
  percentiles and the raw samples.
- `achieved_gbs` is the bytes of the stored collection or cache divided by the median time, a lower bound on the
  bandwidth a read uses, and `peak_bandwidth_gbs` is the GPU's peak (from a table of known GPUs, or
  `--peak_bandwidth_gbs`). For `host`, it is the copy rate from host memory instead.
- Memory is reported as the bytes of the stored collection or cache, and PyTorch's peak allocation from compilation
  or graph capture through the reads, above the allocation before them. A CUDA graph's memory pool is included.
- Results go to `results/profiling/memory_module.jsonl`, one record per cell with its settings, status, timings,
  memory, the environment (GPU, CUDA, torch, git commit) and the arguments. A rerun skips the cells already there and
  retries failed ones. A cell that runs out of memory is recorded as `oom`, and the larger N of the same settings are
  skipped.
