# Troubleshooting

- **A multi-GPU run stalls.** With the NCCL 2.26 bundled with PyTorch 2.7, a multi-GPU run can hang in a collective
  until the NCCL watchdog aborts it after 30 minutes. Check that NCCL 2.27.3 is installed
  (`uv pip show nvidia-nccl-cu12`, or `pip show nvidia-nccl-cu12` in a pip environment; see
  [Installation](../README.md#installation)), then resubmit the job; it resumes from `last.ckpt`.
- **Evaluation is slow when several runs share a machine.** Prompt preparation and scoring are CPU-bound; give each
  `evaluate.py` process about 16 CPU cores, or set `OMP_NUM_THREADS` accordingly.
- **METEOR fails on an offline machine.** METEOR needs NLTK's `wordnet` and `punkt` data, which is downloaded on first
  use; on machines without network access install it beforehand with `python -m nltk.downloader wordnet punkt`.
- **A run cannot find the models offline.** With `HF_HUB_OFFLINE=1`, the decoder and encoder must already be in
  the model cache (`MEMORILLA_MODEL_CACHE` or `--cache_dir`, otherwise the default Hugging Face cache).
