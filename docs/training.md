# Training

`train.py` trains on any list of dataset configs. A YAML recipe from `configs/` sets the defaults and explicit flags
override it:

```bash
python train.py --config configs/stage3_multitask.yaml --wandb_project memorilla
python train.py --config configs/stage3_multitask.yaml --learning_rate 5e-5 --output_dir runs/stage3_lr5e-5
```

Run `python train.py --help` for every option. The main ones are `--datasets`, `--init_memory` (warm start from a
checkpoint directory or `memory.pt`), `--output_dir`, the memory-module options (`--num_memories`, `--num_heads`,
`--num_self_attn_layers`, `--num_cross_attn_layers`, `--dropout`, `--retrieval_init`), the frozen models (`--decoder`,
`--encoder`) and the optimisation settings.

## Recipe

| Recipe | Datasets | Starts from | Epochs | LR | Batch x accumulation x GPUs |
| --- | --- | --- | --- | --- | --- |
| `stage1_enwiki` | `enwiki` | scratch | 1 | 3e-4 | 4 x 8 x 4 |
| `stage2_mixture` | `squad_v2`, `drop`, `coqa`, `pubmedqa`, `quail`, `pwc`, `cnn_dailymail`, `samsum`, `dialogsum`, `msmarco` | Stage 1 | 1 | 3e-4 | 4 x 8 x 4 |
| `stage3_multitask` | `pv4`, `pmv2`, `factkg`, `triviaqa`, `narrativeqa`, `pubmedqa`, `lamp4`, `lamp7` | Stage 2 | 5 | 1e-4 | 2 x 8 x 4 |
| `personalization_pv4` | `pv4` | Stage 3 | 5 | 1e-4 | 2 x 2 x 4 |
| `personalization_pmv2` | `pmv2` | Stage 3 | 5 | 1e-4 | 2 x 2 x 4 |
| `single_task` | one config (default `lamp7`) | Stage 2 | 5 | 1e-4 | 2 x 2 x 4 |
| `factkg` | `factkg` | scratch, with `retrieval_init` | 5 | 1e-4 | 2 x 8 x 1 |

1. **Stage 1** pretrains the module to carry document content to the decoder: the decoder restates about two million
   Wikipedia passages from their memory tokens.
2. **Stage 2** continues on 1.34 million examples from ten reading-comprehension, summarisation and open-domain QA
   datasets, which teaches the module to recall different information from the same kind of store.
3. **Stage 3** fine-tunes one shared module on the training splits of all eight benchmarks and keeps every epoch.
   Benchmarks converge at different speeds, so pick the epoch per benchmark on its validation split where one exists;
   earlier epochs avoid overfitting the benchmarks that converge first:
   ```bash
   python evaluate.py --benchmark factkg --split validation --checkpoint runs/stage3/epoch-01 --output_dir results/stage3-epoch-01
   ```
   For `pv4` and `pmv2` the validation questions come from the training users, while their test users are unseen
   (for `pmv2`, all but one of its 200 test personas).
4. **Personalisation** continues the last Stage 3 epoch on PV4 or PMv2 alone, which keep improving after the other
   benchmarks have converged.

Every recipe uses AdamW (weight decay 0.01) with linear warmup and cosine decay to 10% of the peak learning rate,
bfloat16 mixed precision, gradient clipping at 1.0, the Qwen3-8B decoder and the default memory module (`K = 16`, no
document self-attention; `factkg` also sets `retrieval_init`). Recipes that warm-start read `init_memory` from the YAML
(e.g. `runs/stage2/epoch-00`); pass `--init_memory <dir|memory.pt>` to start from another checkpoint, or
`--init_memory ""` to start from scratch.

On 4 H100 80GB GPUs, Stage 1 takes about 6 hours, Stage 2 about 4, Stage 3 about 9 and each personalisation
recipe about 25 minutes.

## Outputs and resuming

A run writes:

- `<output_dir>/epoch-NN/memory.pt` and `config.json` at the end of every epoch; this is the checkpoint `evaluate.py`
  and `MemoryModule.from_pretrained` read.
- `<output_dir>/last.ckpt`, the resume checkpoint, rewritten after every validation and at the end of every epoch.
- `<output_dir>/logs/version_N/metrics.csv` with the training and validation losses and the learning rate (also sent
  to Weights & Biases with `--wandb_project`).

Rerunning the same command resumes from `last.ckpt`, so a preempted or failed job only needs to be resubmitted, and
rerunning a finished run does nothing. Resuming restores the weights, optimiser state and step count; the remaining
batches are not drawn in the order of an uninterrupted run. Relative output, checkpoint and data paths resolve against
`MEMORILLA_HOME`, while `--config` is read relative to the working directory. Multiple configs are concatenated and
shuffled, and configs with a `validation` split are validated on.

## Multiple GPUs and SLURM

All recipes except `factkg` (one GPU) set `devices: 4`. The effective batch is
`batch_size x gradient_accumulation_steps x devices x num_nodes`, so keep it when you change the GPU count by scaling
the accumulation, e.g. Stage 3 on two GPUs:

```bash
scripts/train.sh configs/stage3_multitask.yaml --devices 2 --gradient_accumulation_steps 16
```

On a single machine Lightning starts one process per GPU (`CUDA_VISIBLE_DEVICES=0,1,2,3 scripts/train.sh ...`).
Under SLURM, Lightning reads the job's environment and expects one task per GPU, so launch with `srun` and
`--ntasks-per-node` equal to `--devices` (running `python train.py` directly inside an allocation stops with a
tasks-per-node mismatch). For several nodes, set `--nodes` and pass `--num_nodes`.

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
srun scripts/train.sh configs/stage3_multitask.yaml
```

## Single-task training and FactKG

To fine-tune on one dataset, start from Stage 2 (or any other checkpoint):

```bash
python train.py --config configs/single_task.yaml --datasets lamp7 --output_dir runs/lamp7
```

`configs/factkg.yaml` trains FactKG from scratch on one GPU with `retrieval_init`: the slots start from the triples most
similar to the claim, and the recipe sets a fact-verification system prompt. Evaluate it with the same prompt, read
from the recipe:

```bash
python evaluate.py --benchmark factkg --checkpoint runs/factkg/epoch-04 --system_prompt_from configs/factkg.yaml
```
