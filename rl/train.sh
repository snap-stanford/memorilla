#!/usr/bin/env bash
# Usage: rl/train.sh <data_dir> <output_dir> [hydra overrides]
# Trains the memory module with GRPO; run inside the SkyRL fork's environment (see the README).
set -euo pipefail

usage="usage: rl/train.sh <data_dir> <output_dir> [hydra overrides]"
rl_dir="$(cd "$(dirname "$0")" && pwd)"
data_dir="$(realpath "${1:?$usage}")"
output_dir="$(realpath -m "${2:?$usage}")"
shift 2

python -m skyrl_train.entrypoints.main_base \
  --config-dir "$rl_dir/configs" --config-name textworld_grpo \
  "data.train_data=['$data_dir/train.parquet']" \
  "data.val_data=['$data_dir/validation.parquet']" \
  "trainer.ckpt_path=$output_dir/checkpoints" \
  "trainer.export_path=$output_dir/exports" \
  "$@"
