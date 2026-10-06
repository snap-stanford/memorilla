#!/usr/bin/env bash
# Usage: rl/evaluate.sh <data_dir> <split> <output_dir> <model> [hydra overrides]
#   model: a training checkpoint (.../global_step_N), a Memorilla memory checkpoint (directory or memory.pt),
#          "untrained" (randomly initialised memory module) or "no-memory" (decoder with its full history only)
# Plays every prompt of <split> three times at temperature 0.6 and writes metrics.json to <output_dir>.
set -euo pipefail

usage="usage: rl/evaluate.sh <data_dir> <split> <output_dir> <model> [hydra overrides]"
rl_dir="$(cd "$(dirname "$0")" && pwd)"
data_dir="$(realpath "${1:?$usage}")"
split="${2:?$usage}"
output_dir="$(realpath -m "${3:?$usage}")"
model="${4:?$usage}"
shift 4

args=(
  "data.val_data=['$data_dir/$split.parquet']"
  "trainer.ckpt_path=$output_dir"
  "trainer.export_path=$output_dir"
)
case "$model" in
  untrained) ;;
  no-memory)
    args+=(
      "~modalities.memorilla"
      "environment.skyrl_gym.textworld.enable_memory=false"
      "environment.skyrl_gym.fast_textworld.enable_memory=false"
    )
    ;;
  *)
    if [[ -d "$model/policy" ]]; then
      args+=("trainer.resume_mode=from_path" "trainer.resume_path=$(realpath "$model")")
    else
      args+=("modalities.memorilla.encoder.kwargs.checkpoint_path=$(realpath "$model")")
    fi
    ;;
esac

python -m skyrl_train.entrypoints.main_base \
  --config-dir "$rl_dir/configs" --config-name textworld_eval \
  "${args[@]}" "$@"
python "$rl_dir/aggregate.py" "$output_dir" --output "$output_dir/metrics.json"
