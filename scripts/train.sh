#!/usr/bin/env bash
# Usage: scripts/train.sh <recipe.yaml> [train.py flags]
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
recipe="${1:?usage: scripts/train.sh <recipe.yaml> [train.py flags]}"
shift
python "$repo/train.py" --config "$recipe" "$@"
