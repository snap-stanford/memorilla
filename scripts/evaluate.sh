#!/usr/bin/env bash
# Usage: scripts/evaluate.sh <checkpoint> [benchmark|all] [evaluate.py flags]
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
checkpoint="${1:?usage: scripts/evaluate.sh <checkpoint> [benchmark|all] [evaluate.py flags]}"
benchmark="${2:-all}"
shift $(( $# > 1 ? 2 : 1 ))
python "$repo/evaluate.py" --checkpoint "$checkpoint" --benchmark "$benchmark" "$@"
