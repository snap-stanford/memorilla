#!/usr/bin/env bash
# Usage: scripts/baselines.sh <closed_book|rag|full_context> [benchmark|all] [evaluate_baselines.py flags]
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
method="${1:?usage: scripts/baselines.sh <closed_book|rag|full_context> [benchmark|all] [flags]}"
benchmark="${2:-all}"
shift $(( $# > 1 ? 2 : 1 ))
python "$repo/evaluate_baselines.py" --method "$method" --benchmark "$benchmark" "$@"
