"""Score TextWorld evaluation rollouts per difficulty tier.

SkyRL writes one JSONL file per tier (``<tier>.jsonl``) under ``<eval_dir>/dumped_evals/<step>_evals/``, one line per
rollout. A rollout is solved when its final turn earns a positive reward. For each tier and overall this reports
avg@k (mean solve rate over all rollouts), pass@k (fraction of episodes solved by at least one of their k rollouts),
the mean total reward per rollout, the fraction of terminal rollouts (ended by the game or the turn limit) and the
fraction cut short by the input or generation budget. Several evaluation directories (e.g. seeds) are averaged. An
episode is one dataset row: a game with its prompt.

Each directory is either the output directory of one evaluation or a single ``dumped_evals/<step>_evals`` directory,
e.g. one of the validation dumps written during training.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np

SUMMARY_FILE = "aggregated_results.jsonl"
OVERALL = "all"
STOP = "stop"
LENGTH = "length"


def load_rollouts(eval_dir: Path) -> list[dict[str, Any]]:
    """Read the rollouts of one evaluation.

    Args:
        eval_dir: The output directory (``trainer.export_path``) of an evaluation run, or one of its
            ``dumped_evals/<step>_evals`` directories.

    Returns:
        The rollout records.

    Raises:
        FileNotFoundError: If the directory holds no dumped rollouts.
        ValueError: If the directory holds the dumps of several evaluations.
    """
    if any(eval_dir.glob("*.jsonl")):
        dump_dirs = [eval_dir]
    else:
        dump_dirs = sorted({path.parent for path in eval_dir.glob("dumped_evals/*/*.jsonl")})
    if len(dump_dirs) > 1:
        names = ", ".join(directory.name for directory in dump_dirs)
        raise ValueError(f"{eval_dir} holds several evaluations ({names}); pass one of their directories instead.")

    files = [path for directory in dump_dirs for path in sorted(directory.glob("*.jsonl")) if path.name != SUMMARY_FILE]
    if not files:
        raise FileNotFoundError(f"No dumped evaluation rollouts under {eval_dir}.")
    return [json.loads(line) for path in files for line in path.read_text().splitlines() if line.strip()]


def _final_reward(rollout: dict[str, Any]) -> float:
    """Return the reward of a rollout's final turn.

    Args:
        rollout: Rollout record whose ``score`` is a per-token list or a scalar.

    Returns:
        The last entry of a list score, or the scalar score.
    """
    score = rollout["score"]
    return float(score[-1]) if isinstance(score, list) else float(score)


def _total_reward(rollout: dict[str, Any]) -> float:
    """Return the total reward of a rollout, the sum of its turn rewards.

    Args:
        rollout: Rollout record whose ``score`` is a per-token list or a scalar.

    Returns:
        The sum of a list score, or the scalar score.
    """
    score = rollout["score"]
    return float(sum(score)) if isinstance(score, list) else float(score)


def score_rollouts(rollouts: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Compute the metrics of one evaluation run.

    Args:
        rollouts: Records with ``score`` (per-token or scalar reward), ``stop_reason``, ``data_source`` and
            ``env_extras`` (``game_file`` and ``extra_info.row_index`` identify the episode).

    Returns:
        Metrics per tier and for all tiers combined.
    """
    groups: dict[str, dict[tuple[str, int | None], list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for rollout in rollouts:
        extras = rollout["env_extras"]
        episode = (extras["game_file"], extras.get("extra_info", {}).get("row_index"))
        groups[rollout["data_source"]][episode].append(rollout)
        groups[OVERALL][episode].append(rollout)

    metrics = {}
    for tier, episodes in groups.items():
        solved = [[_final_reward(rollout) > 0 for rollout in episode_runs] for episode_runs in episodes.values()]
        tier_rollouts = [rollout for episode_runs in episodes.values() for rollout in episode_runs]
        metrics[tier] = {
            "episodes": len(episodes),
            "rollouts": len(tier_rollouts),
            "avg@k": float(np.mean([value for values in solved for value in values])),
            "pass@k": float(np.mean([any(values) for values in solved])),
            "mean_reward": float(np.mean([_total_reward(rollout) for rollout in tier_rollouts])),
            "terminal": float(np.mean([rollout["stop_reason"] == STOP for rollout in tier_rollouts])),
            "truncated": float(np.mean([rollout["stop_reason"] == LENGTH for rollout in tier_rollouts])),
        }
    return metrics


def average(runs: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    """Average metrics over evaluation runs, per tier.

    Args:
        runs: Metrics of each run.

    Returns:
        Mean of every metric over the runs that contain the tier.
    """
    tiers = sorted({tier for run in runs for tier in run}, key=lambda tier: (tier == OVERALL, tier))
    return {
        tier: {name: float(np.mean([run[tier][name] for run in runs if tier in run])) for name in runs[0][OVERALL]}
        for tier in tiers
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("eval_dirs", type=Path, nargs="+", help="Evaluation output directories.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON file for the metrics.")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Print the metrics table and optionally write it as JSON.

    Args:
        args: Parsed command-line arguments.
    """
    runs = [score_rollouts(load_rollouts(eval_dir)) for eval_dir in args.eval_dirs]
    metrics = average(runs)

    print(f"{'tier':<10} {'episodes':>8} {'avg@k':>7} {'pass@k':>7} {'reward':>7} {'terminal':>8} {'truncated':>9}")
    for tier, values in metrics.items():
        print(
            f"{tier:<10} {values['episodes']:>8.0f} {values['avg@k']:>7.3f} {values['pass@k']:>7.3f} "
            f"{values['mean_reward']:>7.2f} {values['terminal']:>8.3f} {values['truncated']:>9.3f}"
        )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {"runs": [str(path) for path in args.eval_dirs], "metrics": metrics}
        args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main(parse_args())
