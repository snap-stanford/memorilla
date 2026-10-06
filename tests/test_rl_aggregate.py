"""CPU tests of the TextWorld rollout metrics."""

import json
from pathlib import Path

import pytest

from rl.aggregate import OVERALL, load_rollouts, score_rollouts


def make_rollout(tier: str, game: str, row: int, score: list[float], stop_reason: str) -> dict:
    """Build one dumped rollout record.

    Args:
        tier: Difficulty tier (``data_source``).
        game: Game file.
        row: Dataset row index of the episode.
        score: Per-token rewards.
        stop_reason: SkyRL stop reason.

    Returns:
        The record.
    """
    return {
        "data_source": tier,
        "env_extras": {"game_file": game, "extra_info": {"row_index": row}},
        "score": score,
        "stop_reason": stop_reason,
    }


ROLLOUTS = [
    make_rollout("medium", "a.z8", 0, [0.0, 0.0, 1.0], "stop"),
    make_rollout("medium", "a.z8", 0, [0.0, 0.0, 0.0], "length"),
    make_rollout("medium", "a.z8", 1, [0.0, 0.0, 0.0], "stop"),
    make_rollout("medium", "a.z8", 1, [0.0, 0.0, 0.0], "length"),
    make_rollout("long", "b.json", 2, [1.0, 1.0, 2.0], "stop"),
    make_rollout("long", "b.json", 2, [0.0, 1.0, 0.0], "length"),
]


def test_score_rollouts() -> None:
    """Episodes are dataset rows, a rollout is solved by a positive final reward, and stop reasons split cleanly."""
    metrics = score_rollouts(ROLLOUTS)

    medium = metrics["medium"]
    assert medium["episodes"] == 2
    assert medium["rollouts"] == 4
    assert medium["avg@k"] == pytest.approx(0.25)
    assert medium["pass@k"] == pytest.approx(0.5)
    assert medium["mean_reward"] == pytest.approx(0.25)
    assert medium["terminal"] == pytest.approx(0.5)

    overall = metrics[OVERALL]
    assert overall["episodes"] == 3
    assert overall["avg@k"] == pytest.approx(2 / 6)
    assert overall["pass@k"] == pytest.approx(2 / 3)
    assert overall["mean_reward"] == pytest.approx(6 / 6)
    assert overall["terminal"] + overall["truncated"] == pytest.approx(1.0)


def test_load_rollouts_skips_summary(tmp_path: Path) -> None:
    """The harness summary file next to the per-tier dumps is not read as rollouts."""
    dump_dir = tmp_path / "dumped_evals" / "global_step_3_evals"
    dump_dir.mkdir(parents=True)
    (dump_dir / "medium.jsonl").write_text("\n".join(json.dumps(rollout) for rollout in ROLLOUTS[:4]) + "\n")
    (dump_dir / "aggregated_results.jsonl").write_text(json.dumps({"eval/all/pass_at_3": 0.5}) + "\n")

    assert len(load_rollouts(tmp_path)) == 4
    assert len(load_rollouts(dump_dir)) == 4
