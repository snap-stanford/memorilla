"""Build the TextWorld RL dataset from generated games.

Every tier is split by game: ``--held_out`` games go to test, the next ``--held_out`` to validation and the rest to
train. Compiled games (``.z8``) are played through Inform7 (``env_class`` ``textworld``) as two episodes, each with a
prompt variant drawn from the game seed, so both episodes can share a variant; spec-only games (``.json``) are played by
the simulator (``fast_textworld``) as one episode with the default prompt. Each split is written as a parquet file
whose rows SkyRL turns into rollouts.
"""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import random
import re
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

SPLITS = ("train", "validation", "test")
GAME_NAME = re.compile(
    r"game_(?P<index>\d+)_(?P<tier>[a-z]+)_w(?P<world>\d+)_o(?P<objects>\d+)_q(?P<quest>\d+)_s(?P<seed>\d+)"
)
EPISODES_PER_COMPILED_GAME = 2
COMPILED_ENV = "textworld"
SIMULATED_ENV = "fast_textworld"
STATS_FILE = "dataset_stats.json"
PROMPT_SEED_SHIFT = 8
PROMPT_SEED_OFFSET = 17
DEFAULT_PROMPT = "memory_first"
PROMPTS: dict[str, tuple[str, str]] = {
    "strict_action": (
        "You are an agent in a text adventure game.\n"
        "Think briefly, then ALWAYS end with exactly one action line in this format:\n"
        "[ACTION: <command>]\n"
        "Use short game commands (look, inventory, examine, take, open, close, go north, etc.).",
        "Start the game and choose the next best action.",
    ),
    "memory_first": (
        "You play TextWorld efficiently.\n"
        "Use memory summaries when useful and avoid repeating failed actions.\n"
        "Return one final action command in the format [ACTION: <command>].",
        "What is your next action?",
    ),
    "compact": (
        "Solve the game step by step.\nOutput must end with [ACTION: <command>].",
        "Play optimally.",
    ),
    "parseable": (
        "You are playing a text-based game.\n"
        "Final line must be parseable as [ACTION: ...] with only a single command.\n"
        "Do not output multiple actions.",
        "Continue the game.",
    ),
}


@dataclass(frozen=True)
class Game:
    """A generated game and the parameters encoded in its file name.

    Attributes:
        path: Game file (``.z8`` compiled game or ``.json`` spec).
        tier: Difficulty tier.
        index: Game index within its generation run.
        world_size: Number of rooms.
        nb_objects: Number of objects.
        quest_length: Length of the quest.
        seed: Seed the game was generated with.
    """

    path: Path
    tier: str
    index: int
    world_size: int
    nb_objects: int
    quest_length: int
    seed: int

    @property
    def compiled(self) -> bool:
        """Whether the game is a compiled ``.z8`` game (played through Inform7)."""
        return self.path.suffix == ".z8"


def find_games(directories: list[Path]) -> list[Game]:
    """Collect games from directories, preferring the compiled ``.z8`` file when both forms exist.

    Args:
        directories: Directories written by ``generate_games.py``.

    Returns:
        The games, one per name.
    """
    games: dict[str, Game] = {}
    for directory in directories:
        for path in sorted(directory.glob("game_*")):
            match = GAME_NAME.fullmatch(path.stem)
            if match is None or path.suffix not in (".z8", ".json"):
                continue
            if path.stem in games and games[path.stem].compiled:
                continue
            games[path.stem] = Game(
                path=path.resolve(),
                tier=match["tier"],
                index=int(match["index"]),
                world_size=int(match["world"]),
                nb_objects=int(match["objects"]),
                quest_length=int(match["quest"]),
                seed=int(match["seed"]),
            )
    return list(games.values())


def split_games(games: list[Game], held_out: int, seed: int) -> dict[str, list[Game]]:
    """Split each tier's games into test, validation and train.

    Tiers are visited compiled tiers first, then alphabetically, and share one shuffle stream. A tier keeps at least
    a third of its games for training.

    Args:
        games: All games.
        held_out: Games per tier in each of validation and test.
        seed: Shuffle seed.

    Returns:
        Games per split.
    """
    by_tier: dict[str, list[Game]] = defaultdict(list)
    for game in games:
        by_tier[game.tier].append(game)

    rng = random.Random(seed)
    splits: dict[str, list[Game]] = {split: [] for split in SPLITS}
    for tier in sorted(by_tier, key=lambda name: (not by_tier[name][0].compiled, name)):
        tier_games = sorted(by_tier[tier], key=lambda game: game.path.name)
        rng.shuffle(tier_games)
        count = min(held_out, len(tier_games) // 3)
        splits["test"] += tier_games[:count]
        splits["validation"] += tier_games[count : 2 * count]
        splits["train"] += tier_games[2 * count :]
    return splits


def prompt_variant(game: Game, episode: int) -> str:
    """Choose the prompt variant of a compiled game's episode, deterministically from the game seed.

    Args:
        game: A compiled game.
        episode: Episode index.

    Returns:
        A key of ``PROMPTS``.
    """
    rng = random.Random((game.seed << PROMPT_SEED_SHIFT) + episode + PROMPT_SEED_OFFSET)
    return list(PROMPTS)[rng.randrange(len(PROMPTS))]


def make_rows(games: list[Game]) -> list[dict[str, Any]]:
    """Turn games into dataset rows.

    Args:
        games: Games of one split.

    Returns:
        Rows with ``prompt``, ``env_class``, ``game_file``, ``data_source`` (the tier) and ``extra_info``.
    """
    rows = []
    for game in sorted(games, key=lambda game: (game.tier, game.index)):
        episodes = EPISODES_PER_COMPILED_GAME if game.compiled else 1
        for episode in range(episodes):
            variant = prompt_variant(game, episode) if game.compiled else DEFAULT_PROMPT
            system, user = PROMPTS[variant]
            rows.append(
                {
                    "prompt": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                    "env_class": COMPILED_ENV if game.compiled else SIMULATED_ENV,
                    "game_file": str(game.path),
                    "data_source": game.tier,
                    "extra_info": {
                        "row_index": len(rows),
                        "game_id": game.path.stem,
                        "tier": game.tier,
                        "world_size": game.world_size,
                        "nb_objects": game.nb_objects,
                        "quest_length": game.quest_length,
                        "game_seed": game.seed,
                        "episode_index": episode,
                        "prompt_variant": variant,
                    },
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--games_dirs", type=Path, nargs="+", required=True, help="Directories of generated games.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Where to write the parquet files.")
    parser.add_argument("--held_out", type=int, default=5, help="Games per tier in validation and in test.")
    parser.add_argument("--seed", type=int, default=42, help="Split seed.")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Write ``train/validation/test.parquet`` and ``dataset_stats.json`` to the output directory.

    Args:
        args: Parsed command-line arguments.
    """
    games = find_games(args.games_dirs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    for split, members in split_games(games, args.held_out, args.seed).items():
        rows = make_rows(members)
        pq.write_table(pa.Table.from_pylist(rows), args.output_dir / f"{split}.parquet")
        stats[split] = {
            "games": dict(sorted(Counter(game.tier for game in members).items())),
            "rows": dict(sorted(Counter(row["data_source"] for row in rows).items())),
        }
        print(f"{split}: {len(members)} games, {len(rows)} rows {stats[split]['rows']}")
    with open(args.output_dir / STATS_FILE, "w") as handle:
        json.dump(stats, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main(parse_args())
