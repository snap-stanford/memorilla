"""Generate TextWorld games for the RL difficulty tiers.

Medium and hard games are compiled with ``tw-make`` (a ``.z8`` game plus its ``.json`` spec) and played through
Inform7. The long tiers (long, mega, huge, extreme) are written as ``.json`` specs only and played by the in-process
simulator of the SkyRL fork. File names encode the tier and the sampled parameters, e.g.
``game_00007_medium_w18_o17_q6_s79826.z8``.
"""

import argparse
from collections.abc import Iterator
from dataclasses import dataclass
import itertools
import multiprocessing
from pathlib import Path
import random
import subprocess

import textworld
from textworld.generator import make_game

GAME_SEED_STRIDE = 9973
MAX_QUEST_BREADTH = 3
ATTEMPTS_PER_GAME = 4
COMPILE_SUFFIXES = (".json", ".ni")


@dataclass(frozen=True)
class Tier:
    """Parameter ranges (inclusive) of a difficulty tier.

    Attributes:
        world_size: Range of the number of rooms.
        nb_objects: Range of the number of objects.
        quest_length: Range of the quest length.
        compiled: Whether games are compiled with ``tw-make`` (otherwise written as ``.json`` specs).
    """

    world_size: tuple[int, int]
    nb_objects: tuple[int, int]
    quest_length: tuple[int, int]
    compiled: bool


TIERS: dict[str, Tier] = {
    "medium": Tier(world_size=(15, 25), nb_objects=(12, 22), quest_length=(6, 10), compiled=True),
    "hard": Tier(world_size=(24, 40), nb_objects=(20, 35), quest_length=(10, 16), compiled=True),
    "long": Tier(world_size=(25, 35), nb_objects=(30, 50), quest_length=(15, 22), compiled=False),
    "mega": Tier(world_size=(35, 50), nb_objects=(50, 90), quest_length=(25, 35), compiled=False),
    "huge": Tier(world_size=(38, 50), nb_objects=(60, 100), quest_length=(40, 55), compiled=False),
    "extreme": Tier(world_size=(50, 65), nb_objects=(80, 130), quest_length=(60, 75), compiled=False),
}


@dataclass(frozen=True)
class GameSpec:
    """Everything needed to build one game.

    Attributes:
        index: Game index in the parameter stream.
        tier: Difficulty tier.
        world_size: Number of rooms.
        nb_objects: Number of objects.
        quest_length: Length of the quest.
        quest_breadth: Number of parallel quest branches (spec-only tiers).
        seed: Seed of the game generator.
        path: Output file (``.z8`` for compiled tiers, ``.json`` otherwise).
    """

    index: int
    tier: str
    world_size: int
    nb_objects: int
    quest_length: int
    quest_breadth: int
    seed: int
    path: Path


def parse_mix(text: str) -> dict[str, float]:
    """Parse a tier mix such as ``medium:0.3,hard:0.7`` into normalised weights.

    Args:
        text: Comma-separated ``tier:weight`` pairs.

    Returns:
        Tier weights summing to one, in the order given.

    Raises:
        ValueError: On an unknown tier or a non-positive weight.
    """
    weights: dict[str, float] = {}
    for item in text.split(","):
        tier, weight = item.split(":")
        tier = tier.strip()
        if tier not in TIERS:
            raise ValueError(f"Unknown tier `{tier}`; choose from {sorted(TIERS)}.")
        if float(weight) <= 0:
            raise ValueError(f"Tier `{tier}` needs a positive weight.")
        weights[tier] = float(weight)
    total = sum(weights.values())
    return {tier: weight / total for tier, weight in weights.items()}


def iter_specs(mix: dict[str, float], seed: int, output_dir: Path) -> Iterator[GameSpec]:
    """Draw the tier and parameters of games ``0, 1, 2, ...`` from one seeded random stream.

    Args:
        mix: Normalised tier weights.
        seed: Seed of the parameter stream; game ``i`` is generated with seed ``seed + 9973 * (i + 1)``.
        output_dir: Directory the games are written to.

    Yields:
        One spec per game index.
    """
    rng = random.Random(seed)
    for index in itertools.count():
        tier = _choose_tier(rng, mix)
        preset = TIERS[tier]
        world_size = rng.randint(*preset.world_size)
        nb_objects = rng.randint(*preset.nb_objects)
        quest_length = rng.randint(*preset.quest_length)
        quest_breadth = 1 if preset.compiled else rng.randint(1, max(1, min(MAX_QUEST_BREADTH, quest_length // 4)))
        game_seed = seed + GAME_SEED_STRIDE * (index + 1)
        name = f"game_{index:05d}_{tier}_w{world_size}_o{nb_objects}_q{quest_length}_s{game_seed}"
        yield GameSpec(
            index=index,
            tier=tier,
            world_size=world_size,
            nb_objects=nb_objects,
            quest_length=quest_length,
            quest_breadth=quest_breadth,
            seed=game_seed,
            path=output_dir / f"{name}{'.z8' if preset.compiled else '.json'}",
        )


def _choose_tier(rng: random.Random, mix: dict[str, float]) -> str:
    """Draw a tier by inverse-CDF sampling over the tiers of ``mix`` in ``TIERS`` order.

    Args:
        rng: Parameter stream.
        mix: Normalised tier weights.

    Returns:
        The drawn tier.
    """
    threshold = rng.random()
    tiers = [tier for tier in TIERS if tier in mix]
    cumulative = 0.0
    for tier in tiers:
        cumulative += mix[tier]
        if threshold <= cumulative:
            return tier
    return tiers[-1]


def build_game(spec: GameSpec) -> tuple[GameSpec, str | None]:
    """Generate one game, skipping it when the file already exists.

    A failed game leaves no files behind, including the ``.json`` spec and Inform7 source that ``tw-make`` writes
    before compiling.

    Args:
        spec: Game to build.

    Returns:
        The spec and an error message, or None on success.
    """
    if spec.path.exists():
        return spec, None
    try:
        if TIERS[spec.tier].compiled:
            command = ["tw-make", "custom", "--world-size", str(spec.world_size), "--nb-objects", str(spec.nb_objects)]
            command += ["--quest-length", str(spec.quest_length), "--seed", str(spec.seed), "--output", str(spec.path)]
            subprocess.run(command, check=True, capture_output=True)
        else:
            options = textworld.GameOptions()
            options.seeds = spec.seed
            options.nb_rooms = spec.world_size
            options.nb_objects = spec.nb_objects
            options.quest_length = spec.quest_length
            options.quest_breadth = spec.quest_breadth
            make_game(options).save(str(spec.path))
    except Exception as error:  # the quest planner fails on some parameter draws; those games are skipped
        for path in (spec.path, *(spec.path.with_suffix(suffix) for suffix in COMPILE_SUFFIXES)):
            path.unlink(missing_ok=True)
        return spec, f"{type(error).__name__}: {error}"
    return spec, None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for the generated games.")
    parser.add_argument("--tiers", required=True, help="Tier mix, e.g. `medium:0.3,hard:0.7` or `long:1`.")
    parser.add_argument("--num_games", type=int, required=True, help="Number of games to produce.")
    parser.add_argument("--seed", type=int, default=42, help="Seed of the parameter stream.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel worker processes.")
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=None,
        help=f"Game indices to try (None tries {ATTEMPTS_PER_GAME} per requested game).",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Generate games in parallel worker processes until ``--num_games`` of them exist.

    Args:
        args: Parsed command-line arguments.
    """
    args.output_dir.mkdir(parents=True, exist_ok=True)
    max_attempts = args.max_attempts if args.max_attempts is not None else ATTEMPTS_PER_GAME * args.num_games
    workers = max(1, args.workers)
    specs = iter_specs(parse_mix(args.tiers), args.seed, args.output_dir)

    built = attempts = 0
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        while built < args.num_games and attempts < max_attempts:
            batch = list(itertools.islice(specs, min(workers, args.num_games - built, max_attempts - attempts)))
            attempts += len(batch)
            for spec, error in pool.imap(build_game, batch):
                built += error is None
                print(spec.path.name if error is None else f"{spec.path.name} skipped: {error}")
    print(f"{built} games in {args.output_dir} from {attempts} game indices")


if __name__ == "__main__":
    main(parse_args())
