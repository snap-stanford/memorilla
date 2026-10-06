"""Export the trained memory module from a SkyRL training checkpoint as a Memorilla checkpoint.

The policy state of a single-GPU SkyRL checkpoint (``global_step_N/policy/model_world_size_1_rank_0.pt``) holds the
frozen decoder and every modality module. This keeps the memory module of one modality and writes ``memory.pt`` and
``config.json``, loadable with ``MemoryModule.from_pretrained`` and usable as an evaluation checkpoint.
"""

import argparse
from dataclasses import asdict, replace
from pathlib import Path

import torch

from memorilla.memory import DEFAULT_NUM_HEADS, MemoryConfig, MemoryModule

POLICY_STATE_FILE = Path("policy") / "model_world_size_1_rank_0.pt"
ENCODER_PREFIX = "_skyrl_modality_encoders.{modality}.memory."


def extract_memory_state(checkpoint: Path, modality: str) -> dict[str, torch.Tensor]:
    """Read the memory-module weights of one modality from a SkyRL checkpoint.

    Args:
        checkpoint: A ``global_step_N`` directory.
        modality: Modality name in the training config.

    Returns:
        The memory module state dict in float32.

    Raises:
        FileNotFoundError: If the single-GPU policy state file is missing.
        KeyError: If the checkpoint holds no memory module for ``modality``.
    """
    state_file = checkpoint / POLICY_STATE_FILE
    if not state_file.exists():
        raise FileNotFoundError(f"Expected the policy state of a single-GPU run at {state_file}.")
    state = torch.load(state_file, map_location="cpu", mmap=True, weights_only=False)

    prefix = ENCODER_PREFIX.format(modality=modality)
    memory_state = {}
    for key, value in state.items():
        if prefix in key:
            tensor = value.to_local() if hasattr(value, "to_local") else value
            memory_state[key.split(prefix, 1)[1]] = tensor.detach().clone().float()
    if not memory_state:
        raise KeyError(f"No `{prefix}*` weights in {state_file}.")
    return memory_state


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("checkpoint", type=Path, help="SkyRL checkpoint directory (global_step_N).")
    parser.add_argument("output_dir", type=Path, help="Directory for memory.pt and config.json.")
    parser.add_argument("--modality", default="memorilla", help="Modality name in the training config.")
    parser.add_argument(
        "--num_heads", type=int, default=DEFAULT_NUM_HEADS, help="Attention heads of the memory module."
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Export the memory module and report its configuration.

    The architecture is inferred from the weights; the number of heads leaves no trace in them and comes from
    ``--num_heads``.

    Args:
        args: Parsed command-line arguments.
    """
    state = extract_memory_state(args.checkpoint, args.modality)
    config = replace(MemoryConfig.from_state_dict(state), num_heads=args.num_heads)
    module = MemoryModule(**asdict(config))
    module.load_state_dict(state, strict=True)
    module.save(args.output_dir)
    print(f"Saved {module.config} to {args.output_dir}")


if __name__ == "__main__":
    main(parse_args())
