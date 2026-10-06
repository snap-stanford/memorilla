"""Tests for the recipes, argument parsing and learning-rate schedule, plus CPU importability of every module."""

import argparse
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from memorilla.utils import read_recipe
from train import MIN_LR_RATIO, MemoryTrainingModule, parse_args

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
RECIPES = sorted(CONFIG_DIR.glob("*.yaml"))
MODULES = [
    "memorilla",
    "memorilla.benchmarks",
    "memorilla.data",
    "memorilla.embeddings",
    "memorilla.llm",
    "memorilla.memory",
    "memorilla.metrics",
    "memorilla.paths",
    "memorilla.utils",
    "train",
    "evaluate_baselines",
]
INFERENCE_MODULES = ["memorilla.vllm", "evaluate"]
STEPS = 100
PEAK_LR = 1e-4
EPOCH_WARNING = "The epoch parameter"


def recipe_args(name: str) -> argparse.Namespace:
    """Parse a recipe from ``configs/`` without further flags.

    Args:
        name: Recipe file name without the extension.

    Returns:
        The parsed training arguments.
    """
    return parse_args(["--config", str(CONFIG_DIR / f"{name}.yaml")])


@pytest.mark.parametrize("recipe", RECIPES, ids=[recipe.stem for recipe in RECIPES])
def test_recipe_parses(recipe: Path) -> None:
    """Every recipe only sets known options and yields a complete configuration."""
    args = recipe_args(recipe.stem)
    assert args.datasets and args.output_dir
    assert isinstance(args.learning_rate, float) and isinstance(args.retrieval_init, bool)
    assert args.num_self_attn_layers == 0


def test_flags_override_recipe() -> None:
    """Explicit flags take precedence over the recipe."""
    args = parse_args(
        ["--config", str(CONFIG_DIR / "single_task.yaml"), "--datasets", "pmv2", "--learning_rate", "5e-5"]
    )
    assert args.datasets == ["pmv2"] and args.learning_rate == 5e-5
    assert args.init_memory == "runs/stage2/epoch-00"


def test_recipe_chain() -> None:
    """Each stage starts from the output of the stage before it."""
    stage1, stage2, stage3 = (recipe_args(name) for name in ("stage1_enwiki", "stage2_mixture", "stage3_multitask"))
    assert stage2.init_memory.startswith(stage1.output_dir + "/")
    assert stage3.init_memory.startswith(stage2.output_dir + "/")
    for name in ("personalization_pv4", "personalization_pmv2"):
        assert recipe_args(name).init_memory.startswith(stage3.output_dir + "/")
    factkg = recipe_args("factkg")
    assert factkg.retrieval_init and factkg.init_memory is None
    assert factkg.system_prompt == read_recipe(CONFIG_DIR / "factkg.yaml")["system_prompt"]
    assert "\n" not in factkg.system_prompt and factkg.system_prompt.endswith("with no other text.")


def learning_rates(warmup_ratio: float) -> list[float]:
    """Step the training optimiser and scheduler through a run of ``STEPS`` optimiser steps.

    Args:
        warmup_ratio: Fraction of the steps spent on warmup.

    Returns:
        The learning rate of every step.
    """
    parameter = torch.nn.Parameter(torch.zeros(1))
    module = SimpleNamespace(
        parameters=lambda: [parameter],
        hparams=SimpleNamespace(learning_rate=PEAK_LR, weight_decay=0.0, warmup_ratio=warmup_ratio),
        trainer=SimpleNamespace(estimated_stepping_batches=STEPS),
    )
    [optimizer], [schedule] = MemoryTrainingModule.configure_optimizers(module)
    rates = []
    for _ in range(STEPS):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        schedule["scheduler"].step()
    return rates


@pytest.mark.filterwarnings(f"ignore:{EPOCH_WARNING}:UserWarning")
@pytest.mark.parametrize("warmup_ratio", [0.0, 0.1])
def test_learning_rate_schedule(warmup_ratio: float) -> None:
    """The rate peaks when the warmup ends (at the first step without warmup), then decays to 10% of the peak."""
    rates = learning_rates(warmup_ratio)
    assert rates[int(warmup_ratio * STEPS)] == pytest.approx(PEAK_LR) and max(rates) == pytest.approx(PEAK_LR)
    assert rates[-1] == pytest.approx(PEAK_LR * MIN_LR_RATIO, rel=0.01)


@pytest.mark.filterwarnings(f"ignore:{EPOCH_WARNING}:UserWarning")
def test_warmup_can_span_the_whole_run() -> None:
    """A warmup over every step rises throughout and steps past the end of the run without error."""
    rates = learning_rates(1.0)
    assert all(earlier < later for earlier, later in zip(rates, rates[1:], strict=False))


def test_unknown_recipe_keys_are_rejected(tmp_path: Path) -> None:
    """A recipe with an unknown key is an error."""
    recipe = tmp_path / "bad.yaml"
    recipe.write_text("datasets: [pv4]\noutput_dir: runs/x\nlearnin_rate: 1.0\n")
    with pytest.raises(SystemExit):
        parse_args(["--config", str(recipe)])


@pytest.mark.parametrize("module", MODULES)
def test_module_imports_without_gpu(module: str) -> None:
    """Every module and entry point imports on a CPU-only machine."""
    importlib.import_module(module)


@pytest.mark.parametrize("module", INFERENCE_MODULES)
def test_inference_modules_import_without_gpu(module: str) -> None:
    """The vLLM-backed modules import on a CPU-only machine wherever vLLM is installed."""
    pytest.importorskip("vllm")
    importlib.import_module(module)
