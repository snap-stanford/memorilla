"""Train a Memorilla memory module against a frozen decoder.

One entry point covers every recipe: pass any list of dataset configs with --datasets, warm-start from an earlier
checkpoint with --init_memory and load defaults from a YAML recipe with --config (explicit flags win).
"""

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, Logger, WandbLogger
from lightning.pytorch.utilities import rank_zero_info, rank_zero_warn
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from transformers import AutoConfig

from memorilla.data import (
    ANSWER_COLUMN,
    COLLECTION_COLUMN,
    DEFAULT_SYSTEM_PROMPT,
    QUESTION_COLUMN,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    MemoryCollator,
    MemoryDataset,
    ensure_data,
    split_files,
)
from memorilla.embeddings import EmbeddingStore, ensure_embeddings, load_split_with_embeddings
from memorilla.llm import MemoryLLM
from memorilla.memory import (
    DEFAULT_DROPOUT,
    DEFAULT_NUM_CROSS_ATTN_LAYERS,
    DEFAULT_NUM_HEADS,
    DEFAULT_NUM_MEMORIES,
    DEFAULT_NUM_SELF_ATTN_LAYERS,
    MemoryModule,
)
from memorilla.paths import DATA_DIR, DATA_REPO, DEFAULT_DECODER, DEFAULT_ENCODER, MODEL_CACHE_DIR, resolve
from memorilla.utils import DEFAULT_SEED, MEMORY_PREFIX, load_memory_state, read_recipe, str2bool

TRAIN_COLUMNS = [COLLECTION_COLUMN, QUESTION_COLUMN, ANSWER_COLUMN]
LAST_CHECKPOINT = "last.ckpt"
LOG_DIR = "logs"
EPOCH_DIR_TEMPLATE = "epoch-{:02d}"
WARMUP_START_FACTOR = 1e-8
MIN_LR_RATIO = 0.1
GRADIENT_CLIP_VAL = 1.0
LOG_EVERY_N_STEPS = 10


class EpochCheckpoint(Callback):
    """Save the memory module to ``<output_dir>/epoch-NN/`` at the end of every training epoch."""

    def __init__(self, output_dir: Path) -> None:
        """Configure the callback.

        Args:
            output_dir: Run directory.
        """
        super().__init__()
        self.output_dir = output_dir

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: "MemoryTrainingModule") -> None:
        """Write ``memory.pt`` and ``config.json`` for the finished epoch and refresh ``last.ckpt``.

        Args:
            trainer: The trainer.
            pl_module: The module being trained.
        """
        if trainer.is_global_zero:
            directory = pl_module.memory.save(self.output_dir / EPOCH_DIR_TEMPLATE.format(trainer.current_epoch))
            rank_zero_info(f"Saved memory module to {directory}")
        trainer.save_checkpoint(self.output_dir / LAST_CHECKPOINT)


class MemoryTrainingModule(pl.LightningModule):
    """Lightning wrapper that trains only the memory module of a ``MemoryLLM``."""

    def __init__(self, args: argparse.Namespace) -> None:
        """Build the memory module (optionally warm-started) and load the frozen decoder.

        Args:
            args: Parsed command-line arguments.
        """
        super().__init__()
        self.save_hyperparameters(vars(args))
        self.strict_loading = False

        embedding_dim = AutoConfig.from_pretrained(args.encoder, cache_dir=args.cache_dir).hidden_size
        output_dim = AutoConfig.from_pretrained(args.decoder, cache_dir=args.cache_dir).hidden_size
        self.memory = MemoryModule(
            embedding_dim=embedding_dim,
            output_dim=output_dim,
            num_memories=args.num_memories,
            num_heads=args.num_heads,
            num_self_attn_layers=args.num_self_attn_layers,
            num_cross_attn_layers=args.num_cross_attn_layers,
            dropout=args.dropout,
            retrieval_init=args.retrieval_init,
        )
        if args.init_memory:
            self._warm_start(resolve(args.init_memory))

        self.llm = MemoryLLM(
            args.decoder,
            self.memory,
            attn_implementation=args.attn_implementation,
            cache_dir=args.cache_dir,
        )
        self.memory.to(dtype=torch.bfloat16).train()
        self.llm.text_model.requires_grad_(False).eval()

    def _warm_start(self, path: Path) -> None:
        """Initialise the memory module from a checkpoint; keys it lacks keep their fresh initialisation.

        Args:
            path: Checkpoint directory or weights file.
        """
        state = load_memory_state(path)
        missing, unexpected = self.memory.load_state_dict(state, strict=False)
        rank_zero_info(f"Initialised memory module from {path} ({len(state) - len(unexpected)} tensors).")
        if missing:
            rank_zero_warn(f"Freshly initialised (absent from {path}): {missing}")
        if unexpected:
            rank_zero_warn(f"Ignored (not part of this architecture): {unexpected}")

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """One optimisation step; the logged training loss is this process's (no cross-GPU synchronisation per step).

        Args:
            batch: Output of ``MemoryCollator``.
            batch_idx: Batch index.

        Returns:
            The answer-token loss.
        """
        loss = self.llm(**batch).loss
        self.log("train_loss", loss.detach(), on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """One validation step; the validation loss is averaged over all processes at the end of validation.

        Args:
            batch: Output of ``MemoryCollator``.
            batch_idx: Batch index.

        Returns:
            The answer-token loss.
        """
        loss = self.llm(**batch).loss
        self.log("val_loss", loss.detach(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[dict[str, Any]]]:
        """AdamW over the memory parameters with linear warmup followed by cosine decay to 10% of the peak rate.

        Without warmup steps (``warmup_ratio`` 0 or too few steps), the decay starts at the peak rate.

        Returns:
            The optimizer and its per-step scheduler.
        """
        optimizer = AdamW(
            [parameter for parameter in self.parameters() if parameter.requires_grad],
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(self.hparams.warmup_ratio * total_steps)
        eta_min = self.hparams.learning_rate * MIN_LR_RATIO
        if warmup_steps == 0:
            scheduler = CosineAnnealingLR(optimizer, T_max=max(total_steps, 1), eta_min=eta_min)
        else:
            warmup = LinearLR(optimizer, start_factor=WARMUP_START_FACTOR, end_factor=1.0, total_iters=warmup_steps)
            decay = CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup_steps, 1), eta_min=eta_min)
            scheduler = SequentialLR(optimizer, schedulers=[warmup, decay], milestones=[warmup_steps])
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Keep only the memory weights in resume checkpoints; the decoder is frozen and reloaded from its source.

        Args:
            checkpoint: The checkpoint being written.
        """
        checkpoint["state_dict"] = {
            key: value for key, value in checkpoint["state_dict"].items() if key.startswith(MEMORY_PREFIX)
        }


def prepare_data(
    args: argparse.Namespace,
) -> tuple[MemoryDataset, MemoryDataset | None, dict[tuple[str, str], EmbeddingStore]]:
    """Fetch the data and embeddings of every config and build the training and validation sets.

    Args:
        args: Parsed command-line arguments.

    Returns:
        The training set, the validation set (None when no config has a validation split) and the embedding stores.

    Raises:
        FileNotFoundError: If a config has no training split.
    """
    train_parts, validation_parts, stores = [], [], {}
    for config in args.datasets:
        root = ensure_data(config, (TRAIN_SPLIT, VALIDATION_SPLIT), repo=args.data_repo, data_dir=args.data_dir)
        splits = [split for split in (TRAIN_SPLIT, VALIDATION_SPLIT) if split_files(config, split, root)]
        if TRAIN_SPLIT not in splits:
            raise FileNotFoundError(f"Config {config!r} has no training split in {args.data_repo}.")
        ensure_embeddings(
            config, splits, repo=args.data_repo, data_dir=args.data_dir, encoder=args.encoder, cache_dir=args.cache_dir
        )

        for split in splits:
            dataset, stores[(config, split)] = load_split_with_embeddings(
                config, split, root, TRAIN_COLUMNS, args.encoder
            )
            (train_parts if split == TRAIN_SPLIT else validation_parts).append((config, split, dataset))

    validation = MemoryDataset(validation_parts) if validation_parts else None
    return MemoryDataset(train_parts), validation, stores


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=str, default=None, help="YAML recipe whose keys set argument defaults.")

    data = parser.add_argument_group("data")
    data.add_argument("--datasets", type=str, nargs="+", default=None, help="Dataset configs to train on.")
    data.add_argument("--data_repo", type=str, default=DATA_REPO, help="Hub dataset repo or local directory.")
    data.add_argument("--data_dir", type=str, default=str(DATA_DIR), help="Local mirror of a Hub data repo.")
    data.add_argument(
        "--system_prompt", type=str, default=DEFAULT_SYSTEM_PROMPT, help="System prompt of every example."
    )
    data.add_argument("--max_length", type=int, default=2048, help="Token limit of a training example.")

    model = parser.add_argument_group("model")
    model.add_argument("--decoder", type=str, default=DEFAULT_DECODER, help="Frozen decoder (Hub id or local path).")
    model.add_argument("--encoder", type=str, default=DEFAULT_ENCODER, help="Frozen encoder of the embeddings.")
    model.add_argument("--num_memories", type=int, default=DEFAULT_NUM_MEMORIES, help="Memory tokens per example.")
    model.add_argument("--num_heads", type=int, default=DEFAULT_NUM_HEADS, help="Attention heads of the memory module.")
    model.add_argument(
        "--num_self_attn_layers",
        type=int,
        default=DEFAULT_NUM_SELF_ATTN_LAYERS,
        help="Self-attention layers over the documents.",
    )
    model.add_argument(
        "--num_cross_attn_layers", type=int, default=DEFAULT_NUM_CROSS_ATTN_LAYERS, help="Refinement blocks."
    )
    model.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT, help="Dropout of the memory module.")
    model.add_argument(
        "--retrieval_init",
        type=str2bool,
        default=False,
        help="Start the slots from the documents most similar to the question.",
    )
    model.add_argument("--init_memory", type=str, default=None, help="Checkpoint directory or memory.pt to start from.")
    model.add_argument(
        "--attn_implementation", type=str, default="flash_attention_2", help="Attention implementation of the decoder."
    )

    optimisation = parser.add_argument_group("optimisation")
    optimisation.add_argument("--learning_rate", type=float, default=1e-4, help="Peak learning rate.")
    optimisation.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay.")
    optimisation.add_argument(
        "--warmup_ratio", type=float, default=0.1, help="Fraction of the optimiser steps spent on warmup."
    )
    optimisation.add_argument("--batch_size", type=int, default=2, help="Examples per GPU and batch.")
    optimisation.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Batches per optimiser step.")
    optimisation.add_argument("--max_epochs", type=int, default=5, help="Training epochs.")
    optimisation.add_argument(
        "--val_check_interval", type=float, default=1.0, help="Fraction of an epoch between validations."
    )
    optimisation.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")

    infrastructure = parser.add_argument_group("infrastructure")
    infrastructure.add_argument("--devices", type=int, default=1, help="GPUs per node (-1 uses all).")
    infrastructure.add_argument("--num_nodes", type=int, default=1, help="Number of nodes.")
    infrastructure.add_argument("--strategy", type=str, default="ddp", help="Lightning distributed strategy.")
    infrastructure.add_argument("--output_dir", type=str, default=None, help="Run directory (resumes if it exists).")
    infrastructure.add_argument("--num_workers", type=int, default=2, help="Data-loader worker processes.")
    infrastructure.add_argument("--cache_dir", type=str, default=MODEL_CACHE_DIR, help="Model cache directory.")
    infrastructure.add_argument("--wandb_project", type=str, default=None, help="Log to this W&B project.")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse arguments, applying the defaults of ``--config`` first.

    Args:
        argv: Arguments (None reads ``sys.argv``).

    Returns:
        The parsed arguments.
    """
    parser = build_parser()
    known, _ = parser.parse_known_args(argv)
    if known.config:
        recipe = read_recipe(known.config)
        destinations = {action.dest for action in parser._actions}
        unknown = sorted(set(recipe) - destinations)
        if unknown:
            parser.error(f"{known.config} sets unknown options: {unknown}")
        parser.set_defaults(**recipe)

    args = parser.parse_args(argv)
    if not args.datasets:
        parser.error("--datasets is required (directly or through --config).")
    if not args.output_dir:
        parser.error("--output_dir is required (directly or through --config).")
    return args


def build_loader(
    dataset: MemoryDataset, collator: MemoryCollator, args: argparse.Namespace, shuffle: bool
) -> DataLoader:
    """Build a data loader.

    Args:
        dataset: Training or validation set.
        collator: Batch collator.
        args: Parsed command-line arguments (batch size and worker count).
        shuffle: Whether to reshuffle every epoch.

    Returns:
        The data loader.
    """
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )


def main(args: argparse.Namespace) -> None:
    """Train the memory module.

    Besides the per-epoch ``epoch-NN`` directories, ``last.ckpt`` holds the single resume checkpoint; it is rewritten
    after every validation and at the end of every epoch. Resuming restores the weights, optimiser state and step
    count; the remaining batches are not drawn in the order of an uninterrupted run. Losses and learning rates are
    written to ``logs/version_N/metrics.csv`` (and to Weights & Biases with ``--wandb_project``).

    Args:
        args: Parsed command-line arguments.
    """
    pl.seed_everything(args.seed)
    torch.set_float32_matmul_precision("medium")
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_set, validation_set, stores = prepare_data(args)
    module = MemoryTrainingModule(args)
    collator = MemoryCollator(
        module.llm.tokenizer,
        stores,
        num_memories=args.num_memories,
        max_length=args.max_length,
        system_prompt=args.system_prompt,
    )

    loggers: list[Logger] = [CSVLogger(save_dir=output_dir, name=LOG_DIR)]
    if args.wandb_project:
        loggers.append(WandbLogger(project=args.wandb_project, name=output_dir.name, save_dir=str(output_dir)))
    resume_checkpoint = ModelCheckpoint(
        dirpath=output_dir, filename=Path(LAST_CHECKPOINT).stem, enable_version_counter=False
    )
    callbacks: list[Callback] = [
        EpochCheckpoint(output_dir),
        resume_checkpoint,
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=args.strategy,
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gradient_clip_val=GRADIENT_CLIP_VAL,
        val_check_interval=args.val_check_interval if validation_set is not None else None,
        log_every_n_steps=LOG_EVERY_N_STEPS,
        logger=loggers,
        callbacks=callbacks,
        default_root_dir=output_dir,
    )
    last_checkpoint = output_dir / LAST_CHECKPOINT
    train_loader = build_loader(train_set, collator, args, shuffle=True)
    validation_loader = None if validation_set is None else build_loader(validation_set, collator, args, shuffle=False)
    trainer.fit(
        module,
        train_dataloaders=train_loader,
        val_dataloaders=validation_loader,
        ckpt_path=last_checkpoint if last_checkpoint.exists() else None,
    )


if __name__ == "__main__":
    main(parse_args())
