#!/usr/bin/env python
from __future__ import annotations

"""Training entrypoint using PyTorch Lightning."""

import argparse
import os
import time
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.tuner.tuning import Tuner
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from ups.data.lightning_datamodule import UPSDataModule
from ups.training.lightning_modules import (
    ConsistencyLightningModule,
    DiffusionLightningModule,
    OperatorLightningModule,
)


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _precision_from_cfg(cfg: dict) -> str:
    training_cfg = cfg.get("training", {}) if isinstance(cfg.get("training"), dict) else {}
    amp_enabled = bool(training_cfg.get("amp", False))
    if not amp_enabled:
        return "32-true"
    amp_dtype = str(training_cfg.get("amp_dtype", "bfloat16")).lower()
    if amp_dtype == "float16":
        return "16-mixed"
    return "bf16-mixed"


def _is_rank_0() -> bool:
    return int(os.environ.get("RANK", 0)) == 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--stage",
        default="operator",
        choices=["all", "operator", "diff_residual", "consistency_distill", "steady_prior"],
        help="Training stage (use 'all' to run all enabled stages sequentially)",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=None,
        help="Override device count (defaults to training.num_gpus from config)",
    )
    parser.add_argument(
        "--tune-batch-size",
        action="store_true",
        help="Run Lightning batch size tuner before training",
    )
    parser.add_argument(
        "--tune-lr",
        action="store_true",
        help="Run Lightning LR finder before training",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Handle "all" stage by running enabled stages sequentially
    if args.stage == "all":
        stages_to_run = []
        for stage_name in ["operator", "diff_residual", "consistency_distill", "steady_prior"]:
            stage_cfg = cfg.get("stages", {}).get(stage_name, {})
            if int(stage_cfg.get("epochs", 0)) > 0:
                stages_to_run.append(stage_name)

        if not stages_to_run:
            if _is_rank_0():
                print("ℹ️  No stages with epochs > 0, exiting.")
            return

        if _is_rank_0():
            print(f"ℹ️  Running stages sequentially: {', '.join(stages_to_run)}")
        for stage_name in stages_to_run:
            if _is_rank_0():
                print(f"\n{'='*70}")
                print(f"▶ Starting stage: {stage_name}")
                print(f"{'='*70}\n")
            # Re-call this script with specific stage
            import sys
            import subprocess
            cmd = [sys.executable, __file__, "--config", args.config, "--stage", stage_name]
            if args.devices is not None:
                cmd.extend(["--devices", str(args.devices)])
            subprocess.run(cmd, check=True)
        if _is_rank_0():
            print(f"\n{'='*70}")
            print(f"✅ All stages complete!")
            print(f"{'='*70}")
        return

    stage = args.stage

    # Disable CUDA graphs if requested (to avoid OOM with FSDP + compile)
    if cfg.get("training", {}).get("torch_inductor_disable_cudagraphs", False):
        os.environ["TORCH_INDUCTOR_DISABLE_CUDAGRAPHS"] = "1"
        if _is_rank_0():
            print("ℹ️  CUDA graphs disabled (TORCH_INDUCTOR_DISABLE_CUDAGRAPHS=1)")

    num_gpus = int(cfg.get("training", {}).get("num_gpus", 1))
    devices = args.devices if args.devices is not None else num_gpus
    accelerator = "gpu" if devices and devices > 0 else "cpu"
    skip_lightning_val = bool(cfg.get("training", {}).get("skip_lightning_val", False))

    # Strategy selection (ddp by default; fsdp if enabled)
    strategy: str | Any = "auto"
    if devices and devices > 1:
        strategy = "ddp"
        if cfg.get("training", {}).get("use_fsdp2", False):
            strategy = "fsdp"

    precision = _precision_from_cfg(cfg)
    training_cfg = cfg.get("training", {}) if isinstance(cfg.get("training"), dict) else {}
    grad_clip = training_cfg.get("grad_clip")
    accum_steps = int(training_cfg.get("accum_steps", 1))
    deterministic = bool(cfg.get("deterministic", False))
    benchmark = bool(cfg.get("benchmark", True))
    if training_cfg.get("dynamo_suppress_errors", False):
        os.environ["TORCHDYNAMO_SUPPRESS_ERRORS"] = "1"
        if _is_rank_0():
            print("ℹ️  TorchDynamo errors will be suppressed (fallback to eager)")
    if devices and devices > 1 and cfg.get("training", {}).get("compile", False):
        if _is_rank_0():
            print("⚠️  Warning: compile + multi-GPU may be unstable on some stacks. Disable if issues arise.")

    stage_cfg = cfg.get("stages", {}).get(stage, {}) if isinstance(cfg.get("stages"), dict) else {}
    epochs = int(stage_cfg.get("epochs", 1))

    # Model + Data
    stage_cfg = cfg.get("stages", {}).get(stage, {}) if isinstance(cfg.get("stages"), dict) else {}
    stage_epochs = int(stage_cfg.get("epochs", 0) or 0)

    if stage_epochs <= 0:
        if _is_rank_0():
            print(f"ℹ️  Stage {stage} has epochs<=0, skipping.")
        return

    if stage == "operator":
        model = OperatorLightningModule(cfg)
    elif stage == "diff_residual":
        operator_ckpt = os.environ.get("OPERATOR_CKPT")
        model = DiffusionLightningModule(cfg, operator_ckpt=operator_ckpt)
    elif stage == "consistency_distill":
        operator_ckpt = os.environ.get("OPERATOR_CKPT")
        diffusion_ckpt = os.environ.get("DIFFUSION_CKPT")
        model = ConsistencyLightningModule(cfg, operator_ckpt=operator_ckpt, diffusion_ckpt=diffusion_ckpt)
    else:
        raise NotImplementedError(f"Stage {stage} not implemented for Lightning")

    datamodule = UPSDataModule(cfg)

    # Logging
    wandb_cfg = cfg.get("logging", {}).get("wandb", {}) if isinstance(cfg.get("logging"), dict) else {}
    logger = None
    if wandb_cfg.get("enabled", True):
        logger = WandbLogger(
            project=wandb_cfg.get("project", "universal-simulator"),
            entity=wandb_cfg.get("entity"),
            name=wandb_cfg.get("run_name"),
            tags=wandb_cfg.get("tags", []),
            log_model=False,
        )

    ckpt_dir = Path(cfg.get("checkpoint", {}).get("dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    callbacks: list[pl.Callback] = []
    if not skip_lightning_val:
        # Honor optional checkpoint_interval; if null/0, rely on final checkpoint
        ckpt_interval = stage_cfg.get("checkpoint_interval", training_cfg.get("checkpoint_interval"))
        every_n_epochs = None
        if ckpt_interval is not None:
            try:
                ckpt_int = int(ckpt_interval)
                if ckpt_int > 0:
                    every_n_epochs = ckpt_int
            except (TypeError, ValueError):
                every_n_epochs = None
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(ckpt_dir),
                filename=f"{stage}-{{epoch:02d}}-{{val_nrmse:.4f}}",
                monitor="val/nrmse",
                mode="min",
                save_top_k=3,
                every_n_epochs=every_n_epochs,
            )
        )
        patience = stage_cfg.get("patience", cfg.get("training", {}).get("patience"))
        if patience is not None:
            callbacks.append(
                EarlyStopping(
                    monitor="val/nrmse",
                    patience=int(patience),
                    mode="min",
                )
            )

    profile_mode = str(training_cfg.get("lightning_profile", "none")).lower()
    profiler = None
    if profile_mode == "profiler":
        from pytorch_lightning.profilers import PyTorchProfiler

        profiler = PyTorchProfiler()

    class WandbEpochTimer(pl.Callback):
        def __init__(self, batch_size: int, accum: int, world_size: int):
            super().__init__()
            self.batch_size = batch_size
            self.accum = accum
            self.world_size = world_size
            self.epoch_start = None

        def on_train_epoch_start(self, trainer, pl_module):
            self.epoch_start = time.time()

        def on_train_epoch_end(self, trainer, pl_module):
            if self.epoch_start is None:
                return
            duration = time.time() - self.epoch_start
            total_batches = trainer.num_training_batches
            effective_batch = self.batch_size * max(1, self.world_size) * max(1, self.accum)
            throughput = (total_batches * effective_batch) / duration if duration > 0 else 0.0
            if trainer.logger:
                trainer.logger.log_metrics(
                    {
                        "time/epoch_seconds": duration,
                        "time/throughput_samples_per_s": throughput,
                    },
                    step=trainer.global_step,
                )

    world_size = devices if isinstance(devices, int) else (devices[0] if isinstance(devices, (list, tuple)) else 1)
    epoch_timer_cb = WandbEpochTimer(
        batch_size=int(training_cfg.get("batch_size", 1)),
        accum=accum_steps,
        world_size=world_size or 1,
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=precision,
        logger=logger,
        callbacks=callbacks + [epoch_timer_cb],
        gradient_clip_val=grad_clip,
        accumulate_grad_batches=accum_steps,
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True,
        num_sanity_val_steps=0,  # avoid extra val dataloader build/cache preload before training
        limit_val_batches=0 if skip_lightning_val else None,
        # Note: replace_sampler_ddp removed in Lightning 2.0+
        # DDP now preserves custom samplers by default
        deterministic=deterministic,
        benchmark=benchmark,
        enable_checkpointing=not skip_lightning_val,
        profiler=profiler,
    )

    # Optional tuners
    tune_bs_flag = args.tune_batch_size or bool(training_cfg.get("tune_batch_size", False))
    tune_lr_flag = args.tune_lr or bool(training_cfg.get("tune_lr", False))
    if devices and isinstance(devices, int) and devices > 1 and (tune_bs_flag or tune_lr_flag):
        if tune_bs_flag:
            if _is_rank_0():
                print("ℹ️  Skipping batch size tuner: not supported for distributed strategies")
        if tune_lr_flag:
            if _is_rank_0():
                print("ℹ️  Skipping LR finder: not supported for distributed strategies")
        tune_bs_flag = False
        tune_lr_flag = False

    if tune_bs_flag:
        try:
            tuner = Tuner(trainer)
            new_bs = tuner.scale_batch_size(model, datamodule=datamodule, mode="power")
            cfg.setdefault("training", {})["batch_size"] = new_bs
            # Rebuild datamodule with new batch size
            datamodule = UPSDataModule(cfg)
            if _is_rank_0():
                print(f"✓ Tuned batch size: {new_bs}")
        except Exception as exc:
            if _is_rank_0():
                print(f"⚠️  Batch size tuner failed: {exc}")
    if tune_lr_flag:
        try:
            tuner = Tuner(trainer)
            lr_finder = tuner.lr_find(model, datamodule=datamodule)
            suggestion = lr_finder.suggestion()
            if _is_rank_0():
                print(f"ℹ️  Suggested LR: {suggestion}")
            # Apply suggested LR to optimizer config if possible
            stage_cfg.setdefault("optimizer", {})
            stage_cfg["optimizer"]["lr"] = float(suggestion)
        except Exception as exc:
            if _is_rank_0():
                print(f"⚠️  LR finder failed: {exc}")

    trainer.fit(model, datamodule=datamodule)

    # Always persist a final checkpoint for the stage (Lightning ModelCheckpoint sometimes skips)
    final_ckpt_path = ckpt_dir / f"{stage}_last.ckpt"
    try:
        trainer.save_checkpoint(str(final_ckpt_path))
        if _is_rank_0():
            print(f"✅ Saved final checkpoint to {final_ckpt_path}")
    except Exception as exc:
        if _is_rank_0():
            print(f"⚠️  Failed to save final checkpoint: {exc}")

    # Optional test step (skip if not implemented)
    try:
        # Force test split and disable preloading for eval
        if isinstance(cfg.get("data"), dict):
            cfg["data"]["split"] = "test"
        if isinstance(cfg.get("training"), dict):
            cfg["training"]["use_preloaded_cache"] = False
        
        # Recreate datamodule for test phase
        test_datamodule = UPSDataModule(cfg)
        trainer.test(model, datamodule=test_datamodule)
    except Exception as e:
        if _is_rank_0():
            print(f"ℹ️  Skipping test (test_step not implemented): {e}")


if __name__ == "__main__":
    main()
