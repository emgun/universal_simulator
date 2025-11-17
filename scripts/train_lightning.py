#!/usr/bin/env python
from __future__ import annotations

"""Training entrypoint using PyTorch Lightning."""

import argparse
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from ups.data.lightning_datamodule import UPSDataModule
from ups.training.lightning_modules import OperatorLightningModule


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--stage",
        default="operator",
        choices=["operator"],
        help="Training stage (Lightning currently supports operator stage)",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=None,
        help="Override device count (defaults to training.num_gpus from config)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    stage = args.stage

    num_gpus = int(cfg.get("training", {}).get("num_gpus", 1))
    devices = args.devices if args.devices is not None else num_gpus
    accelerator = "gpu" if devices and devices > 0 else "cpu"

    # Strategy selection (ddp by default; fsdp if enabled)
    strategy: str | Any = "auto"
    if devices and devices > 1:
        strategy = "ddp"
        if cfg.get("training", {}).get("use_fsdp2", False):
            strategy = "fsdp"

    precision = _precision_from_cfg(cfg)
    grad_clip = cfg.get("training", {}).get("grad_clip")
    accum_steps = int(cfg.get("training", {}).get("accum_steps", 1))
    deterministic = bool(cfg.get("deterministic", False))
    benchmark = bool(cfg.get("benchmark", True))

    stage_cfg = cfg.get("stages", {}).get(stage, {}) if isinstance(cfg.get("stages"), dict) else {}
    epochs = int(stage_cfg.get("epochs", 1))

    # Model + Data
    if stage == "operator":
        model = OperatorLightningModule(cfg)
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
    callbacks = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename=f"{stage}-{{epoch:02d}}-{{val_nrmse:.4f}}",
            monitor="val/nrmse",
            mode="min",
            save_top_k=3,
        ),
    ]
    patience = stage_cfg.get("patience", cfg.get("training", {}).get("patience"))
    if patience is not None:
        callbacks.append(
            EarlyStopping(
                monitor="val/nrmse",
                patience=int(patience),
                mode="min",
            )
        )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=precision,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=grad_clip,
        accumulate_grad_batches=accum_steps,
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True,
        num_sanity_val_steps=0,  # Skip sanity check
        limit_val_batches=0,  # Disable validation (val data from WandB artifacts)
        # Note: replace_sampler_ddp removed in Lightning 2.0+
        # DDP now preserves custom samplers by default
        deterministic=deterministic,
        benchmark=benchmark,
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
