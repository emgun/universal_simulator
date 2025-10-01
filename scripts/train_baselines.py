#!/usr/bin/env python
from __future__ import annotations

"""Train simple latent-space baselines for benchmarking."""

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, Optional

import sys

import numpy as np
import torch
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import yaml
import math

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ups.baselines.models import BaselineConfig, build_baseline
from ups.core.latent_state import LatentState
from ups.data.latent_pairs import build_latent_pair_loader, unpack_batch
from ups.utils.monitoring import init_monitoring_session


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TrainingLogger:
    def __init__(self, cfg: Dict, baseline_name: str) -> None:
        baseline_cfg = cfg.get("baseline", {})
        log_path = baseline_cfg.get("log_path", "reports/baseline_log.jsonl")
        self.session = init_monitoring_session(cfg, component=f"baseline-{baseline_name}", file_path=log_path)
        self.stage = f"baseline_{baseline_name}"
        self.session.log({"stage": self.stage, "event": "config", "config": cfg})

    def log(self, epoch: int, loss: float, lr: float | None) -> None:
        self.session.log({
            "stage": self.stage,
            "epoch": epoch,
            "loss": loss,
            "lr": lr,
        })

    def log_eval(self, split: str, metrics: Dict[str, float]) -> None:
        entry = {f"metric_{k}": v for k, v in metrics.items()}
        entry.update({"stage": self.stage, "event": f"eval_{split}", "split": split})
        self.session.log(entry)

    def close(self) -> None:
        self.session.finish()


def _create_optimizer(cfg: Dict, model: torch.nn.Module) -> torch.optim.Optimizer:
    opt_cfg = cfg.get("optimizer", {})
    lr = opt_cfg.get("lr", 1e-3)
    weight_decay = opt_cfg.get("weight_decay", 0.0)
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def _create_scheduler(optimizer: torch.optim.Optimizer, cfg: Dict):
    sched_cfg = cfg.get("scheduler")
    if not sched_cfg:
        return None
    name = sched_cfg.get("name", "steplr").lower()
    if name == "steplr":
        return lr_scheduler.StepLR(
            optimizer,
            step_size=sched_cfg.get("step_size", 1),
            gamma=sched_cfg.get("gamma", 0.5),
        )
    if name == "cosineannealinglr":
        return lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=sched_cfg.get("t_max", 10),
            eta_min=sched_cfg.get("eta_min", 0.0),
        )
    raise ValueError(f"Unsupported scheduler '{name}'")


def _evaluate_split(model: torch.nn.Module, cfg: Dict, split: str) -> Dict[str, float]:
    eval_cfg = copy.deepcopy(cfg)
    data_cfg = eval_cfg.setdefault("data", {})
    data_cfg["split"] = split
    loader = build_latent_pair_loader(eval_cfg)
    device = next(model.parameters()).device if any(p.requires_grad for p in model.parameters()) else torch.device("cpu")
    model.eval()
    total_abs = 0.0
    total_sq = 0.0
    total = 0
    with torch.no_grad():
        for batch in loader:
            z0, z1, cond = unpack_batch(batch)
            cond_device = {k: v.to(device) for k, v in cond.items()}
            pred = model(z0.to(device), cond_device)
            diff = pred - z1.to(device)
            total_abs += diff.abs().sum().item()
            total_sq += diff.pow(2).sum().item()
            total += diff.numel()
    if total == 0:
        return {"mse": 0.0, "mae": 0.0, "rmse": 0.0}
    mse = total_sq / total
    mae = total_abs / total
    rmse = math.sqrt(mse)
    return {"mse": mse, "mae": mae, "rmse": rmse}


def train_baseline(cfg: Dict, baseline_name: str) -> Path:
    loader = build_latent_pair_loader(cfg)
    latent_cfg = cfg.get("latent", {})
    baseline_cfg = BaselineConfig(latent_dim=latent_cfg.get("dim", 32), tokens=latent_cfg.get("tokens", 64))
    model = build_baseline(baseline_name, baseline_cfg)
    params = list(model.parameters())
    optimizer: Optional[torch.optim.Optimizer] = None
    scheduler = None
    patience = cfg.get("baseline", {}).get("patience")
    epochs = cfg.get("baseline", {}).get("epochs", 3)

    if params:
        optimizer = _create_optimizer(cfg.get("baseline", {}), model)
        scheduler = _create_scheduler(optimizer, cfg.get("baseline", {}))
    else:
        epochs = 0

    logger = TrainingLogger(cfg, baseline_name)

    device = torch.device("cpu")
    model.to(device)

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_since_improve = 0

    if epochs == 0:
        logger.log(epoch=0, loss=0.0, lr=None)
        ckpt_dir = Path(cfg.get("checkpoint", {}).get("dir", "checkpoints"))
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"baseline_{baseline_name}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved baseline checkpoint to {ckpt_path}")
        logger.close()
        return ckpt_path

    for epoch in range(epochs):
        epoch_loss = 0.0
        batches = 0
        for batch in loader:
            z0, z1, cond = unpack_batch(batch)
            cond_device = {k: v.to(device) for k, v in cond.items()}
            z0 = z0.to(device)
            target = z1.to(device)
            pred = model(z0, cond_device)
            loss = F.mse_loss(pred, target)
            assert optimizer is not None
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1
        mean_loss = epoch_loss / max(batches, 1)
        lr = optimizer.param_groups[0].get("lr") if optimizer and optimizer.param_groups else None
        logger.log(epoch=epoch, loss=mean_loss, lr=lr)

        if mean_loss + 1e-6 < best_loss:
            best_loss = mean_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if patience is not None and epochs_since_improve > patience:
                break
        if scheduler is not None:
            scheduler.step()

    model.load_state_dict(best_state)
    for split in ("val", "test"):
        metrics = _evaluate_split(model, cfg, split)
        logger.log_eval(split, metrics)
    logger.close()

    ckpt_dir = Path(cfg.get("checkpoint", {}).get("dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"baseline_{baseline_name}.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved baseline checkpoint to {ckpt_path}")
    return ckpt_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train latent baselines")
    parser.add_argument("--config", default="configs/train_multi_pde.yaml")
    parser.add_argument("--baseline", choices=["identity", "linear", "mlp"], default="linear")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)
    train_baseline(cfg, args.baseline)


if __name__ == "__main__":
    main()
