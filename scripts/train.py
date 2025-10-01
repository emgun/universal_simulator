#!/usr/bin/env python
from __future__ import annotations

"""Training entrypoint for latent operator stages."""

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Dict, Optional

import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import yaml

try:
    import wandb
except ImportError:
    wandb = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from ups.core.latent_state import LatentState
from ups.models.latent_operator import LatentOperator, LatentOperatorConfig
from ups.core.blocks_pdet import PDETransformerConfig
from ups.models.diffusion_residual import DiffusionResidual, DiffusionResidualConfig
from ups.training.consistency_distill import DistillationConfig, distillation_loss
from ups.models.steady_prior import SteadyPrior, SteadyPriorConfig
from ups.data.latent_pairs import build_latent_pair_loader, unpack_batch
from ups.utils.monitoring import init_monitoring_session


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def set_seed(cfg: Dict) -> None:
    seed = cfg.get("seed", 17)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_checkpoint_dir(cfg: dict) -> Path:
    ckpt_cfg = cfg.get("checkpoint", {})
    directory = Path(ckpt_cfg.get("dir", "checkpoints"))
    directory.mkdir(parents=True, exist_ok=True)
    return directory


class TrainingLogger:
    def __init__(self, cfg: Dict[str, Dict], stage: str) -> None:
        training_cfg = cfg.get("training", {})
        log_path = training_cfg.get("log_path", "reports/training_log.jsonl")
        self.stage = stage
        self.session = init_monitoring_session(cfg, component=f"training-{stage}", file_path=log_path)
        self.session.log({"stage": self.stage, "event": "config", "config": cfg})

    def log(
        self,
        *,
        epoch: int,
        loss: float,
        optimizer: torch.optim.Optimizer,
        patience_counter: Optional[int] = None,
    ) -> None:
        lr = optimizer.param_groups[0].get("lr") if optimizer.param_groups else None
        entry = {
            "stage": self.stage,
            "epoch": epoch,
            "loss": loss,
            "lr": lr,
        }
        if patience_counter is not None:
            entry["epochs_since_improve"] = patience_counter
        self.session.log(entry)

    def close(self) -> None:
        self.session.finish()


def dataset_loader(cfg: dict) -> DataLoader:
    data_cfg = cfg.get("data", {})
    if not (data_cfg.get("task") or data_cfg.get("kind")):
        raise ValueError(
            "Training now requires a real dataset configuration. Set data.task for PDEBench or data.kind for Zarr datasets."
        )
    return build_latent_pair_loader(cfg)


def make_operator(cfg: dict) -> LatentOperator:
    latent_cfg = cfg.get("latent", {})
    dim = latent_cfg.get("dim", 32)
    pdet_cfg = cfg.get("operator", {}).get("pdet", {})
    if not pdet_cfg:
        pdet_cfg = {
            "input_dim": dim,
            "hidden_dim": dim * 2,
            "depths": [1, 1, 1],
            "group_size": max(dim // 2, 4),
            "num_heads": 4,
        }
    config = LatentOperatorConfig(
        latent_dim=dim,
        pdet=PDETransformerConfig(**pdet_cfg),
        time_embed_dim=dim,
    )
    return LatentOperator(config)


def _create_optimizer(cfg: dict, model: nn.Module, stage: str) -> torch.optim.Optimizer:
    stage_cfg = cfg.get("stages", {}).get(stage, {}) if isinstance(cfg.get("stages"), dict) else {}
    opt_cfg = stage_cfg.get("optimizer") or cfg.get("optimizer", {})
    name = opt_cfg.get("name", "adam").lower()
    lr = opt_cfg.get("lr", 1e-3)
    weight_decay = opt_cfg.get("weight_decay", 0.0)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        momentum = opt_cfg.get("momentum", 0.9)
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer '{name}'")


def _create_scheduler(optimizer: torch.optim.Optimizer, cfg: dict, stage: str):
    stage_cfg = cfg.get("stages", {}).get(stage, {})
    sched_cfg = stage_cfg.get("scheduler") or cfg.get("optimizer", {}).get("scheduler")
    if not sched_cfg:
        return None
    name = sched_cfg.get("name", "steplr").lower()
    if name == "steplr":
        step_size = sched_cfg.get("step_size", 1)
        gamma = sched_cfg.get("gamma", 0.5)
        return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if name == "cosineannealinglr":
        t_max = sched_cfg.get("t_max", 10)
        eta_min = sched_cfg.get("eta_min", 0.0)
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
    raise ValueError(f"Unsupported scheduler '{name}'")


def _get_patience(cfg: dict, stage: str) -> Optional[int]:
    stage_cfg = cfg.get("stages", {}).get(stage, {})
    if "patience" in stage_cfg:
        return stage_cfg["patience"]
    training_cfg = cfg.get("training", {})
    return training_cfg.get("patience")


def _should_stop(patience: Optional[int], epochs_since_improve: int) -> bool:
    if patience is None:
        return False
    return epochs_since_improve > patience


def train_operator(cfg: dict) -> None:
    loader = dataset_loader(cfg)
    operator = make_operator(cfg)
    train_cfg = cfg.get("training", {})
    dt = train_cfg.get("dt", 0.1)
    stage_cfg = cfg.get("stages", {}).get("operator", {})
    epochs = stage_cfg.get("epochs", 1)
    optimizer = _create_optimizer(cfg, operator, "operator")
    scheduler = _create_scheduler(optimizer, cfg, "operator")
    patience = _get_patience(cfg, "operator")
    logger = TrainingLogger(cfg, stage="operator")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    operator.to(device)
    dt_tensor = torch.tensor(dt, device=device)
    best_loss = float("inf")
    best_state = copy.deepcopy(operator.state_dict())
    epochs_since_improve = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        batches = 0
        for batch in loader:
            z0, z1, cond = unpack_batch(batch)
            cond_device = {k: v.to(device) for k, v in cond.items()}
            state = LatentState(z=z0.to(device), t=torch.tensor(0.0, device=device), cond=cond_device)
            target = z1.to(device)
            next_state = operator(state, dt_tensor)
            loss = F.mse_loss(next_state.z, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1
        mean_loss = epoch_loss / max(batches, 1)
        logger.log(epoch=epoch, loss=mean_loss, optimizer=optimizer, patience_counter=epochs_since_improve)
        if mean_loss + 1e-6 < best_loss:
            best_loss = mean_loss
            best_state = copy.deepcopy(operator.state_dict())
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if _should_stop(patience, epochs_since_improve):
                break
        if scheduler is not None:
            scheduler.step()
    operator.load_state_dict(best_state)
    logger.close()
    checkpoint_dir = ensure_checkpoint_dir(cfg)
    torch.save(operator.state_dict(), checkpoint_dir / "operator.pt")
    print("Saved operator checkpoint.")
    
    # Send W&B alert
    if wandb is not None and wandb.run is not None:
        try:
            wandb.alert(
                title="✅ Operator Training Complete",
                text=f"Final loss: {best_loss:.6f} | Ready for diffusion stage",
                level=wandb.AlertLevel.INFO
            )
        except Exception:
            pass


def train_diffusion(cfg: dict) -> None:
    loader = dataset_loader(cfg)
    checkpoint_dir = ensure_checkpoint_dir(cfg)
    operator = make_operator(cfg)
    op_path = checkpoint_dir / "operator.pt"
    if op_path.exists():
        operator.load_state_dict(torch.load(op_path, map_location="cpu"))
    operator.eval()

    latent_dim = cfg.get("latent", {}).get("dim", 32)
    stage_cfg = cfg.get("stages", {}).get("diff_residual", {})
    diff = DiffusionResidual(DiffusionResidualConfig(latent_dim=latent_dim, hidden_dim=latent_dim * 2))
    optimizer = _create_optimizer(cfg, diff, "diff_residual")
    scheduler = _create_scheduler(optimizer, cfg, "diff_residual")
    patience = _get_patience(cfg, "diff_residual")
    dt = cfg.get("training", {}).get("dt", 0.1)
    epochs = stage_cfg.get("epochs", 1)
    logger = TrainingLogger(cfg, stage="diffusion_residual")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    diff.to(device)
    operator.to(device)
    dt_tensor = torch.tensor(dt, device=device)
    best_loss = float("inf")
    best_state = copy.deepcopy(diff.state_dict())
    epochs_since_improve = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        batches = 0
        for batch in loader:
            z0, z1, cond = unpack_batch(batch)
            cond_device = {k: v.to(device) for k, v in cond.items()}
            state = LatentState(z=z0.to(device), t=torch.tensor(0.0, device=device), cond=cond_device)
            target = z1.to(device)
            with torch.no_grad():
                predicted = operator(state, dt_tensor)
            residual_target = target - predicted.z
            tau_tensor = torch.full((z0.size(0),), 0.5, device=device)
            drift = diff(predicted, tau_tensor)
            loss = F.mse_loss(drift, residual_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1
        mean_loss = epoch_loss / max(batches, 1)
        logger.log(epoch=epoch, loss=mean_loss, optimizer=optimizer, patience_counter=epochs_since_improve)
        if mean_loss + 1e-6 < best_loss:
            best_loss = mean_loss
            best_state = copy.deepcopy(diff.state_dict())
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if _should_stop(patience, epochs_since_improve):
                break
        if scheduler is not None:
            scheduler.step()
    diff.load_state_dict(best_state)
    logger.close()
    torch.save(diff.state_dict(), checkpoint_dir / "diffusion_residual.pt")
    print("Saved diffusion residual checkpoint.")
    
    # Send W&B alert
    if wandb is not None and wandb.run is not None:
        try:
            wandb.alert(
                title="✅ Diffusion Residual Training Complete",
                text=f"Final loss: {best_loss:.6f} | Ready for consistency distillation",
                level=wandb.AlertLevel.INFO
            )
        except Exception:
            pass


def train_consistency(cfg: dict) -> None:
    loader = dataset_loader(cfg)
    checkpoint_dir = ensure_checkpoint_dir(cfg)
    operator = make_operator(cfg)
    op_path = checkpoint_dir / "operator.pt"
    if op_path.exists():
        operator.load_state_dict(torch.load(op_path, map_location="cpu"))
    operator.eval()
    
    latent_dim = cfg.get("latent", {}).get("dim", 32)
    stage_cfg = cfg.get("stages", {}).get("consistency_distill", {})
    diff = DiffusionResidual(DiffusionResidualConfig(latent_dim=latent_dim, hidden_dim=latent_dim * 2))
    diff_path = checkpoint_dir / "diffusion_residual.pt"
    if diff_path.exists():
        diff.load_state_dict(torch.load(diff_path, map_location="cpu"))
    
    epochs = stage_cfg.get("epochs", 1)
    optimizer = _create_optimizer(cfg, diff, "consistency_distill")
    scheduler = _create_scheduler(optimizer, cfg, "consistency_distill")
    patience = _get_patience(cfg, "consistency_distill")
    logger = TrainingLogger(cfg, stage="consistency_distill")
    dt = cfg.get("training", {}).get("dt", 0.1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    diff.to(device)
    operator.to(device)
    dt_tensor = torch.tensor(dt, device=device)

    def teacher_fn(state: LatentState, tau: torch.Tensor) -> LatentState:
        return operator(state, dt_tensor)

    def student_fn(state: LatentState, tau: torch.Tensor) -> LatentState:
        predicted = operator(state, dt_tensor)
        tau_vec = tau.expand(predicted.z.size(0))
        drift = diff(predicted, tau_vec)
        return LatentState(z=predicted.z + drift, t=predicted.t, cond=predicted.cond)

    best_loss = float("inf")
    best_state = copy.deepcopy(diff.state_dict())
    epochs_since_improve = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        batches = 0
        for batch in loader:
            z0, _, cond = unpack_batch(batch)
            cond_device = {k: v.to(device) for k, v in cond.items()}
            state = LatentState(z=z0.to(device), t=torch.tensor(0.0, device=device), cond=cond_device)
            loss = distillation_loss(teacher_fn, student_fn, state, DistillationConfig(taus=[0.25, 0.5, 0.75]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1
        mean_loss = epoch_loss / max(batches, 1)
        logger.log(epoch=epoch, loss=mean_loss, optimizer=optimizer, patience_counter=epochs_since_improve)
        if mean_loss + 1e-6 < best_loss:
            best_loss = mean_loss
            best_state = copy.deepcopy(diff.state_dict())
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if _should_stop(patience, epochs_since_improve):
                break
        if scheduler is not None:
            scheduler.step()
    diff.load_state_dict(best_state)
    logger.close()
    torch.save(diff.state_dict(), checkpoint_dir / "diffusion_residual.pt")
    print("Updated diffusion residual via consistency distillation.")
    
    # Send W&B alert
    if wandb is not None and wandb.run is not None:
        try:
            wandb.alert(
                title="✅ Consistency Distillation Complete",
                text=f"Final loss: {best_loss:.6f} | Ready for steady prior training",
                level=wandb.AlertLevel.INFO
            )
        except Exception:
            pass


def train_steady_prior(cfg: dict) -> None:
    loader = dataset_loader(cfg)
    latent_dim = cfg.get("latent", {}).get("dim", 32)
    stage_cfg = cfg.get("stages", {}).get("steady_prior", {})
    prior = SteadyPrior(SteadyPriorConfig(latent_dim=latent_dim, hidden_dim=latent_dim * 2, num_steps=4))
    optimizer = _create_optimizer(cfg, prior, "steady_prior")
    scheduler = _create_scheduler(optimizer, cfg, "steady_prior")
    patience = _get_patience(cfg, "steady_prior")
    epochs = stage_cfg.get("epochs", 1)
    logger = TrainingLogger(cfg, stage="steady_prior")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    prior.to(device)
    best_loss = float("inf")
    best_state = copy.deepcopy(prior.state_dict())
    epochs_since_improve = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        batches = 0
        for batch in loader:
            z0, z1, cond = unpack_batch(batch)
            cond_device = {k: v.to(device) for k, v in cond.items()}
            state = LatentState(z=z0.to(device), t=torch.tensor(0.0, device=device), cond=cond_device)
            refined = prior(state)
            loss = F.mse_loss(refined.z, z1.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1
        mean_loss = epoch_loss / max(batches, 1)
        logger.log(epoch=epoch, loss=mean_loss, optimizer=optimizer, patience_counter=epochs_since_improve)
        if mean_loss + 1e-6 < best_loss:
            best_loss = mean_loss
            best_state = copy.deepcopy(prior.state_dict())
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if _should_stop(patience, epochs_since_improve):
                break
        if scheduler is not None:
            scheduler.step()
    prior.load_state_dict(best_state)
    logger.close()
    checkpoint_dir = ensure_checkpoint_dir(cfg)
    torch.save(prior.state_dict(), checkpoint_dir / "steady_prior.pt")
    print("Saved steady prior checkpoint.")
    
    # Send W&B alert
    if wandb is not None and wandb.run is not None:
        try:
            wandb.alert(
                title="🎉 All Training Stages Complete!",
                text=f"Steady prior final loss: {best_loss:.6f} | Full pipeline ready for evaluation",
                level=wandb.AlertLevel.SUCCESS
            )
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_multi_pde.yaml")
    parser.add_argument(
        "--stage",
        choices=["operator", "diff_residual", "consistency_distill", "steady_prior"],
        required=True,
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg)
    stage = args.stage
    if stage == "operator":
        train_operator(cfg)
    elif stage == "diff_residual":
        train_diffusion(cfg)
    elif stage == "consistency_distill":
        train_consistency(cfg)
    elif stage == "steady_prior":
        train_steady_prior(cfg)


if __name__ == "__main__":
    main()
