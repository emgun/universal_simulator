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
from torch.cuda.amp import autocast, GradScaler
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
from ups.utils.monitoring import init_monitoring_session, MonitoringSession


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
    def __init__(self, cfg: Dict[str, Dict], stage: str, global_step: int = 0, shared_run=None) -> None:
        training_cfg = cfg.get("training", {})
        log_path = training_cfg.get("log_path", "reports/training_log.jsonl")
        self.stage = stage
        self.global_step = global_step
        
        # Use shared run if provided, otherwise create new one
        if shared_run is not None:
            self.session = MonitoringSession(file_path=Path(log_path) if log_path else None, run=shared_run, component=f"training-{stage}")
            self.owns_run = False
        else:
            self.session = init_monitoring_session(cfg, component=f"training-{stage}", file_path=log_path)
            self.owns_run = True

    def log(
        self,
        *,
        epoch: int,
        loss: float,
        optimizer: torch.optim.Optimizer,
        patience_counter: Optional[int] = None,
        grad_norm: Optional[float] = None,
        epoch_time: Optional[float] = None,
        best_loss: Optional[float] = None,
    ) -> None:
        lr = optimizer.param_groups[0].get("lr") if optimizer.param_groups else None
        self.global_step += 1
        
        # Log with stage-specific prefixes for better W&B charts
        entry = {
            f"{self.stage}/loss": loss,
            f"{self.stage}/epoch": epoch,
            f"{self.stage}/lr": lr,
            "global_step": self.global_step,
        }
        
        # Add optional metrics
        if patience_counter is not None:
            entry[f"{self.stage}/epochs_since_improve"] = patience_counter
        if grad_norm is not None:
            entry[f"{self.stage}/grad_norm"] = grad_norm
        if epoch_time is not None:
            entry[f"{self.stage}/epoch_time_sec"] = epoch_time
        if best_loss is not None:
            entry[f"{self.stage}/best_loss"] = best_loss
        
        self.session.log(entry)

    def close(self) -> None:
        # Only finish the run if we own it
        if self.owns_run:
            self.session.finish()
    
    def get_global_step(self) -> int:
        return self.global_step


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
    if name in {"reducelronplateau", "reducelr", "plateau"}:
        mode = sched_cfg.get("mode", "min")
        factor = sched_cfg.get("factor", 0.5)
        patience = sched_cfg.get("patience", 3)
        threshold = sched_cfg.get("threshold", 1e-3)
        threshold_mode = sched_cfg.get("threshold_mode", "rel")
        cooldown = sched_cfg.get("cooldown", 0)
        min_lr = sched_cfg.get("min_lr", 0.0)
        eps = sched_cfg.get("eps", 1e-8)
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            threshold=threshold,
            threshold_mode=threshold_mode,
            cooldown=cooldown,
            min_lr=min_lr,
            eps=eps,
        )
    raise ValueError(f"Unsupported scheduler '{name}'")


def _amp_enabled(cfg: Dict) -> bool:
    return bool(cfg.get("training", {}).get("amp", False)) and torch.cuda.is_available()


def _grad_clip_value(cfg: Dict, stage: str) -> Optional[float]:
    # Stage-specific override takes precedence; fallback to training.grad_clip
    stage_cfg = cfg.get("stages", {}).get(stage, {}) if isinstance(cfg.get("stages"), dict) else {}
    if "grad_clip" in stage_cfg:
        return stage_cfg.get("grad_clip")
    return cfg.get("training", {}).get("grad_clip")


def _get_ema_decay(cfg: Dict, stage: str) -> Optional[float]:
    stage_cfg = cfg.get("stages", {}).get(stage, {}) if isinstance(cfg.get("stages"), dict) else {}
    if "ema_decay" in stage_cfg:
        return stage_cfg.get("ema_decay")
    return cfg.get("training", {}).get("ema_decay")


def _init_ema(model: nn.Module) -> nn.Module:
    ema = copy.deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)
    ema.eval()
    return ema


@torch.no_grad()
def _update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
        p_ema.mul_(decay).add_(p.data, alpha=1.0 - decay)


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


def _stage_epochs(cfg: dict, stage: str) -> int:
    """Helper to read configured epochs for a stage; defaults to 0 when unset."""
    try:
        value = cfg.get("stages", {}).get(stage, {}).get("epochs", 0)
        return int(value) if value is not None else 0
    except Exception:
        return 0


def train_operator(cfg: dict, shared_run=None, global_step: int = 0) -> None:
    loader = dataset_loader(cfg)
    operator = make_operator(cfg)
    train_cfg = cfg.get("training", {})
    dt = train_cfg.get("dt", 0.1)
    stage_cfg = cfg.get("stages", {}).get("operator", {})
    epochs = stage_cfg.get("epochs", 1)
    optimizer = _create_optimizer(cfg, operator, "operator")
    scheduler = _create_scheduler(optimizer, cfg, "operator")
    patience = _get_patience(cfg, "operator")
    logger = TrainingLogger(cfg, stage="operator", global_step=global_step, shared_run=shared_run)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    operator.to(device)
    dt_tensor = torch.tensor(dt, device=device)
    best_loss = float("inf")
    best_state = copy.deepcopy(operator.state_dict())
    # AMP + EMA setup
    use_amp = _amp_enabled(cfg)
    scaler = GradScaler(enabled=use_amp)
    ema_decay = _get_ema_decay(cfg, "operator")
    ema_model = _init_ema(operator) if ema_decay else None
    clip_val = _grad_clip_value(cfg, "operator")
    epochs_since_improve = 0
    
    import time
    accum_steps = max(1, int(cfg.get("training", {}).get("accum_steps", 1)))
    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        total_grad_norm = 0.0
        batches = 0
        grad_steps = 0
        num_batches = len(loader)
        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(loader):
            z0, z1, cond = unpack_batch(batch)
            cond_device = {k: v.to(device) for k, v in cond.items()}
            state = LatentState(z=z0.to(device), t=torch.tensor(0.0, device=device), cond=cond_device)
            target = z1.to(device)
            try:
                with autocast(enabled=use_amp):
                    next_state = operator(state, dt_tensor)
                    loss = F.mse_loss(next_state.z, target)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print("Warning: OOM encountered in operator step, skipping batch")
                    continue
                raise
            loss_value = loss.detach().item()
            if use_amp:
                scaler.scale(loss / accum_steps).backward()
            else:
                (loss / accum_steps).backward()
            do_step = ((i + 1) % accum_steps == 0) or ((i + 1) == num_batches)
            if do_step:
                if use_amp:
                    if clip_val is not None:
                        scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(operator.parameters(), float('inf') if clip_val is None else clip_val)
                    total_grad_norm += float(grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(operator.parameters(), float('inf') if clip_val is None else clip_val)
                    total_grad_norm += grad_norm.item()
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                grad_steps += 1
                if ema_model is not None and ema_decay is not None:
                    _update_ema(ema_model, operator, ema_decay)
            epoch_loss += loss_value
            batches += 1
        
        epoch_time = time.time() - epoch_start
        mean_loss = epoch_loss / max(batches, 1)
        mean_grad_norm = total_grad_norm / max(grad_steps, 1)
        
        logger.log(
            epoch=epoch,
            loss=mean_loss,
            optimizer=optimizer,
            patience_counter=epochs_since_improve,
            grad_norm=mean_grad_norm,
            epoch_time=epoch_time,
            best_loss=best_loss,
        )
        if mean_loss + 1e-6 < best_loss:
            best_loss = mean_loss
            best_state = copy.deepcopy(operator.state_dict())
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if _should_stop(patience, epochs_since_improve):
                break
        if scheduler is not None:
            if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                scheduler.step(mean_loss)
            else:
                scheduler.step()
    operator.load_state_dict(best_state)
    logger.close()
    checkpoint_dir = ensure_checkpoint_dir(cfg)
    operator_path = checkpoint_dir / "operator.pt"
    torch.save(operator.state_dict(), operator_path)
    print(f"Saved operator checkpoint to {operator_path}")
    if ema_model is not None:
        operator_ema_path = checkpoint_dir / "operator_ema.pt"
        torch.save(ema_model.state_dict(), operator_ema_path)
        print(f"Saved operator EMA checkpoint to {operator_ema_path}")
    
    # Upload checkpoint to W&B
    if wandb is not None and wandb.run is not None:
        wandb.save(str(operator_path), base_path=str(checkpoint_dir.parent))
        print(f"Uploaded operator checkpoint to W&B")
    
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


def train_diffusion(cfg: dict, shared_run=None, global_step: int = 0) -> None:
    loader = dataset_loader(cfg)
    checkpoint_dir = ensure_checkpoint_dir(cfg)
    
    # Determine device FIRST
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create operator and load checkpoint directly to target device
    operator = make_operator(cfg)
    op_path = checkpoint_dir / "operator.pt"
    if op_path.exists():
        operator.load_state_dict(torch.load(op_path, map_location=device))
    _ensure_model_on_device(operator, device)
    operator.eval()

    latent_dim = cfg.get("latent", {}).get("dim", 32)
    stage_cfg = cfg.get("stages", {}).get("diff_residual", {})
    diff = DiffusionResidual(DiffusionResidualConfig(latent_dim=latent_dim, hidden_dim=latent_dim * 2))
    _ensure_model_on_device(diff, device)
    
    optimizer = _create_optimizer(cfg, diff, "diff_residual")
    scheduler = _create_scheduler(optimizer, cfg, "diff_residual")
    patience = _get_patience(cfg, "diff_residual")
    dt = cfg.get("training", {}).get("dt", 0.1)
    epochs = stage_cfg.get("epochs", 1)
    logger = TrainingLogger(cfg, stage="diffusion_residual", global_step=global_step, shared_run=shared_run)
    dt_tensor = torch.tensor(dt, device=device)
    best_loss = float("inf")
    best_state = copy.deepcopy(diff.state_dict())
    # AMP + EMA setup
    use_amp = _amp_enabled(cfg)
    scaler = GradScaler(enabled=use_amp)
    ema_decay = _get_ema_decay(cfg, "diff_residual")
    ema_model = _init_ema(diff) if ema_decay else None
    clip_val = _grad_clip_value(cfg, "diff_residual")
    epochs_since_improve = 0
    
    import time
    accum_steps = max(1, int(cfg.get("training", {}).get("accum_steps", 1)))
    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        total_grad_norm = 0.0
        batches = 0
        grad_steps = 0
        optimizer.zero_grad(set_to_none=True)
        num_batches = len(loader)
        for i, batch in enumerate(loader):
            z0, z1, cond = unpack_batch(batch)
            cond_device = {k: v.to(device) for k, v in cond.items()}
            state = LatentState(z=z0.to(device), t=torch.tensor(0.0, device=device), cond=cond_device)
            target = z1.to(device)
            try:
                with torch.no_grad():
                    predicted = operator(state, dt_tensor)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print("Warning: OOM encountered in operator forward (teacher), skipping batch")
                    continue
                raise
            residual_target = target - predicted.z
            tau_tensor = torch.full((z0.size(0),), 0.5, device=device)
            try:
                with autocast(enabled=use_amp):
                    drift = diff(predicted, tau_tensor)
                    loss = F.mse_loss(drift, residual_target)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print("Warning: OOM encountered in diffusion step, skipping batch")
                    continue
                raise
            loss_value = loss.detach().item()
            if use_amp:
                scaler.scale(loss / accum_steps).backward()
            else:
                (loss / accum_steps).backward()
            do_step = ((i + 1) % accum_steps == 0) or ((i + 1) == num_batches)
            if do_step:
                if use_amp:
                    if clip_val is not None:
                        scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(diff.parameters(), float('inf') if clip_val is None else clip_val)
                    total_grad_norm += float(grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(diff.parameters(), float('inf') if clip_val is None else clip_val)
                    total_grad_norm += grad_norm.item()
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                grad_steps += 1
                if ema_model is not None and ema_decay is not None:
                    _update_ema(ema_model, diff, ema_decay)
            epoch_loss += loss.item()
            batches += 1
        
        epoch_time = time.time() - epoch_start
        mean_loss = epoch_loss / max(batches, 1)
        mean_grad_norm = total_grad_norm / max(grad_steps, 1)
        
        logger.log(
            epoch=epoch,
            loss=mean_loss,
            optimizer=optimizer,
            patience_counter=epochs_since_improve,
            grad_norm=mean_grad_norm,
            epoch_time=epoch_time,
            best_loss=best_loss,
        )
        if mean_loss + 1e-6 < best_loss:
            best_loss = mean_loss
            best_state = copy.deepcopy(diff.state_dict())
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if _should_stop(patience, epochs_since_improve):
                break
        if scheduler is not None:
            if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                scheduler.step(mean_loss)
            else:
                scheduler.step()
    diff.load_state_dict(best_state)
    logger.close()
    diffusion_path = checkpoint_dir / "diffusion_residual.pt"
    torch.save(diff.state_dict(), diffusion_path)
    print(f"Saved diffusion residual checkpoint to {diffusion_path}")
    if ema_model is not None:
        diffusion_ema_path = checkpoint_dir / "diffusion_residual_ema.pt"
        torch.save(ema_model.state_dict(), diffusion_ema_path)
        print(f"Saved diffusion EMA checkpoint to {diffusion_ema_path}")
    
    # Upload checkpoint to W&B
    if wandb is not None and wandb.run is not None:
        wandb.save(str(diffusion_path), base_path=str(checkpoint_dir.parent))
        print(f"Uploaded diffusion checkpoint to W&B")
    
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


def _ensure_model_on_device(model: nn.Module, device: torch.device) -> None:
    """Aggressively ensure all model parameters and buffers are on the correct device."""
    model.to(device)
    # Force all parameters to device
    for param in model.parameters():
        param.data = param.data.to(device)
        if param.grad is not None:
            param.grad.data = param.grad.data.to(device)
    # Force all buffers to device
    for buffer in model.buffers():
        buffer.data = buffer.data.to(device)


def train_consistency(cfg: dict, shared_run=None, global_step: int = 0) -> None:
    # Use smaller batch size for consistency stage to avoid OOM
    # This stage needs both operator and diffusion models loaded
    cfg_copy = copy.deepcopy(cfg)
    original_batch_size = cfg_copy.get("training", {}).get("batch_size", 32)
    consistency_batch_size = cfg_copy.get("stages", {}).get("consistency_distill", {}).get("batch_size", 8)
    cfg_copy.setdefault("training", {})["batch_size"] = consistency_batch_size
    
    loader = dataset_loader(cfg_copy)
    checkpoint_dir = ensure_checkpoint_dir(cfg)
    
    # Determine device FIRST
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create operator and load checkpoint directly to target device
    operator = make_operator(cfg)
    op_path = checkpoint_dir / "operator.pt"
    if op_path.exists():
        operator.load_state_dict(torch.load(op_path, map_location=device))
    _ensure_model_on_device(operator, device)
    operator.eval()
    
    # Create diffusion model and load checkpoint directly to target device
    latent_dim = cfg.get("latent", {}).get("dim", 32)
    stage_cfg = cfg.get("stages", {}).get("consistency_distill", {})
    diff = DiffusionResidual(DiffusionResidualConfig(latent_dim=latent_dim, hidden_dim=latent_dim * 2))
    diff_path = checkpoint_dir / "diffusion_residual.pt"
    if diff_path.exists():
        diff.load_state_dict(torch.load(diff_path, map_location=device))
    _ensure_model_on_device(diff, device)
    
    epochs = stage_cfg.get("epochs", 1)
    optimizer = _create_optimizer(cfg, diff, "consistency_distill")
    scheduler = _create_scheduler(optimizer, cfg, "consistency_distill")
    patience = _get_patience(cfg, "consistency_distill")
    logger = TrainingLogger(cfg, stage="consistency_distill", global_step=global_step, shared_run=shared_run)
    dt = cfg.get("training", {}).get("dt", 0.1)
    
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
    use_amp = _amp_enabled(cfg)
    scaler = GradScaler(enabled=use_amp)
    ema_decay = _get_ema_decay(cfg, "consistency_distill")
    ema_model = _init_ema(diff) if ema_decay else None
    clip_val = _grad_clip_value(cfg, "consistency_distill")
    epochs_since_improve = 0
    
    # Get micro-batch size for gradient accumulation
    distill_micro = cfg.get("training", {}).get("distill_micro_batch")
    
    import time
    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        total_grad_norm = 0.0
        batches = 0
        
        for batch in loader:
            z0, _, cond = unpack_batch(batch)
            batch_size = z0.shape[0]
            micro = distill_micro or batch_size
            num_chunks = max(1, (batch_size + micro - 1) // micro)
            optimizer.zero_grad(set_to_none=True)
            batch_loss_value = 0.0
            for start in range(0, batch_size, micro):
                end = min(start + micro, batch_size)
                chunk_weight = (end - start) / batch_size
                z_chunk = z0[start:end].to(device)
                chunk_cond = {k: v[start:end].to(device) for k, v in cond.items()}
                state = LatentState(z=z_chunk, t=torch.tensor(0.0, device=device), cond=chunk_cond)
                try:
                    with autocast(enabled=use_amp):
                        loss_chunk = distillation_loss(
                            teacher_fn,
                            student_fn,
                            state,
                            DistillationConfig(taus=[0.25, 0.5, 0.75]),
                            device=device,
                        )
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        print("Warning: OOM in consistency distill chunk, skipping chunk")
                        continue
                    raise
                if use_amp:
                    scaler.scale(loss_chunk * chunk_weight).backward()
                else:
                    (loss_chunk * chunk_weight).backward()
                batch_loss_value += loss_chunk.item() * chunk_weight
            if use_amp:
                if clip_val is not None:
                    scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(diff.parameters(), float('inf') if clip_val is None else clip_val)
                total_grad_norm += float(grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(diff.parameters(), float('inf') if clip_val is None else clip_val)
                total_grad_norm += grad_norm.item()
                optimizer.step()
            if ema_model is not None and ema_decay is not None:
                _update_ema(ema_model, diff, ema_decay)
            epoch_loss += batch_loss_value
            batches += 1
        
        epoch_time = time.time() - epoch_start
        mean_loss = epoch_loss / max(batches, 1)
        mean_grad_norm = total_grad_norm / max(batches, 1)
        
        logger.log(
            epoch=epoch,
            loss=mean_loss,
            optimizer=optimizer,
            patience_counter=epochs_since_improve,
            grad_norm=mean_grad_norm,
            epoch_time=epoch_time,
            best_loss=best_loss,
        )
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
    diffusion_path = checkpoint_dir / "diffusion_residual.pt"
    torch.save(diff.state_dict(), diffusion_path)
    print(f"Updated diffusion residual via consistency distillation to {diffusion_path}")
    if ema_model is not None:
        diffusion_ema_path = checkpoint_dir / "diffusion_residual_ema.pt"
        torch.save(ema_model.state_dict(), diffusion_ema_path)
        print(f"Saved diffusion EMA checkpoint to {diffusion_ema_path}")
    
    # Upload updated checkpoint to W&B
    if wandb is not None and wandb.run is not None:
        wandb.save(str(diffusion_path), base_path=str(checkpoint_dir.parent))
        print(f"Uploaded updated diffusion checkpoint to W&B")
    
    # Clean up operator from memory
    del operator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
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


def train_steady_prior(cfg: dict, shared_run=None, global_step: int = 0) -> None:
    latent_dim = cfg.get("latent", {}).get("dim", 32)
    stage_cfg = cfg.get("stages", {}).get("steady_prior", {})
    epochs = stage_cfg.get("epochs", 0)

    # Early exit when disabled
    if epochs <= 0:
        print("Skipping steady_prior stage (epochs<=0)")
        return

    loader = dataset_loader(cfg)
    prior = SteadyPrior(SteadyPriorConfig(latent_dim=latent_dim, hidden_dim=latent_dim * 2, num_steps=4))
    optimizer = _create_optimizer(cfg, prior, "steady_prior")
    scheduler = _create_scheduler(optimizer, cfg, "steady_prior")
    patience = _get_patience(cfg, "steady_prior")
    logger = TrainingLogger(cfg, stage="steady_prior", global_step=global_step, shared_run=shared_run)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    prior.to(device)
    best_loss = float("inf")
    best_state = copy.deepcopy(prior.state_dict())
    epochs_since_improve = 0
    
    import time
    accum_steps = max(1, int(cfg.get("training", {}).get("accum_steps", 1)))
    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        total_grad_norm = 0.0
        batches = 0
        grad_steps = 0
        optimizer.zero_grad(set_to_none=True)
        num_batches = len(loader)
        for i, batch in enumerate(loader):
            z0, z1, cond = unpack_batch(batch)
            cond_device = {k: v.to(device) for k, v in cond.items()}
            state = LatentState(z=z0.to(device), t=torch.tensor(0.0, device=device), cond=cond_device)
            refined = prior(state)
            loss = F.mse_loss(refined.z, z1.to(device))
            (loss / accum_steps).backward()
            do_step = ((i + 1) % accum_steps == 0) or ((i + 1) == num_batches)
            if do_step:
                grad_norm = torch.nn.utils.clip_grad_norm_(prior.parameters(), float('inf'))
                total_grad_norm += grad_norm.item()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                grad_steps += 1
            epoch_loss += loss.item()
            batches += 1
        
        epoch_time = time.time() - epoch_start
        mean_loss = epoch_loss / max(batches, 1)
        mean_grad_norm = total_grad_norm / max(grad_steps, 1)
        
        logger.log(
            epoch=epoch,
            loss=mean_loss,
            optimizer=optimizer,
            patience_counter=epochs_since_improve,
            grad_norm=mean_grad_norm,
            epoch_time=epoch_time,
            best_loss=best_loss,
        )
        if mean_loss + 1e-6 < best_loss:
            best_loss = mean_loss
            best_state = copy.deepcopy(prior.state_dict())
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if _should_stop(patience, epochs_since_improve):
                break
        if scheduler is not None:
            if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                scheduler.step(mean_loss)
            else:
                scheduler.step()
    prior.load_state_dict(best_state)
    logger.close()
    checkpoint_dir = ensure_checkpoint_dir(cfg)
    prior_path = checkpoint_dir / "steady_prior.pt"
    torch.save(prior.state_dict(), prior_path)
    print(f"Saved steady prior checkpoint to {prior_path}")
    
    # Upload checkpoint to W&B
    if wandb is not None and wandb.run is not None:
        wandb.save(str(prior_path), base_path=str(checkpoint_dir.parent))
        print(f"Uploaded steady prior checkpoint to W&B")
    
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


def train_all_stages(cfg: dict) -> None:
    """Run all training stages in sequence with shared W&B run for better charts."""
    # Initialize W&B once for all stages
    logging_cfg = cfg.get("logging", {})
    wandb_cfg = logging_cfg.get("wandb", {})
    shared_run = None
    
    if wandb_cfg.get("enabled") and wandb is not None:
        shared_run = wandb.init(
            project=wandb_cfg.get("project", "universal-simulator"),
            name=wandb_cfg.get("run_name", "full-pipeline"),
            config=cfg,
            tags=wandb_cfg.get("tags", []) + ["full-pipeline"],
            group=wandb_cfg.get("group"),
            job_type="multi-stage-training",
        )
        # Define metric relationships for better charting
        if shared_run:
            wandb.define_metric("global_step")
            wandb.define_metric("operator/*", step_metric="global_step")
            wandb.define_metric("diffusion_residual/*", step_metric="global_step")
            wandb.define_metric("consistency_distill/*", step_metric="global_step")
            wandb.define_metric("steady_prior/*", step_metric="global_step")
            
            # Log system info
            import torch
            if torch.cuda.is_available():
                gpu_info = {
                    "gpu_name": torch.cuda.get_device_name(0),
                    "gpu_count": torch.cuda.device_count(),
                    "cuda_version": torch.version.cuda,
                }
                wandb.config.update(gpu_info)
            
            # Watch gradients and model parameters (optional, can be heavy)
            # wandb.watch(models, log="all", log_freq=100)
    
    global_step = 0
    
    # Stage 1: Operator
    op_epochs = _stage_epochs(cfg, "operator")
    if op_epochs > 0:
        print("\n" + "="*50)
        print("STAGE 1/4: Training Operator")
        print("="*50)
        train_operator(cfg, shared_run=shared_run, global_step=global_step)
        global_step += op_epochs
    else:
        print("\n" + "="*50)
        print("STAGE 1/4: Skipping Operator (epochs<=0)")
        print("="*50)
    
    # Clear GPU cache between stages
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✓ Cleared GPU cache")
    
    # Stage 2: Diffusion Residual
    diff_epochs = _stage_epochs(cfg, "diff_residual")
    if diff_epochs > 0:
        print("\n" + "="*50)
        print("STAGE 2/4: Training Diffusion Residual")
        print("="*50)
        train_diffusion(cfg, shared_run=shared_run, global_step=global_step)
        global_step += diff_epochs
    else:
        print("\n" + "="*50)
        print("STAGE 2/4: Skipping Diffusion Residual (epochs<=0)")
        print("="*50)
    
    # Clear GPU cache between stages
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✓ Cleared GPU cache")
    
    # Stage 3: Consistency Distillation
    distill_epochs = _stage_epochs(cfg, "consistency_distill")
    if distill_epochs > 0:
        print("\n" + "="*50)
        print("STAGE 3/4: Consistency Distillation")
        print("="*50)
        train_consistency(cfg, shared_run=shared_run, global_step=global_step)
        global_step += distill_epochs
    else:
        print("\n" + "="*50)
        print("STAGE 3/4: Skipping Consistency Distillation (epochs<=0)")
        print("="*50)
    
    # Clear GPU cache between stages
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✓ Cleared GPU cache")
    
    # Stage 4: Steady Prior
    steady_epochs = _stage_epochs(cfg, "steady_prior")
    if steady_epochs > 0:
        print("\n" + "="*50)
        print("STAGE 4/4: Training Steady Prior")
        print("="*50)
        train_steady_prior(cfg, shared_run=shared_run, global_step=global_step)
    else:
        print("\n" + "="*50)
        print("STAGE 4/4: Skipping Steady Prior (epochs<=0)")
        print("="*50)
    
    # Log final summary
    if shared_run:
        # Load final checkpoints to get model sizes
        checkpoint_dir = ensure_checkpoint_dir(cfg)
        import os
        summary = {
            "summary/total_training_complete": 1,
            "summary/operator_checkpoint_size_mb": os.path.getsize(checkpoint_dir / "operator.pt") / 1e6 if (checkpoint_dir / "operator.pt").exists() else 0,
            "summary/diffusion_checkpoint_size_mb": os.path.getsize(checkpoint_dir / "diffusion_residual.pt") / 1e6 if (checkpoint_dir / "diffusion_residual.pt").exists() else 0,
            "summary/steady_prior_checkpoint_size_mb": os.path.getsize(checkpoint_dir / "steady_prior.pt") / 1e6 if (checkpoint_dir / "steady_prior.pt").exists() else 0,
        }
        shared_run.log(summary)
        shared_run.finish()
    
    print("\n" + "="*50)
    print("✅ All training stages complete!")
    print("="*50)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_multi_pde.yaml")
    parser.add_argument(
        "--stage",
        choices=["operator", "diff_residual", "consistency_distill", "steady_prior", "all"],
        required=True,
        help="Training stage to run, or 'all' to run full pipeline"
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg)
    stage = args.stage
    
    if stage == "all":
        train_all_stages(cfg)
    elif stage == "operator":
        train_operator(cfg)
    elif stage == "diff_residual":
        train_diffusion(cfg)
    elif stage == "consistency_distill":
        train_consistency(cfg)
    elif stage == "steady_prior":
        train_steady_prior(cfg)


if __name__ == "__main__":
    main()
