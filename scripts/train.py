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
import torch.multiprocessing as mp

try:
    import wandb
except ImportError:
    wandb = None

# Ensure CUDA + DataLoader workers use a safe start method
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from ups.core.latent_state import LatentState
from ups.models.latent_operator import LatentOperator, LatentOperatorConfig
from ups.core.blocks_pdet import PDETransformerConfig
from ups.models.diffusion_residual import DiffusionResidual, DiffusionResidualConfig
from ups.training.consistency_distill import DistillationConfig, distillation_loss
from ups.models.steady_prior import SteadyPrior, SteadyPriorConfig
from ups.data.latent_pairs import build_latent_pair_loader, unpack_batch
from ups.utils.monitoring import init_monitoring_session, MonitoringSession

# ---- Auxiliary training losses ----
def _nrmse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mse = torch.mean((pred - target) ** 2)
    denom = torch.mean(target ** 2) + eps
    return torch.sqrt(mse / denom)

def _spectral_energy_loss(pred: torch.Tensor, target: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    """Relative spectral energy difference along the given axis (default: token axis).

    cuFFT requires power-of-two signal sizes when using half precision. Temporarily
    disable autocast and promote to float32 before taking the FFT so non-power-of-two
    token counts (e.g., 48) do not trigger runtime errors. Cast the result back to the
    original dtype for downstream losses.
    """
    with torch.cuda.amp.autocast(enabled=False):
        pred_fft = torch.fft.rfft(pred.float(), dim=dim)
        tgt_fft = torch.fft.rfft(target.float(), dim=dim)
        pred_energy = torch.mean(pred_fft.abs() ** 2)
        tgt_energy = torch.mean(tgt_fft.abs() ** 2)
        loss = torch.abs(pred_energy - tgt_energy) / (tgt_energy + eps)
    return loss.to(pred.dtype)


def _strip_compiled_prefix(state_dict: dict) -> dict:
    """Strip _orig_mod. prefix from state dict keys (from torch.compile)."""
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("_orig_mod.", "") if key.startswith("_orig_mod.") else key
        new_state_dict[new_key] = value
    return new_state_dict


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def set_seed(cfg: Dict) -> None:
    """Set random seed and configure determinism settings.

    Args:
        cfg: Config dict with optional 'seed', 'deterministic', and 'benchmark' keys
    """
    seed = cfg.get("seed", 17)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Configure PyTorch determinism
    deterministic = cfg.get("deterministic", False)
    benchmark = cfg.get("benchmark", True)

    if deterministic:
        # Set CUBLAS workspace config for deterministic CuBLAS operations
        import os
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"✓ Deterministic mode enabled (seed={seed}, CUBLAS workspace configured)")
    else:
        torch.backends.cudnn.benchmark = benchmark
        print(f"✓ Seed set to {seed} (deterministic={deterministic}, benchmark={benchmark})")


def ensure_checkpoint_dir(cfg: dict) -> Path:
    ckpt_cfg = cfg.get("checkpoint", {})
    directory = Path(ckpt_cfg.get("dir", "checkpoints"))
    directory.mkdir(parents=True, exist_ok=True)
    return directory


class TrainingLogger:
    def __init__(self, cfg: Dict[str, Dict], stage: str, global_step: int = 0, wandb_ctx=None) -> None:
        """Training logger that writes to file and optionally to WandB.

        Args:
            cfg: Training configuration
            stage: Training stage name (operator, diffusion_residual, etc.)
            global_step: Initial global step counter
            wandb_ctx: Optional WandBContext for logging (recommended)
        """
        training_cfg = cfg.get("training", {})
        log_path = training_cfg.get("log_path", "reports/training_log.jsonl")
        self.stage = stage
        self.global_step = global_step
        self.wandb_ctx = wandb_ctx
        self.log_path = Path(log_path) if log_path else None

        # Create log file if specified
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

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

        # Log to file (JSONL format)
        if self.log_path:
            entry = {
                "stage": self.stage,
                "loss": loss,
                "epoch": epoch,
                "lr": lr,
                "global_step": self.global_step,
            }
            if patience_counter is not None:
                entry["epochs_since_improve"] = patience_counter
            if grad_norm is not None:
                entry["grad_norm"] = grad_norm
            if epoch_time is not None:
                entry["epoch_time_sec"] = epoch_time
            if best_loss is not None:
                entry["best_loss"] = best_loss

            try:
                with self.log_path.open("a", encoding="utf-8") as fh:
                    import json
                    fh.write(json.dumps(entry) + "\n")
            except Exception:
                pass

        # Log to WandB using clean context (proper time series!)
        if self.wandb_ctx:
            self.wandb_ctx.log_training_metric(self.stage, "loss", loss, step=self.global_step)
            if lr is not None:
                self.wandb_ctx.log_training_metric(self.stage, "lr", lr, step=self.global_step)
            if patience_counter is not None:
                self.wandb_ctx.log_training_metric(self.stage, "epochs_since_improve", patience_counter, step=self.global_step)
            if grad_norm is not None:
                self.wandb_ctx.log_training_metric(self.stage, "grad_norm", grad_norm, step=self.global_step)
            if epoch_time is not None:
                self.wandb_ctx.log_training_metric(self.stage, "epoch_time_sec", epoch_time, step=self.global_step)
            if best_loss is not None:
                self.wandb_ctx.log_training_metric(self.stage, "best_loss", best_loss, step=self.global_step)

    def close(self) -> None:
        # No longer owns wandb run - orchestrator manages it
        pass

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


def _maybe_compile(model: nn.Module, cfg: Dict, name: str) -> nn.Module:
    """Optionally compile a model with torch.compile when enabled and available.

    Controlled by training.compile bool. Falls back silently if unavailable.
    """
    try:
        compile_enabled = bool(cfg.get("training", {}).get("compile", False))
    except Exception:
        compile_enabled = False
    if not compile_enabled:
        return model
    
    # Skip compilation for teacher models (eval-only) to avoid CUDA graph issues
    if "teacher" in name:
        return model
        
    try:
        import torch

        # Use different compile modes based on model type and training config:
        # - "default" for diffusion models (avoids CUDA graph issues with complex control flow)
        # - "default" for operators with rollout loss (multi-step prediction breaks CUDA graphs)
        # - "reduce-overhead" for operators without rollout (aggressive CUDA graph optimization)
        training_cfg = cfg.get("training", {})
        has_rollout = training_cfg.get("rollout_horizon", 0) > 0 and training_cfg.get("lambda_rollout", 0) > 0
        
        if "diffusion" in name.lower():
            compile_mode = "default"
        elif "operator" in name.lower() and has_rollout:
            compile_mode = "default"  # Rollout breaks CUDA graphs
        else:
            compile_mode = "reduce-overhead"
        
        compiled = torch.compile(model, mode=compile_mode, fullgraph=False)
        return compiled
    except Exception:
        # If torch.compile is unavailable or fails, just return the original model
        return model

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


def _sample_tau(batch_size: int, device: torch.device, cfg: Dict) -> torch.Tensor:
    dist_cfg = cfg.get("training", {}).get("tau_distribution")
    if dist_cfg:
        dist_type = str(dist_cfg.get("type", "")).lower()
        if dist_type == "beta":
            alpha = float(dist_cfg.get("alpha", 1.0))
            beta = float(dist_cfg.get("beta", 1.0))
            beta_dist = torch.distributions.Beta(alpha, beta)
            samples = beta_dist.sample((batch_size,))
            return samples.to(device=device)
    return torch.rand(batch_size, device=device)


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


def train_operator(cfg: dict, wandb_ctx=None, global_step: int = 0) -> None:
    loader = dataset_loader(cfg)
    operator = make_operator(cfg)
    train_cfg = cfg.get("training", {})
    dt = train_cfg.get("dt", 0.1)
    stage_cfg = cfg.get("stages", {}).get("operator", {})
    epochs = stage_cfg.get("epochs", 1)
    optimizer = _create_optimizer(cfg, operator, "operator")
    scheduler = _create_scheduler(optimizer, cfg, "operator")
    patience = _get_patience(cfg, "operator")
    logger = TrainingLogger(cfg, stage="operator", global_step=global_step, wandb_ctx=wandb_ctx)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    operator.to(device)
    operator = _maybe_compile(operator, cfg, "operator")
    dt_tensor = torch.tensor(dt, device=device)
    best_loss = float("inf")
    best_state = copy.deepcopy(operator.state_dict())
    # AMP + EMA setup
    use_amp = _amp_enabled(cfg)
    scaler = GradScaler(enabled=use_amp)
    ema_decay = _get_ema_decay(cfg, "operator")
    ema_model = _init_ema(operator) if ema_decay else None
    best_ema_state = copy.deepcopy(ema_model.state_dict()) if ema_model is not None else None
    clip_val = _grad_clip_value(cfg, "operator")
    epochs_since_improve = 0

    import time
    accum_steps = max(1, int(cfg.get("training", {}).get("accum_steps", 1)))
    lam_spec = float(cfg.get("training", {}).get("lambda_spectral", 0.0) or 0.0)
    lam_rel = float(cfg.get("training", {}).get("lambda_relative", 0.0) or 0.0)
    lam_rollout = float(cfg.get("training", {}).get("lambda_rollout", 0.0) or 0.0)
    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        total_grad_norm = 0.0
        batches = 0
        grad_steps = 0
        num_batches = len(loader)
        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(loader):
            unpacked = unpack_batch(batch)
            if len(unpacked) == 4:
                z0, z1, cond, future = unpacked
            else:
                z0, z1, cond = unpacked
                future = None
            cond_device = {k: v.to(device) for k, v in cond.items()}
            state = LatentState(z=z0.to(device), t=torch.tensor(0.0, device=device), cond=cond_device)
            target = z1.to(device)
            try:
                with autocast(enabled=use_amp):
                    next_state = operator(state, dt_tensor)
                    base = F.mse_loss(next_state.z, target)
                    extra = 0.0
                    if lam_spec > 0.0:
                        extra = extra + lam_spec * _spectral_energy_loss(next_state.z, target, dim=1)
                    if lam_rel > 0.0:
                        extra = extra + lam_rel * _nrmse(next_state.z, target)
                    loss = base + extra
                    if lam_rollout > 0.0 and future is not None and future.numel() > 0:
                        rollout_targets = future.to(device)
                        rollout_state = next_state
                        rollout_loss = 0.0
                        steps = rollout_targets.shape[1]
                        for step in range(steps):
                            rollout_state = operator(rollout_state, dt_tensor)
                            target_step = rollout_targets[:, step]
                            rollout_loss = rollout_loss + F.mse_loss(rollout_state.z, target_step)
                        rollout_loss = rollout_loss / max(steps, 1)
                        loss = loss + lam_rollout * rollout_loss
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
            if ema_model is not None:
                best_ema_state = copy.deepcopy(ema_model.state_dict())
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
        to_save = best_ema_state if best_ema_state is not None else ema_model.state_dict()
        torch.save(to_save, operator_ema_path)
        print(f"Saved operator EMA checkpoint to {operator_ema_path}")
    
    # Upload checkpoint to W&B (clean way!)
    if wandb_ctx:
        wandb_ctx.save_file(operator_path)
        print(f"Uploaded operator checkpoint to W&B")
        if ema_model is not None:
            wandb_ctx.save_file(operator_ema_path)

    # Send W&B alert
    if wandb_ctx:
        wandb_ctx.alert(
            title="✅ Operator Training Complete",
            text=f"Final loss: {best_loss:.6f} | Ready for diffusion stage",
            level="INFO"
        )


def train_diffusion(cfg: dict, wandb_ctx=None, global_step: int = 0) -> None:
    loader = dataset_loader(cfg)
    checkpoint_dir = ensure_checkpoint_dir(cfg)
    
    # Determine device FIRST
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create operator and load checkpoint directly to target device
    operator = make_operator(cfg)
    op_path = checkpoint_dir / "operator.pt"
    if op_path.exists():
        operator_state = torch.load(op_path, map_location="cpu")
        operator_state = _strip_compiled_prefix(operator_state)
        operator.load_state_dict(operator_state)
    _ensure_model_on_device(operator, device)
    operator = _maybe_compile(operator, cfg, "operator_teacher")
    operator.eval()

    latent_dim = cfg.get("latent", {}).get("dim", 32)
    stage_cfg = cfg.get("stages", {}).get("diff_residual", {})
    # Read hidden_dim from config, fallback to latent_dim * 2 for backward compatibility
    hidden_dim = cfg.get("diffusion", {}).get("hidden_dim", latent_dim * 2)
    diff = DiffusionResidual(DiffusionResidualConfig(latent_dim=latent_dim, hidden_dim=hidden_dim))
    _ensure_model_on_device(diff, device)
    diff = _maybe_compile(diff, cfg, "diffusion_residual")
    
    optimizer = _create_optimizer(cfg, diff, "diff_residual")
    scheduler = _create_scheduler(optimizer, cfg, "diff_residual")
    patience = _get_patience(cfg, "diff_residual")
    dt = cfg.get("training", {}).get("dt", 0.1)
    epochs = stage_cfg.get("epochs", 1)
    checkpoint_interval = int(cfg.get("training", {}).get("checkpoint_interval", 0) or 0)
    logger = TrainingLogger(cfg, stage="diffusion_residual", global_step=global_step, wandb_ctx=wandb_ctx)
    dt_tensor = torch.tensor(dt, device=device)
    best_loss = float("inf")
    best_state = copy.deepcopy(diff.state_dict())
    # AMP + EMA setup
    use_amp = _amp_enabled(cfg)
    scaler = GradScaler(enabled=use_amp)
    ema_decay = _get_ema_decay(cfg, "diff_residual")
    ema_model = _init_ema(diff) if ema_decay else None
    best_ema_state = copy.deepcopy(ema_model.state_dict()) if ema_model is not None else None
    clip_val = _grad_clip_value(cfg, "diff_residual")
    epochs_since_improve = 0
    
    import time
    accum_steps = max(1, int(cfg.get("training", {}).get("accum_steps", 1)))
    lam_spec = float(cfg.get("training", {}).get("lambda_spectral", 0.0) or 0.0)
    lam_rel = float(cfg.get("training", {}).get("lambda_relative", 0.0) or 0.0)
    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        total_grad_norm = 0.0
        batches = 0
        grad_steps = 0
        optimizer.zero_grad(set_to_none=True)
        num_batches = len(loader)
        for i, batch in enumerate(loader):
            unpacked = unpack_batch(batch)
            if len(unpacked) == 4:
                z0, z1, cond, _ = unpacked
            else:
                z0, z1, cond = unpacked
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
            # Sample per-sample tau in (0,1) to broaden supervision
            tau_tensor = _sample_tau(z0.size(0), device, cfg)
            try:
                with autocast(enabled=use_amp):
                    drift = diff(predicted, tau_tensor)
                    base = F.mse_loss(drift, residual_target)
                    extra = 0.0
                    if lam_spec > 0.0:
                        extra = extra + lam_spec * _spectral_energy_loss(drift, residual_target, dim=1)
                    if lam_rel > 0.0:
                        extra = extra + lam_rel * _nrmse(drift, residual_target)
                    loss = base + extra
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
            if ema_model is not None:
                best_ema_state = copy.deepcopy(ema_model.state_dict())
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

        if checkpoint_interval and (epoch + 1) % checkpoint_interval == 0:
            epoch_ckpt = checkpoint_dir / f"diffusion_residual_epoch_{epoch + 1}.pt"
            torch.save(diff.state_dict(), epoch_ckpt)
            if ema_model is not None:
                ema_epoch_ckpt = checkpoint_dir / f"diffusion_residual_ema_epoch_{epoch + 1}.pt"
                torch.save(ema_model.state_dict(), ema_epoch_ckpt)
    diff.load_state_dict(best_state)
    logger.close()
    diffusion_path = checkpoint_dir / "diffusion_residual.pt"
    torch.save(diff.state_dict(), diffusion_path)
    print(f"Saved diffusion residual checkpoint to {diffusion_path}")
    if ema_model is not None:
        diffusion_ema_path = checkpoint_dir / "diffusion_residual_ema.pt"
        torch.save(best_ema_state if best_ema_state is not None else ema_model.state_dict(), diffusion_ema_path)
        print(f"Saved diffusion EMA checkpoint to {diffusion_ema_path}")
    
    # Upload checkpoint to W&B (clean way!)
    if wandb_ctx:
        wandb_ctx.save_file(diffusion_path)
        print(f"Uploaded diffusion checkpoint to W&B")
        if ema_model is not None:
            wandb_ctx.save_file(diffusion_ema_path)

    # Send W&B alert
    if wandb_ctx:
        wandb_ctx.alert(
            title="✅ Diffusion Residual Training Complete",
            text=f"Final loss: {best_loss:.6f} | Ready for consistency distillation",
            level="INFO"
        )


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


def _distill_forward_and_loss_compiled(
    teacher_z_chunk: torch.Tensor,
    teacher_cond_chunk: dict,
    num_taus: int,
    diff_model: nn.Module,
    t_value: torch.Tensor,
    tau_seed: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """OPTIMIZATION #1: Compiled distillation forward pass.

    Fuses tau expansion + diffusion forward + loss into single compiled graph.
    Expected speedup: ~1.3-1.5x via kernel fusion and reduced Python overhead.
    """
    Bc, T, D = teacher_z_chunk.shape

    # Tau expansion
    z_tiled = (
        teacher_z_chunk.unsqueeze(1)
        .expand(Bc, num_taus, T, D)
        .reshape(Bc * num_taus, T, D)
        .contiguous()
    )

    cond_tiled = {
        k: v.repeat_interleave(num_taus, dim=0)
        for k, v in teacher_cond_chunk.items()
    }

    tau_flat = tau_seed.repeat(Bc).to(z_tiled.dtype)

    # Diffusion forward
    tiled_state = LatentState(z=z_tiled, t=t_value, cond=cond_tiled)
    drift = diff_model(tiled_state, tau_flat)

    # Loss computation
    z_tiled_cast = z_tiled.to(drift.dtype)
    student_z = z_tiled_cast + drift
    loss = torch.nn.functional.mse_loss(student_z, z_tiled_cast)

    return loss


# Try to compile the distillation function if torch.compile is available
try:
    _distill_forward_and_loss = torch.compile(
        _distill_forward_and_loss_compiled,
        mode="reduce-overhead",
        fullgraph=False,  # Allow graph breaks for LatentState
    )
    _COMPILE_AVAILABLE = True
except Exception:
    _distill_forward_and_loss = _distill_forward_and_loss_compiled
    _COMPILE_AVAILABLE = False


def train_consistency(cfg: dict, wandb_ctx=None, global_step: int = 0) -> None:
    # Use smaller batch size for consistency stage to avoid OOM
    # This stage needs both operator and diffusion models loaded
    cfg_copy = copy.deepcopy(cfg)
    original_batch_size = cfg_copy.get("training", {}).get("batch_size", 32)
    consistency_batch_size = cfg_copy.get("stages", {}).get("consistency_distill", {}).get("batch_size", 8)
    training_cfg_copy = cfg_copy.setdefault("training", {})
    training_cfg_copy["batch_size"] = consistency_batch_size
    # Enable pin_memory for faster transfers
    training_cfg_copy["pin_memory"] = True
    # OPTIMIZATION #3: Persistent workers to avoid respawning overhead
    training_cfg_copy["persistent_workers"] = True if training_cfg_copy.get("num_workers", 0) > 0 else False
    training_cfg_copy["prefetch_factor"] = 4  # Increase prefetch
    
    loader = dataset_loader(cfg_copy)
    checkpoint_dir = ensure_checkpoint_dir(cfg)

    # Determine device FIRST
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create operator and load checkpoint directly to target device
    operator = make_operator(cfg)
    op_path = checkpoint_dir / "operator.pt"
    if op_path.exists():
        operator_state = torch.load(op_path, map_location="cpu")
        operator_state = _strip_compiled_prefix(operator_state)
        operator.load_state_dict(operator_state)
    _ensure_model_on_device(operator, device)
    operator = _maybe_compile(operator, cfg, "operator_teacher")
    operator.eval()

    # Create diffusion model and load checkpoint directly to target device
    latent_dim = cfg.get("latent", {}).get("dim", 32)
    stage_cfg = cfg.get("stages", {}).get("consistency_distill", {})
    tau_schedule = stage_cfg.get("tau_schedule")
    target_loss = float(stage_cfg.get("target_loss") or cfg.get("training", {}).get("distill_target_loss", 0.0) or 0.0)
    # Read hidden_dim from config, fallback to latent_dim * 2 for backward compatibility
    hidden_dim = cfg.get("diffusion", {}).get("hidden_dim", latent_dim * 2)
    diff = DiffusionResidual(DiffusionResidualConfig(latent_dim=latent_dim, hidden_dim=hidden_dim))
    diff_path = checkpoint_dir / "diffusion_residual.pt"
    if diff_path.exists():
        diff_state = torch.load(diff_path, map_location="cpu")
        diff_state = _strip_compiled_prefix(diff_state)
        diff.load_state_dict(diff_state)
    _ensure_model_on_device(diff, device)
    diff = _maybe_compile(diff, cfg, "diffusion_residual")

    print(f"Consistency distillation optimizations enabled:")
    print(f"  - Async GPU transfers: enabled")
    print(f"  - Adaptive tau schedule: {tau_schedule if tau_schedule else 'using base num_taus'}")
    print(f"  - Micro-batch size: {cfg.get('training', {}).get('distill_micro_batch', 'auto')}")
    
    epochs = stage_cfg.get("epochs", 1)
    optimizer = _create_optimizer(cfg, diff, "consistency_distill")
    scheduler = _create_scheduler(optimizer, cfg, "consistency_distill")
    patience = _get_patience(cfg, "consistency_distill")
    logger = TrainingLogger(cfg, stage="consistency_distill", global_step=global_step, wandb_ctx=wandb_ctx)
    dt = cfg.get("training", {}).get("dt", 0.1)
    
    dt_tensor = torch.tensor(dt, device=device)

    # Teacher/student are inlined below to enable reuse and vectorized taus

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
    base_num_taus = int(cfg.get("training", {}).get("distill_num_taus", 3) or 3)

    # Log optimizations applied
    print("Consistency distillation optimizations:")
    print(f"  - Teacher caching: ENABLED (computed once per batch)")
    print(f"  - AMP for teacher: {'ENABLED' if use_amp else 'DISABLED'}")
    print(f"  - Async GPU transfers: ENABLED")
    print(f"  - torch.compile: {'ENABLED' if _COMPILE_AVAILABLE else 'DISABLED'}")
    print(f"  - Persistent workers: {training_cfg_copy.get('persistent_workers', False)}")
    print(f"  - Prefetch factor: {training_cfg_copy.get('prefetch_factor', 2)}")
    print(f"  - Micro-batch size: {distill_micro}")
    print(f"  - Base num taus: {base_num_taus}")
    if tau_schedule:
        print(f"  - Tau schedule: {tau_schedule}")

    import time
    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        total_grad_norm = 0.0
        batches = 0
        
        num_taus_epoch = base_num_taus
        if tau_schedule:
            idx = min(epoch, len(tau_schedule) - 1)
            scheduled = tau_schedule[idx]
            if scheduled:
                num_taus_epoch = int(scheduled)

        for batch in loader:
            unpacked = unpack_batch(batch)
            if len(unpacked) == 4:
                z0, _, cond, _ = unpacked
            else:
                z0, _, cond = unpacked
            batch_size = z0.shape[0]
            micro = distill_micro or batch_size
            optimizer.zero_grad(set_to_none=True)
            batch_loss_value = 0.0

            # OPTIMIZATION #1: Compute teacher predictions ONCE per batch (outside micro-batch loop)
            # Moves teacher forward from inside loop (called N times) to outside (called 1 time)
            # Expected speedup: ~2x (reduces teacher calls by 50% when micro < batch_size)
            z0_device = z0.to(device, non_blocking=True)
            cond_device = {k: v.to(device, non_blocking=True) for k, v in cond.items()}
            full_batch_state = LatentState(z=z0_device, t=torch.tensor(0.0, device=device), cond=cond_device)

            # OPTIMIZATION #6: Use AMP for teacher forward (even though no gradients)
            # Reduces teacher forward time by ~20%, overall ~8% speedup
            with torch.no_grad(), autocast(enabled=use_amp):
                teacher_full = operator(full_batch_state, dt_tensor)

            for start in range(0, batch_size, micro):
                end = min(start + micro, batch_size)
                chunk_weight = (end - start) / batch_size

                # Slice pre-computed teacher predictions
                teacher_z_chunk = teacher_full.z[start:end]
                teacher_cond_chunk = {k: v[start:end] for k, v in teacher_full.cond.items()}

                try:
                    # Sample tau values
                    tau_seed = _sample_tau(num_taus_epoch, device, cfg)

                    # Use compiled forward function
                    with autocast(enabled=use_amp):
                        loss_chunk = _distill_forward_and_loss(
                            teacher_z_chunk,
                            teacher_cond_chunk,
                            num_taus_epoch,
                            diff,
                            teacher_full.t,
                            tau_seed,
                            device,
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
        if target_loss and best_loss <= target_loss:
            print(f"Consistency distill reached target loss {best_loss:.6f} <= {target_loss:.6f}; stopping early")
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
    
    # Upload updated checkpoint to W&B (clean way!)
    if wandb_ctx:
        wandb_ctx.save_file(diffusion_path)
        print(f"Uploaded updated diffusion checkpoint to W&B")
        if ema_model is not None:
            wandb_ctx.save_file(diffusion_ema_path)

    # Clean up operator from memory
    del operator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Send W&B alert
    if wandb_ctx:
        wandb_ctx.alert(
            title="✅ Consistency Distillation Complete",
            text=f"Final loss: {best_loss:.6f} | Ready for steady prior training",
            level="INFO"
        )


def train_steady_prior(cfg: dict, wandb_ctx=None, global_step: int = 0) -> None:
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
    logger = TrainingLogger(cfg, stage="steady_prior", global_step=global_step, wandb_ctx=wandb_ctx)
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
            unpacked = unpack_batch(batch)
            if len(unpacked) == 4:
                z0, z1, cond, _ = unpacked
            else:
                z0, z1, cond = unpacked
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
    
    # Upload checkpoint to W&B (clean way!)
    if wandb_ctx:
        wandb_ctx.save_file(prior_path)
        print(f"Uploaded steady prior checkpoint to W&B")

    # Send W&B alert
    if wandb_ctx:
        wandb_ctx.alert(
            title="🎉 All Training Stages Complete!",
            text=f"Steady prior final loss: {best_loss:.6f} | Full pipeline ready for evaluation",
            level="INFO"
        )


def _run_evaluation(cfg: dict, checkpoint_dir: Path, eval_mode: str = "baseline", wandb_ctx=None) -> dict:
    """Run evaluation and return metrics. Mode can be 'baseline' or 'ttc'."""
    from ups.eval.pdebench_runner import evaluate_latent_operator
    from ups.inference.rollout_ttc import TTCConfig, build_reward_model_from_config
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load models
    operator = make_operator(cfg)
    op_path = checkpoint_dir / "operator.pt"
    if op_path.exists():
        operator_state = torch.load(op_path, map_location="cpu")
        operator_state = _strip_compiled_prefix(operator_state)
        operator.load_state_dict(operator_state)
    operator = operator.to(device)
    operator.eval()
    
    # Load diffusion if available
    diffusion = None
    diff_path = checkpoint_dir / "diffusion_residual.pt"
    if diff_path.exists():
        latent_dim = cfg.get("latent", {}).get("dim", 32)
        hidden_dim = cfg.get("diffusion", {}).get("hidden_dim", latent_dim * 2)
        diffusion = DiffusionResidual(DiffusionResidualConfig(latent_dim=latent_dim, hidden_dim=hidden_dim))
        diff_state = torch.load(diff_path, map_location="cpu")
        diff_state = _strip_compiled_prefix(diff_state)
        diffusion.load_state_dict(diff_state)
        diffusion = diffusion.to(device)
        diffusion.eval()
    
    # Setup TTC if requested
    tau = cfg.get("training", {}).get("tau", 0.5)
    ttc_cfg = None
    reward_model = None
    
    if eval_mode == "ttc" and cfg.get("ttc", {}).get("enabled"):
        ttc_dict = cfg.get("ttc", {})
        # Use direct constructor like other places in codebase (evaluate.py, pdebench_runner.py)
        ttc_cfg = TTCConfig(
            steps=ttc_dict.get("steps", 1),
            dt=ttc_dict.get("dt", cfg.get("training", {}).get("dt", 0.1)),
            candidates=ttc_dict.get("candidates", 4),
            beam_width=ttc_dict.get("beam_width", 1),
            horizon=ttc_dict.get("horizon", 1),
            tau_range=tuple(ttc_dict.get("tau_range", [0.3, 0.7])) if "tau_range" in ttc_dict else (0.3, 0.7),
            residual_threshold=ttc_dict.get("residual_threshold"),
            max_evaluations=ttc_dict.get("max_evaluations"),
            gamma=ttc_dict.get("gamma", 1.0),
            device=device,
        )
        reward_model = build_reward_model_from_config(ttc_dict, latent_dim, device)
    
    # Change data split to test for evaluation
    eval_cfg = copy.deepcopy(cfg)
    eval_cfg["data"]["split"] = "test"
    
    # Run evaluation
    print(f"\nRunning evaluation (mode: {eval_mode})...")
    report, details = evaluate_latent_operator(
        cfg=eval_cfg,
        operator=operator,
        diffusion=diffusion,
        tau=tau,
        device=device,
        ttc_config=ttc_cfg,
        reward_model=reward_model,
        return_details=True,
    )
    
    # Add TTC flag to report
    report.extra["ttc"] = (eval_mode == "ttc")
    
    return {"report": report, "details": details}


def _log_evaluation_summary(wandb_ctx, baseline_metrics: dict, ttc_metrics: dict = None) -> None:
    """Log evaluation results and summary to WandB using WandBContext.

    Args:
        wandb_ctx: WandBContext instance (or None to skip logging)
        baseline_metrics: Baseline evaluation results
        ttc_metrics: Optional TTC evaluation results
    """
    if not wandb_ctx or not wandb_ctx.enabled:
        return

    baseline_report = baseline_metrics["report"]

    # Log baseline metrics to SUMMARY (scalars, not time series!)
    baseline_vals = {
        "baseline_mse": baseline_report.metrics.get("mse"),
        "baseline_mae": baseline_report.metrics.get("mae"),
        "baseline_rmse": baseline_report.metrics.get("rmse"),
        "baseline_nrmse": baseline_report.metrics.get("nrmse"),
        "baseline_rel_l2": baseline_report.metrics.get("rel_l2"),
    }

    # Add physics metrics if present
    for physics_key in ["conservation_gap", "bc_violation", "negativity_penalty"]:
        if physics_key in baseline_report.metrics:
            baseline_vals[f"baseline_{physics_key}"] = baseline_report.metrics[physics_key]

    wandb_ctx.log_eval_summary(baseline_vals, prefix="eval")

    # Log TTC metrics if available
    ttc_improvement_pct = None
    if ttc_metrics:
        ttc_report = ttc_metrics["report"]
        ttc_vals = {
            "ttc_mse": ttc_report.metrics.get("mse"),
            "ttc_mae": ttc_report.metrics.get("mae"),
            "ttc_rmse": ttc_report.metrics.get("rmse"),
            "ttc_nrmse": ttc_report.metrics.get("nrmse"),
            "ttc_rel_l2": ttc_report.metrics.get("rel_l2"),
        }

        # Add TTC physics metrics
        for physics_key in ["conservation_gap", "bc_violation", "negativity_penalty"]:
            if physics_key in ttc_report.metrics:
                ttc_vals[f"ttc_{physics_key}"] = ttc_report.metrics[physics_key]

        wandb_ctx.log_eval_summary(ttc_vals, prefix="eval")

        # Compute TTC improvement
        baseline_nrmse = baseline_report.metrics.get("nrmse", 1.0)
        ttc_nrmse = ttc_report.metrics.get("nrmse", 1.0)
        ttc_improvement_pct = ((baseline_nrmse - ttc_nrmse) / baseline_nrmse) * 100

        wandb_ctx.log_eval_summary({"ttc_improvement_pct": ttc_improvement_pct}, prefix="eval")
    
    # Create summary table
    summary_md = "## Evaluation Summary\n\n"
    summary_md += "| Metric | Baseline | TTC | Improvement |\n"
    summary_md += "|--------|----------|-----|-------------|\n"
    
    for metric_name in ["mse", "mae", "rmse", "nrmse", "rel_l2"]:
        baseline_val = baseline_metrics["report"].metrics.get(metric_name, 0)
        if ttc_metrics:
            ttc_val = ttc_metrics["report"].metrics.get(metric_name, 0)
            improv = ((baseline_val - ttc_val) / baseline_val) * 100 if baseline_val > 0 else 0
            summary_md += f"| {metric_name.upper()} | {baseline_val:.6f} | {ttc_val:.6f} | {improv:.1f}% |\n"
        else:
            summary_md += f"| {metric_name.upper()} | {baseline_val:.6f} | - | - |\n"
    
    # Log comprehensive comparison tables
    if wandb_ctx:
        # Accuracy metrics table
        accuracy_rows = []
        for metric in ["mse", "mae", "rmse", "nrmse", "rel_l2"]:
            base_val = baseline_metrics["report"].metrics.get(metric)
            if ttc_metrics:
                ttc_val = ttc_metrics["report"].metrics.get(metric)
                if base_val is not None and ttc_val is not None and base_val != 0:
                    improvement_pct = ((base_val - ttc_val) / base_val) * 100.0
                else:
                    improvement_pct = None
                accuracy_rows.append([
                    metric.upper(),
                    f"{base_val:.6f}" if base_val is not None else "N/A",
                    f"{ttc_val:.6f}" if ttc_val is not None else "N/A",
                    f"{improvement_pct:.1f}%" if improvement_pct is not None else "N/A",
                ])
            else:
                accuracy_rows.append([
                    metric.upper(),
                    f"{base_val:.6f}" if base_val is not None else "N/A",
                    "N/A",
                    "N/A",
                ])

        wandb_ctx.log_table(
            "Training Evaluation Summary",
            columns=["Metric", "Baseline", "TTC", "Improvement"],
            data=accuracy_rows
        )

        # Physics metrics table (if present)
        physics_rows = []
        for physics_key in ["conservation_gap", "bc_violation", "negativity_penalty"]:
            base_val = baseline_report.metrics.get(physics_key)
            if base_val is not None:
                if ttc_metrics:
                    ttc_val = ttc_metrics["report"].metrics.get(physics_key)
                    physics_rows.append([
                        physics_key.replace("_", " ").title(),
                        f"{base_val:.6f}",
                        f"{ttc_val:.6f}" if ttc_val is not None else "N/A",
                    ])
                else:
                    physics_rows.append([
                        physics_key.replace("_", " ").title(),
                        f"{base_val:.6f}",
                        "N/A",
                    ])

        if physics_rows:
            wandb_ctx.log_table(
                "Training Physics Diagnostics",
                columns=["Physics Check", "Baseline", "TTC"],
                data=physics_rows
            )


def train_all_stages(cfg: dict, wandb_ctx=None) -> None:
    """Run all training stages in sequence with clean WandB context.

    Args:
        cfg: Training configuration
        wandb_ctx: Optional WandBContext (if not provided, will try to load from env)
    """
    # Load or create WandB context
    if wandb_ctx is None:
        # Try to load from environment (subprocess mode)
        from ups.utils.wandb_context import create_wandb_context, save_wandb_context
        import datetime
        import os
        import json
        from pathlib import Path

        # Standalone mode: create new WandB context
        logging_cfg = cfg.get("logging", {})
        wandb_cfg = logging_cfg.get("wandb", {})
        if wandb_cfg.get("enabled", True):
            run_id = f"train-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            wandb_ctx = create_wandb_context(cfg, run_id=run_id, mode="online")

            # Save context to file for evaluation subprocess
            context_file_path = os.environ.get("WANDB_CONTEXT_FILE")
            if context_file_path and wandb_ctx and wandb_ctx.enabled:
                save_wandb_context(wandb_ctx, Path(context_file_path))
                print(f"✓ Saved WandB context to {context_file_path}")

            # Save WandB info for orchestrator
            wandb_info_path = os.environ.get("FAST_TO_SOTA_WANDB_INFO")
            if wandb_info_path and wandb_ctx and wandb_ctx.run:
                wandb_info = {
                    "id": wandb_ctx.run.id,
                    "name": wandb_ctx.run.name,
                    "project": wandb_ctx.run.project,
                    "entity": wandb_ctx.run.entity,
                    "url": wandb_ctx.run.url,
                }
                Path(wandb_info_path).write_text(json.dumps(wandb_info, indent=2))
                print(f"✓ Saved WandB info to {wandb_info_path}")

    # Log system info to config
    if wandb_ctx and wandb_ctx.enabled and torch.cuda.is_available():
        wandb_ctx.update_config({
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_count": torch.cuda.device_count(),
            "cuda_version": torch.version.cuda,
        })

    global_step = 0
    
    # Stage 1: Operator
    op_epochs = _stage_epochs(cfg, "operator")
    if op_epochs > 0:
        print("\n" + "="*50)
        print("STAGE 1/4: Training Operator")
        print("="*50)
        train_operator(cfg, wandb_ctx=wandb_ctx, global_step=global_step)
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
        train_diffusion(cfg, wandb_ctx=wandb_ctx, global_step=global_step)
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
        train_consistency(cfg, wandb_ctx=wandb_ctx, global_step=global_step)
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
        train_steady_prior(cfg, wandb_ctx=wandb_ctx, global_step=global_step)
    else:
        print("\n" + "="*50)
        print("STAGE 4/4: Skipping Steady Prior (epochs<=0)")
        print("="*50)
    
    # Stage 5: Evaluation (optional, controlled by config)
    checkpoint_dir = ensure_checkpoint_dir(cfg)
    run_eval = cfg.get("evaluation", {}).get("enabled", True)  # Default to True for convenience
    
    if run_eval:
        print("\n" + "="*50)
        print("STAGE 5/5: Evaluation on Test Set")
        print("="*50)
        
        try:
            # Run baseline evaluation
            print("\n📊 Running baseline evaluation...")
            baseline_results = _run_evaluation(cfg, checkpoint_dir, eval_mode="baseline", wandb_ctx=wandb_ctx)
            baseline_report = baseline_results["report"]
            
            print(f"Baseline Results:")
            print(f"  MSE:   {baseline_report.metrics.get('mse', 0):.6f}")
            print(f"  MAE:   {baseline_report.metrics.get('mae', 0):.6f}")
            print(f"  RMSE:  {baseline_report.metrics.get('rmse', 0):.6f}")
            print(f"  NRMSE: {baseline_report.metrics.get('nrmse', 0):.6f}")
            
            # Run TTC evaluation if configured
            ttc_results = None
            if cfg.get("ttc", {}).get("enabled", False):
                print("\n📊 Running TTC evaluation...")
                ttc_results = _run_evaluation(cfg, checkpoint_dir, eval_mode="ttc", wandb_ctx=wandb_ctx)
                ttc_report = ttc_results["report"]
                
                print(f"TTC Results:")
                print(f"  MSE:   {ttc_report.metrics.get('mse', 0):.6f}")
                print(f"  MAE:   {ttc_report.metrics.get('mae', 0):.6f}")
                print(f"  RMSE:  {ttc_report.metrics.get('rmse', 0):.6f}")
                print(f"  NRMSE: {ttc_report.metrics.get('nrmse', 0):.6f}")
                
                # Compute improvement
                baseline_nrmse = baseline_report.metrics.get('nrmse', 1.0)
                ttc_nrmse = ttc_report.metrics.get('nrmse', 1.0)
                improvement = ((baseline_nrmse - ttc_nrmse) / baseline_nrmse) * 100
                print(f"\n  TTC Improvement: {improvement:.1f}%")
            
            # Log to WandB using clean context
            if wandb_ctx:
                _log_evaluation_summary(wandb_ctx, baseline_results, ttc_results)
                
                # Save reports as artifacts
                import os
                report_dir = Path("reports")
                report_dir.mkdir(parents=True, exist_ok=True)
                
                # Save baseline report
                baseline_json = report_dir / "eval_baseline.json"
                baseline_report.to_json(baseline_json)
                if wandb_ctx and wandb_ctx.run is not None:
                    artifact = wandb.Artifact(name=f"eval-baseline-{wandb_ctx.run.id}", type="evaluation")
                    artifact.add_file(str(baseline_json))
                    wandb_ctx.log_artifact(artifact)

                # Save TTC report if available
                if ttc_results:
                    ttc_json = report_dir / "eval_ttc.json"
                    ttc_report.to_json(ttc_json)
                    if wandb_ctx and wandb_ctx.run is not None:
                        artifact = wandb.Artifact(name=f"eval-ttc-{wandb_ctx.run.id}", type="evaluation")
                        artifact.add_file(str(ttc_json))
                        wandb_ctx.log_artifact(artifact)
                
        except Exception as e:
            print(f"\n⚠️  Evaluation failed: {e}")
            if wandb_ctx and wandb_ctx.enabled:
                wandb_ctx.log_eval_summary({"error": str(e)}, prefix="eval")
    else:
        print("\n" + "="*50)
        print("STAGE 5/5: Skipping Evaluation (disabled in config)")
        print("="*50)
    
    # Log final summary
    if wandb_ctx and wandb_ctx.enabled:
        # Load final checkpoints to get model sizes
        import os
        summary = {
            "total_training_complete": 1,
            "operator_checkpoint_size_mb": os.path.getsize(checkpoint_dir / "operator.pt") / 1e6 if (checkpoint_dir / "operator.pt").exists() else 0,
            "diffusion_checkpoint_size_mb": os.path.getsize(checkpoint_dir / "diffusion_residual.pt") / 1e6 if (checkpoint_dir / "diffusion_residual.pt").exists() else 0,
            "steady_prior_checkpoint_size_mb": os.path.getsize(checkpoint_dir / "steady_prior.pt") / 1e6 if (checkpoint_dir / "steady_prior.pt").exists() else 0,
        }
        wandb_ctx.log_eval_summary(summary, prefix="summary")

        # Generate final report summary
        print("\n" + "="*50)
        print("📝 WandB Summary Generated")
        print("="*50)
        print(f"View full results at: {wandb_ctx.run.url}")

        # Training run owns its own lifecycle - call finish()
        wandb_ctx.finish()
    
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
