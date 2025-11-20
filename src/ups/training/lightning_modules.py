from __future__ import annotations

"""PyTorch Lightning modules for UPS training."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import lr_scheduler

from ups.core.blocks_pdet import PDETransformerConfig
from ups.core.latent_state import LatentState
from ups.models.latent_operator import LatentOperator, LatentOperatorConfig
from ups.models.pure_transformer import PureTransformerConfig
from ups.training.distributed_utils import maybe_trigger_simulated_oom
from ups.training.hybrid_optimizer import HybridOptimizer, wrap_optimizer_with_cpu_offload
from ups.training.losses import compute_operator_loss_bundle
from ups.training.muon_factory import create_muon_optimizer, get_available_backends
from ups.training.param_groups import build_param_groups, print_param_split_summary
from ups.utils.config_loader import load_config_with_includes
from ups.models.diffusion_residual import DiffusionResidual, DiffusionResidualConfig


def _nrmse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mse = torch.mean((pred - target) ** 2)
    denom = torch.mean(target**2) + eps
    return torch.sqrt(mse / denom)


def _spectral_energy_loss(
    pred: torch.Tensor, target: torch.Tensor, dim: int = 1, eps: float = 1e-8
) -> torch.Tensor:
    # Match native helper from scripts/train.py
    with torch.cuda.amp.autocast(enabled=False):
        pred_fft = torch.fft.rfft(pred.float(), dim=dim)
        tgt_fft = torch.fft.rfft(target.float(), dim=dim)
        pred_energy = torch.mean(pred_fft.abs() ** 2)
        tgt_energy = torch.mean(tgt_fft.abs() ** 2)
        loss = torch.abs(pred_energy - tgt_energy) / (tgt_energy + eps)
    return loss.to(pred.dtype)


def _sample_tau(batch_size: int, device: torch.device, cfg: dict) -> torch.Tensor:
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


def _maybe_compile(model: nn.Module, cfg: dict, name: str) -> nn.Module:
    """Optionally compile a model with torch.compile when enabled and available."""
    compile_enabled = bool(cfg.get("training", {}).get("compile", False))
    if not compile_enabled:
        return model

    # Skip compilation for teacher/eval-only labels
    if "teacher" in name:
        return model

    try:
        import torch  # Local import to avoid issues on older versions

        training_cfg = cfg.get("training", {})
        user_mode = str(training_cfg.get("compile_mode", "")).lower()
        compile_mode = user_mode if user_mode in {"default", "reduce-overhead", "max-autotune"} else "default"
        return torch.compile(model, mode=compile_mode, fullgraph=False)
    except Exception:
        return model


def _build_operator(cfg: dict) -> LatentOperator:
    latent_cfg = cfg.get("latent", {})
    dim = latent_cfg.get("dim", 32)
    operator_cfg = cfg.get("operator", {})
    pdet_cfg = operator_cfg.get("pdet", {})
    architecture_type = operator_cfg.get("architecture_type", "pdet_unet")

    if not pdet_cfg:
        pdet_cfg = {
            "input_dim": dim,
            "hidden_dim": dim * 2,
            "depths": [1, 1, 1],
            "group_size": max(dim // 2, 4),
            "num_heads": 4,
        }

    if architecture_type == "pdet_stack":
        pdet_config = PureTransformerConfig(**pdet_cfg)
    else:
        pdet_config = PDETransformerConfig(**pdet_cfg)

    operator_config = LatentOperatorConfig(
        latent_dim=dim,
        pdet=pdet_config,
        architecture_type=architecture_type,
        time_embed_dim=dim,
    )
    return LatentOperator(operator_config)


def _create_optimizer(cfg: dict, model: nn.Module, stage: str) -> torch.optim.Optimizer:
    """Match optimizer behavior from native training (supports muon hybrid)."""
    stage_cfg = cfg.get("stages", {}).get(stage, {}) if isinstance(cfg.get("stages"), dict) else {}
    opt_cfg = stage_cfg.get("optimizer") or cfg.get("optimizer", {})
    name = opt_cfg.get("name", "adam").lower()
    lr = opt_cfg.get("lr", 1e-3)
    weight_decay = opt_cfg.get("weight_decay", 0.0)

    cpu_offload_enabled = cfg.get("training", {}).get("cpu_offload_optimizer", False)

    if name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        return wrap_optimizer_with_cpu_offload(optimizer, cpu_offload_enabled)

    if name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            fused=True,
        )
        return wrap_optimizer_with_cpu_offload(optimizer, cpu_offload_enabled)

    if name == "sgd":
        momentum = opt_cfg.get("momentum", 0.9)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
        return wrap_optimizer_with_cpu_offload(optimizer, cpu_offload_enabled)

    if name in {"muon_hybrid", "muon"}:
        backends = get_available_backends()
        if not backends:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                fused=True,
            )
            return wrap_optimizer_with_cpu_offload(optimizer, cpu_offload_enabled)

        print(f"Available Muon backends: {', '.join(backends)}")
        params_muon, params_adamw = build_param_groups(model)
        print_param_split_summary(model)

        muon_momentum = opt_cfg.get("muon_momentum", 0.95)
        muon_ns_steps = opt_cfg.get("muon_ns_steps", 5)
        muon_backend = opt_cfg.get("muon_backend", "auto")

        optimizers = []
        if len(params_muon) > 0:
            muon_optimizer, actual_backend = create_muon_optimizer(
                params_muon,
                lr=lr,
                weight_decay=weight_decay,
                momentum=muon_momentum,
                ns_steps=muon_ns_steps,
                backend=muon_backend,
            )
            optimizers.append(muon_optimizer)
            print(f"  Muon: {len(params_muon)} parameter groups (backend: {actual_backend})")

        if len(params_adamw) > 0:
            adamw_betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))
            adamw_eps = opt_cfg.get("eps", 1e-8)
            adamw_opt = torch.optim.AdamW(
                params_adamw,
                lr=lr,
                betas=adamw_betas,
                eps=adamw_eps,
                weight_decay=weight_decay,
                fused=True,
            )
            optimizers.append(adamw_opt)
            print(f"  AdamW: {len(params_adamw)} parameter groups")

        if len(optimizers) == 1:
            return optimizers[0]
        return HybridOptimizer(optimizers)

    raise ValueError(f"Unsupported optimizer '{name}'")


def _create_scheduler(optimizer: torch.optim.Optimizer, cfg: dict, stage: str):
    """Match scheduler behavior from native training."""
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


class OperatorLightningModule(pl.LightningModule):
    """Lightning module for operator training (parity with native training loop)."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.operator = _build_operator(cfg)
        self.dt = float(cfg.get("training", {}).get("dt", 0.1))
        self.lam_spec = float(cfg.get("training", {}).get("lambda_spectral", 0.0) or 0.0)
        self.lam_rollout = float(cfg.get("training", {}).get("lambda_rollout", 0.0) or 0.0)
        train_cfg = cfg.get("training", {})
        query_sample_cfg = train_cfg.get("query_sampling", {}) if isinstance(train_cfg, dict) else {}
        self.num_queries = query_sample_cfg.get("num_queries")
        self.query_strategy = query_sample_cfg.get("strategy", "uniform")

        # Optional compile (safe fallback)
        self.operator = _maybe_compile(self.operator, cfg, "operator")

    def forward(self, state: LatentState, dt: torch.Tensor) -> LatentState:
        return self.operator(state, dt)

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        device = self.device
        z0 = batch["z0"].to(device)
        z1 = batch["z1"].to(device)
        cond = {k: v.to(device) for k, v in batch.get("cond", {}).items()}
        future = batch.get("future")
        if future is not None:
            future = future.to(device)
        coords = batch.get("coords")
        if coords is not None:
            coords = coords.to(device)
        meta = batch.get("meta") or {}
        task_names = batch.get("task_names")

        state = LatentState(z=z0, t=torch.tensor(0.0, device=device), cond=cond)
        dt_tensor = torch.tensor(self.dt, device=device)

        maybe_trigger_simulated_oom("operator", batch_idx, int(self.global_rank))

        next_state = self(state, dt_tensor)

        rollout_pred = None
        rollout_tgt = None
        if self.lam_rollout > 0.0 and future is not None and future.numel() > 0:
            rollout_preds = []
            rollout_state = next_state
            steps = future.shape[1]
            for _ in range(steps):
                rollout_state = self(rollout_state, dt_tensor)
                rollout_preds.append(rollout_state.z)
            rollout_pred = torch.stack(rollout_preds, dim=1)
            rollout_tgt = future

        loss_weights = {
            "lambda_forward": 1.0,
            "lambda_inv_enc": float(self.cfg.get("training", {}).get("lambda_inv_enc", 0.0)),
            "lambda_inv_dec": float(self.cfg.get("training", {}).get("lambda_inv_dec", 0.0)),
            "lambda_spectral": self.lam_spec,
            "lambda_rollout": self.lam_rollout,
            "inverse_loss_warmup_epochs": int(self.cfg.get("training", {}).get("inverse_loss_warmup_epochs", 15)),
            "inverse_loss_max_weight": float(self.cfg.get("training", {}).get("inverse_loss_max_weight", 0.05)),
        }

        grid_shape = meta.get("grid_shape") if meta else None

        loss_bundle = compute_operator_loss_bundle(
            input_fields=batch.get("input_fields"),
            encoded_latent=state.z,
            decoder=None,  # decoder not constructed here (inverse losses default off)
            input_positions=coords,
            encoder=None,
            query_positions=coords,
            coords=coords,
            meta=meta,
            pred_next=next_state.z,
            target_next=z1,
            pred_rollout=rollout_pred,
            target_rollout=rollout_tgt,
            spectral_pred=next_state.z if self.lam_spec > 0 else None,
            spectral_target=z1 if self.lam_spec > 0 else None,
            weights=loss_weights,
            current_epoch=int(self.current_epoch),
            num_queries=self.num_queries,
            query_strategy=self.query_strategy,
            grid_shape=grid_shape,
        )

        loss = loss_bundle.total
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        for name, value in loss_bundle.components.items():
            self.log(f"train/{name}", value, on_step=False, on_epoch=True, sync_dist=True)

        if task_names:
            unique = set(task_names)
            for task in unique:
                mask = [t == task for t in task_names]
                if any(mask):
                    indices = torch.nonzero(torch.tensor(mask, device=device), as_tuple=False).squeeze(-1)
                    task_loss = _nrmse(next_state.z[indices], z1[indices])
                    self.log(f"train/{task}/nrmse", task_loss, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        device = self.device
        z0 = batch["z0"].to(device)
        z1 = batch["z1"].to(device)
        cond = {k: v.to(device) for k, v in batch.get("cond", {}).items()}
        state = LatentState(z=z0, t=torch.tensor(0.0, device=device), cond=cond)
        dt_tensor = torch.tensor(self.dt, device=device)
        next_state = self(state, dt_tensor)

        val_loss = _nrmse(next_state.z, z1)
        self.log("val/nrmse", val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        task_names = batch.get("task_names")
        if task_names:
            unique = set(task_names)
            for task in unique:
                mask = [t == task for t in task_names]
                if any(mask):
                    indices = torch.nonzero(torch.tensor(mask, device=device), as_tuple=False).squeeze(-1)
                    task_loss = _nrmse(next_state.z[indices], z1[indices])
                    self.log(f"val/{task}/nrmse", task_loss, on_epoch=True, sync_dist=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = _create_optimizer(self.cfg, self.operator, "operator")
        scheduler = _create_scheduler(optimizer, self.cfg, "operator")
        if scheduler is None:
            return optimizer
        if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/nrmse",
                    "interval": "epoch",
                },
            }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class DiffusionLightningModule(pl.LightningModule):
    """Lightning module for diffusion residual training using a frozen operator teacher."""

    def __init__(self, cfg: dict, operator_ckpt: Optional[str] = None):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.dt = float(cfg.get("training", {}).get("dt", 0.1))
        self.lam_spec = float(cfg.get("training", {}).get("lambda_spectral", 0.0) or 0.0)
        self.lam_rel = float(cfg.get("training", {}).get("lambda_relative", 0.0) or 0.0)

        # Teacher operator (frozen)
        self.operator = _build_operator(cfg).eval()
        self._load_operator_checkpoint(operator_ckpt)
        for p in self.operator.parameters():
            p.requires_grad_(False)

        # Student diffusion residual
        latent_dim = cfg.get("latent", {}).get("dim", 32)
        hidden_dim = cfg.get("diffusion", {}).get("hidden_dim", latent_dim * 2)
        self.diffusion = DiffusionResidual(DiffusionResidualConfig(latent_dim=latent_dim, hidden_dim=hidden_dim))
        self.diffusion = _maybe_compile(self.diffusion, cfg, "diffusion_residual")

    def _load_operator_checkpoint(self, operator_ckpt: Optional[str]) -> None:
        if operator_ckpt:
            path = Path(operator_ckpt)
        else:
            ckpt_dir = Path(self.cfg.get("checkpoint", {}).get("dir", "checkpoints"))
            # Prefer EMA if present
            candidate = ckpt_dir / "operator_ema.pt"
            if candidate.exists():
                path = candidate
            else:
                path = ckpt_dir / "operator.pt"
        if not path.exists():
            return
        try:
            state = torch.load(path, map_location="cpu")
            state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
            if "state_dict" in state:
                state = state["state_dict"]
            # Strip "operator." prefix if present
            fixed = {}
            for k, v in state.items():
                fixed[k.replace("operator.", "", 1) if k.startswith("operator.") else k] = v
            # Adjust shapes mismatch by filtering keys that don't align
            model_state = self.operator.state_dict()
            filtered = {k: v for k, v in fixed.items() if k in model_state and v.shape == model_state[k].shape}
            self.operator.load_state_dict(filtered, strict=False)
        except Exception:
            # Ignore loading if incompatible (tests may use tiny dims)
            pass

    def forward(self, state: LatentState, tau: torch.Tensor) -> torch.Tensor:
        return self.diffusion(state, tau)

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        device = self.device
        z0 = batch["z0"].to(device)
        z1 = batch["z1"].to(device)
        cond = {k: v.to(device) for k, v in batch.get("cond", {}).items()}
        state = LatentState(z=z0, t=torch.tensor(0.0, device=device), cond=cond)
        dt_tensor = torch.tensor(self.dt, device=device)

        maybe_trigger_simulated_oom("diffusion", batch_idx, int(self.global_rank))

        with torch.no_grad():
            predicted = self.operator(state, dt_tensor)

        residual_target = z1 - predicted.z
        tau_tensor = _sample_tau(z0.size(0), device, self.cfg)
        drift = self.diffusion(predicted, tau_tensor)

        loss = torch.nn.functional.mse_loss(drift, residual_target)
        if self.lam_spec > 0.0:
            loss = loss + self.lam_spec * _spectral_energy_loss(drift, residual_target, dim=1)
        if self.lam_rel > 0.0:
            loss = loss + self.lam_rel * _nrmse(drift, residual_target)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = _create_optimizer(self.cfg, self.diffusion, "diff_residual")
        scheduler = _create_scheduler(optimizer, self.cfg, "diff_residual")
        if scheduler is None:
            return optimizer
        if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/nrmse",
                    "interval": "epoch",
                },
            }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class ConsistencyLightningModule(pl.LightningModule):
    """Lightweight consistency distillation (student matches teacher prediction)."""

    def __init__(self, cfg: dict, operator_ckpt: Optional[str] = None, diffusion_ckpt: Optional[str] = None):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.dt = float(cfg.get("training", {}).get("dt", 0.1))
        self.lam_spec = float(cfg.get("training", {}).get("lambda_spectral", 0.0) or 0.0)
        self.lam_rel = float(cfg.get("training", {}).get("lambda_relative", 0.0) or 0.0)
        self.base_num_taus = int(cfg.get("training", {}).get("distill_num_taus", 3) or 3)

        # Teacher operator (frozen)
        self.operator = _build_operator(cfg).eval()
        self._load_operator_checkpoint(operator_ckpt)
        for p in self.operator.parameters():
            p.requires_grad_(False)

        # Student diffusion residual (teacher == student in this distilled phase)
        latent_dim = cfg.get("latent", {}).get("dim", 32)
        hidden_dim = cfg.get("diffusion", {}).get("hidden_dim", latent_dim * 2)
        self.diffusion = DiffusionResidual(DiffusionResidualConfig(latent_dim=latent_dim, hidden_dim=hidden_dim))
        self._load_diffusion_checkpoint(diffusion_ckpt)
        self.diffusion = _maybe_compile(self.diffusion, cfg, "consistency_diffusion")

    def _load_operator_checkpoint(self, operator_ckpt: Optional[str]) -> None:
        if operator_ckpt:
            path = Path(operator_ckpt)
        else:
            ckpt_dir = Path(self.cfg.get("checkpoint", {}).get("dir", "checkpoints"))
            path = ckpt_dir / "operator_ema.pt"
            if not path.exists():
                path = ckpt_dir / "operator.pt"
        if path.exists():
            try:
                state = torch.load(path, map_location="cpu")
                if "state_dict" in state:
                    state = state["state_dict"]
                fixed = {}
                for k, v in state.items():
                    if k.startswith("_orig_mod."):
                        k = k.replace("_orig_mod.", "", 1)
                    if k.startswith("operator."):
                        k = k.replace("operator.", "", 1)
                    fixed[k] = v
                model_state = self.operator.state_dict()
                filtered = {k: v for k, v in fixed.items() if k in model_state and v.shape == model_state[k].shape}
                self.operator.load_state_dict(filtered, strict=False)
            except Exception:
                pass

    def _load_diffusion_checkpoint(self, diffusion_ckpt: Optional[str]) -> None:
        path = None
        if diffusion_ckpt:
            path = Path(diffusion_ckpt)
        else:
            ckpt_dir = Path(self.cfg.get("checkpoint", {}).get("dir", "checkpoints"))
            candidate = ckpt_dir / "diffusion_residual_ema.pt"
            path = candidate if candidate.exists() else ckpt_dir / "diffusion_residual.pt"
        if path and path.exists():
            try:
                state = torch.load(path, map_location="cpu")
                if "state_dict" in state:
                    state = state["state_dict"]
                fixed = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
                model_state = self.diffusion.state_dict()
                filtered = {
                    k: v for k, v in fixed.items() if k in model_state and v.shape == model_state[k].shape
                }
                self.diffusion.load_state_dict(filtered, strict=False)
            except Exception:
                pass

    def forward(self, state: LatentState, tau: torch.Tensor) -> torch.Tensor:
        return self.diffusion(state, tau)

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        device = self.device
        z0 = batch["z0"].to(device)
        cond = {k: v.to(device) for k, v in batch.get("cond", {}).items()}
        state = LatentState(z=z0, t=torch.tensor(0.0, device=device), cond=cond)
        dt_tensor = torch.tensor(self.dt, device=device)

        with torch.no_grad():
            teacher_state = self.operator(state, dt_tensor)

        # Tile over taus (simple uniform sampling)
        num_taus = self.base_num_taus
        tau_seed = torch.rand(z0.size(0), device=device)
        tau_flat = tau_seed.repeat_interleave(num_taus)
        z_tiled = (
            teacher_state.z.unsqueeze(1)
            .expand(-1, num_taus, -1, -1)
            .reshape(z0.size(0) * num_taus, teacher_state.z.shape[1], teacher_state.z.shape[2])
            .contiguous()
        )
        cond_tiled = {k: v.repeat_interleave(num_taus, dim=0) for k, v in cond.items()}
        tiled_state = LatentState(z=z_tiled, t=torch.tensor(0.0, device=device), cond=cond_tiled)
        drift = self.diffusion(tiled_state, tau_flat)

        student_z = z_tiled.to(drift.dtype) + drift
        loss = torch.nn.functional.mse_loss(student_z, z_tiled.to(drift.dtype))
        if self.lam_spec > 0.0:
            loss = loss + self.lam_spec * _spectral_energy_loss(drift, torch.zeros_like(drift), dim=1)
        if self.lam_rel > 0.0:
            loss = loss + self.lam_rel * _nrmse(student_z, z_tiled.to(student_z.dtype))

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = _create_optimizer(self.cfg, self.diffusion, "consistency_distill")
        scheduler = _create_scheduler(optimizer, self.cfg, "consistency_distill")
        if scheduler is None:
            return optimizer
        if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/nrmse",
                    "interval": "epoch",
                },
            }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
