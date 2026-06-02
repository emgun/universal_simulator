#!/usr/bin/env python
from __future__ import annotations

"""Training entrypoint for latent operator stages."""

import argparse
import copy
import random
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.multiprocessing as mp
import yaml
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import UninitializedParameter
from torch.optim import lr_scheduler
from torch.utils.data import ConcatDataset, DataLoader, Dataset

try:
    import wandb
except ImportError:
    wandb = None

# Ensure CUDA + DataLoader workers use a safe start method
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ups.core.blocks_pdet import PDETransformerConfig
from ups.core.conditioning import ConditioningConfig
from ups.core.latent_state import LatentState
from ups.data.latent_pairs import (
    build_latent_pair_loader,
    conditioning_source_dims_from_sample,
    infer_channel_count,
    infer_grid_shape,
    make_grid_coords,
    pdebench_condition_step,
    pdebench_conditioning_extras,
    unpack_batch,
)
from ups.data.pdebench import PDEBenchConfig, PDEBenchDataset
from ups.io.decoder_anypoint import AnyPointDecoder, AnyPointDecoderConfig
from ups.io.enc_grid import GridEncoder, GridEncoderConfig
from ups.models.diffusion_residual import DiffusionResidual, DiffusionResidualConfig
from ups.models.latent_operator import LatentOperator, LatentOperatorConfig
from ups.models.steady_prior import SteadyPrior, SteadyPriorConfig
from ups.training.losses import semigroup_consistency_loss
from ups.utils.monitoring import MonitoringSession, init_monitoring_session


# ---- Auxiliary training losses ----
def _nrmse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mse = torch.mean((pred - target) ** 2)
    denom = torch.mean(target**2) + eps
    return torch.sqrt(mse / denom)


def _spectral_energy_loss(
    pred: torch.Tensor, target: torch.Tensor, dim: int = 1, eps: float = 1e-8
) -> torch.Tensor:
    """Relative spectral energy difference along the given axis (default: token axis)."""
    pred_fft = torch.fft.rfft(pred, dim=dim)
    tgt_fft = torch.fft.rfft(target, dim=dim)
    pred_energy = torch.mean(pred_fft.abs() ** 2)
    tgt_energy = torch.mean(tgt_fft.abs() ** 2)
    return torch.abs(pred_energy - tgt_energy) / (tgt_energy + eps)


def _decoded_field_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    previous: torch.Tensor,
    *,
    stage_cfg: dict[str, Any],
) -> torch.Tensor:
    loss = F.mse_loss(pred, target)
    residual_weight = float(stage_cfg.get("lambda_persistence_residual", 0.0) or 0.0)
    if residual_weight > 0.0:
        loss = loss + residual_weight * F.mse_loss(pred - previous, target - previous)
    spectral_weight = float(stage_cfg.get("lambda_spectral", 0.0) or 0.0)
    if spectral_weight > 0.0:
        loss = loss + spectral_weight * _spectral_energy_loss(pred, target, dim=1)
    residual_spectral_weight = float(
        stage_cfg.get("lambda_persistence_residual_spectral", 0.0) or 0.0
    )
    if residual_spectral_weight > 0.0:
        loss = loss + residual_spectral_weight * _spectral_energy_loss(
            pred - previous, target - previous, dim=1
        )
    relative_weight = float(stage_cfg.get("lambda_relative", 0.0) or 0.0)
    if relative_weight > 0.0:
        loss = loss + relative_weight * _nrmse(pred, target)
    return loss


def _task_loss_weight(stage_cfg: dict[str, Any], task_name: str | None) -> float:
    if not task_name:
        return 1.0
    raw_weights = stage_cfg.get("task_loss_weights", {})
    if raw_weights is None:
        return 1.0
    if not isinstance(raw_weights, Mapping):
        raise ValueError("task_loss_weights must be a mapping from task name to weight")
    weight = float(raw_weights.get(str(task_name), 1.0))
    if weight < 0.0:
        raise ValueError("task_loss_weights values must be non-negative")
    return weight


def _decoded_rollout_training_loss(
    decoded_losses: Sequence[torch.Tensor],
    *,
    stage_cfg: dict[str, Any],
    lambda_rollout: float,
) -> torch.Tensor:
    if not decoded_losses:
        raise ValueError("decoded_losses must be non-empty")
    loss = decoded_losses[0]
    if len(decoded_losses) <= 1:
        return loss

    rollout_losses = torch.stack(list(decoded_losses[1:]))
    horizon_power = float(stage_cfg.get("rollout_loss_horizon_power", 0.0) or 0.0)
    if horizon_power < 0.0:
        raise ValueError("rollout_loss_horizon_power must be non-negative")
    if horizon_power > 0.0:
        step_count = len(decoded_losses)
        steps = torch.arange(
            2, step_count + 1, device=rollout_losses.device, dtype=rollout_losses.dtype
        )
        weights = torch.pow(steps / float(step_count), horizon_power)
        rollout_loss = (rollout_losses * weights).sum() / weights.sum()
    else:
        rollout_loss = rollout_losses.mean()
    return loss + lambda_rollout * rollout_loss


def _decoded_rollout_training_window(
    fields: torch.Tensor,
    *,
    rollout_steps: int,
    stage_cfg: dict[str, Any],
) -> tuple[torch.Tensor, int]:
    """Select the temporal training window for decoded rollout supervision."""
    max_steps = min(rollout_steps, max(fields.shape[1] - 1, 0))
    if max_steps <= 0:
        return fields[:, :1], 0

    strategy = str(stage_cfg.get("rollout_start_strategy", "zero") or "zero")
    if strategy in {"zero", "first"}:
        start = 0
    elif strategy == "latest":
        start = max(0, fields.shape[1] - (max_steps + 1))
    else:
        raise ValueError("rollout_start_strategy must be one of: zero, first, latest")

    end = start + max_steps + 1
    return fields[:, start:end], start


def load_config(path: str) -> dict:
    try:
        from ups.utils.config_loader import load_config_with_includes

        return load_config_with_includes(path)
    except ImportError:
        with open(path, encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}


def set_seed(cfg: dict) -> None:
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
    def __init__(
        self, cfg: dict[str, dict], stage: str, global_step: int = 0, shared_run=None
    ) -> None:
        training_cfg = cfg.get("training", {})
        log_path = training_cfg.get("log_path", "reports/training_log.jsonl")
        self.stage = stage
        self.global_step = global_step

        # Use shared run if provided, otherwise create new one
        if shared_run is not None:
            self.session = MonitoringSession(
                file_path=Path(log_path) if log_path else None,
                run=shared_run,
                component=f"training-{stage}",
            )
            self.owns_run = False
        else:
            self.session = init_monitoring_session(
                cfg, component=f"training-{stage}", file_path=log_path
            )
            self.owns_run = True

    def log(
        self,
        *,
        epoch: int,
        loss: float,
        optimizer: torch.optim.Optimizer,
        patience_counter: int | None = None,
        grad_norm: float | None = None,
        epoch_time: float | None = None,
        best_loss: float | None = None,
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


def dataset_loader(cfg: dict, *, encoder_override=None) -> DataLoader:
    data_cfg = cfg.get("data", {})
    if not (data_cfg.get("task") or data_cfg.get("kind")):
        raise ValueError(
            "Training now requires a real dataset configuration. Set data.task for PDEBench or data.kind for Zarr datasets."
        )
    return build_latent_pair_loader(cfg, encoder_override=encoder_override)


def _supports_grid_codec(cfg: dict) -> bool:
    data_cfg = cfg.get("data", {})
    tasks = data_cfg.get("task")
    if isinstance(tasks, str):
        return True
    return (
        isinstance(tasks, (list, tuple))
        and len(tasks) > 0
        and all(isinstance(task, str) for task in tasks)
    )


def _pdebench_tasks(cfg: dict) -> list[str]:
    data_cfg = cfg.get("data", {})
    tasks = data_cfg.get("task")
    if isinstance(tasks, str):
        return [tasks]
    if isinstance(tasks, (list, tuple)) and tasks and all(isinstance(task, str) for task in tasks):
        return [str(task) for task in tasks]
    raise ValueError(
        "Grid codec support currently requires one PDEBench task or a non-empty list of task names"
    )


def _pdebench_codec_context(cfg: dict) -> tuple[list[dict[str, Any]], int, str]:
    data_cfg = cfg.get("data", {})
    field_name = data_cfg.get("field_name", "u")
    specs: list[dict[str, Any]] = []
    for task in _pdebench_tasks(cfg):
        dataset = PDEBenchDataset(
            PDEBenchConfig(
                task=task,
                split=data_cfg.get("split", "train"),
                root=data_cfg.get("root"),
                param_keys=tuple(data_cfg.get("param_keys", ())),
                bc_keys=tuple(data_cfg.get("bc_keys", ())),
                max_samples=data_cfg.get("max_samples"),
            )
        )
        sample_fields = dataset.fields[0]
        grid_shape = infer_grid_shape(sample_fields)
        channels = infer_channel_count(sample_fields, grid_shape)
        specs.append(
            {"task": task, "dataset": dataset, "grid_shape": grid_shape, "channels": channels}
        )
    channels = int(specs[0]["channels"])
    if any(int(spec["channels"]) != channels for spec in specs[1:]):
        raise ValueError(
            "Multi-task grid codec training currently requires all tasks to share the same channel count"
        )
    return specs, channels, field_name


def _auto_conditioning_sources(cfg: dict[str, Any]) -> dict[str, int]:
    specs, _, _ = _pdebench_codec_context(cfg)
    task_vocab = tuple(str(spec["task"]) for spec in specs) if len(specs) > 1 else None
    data_cfg = cfg.get("data", {})
    param_vocab = tuple(data_cfg.get("param_keys", ()))
    bc_vocab = tuple(data_cfg.get("bc_keys", ()))
    sources: dict[str, int] = {}
    for spec in specs:
        sample = spec["dataset"][0]
        sample_sources = conditioning_source_dims_from_sample(
            sample,
            grid_shape=spec["grid_shape"],
            task_name=str(spec["task"]),
            task_vocab=task_vocab,
            param_vocab=param_vocab,
            bc_vocab=bc_vocab,
        )
        for key, dim in sample_sources.items():
            sources[key] = max(sources.get(key, 0), int(dim))
    return sources


def _pdebench_grid_spec(cfg: dict) -> tuple[PDEBenchDataset, tuple[int, int], int, str]:
    specs, channels, field_name = _pdebench_codec_context(cfg)
    first = specs[0]
    return first["dataset"], first["grid_shape"], channels, field_name


def make_encoder(cfg: dict[str, Any]) -> GridEncoder:
    _, channels, field_name = _pdebench_codec_context(cfg)
    data_cfg = cfg.get("data", {})
    latent_cfg = cfg.get("latent", {})
    encoder_cfg = GridEncoderConfig(
        patch_size=data_cfg.get("patch_size", 4),
        latent_dim=latent_cfg.get("dim", 32),
        latent_len=latent_cfg.get("tokens", 16),
        field_channels={field_name: channels},
    )
    return GridEncoder(encoder_cfg)


def make_decoder(cfg: dict[str, Any]) -> AnyPointDecoder:
    _, channels, field_name = _pdebench_codec_context(cfg)
    latent_dim = cfg.get("latent", {}).get("dim", 32)
    decoder_cfg = cfg.get("decoder", {})
    hidden_dim = decoder_cfg.get("hidden_dim", max(latent_dim * 2, 64))
    return AnyPointDecoder(
        AnyPointDecoderConfig(
            latent_dim=latent_dim,
            query_dim=2,
            hidden_dim=hidden_dim,
            num_layers=decoder_cfg.get("num_layers", 2),
            num_heads=decoder_cfg.get("num_heads", 4),
            frequencies=tuple(decoder_cfg.get("frequencies", (1.0, 2.0, 4.0))),
            mlp_hidden_dim=decoder_cfg.get("mlp_hidden_dim", hidden_dim),
            output_channels={field_name: channels},
        )
    )


def _flatten_field_step(field_step: torch.Tensor, grid_shape: tuple[int, int]) -> torch.Tensor:
    H, W = grid_shape
    if field_step.dim() == 3:
        if tuple(field_step.shape[-2:]) == (H, W):
            data = field_step
        elif tuple(field_step.shape[:2]) == (H, W):
            data = field_step.permute(2, 0, 1)
        else:
            if field_step.shape[-1] <= 8:
                data = field_step.permute(2, 0, 1)
            else:
                data = field_step
    elif field_step.dim() == 2:
        if tuple(field_step.shape) == (H, W):
            data = field_step.unsqueeze(0)
        else:
            data = field_step.unsqueeze(0).unsqueeze(1)
    elif field_step.dim() == 1:
        data = field_step.unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError(f"Unsupported field step shape {tuple(field_step.shape)}")

    return data.contiguous().view(1, data.shape[0], H * W).transpose(1, 2)


def _encode_sequence_batch(
    encoder: GridEncoder,
    fields: torch.Tensor,
    coords: torch.Tensor,
    grid_shape: tuple[int, int],
    *,
    field_name: str,
    device: torch.device,
) -> torch.Tensor:
    flat_steps = []
    for batch_idx in range(fields.shape[0]):
        for step_idx in range(fields.shape[1]):
            flat_steps.append(_flatten_field_step(fields[batch_idx, step_idx], grid_shape))
    flattened = torch.cat(flat_steps, dim=0).to(device)
    coords_batch = coords.expand(flattened.shape[0], -1, -1)
    with torch.no_grad():
        latent = encoder({field_name: flattened}, coords_batch, meta={"grid_shape": grid_shape})
    return latent


def _materialize_encoder(encoder: GridEncoder, cfg: dict[str, Any], device: torch.device) -> None:
    dataset, grid_shape, _, field_name = _pdebench_grid_spec(cfg)
    sample = dataset[0]["fields"].float()
    if sample.dim() >= 2:
        field_step = sample[0]
    else:
        field_step = sample
    flattened = _flatten_field_step(field_step, grid_shape).to(device)
    coords = make_grid_coords(grid_shape, device)
    with torch.no_grad():
        encoder.to(device)
        encoder({field_name: flattened}, coords, meta={"grid_shape": grid_shape})


def _freeze_initialized_parameters(module: nn.Module) -> None:
    for param in module.parameters():
        if isinstance(param, UninitializedParameter):
            continue
        param.requires_grad_(False)


def _flatten_field_batch(fields: torch.Tensor, grid_shape: tuple[int, int]) -> torch.Tensor:
    flat_steps = []
    for batch_idx in range(fields.shape[0]):
        for step_idx in range(fields.shape[1]):
            flat_steps.append(_flatten_field_step(fields[batch_idx, step_idx], grid_shape))
    flat = torch.cat(flat_steps, dim=0)
    return flat.view(fields.shape[0], fields.shape[1], flat.shape[1], flat.shape[2])


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
    conditioning = None
    conditioning_cfg = cfg.get("operator", {}).get("conditioning", {})
    sources = conditioning_cfg.get("sources")
    if sources:
        conditioning = ConditioningConfig(
            latent_dim=dim,
            hidden_dim=int(conditioning_cfg.get("hidden_dim", max(dim * 2, 64))),
            sources={str(key): int(value) for key, value in sources.items()},
        )
    elif bool(cfg.get("training", {}).get("auto_conditioning", False)):
        auto_sources = _auto_conditioning_sources(cfg)
        conditioning = ConditioningConfig(
            latent_dim=dim,
            hidden_dim=int(conditioning_cfg.get("hidden_dim", max(dim * 2, 64))),
            sources=auto_sources,
        )

    config = LatentOperatorConfig(
        latent_dim=dim,
        pdet=PDETransformerConfig(**pdet_cfg),
        conditioning=conditioning,
        time_embed_dim=dim,
    )
    return LatentOperator(config)


def _grid_structured_conditioning(
    cfg: dict[str, Any],
    *,
    grid_shape: tuple[int, int],
    batch_size: int,
    device: torch.device,
    task_name: str | None = None,
    params: dict[str, Any] | None = None,
    bc: dict[str, Any] | None = None,
    step: int = 0,
) -> dict[str, torch.Tensor]:
    if not bool(cfg.get("training", {}).get("auto_conditioning", False)):
        return {}
    data_cfg = cfg.get("data", {})
    task = data_cfg.get("task")
    resolved_task = (
        task_name if task_name is not None else (task if isinstance(task, str) else None)
    )
    task_vocab = tuple(str(name) for name in task) if isinstance(task, (list, tuple)) else None
    extras = pdebench_conditioning_extras(
        task_name=resolved_task, grid_shape=grid_shape, task_vocab=task_vocab
    )
    cond = pdebench_condition_step(
        params,
        bc,
        batch_size=batch_size,
        step=step,
        extras=extras,
        param_vocab=tuple(data_cfg.get("param_keys", ())),
        bc_vocab=tuple(data_cfg.get("bc_keys", ())),
    )
    return {key: value.to(device) for key, value in cond.items()}


class _NamedPDEBenchDataset(Dataset):
    def __init__(self, base: PDEBenchDataset, task_name: str) -> None:
        self.base = base
        self.task_name = task_name

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = dict(self.base[idx])
        sample["task_name"] = self.task_name
        return sample


def _raw_pdebench_dataset(cfg: dict[str, Any]) -> tuple[Dataset, str]:
    specs, _, field_name = _pdebench_codec_context(cfg)
    datasets = [_NamedPDEBenchDataset(spec["dataset"], str(spec["task"])) for spec in specs]
    if len(datasets) == 1:
        return datasets[0], field_name
    return ConcatDataset(datasets), field_name


def _task_name_from_batch(batch: dict[str, Any]) -> str | None:
    task_name = batch.get("task_name")
    if isinstance(task_name, str):
        return task_name
    if isinstance(task_name, (list, tuple)) and task_name:
        return str(task_name[0])
    return None


def _raw_collate(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return batch


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
        return torch.optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
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


def _amp_enabled(cfg: dict) -> bool:
    return bool(cfg.get("training", {}).get("amp", False)) and torch.cuda.is_available()


def _autocast(device: torch.device, enabled: bool):
    if not hasattr(torch, "amp") or not hasattr(torch.amp, "autocast"):
        if device.type == "cuda":
            return torch.cuda.amp.autocast(enabled=enabled)
        from contextlib import nullcontext

        return nullcontext()
    return torch.amp.autocast(device_type=device.type, enabled=enabled)


def _grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _maybe_compile(model: nn.Module, cfg: dict, name: str) -> nn.Module:
    """Optionally compile a model with torch.compile when enabled and available.

    Controlled by training.compile bool. Falls back silently if unavailable.
    """
    try:
        compile_enabled = bool(cfg.get("training", {}).get("compile", False))
    except Exception:
        compile_enabled = False
    if not compile_enabled:
        return model
    try:
        import torch

        # Reduce overhead mode is a good default for training loops
        compiled = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        return compiled
    except Exception:
        # If torch.compile is unavailable or fails, just return the original model
        return model


def _grad_clip_value(cfg: dict, stage: str) -> float | None:
    # Stage-specific override takes precedence; fallback to training.grad_clip
    stage_cfg = cfg.get("stages", {}).get(stage, {}) if isinstance(cfg.get("stages"), dict) else {}
    if "grad_clip" in stage_cfg:
        return stage_cfg.get("grad_clip")
    return cfg.get("training", {}).get("grad_clip")


def _get_ema_decay(cfg: dict, stage: str) -> float | None:
    stage_cfg = cfg.get("stages", {}).get(stage, {}) if isinstance(cfg.get("stages"), dict) else {}
    if "ema_decay" in stage_cfg:
        return stage_cfg.get("ema_decay")
    return cfg.get("training", {}).get("ema_decay")


def _semigroup_loss_from_batch(
    operator: nn.Module,
    batch,
    device: torch.device,
    dt_tensor: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(batch, dict):
        return torch.tensor(0.0, device=device)
    z_seq = batch.get("z_seq")
    seq_lens = batch.get("seq_lens")
    if z_seq is None or seq_lens is None:
        return torch.tensor(0.0, device=device)

    z_seq = z_seq.to(device)
    seq_lens = seq_lens.to(device=device, dtype=torch.long)
    if z_seq.dim() != 4 or z_seq.shape[1] < 3:
        return z_seq.new_tensor(0.0)

    valid = torch.arange(z_seq.shape[1] - 2, device=device).unsqueeze(0) < (seq_lens - 2).unsqueeze(
        1
    )
    if not torch.any(valid):
        return z_seq.new_tensor(0.0)

    start = z_seq[:, :-2]
    target = z_seq[:, 2:]
    valid_flat = valid.reshape(-1)
    start_flat = start.reshape(-1, *start.shape[2:])[valid_flat]
    target_flat = target.reshape(-1, *target.shape[2:])[valid_flat]

    state0 = LatentState(z=start_flat, t=torch.tensor(0.0, device=device), cond={})
    mid = operator(state0, dt_tensor).z
    composed = operator(LatentState(z=mid, t=dt_tensor, cond={}), dt_tensor).z
    direct = operator(state0, dt_tensor * 2.0).z

    consistency = semigroup_consistency_loss(direct, composed)
    target_fit = 0.5 * (F.mse_loss(direct, target_flat) + F.mse_loss(composed, target_flat))
    return consistency + target_fit


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


def _get_patience(cfg: dict, stage: str) -> int | None:
    stage_cfg = cfg.get("stages", {}).get(stage, {})
    if "patience" in stage_cfg:
        return stage_cfg["patience"]
    training_cfg = cfg.get("training", {})
    return training_cfg.get("patience")


def _should_stop(patience: int | None, epochs_since_improve: int) -> bool:
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
    train_cfg = cfg.get("training", {})
    lam_semigroup = float(train_cfg.get("lambda_semigroup", 0.0) or 0.0)
    loader_cfg = cfg
    export_encoder = None
    checkpoint_dir = ensure_checkpoint_dir(cfg)
    if _supports_grid_codec(cfg):
        export_encoder = make_encoder(cfg)
        _materialize_encoder(
            export_encoder, cfg, torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
    if lam_semigroup > 0.0 and not bool(train_cfg.get("preserve_sequences", False)):
        loader_cfg = copy.deepcopy(cfg)
        loader_cfg.setdefault("training", {})["preserve_sequences"] = True
    loader = dataset_loader(loader_cfg, encoder_override=export_encoder)
    operator = make_operator(cfg)
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
    operator = _maybe_compile(operator, cfg, "operator")
    dt_tensor = torch.tensor(dt, device=device)
    best_loss = float("inf")
    best_state = copy.deepcopy(operator.state_dict())
    # AMP + EMA setup
    use_amp = _amp_enabled(cfg)
    scaler = _grad_scaler(use_amp)
    ema_decay = _get_ema_decay(cfg, "operator")
    ema_model = _init_ema(operator) if ema_decay else None
    clip_val = _grad_clip_value(cfg, "operator")
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
        num_batches = len(loader)
        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(loader):
            z0, z1, cond = unpack_batch(batch)
            cond_device = {k: v.to(device) for k, v in cond.items()}
            state = LatentState(
                z=z0.to(device), t=torch.tensor(0.0, device=device), cond=cond_device
            )
            target = z1.to(device)
            try:
                with _autocast(device, use_amp):
                    next_state = operator(state, dt_tensor)
                    base = F.mse_loss(next_state.z, target)
                    extra = 0.0
                    if lam_spec > 0.0:
                        extra = extra + lam_spec * _spectral_energy_loss(
                            next_state.z, target, dim=1
                        )
                    if lam_rel > 0.0:
                        extra = extra + lam_rel * _nrmse(next_state.z, target)
                    if lam_semigroup > 0.0:
                        extra = extra + lam_semigroup * _semigroup_loss_from_batch(
                            operator, batch, device, dt_tensor
                        )
                    loss = base + extra
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
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        operator.parameters(), float("inf") if clip_val is None else clip_val
                    )
                    total_grad_norm += float(grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        operator.parameters(), float("inf") if clip_val is None else clip_val
                    )
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
    operator_path = checkpoint_dir / "operator.pt"
    torch.save(operator.state_dict(), operator_path)
    print(f"Saved operator checkpoint to {operator_path}")
    if export_encoder is not None:
        encoder_path = checkpoint_dir / "encoder.pt"
        torch.save(export_encoder.state_dict(), encoder_path)
        print(f"Saved encoder checkpoint to {encoder_path}")
    if ema_model is not None:
        operator_ema_path = checkpoint_dir / "operator_ema.pt"
        torch.save(ema_model.state_dict(), operator_ema_path)
        print(f"Saved operator EMA checkpoint to {operator_ema_path}")

    # Upload checkpoint to W&B
    if wandb is not None and wandb.run is not None:
        wandb.save(str(operator_path), base_path=str(checkpoint_dir.parent))
        print("Uploaded operator checkpoint to W&B")

    # Send W&B alert
    if wandb is not None and wandb.run is not None:
        try:
            wandb.alert(
                title="✅ Operator Training Complete",
                text=f"Final loss: {best_loss:.6f} | Ready for the next training stage",
                level=wandb.AlertLevel.INFO,
            )
        except Exception:
            pass


def train_decoder(cfg: dict, shared_run=None, global_step: int = 0) -> None:
    if not _supports_grid_codec(cfg):
        raise ValueError("Decoder training currently supports single-task PDEBench grid data only")

    checkpoint_dir = ensure_checkpoint_dir(cfg)
    encoder_path = checkpoint_dir / "encoder.pt"
    if not encoder_path.exists():
        raise FileNotFoundError(
            f"Decoder training requires {encoder_path}. Run operator training first so the latent encoder is checkpointed."
        )

    dataset, field_name = _raw_pdebench_dataset(cfg)
    decoder = make_decoder(cfg)
    encoder = make_encoder(cfg)
    encoder.load_state_dict(torch.load(encoder_path, map_location="cpu", weights_only=False))
    encoder.eval()
    _freeze_initialized_parameters(encoder)

    stage_cfg = cfg.get("stages", {}).get("decoder", {})
    epochs = stage_cfg.get("epochs", 0)
    if epochs <= 0:
        print("Skipping decoder stage (epochs<=0)")
        return

    optimizer = _create_optimizer(cfg, decoder, "decoder")
    scheduler = _create_scheduler(optimizer, cfg, "decoder")
    patience = _get_patience(cfg, "decoder")
    logger = TrainingLogger(cfg, stage="decoder", global_step=global_step, shared_run=shared_run)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    encoder.to(device)
    decoder.to(device)

    is_multitask = isinstance(cfg.get("data", {}).get("task"), (list, tuple))
    loader = DataLoader(
        dataset,
        batch_size=cfg.get("training", {}).get("batch_size", 16),
        shuffle=True,
        collate_fn=_raw_collate if is_multitask else None,
    )
    best_loss = float("inf")
    best_state = copy.deepcopy(decoder.state_dict())
    epochs_since_improve = 0

    import time

    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        batches = 0

        for batch in loader:
            if isinstance(batch, list):
                sample_losses = []
                for sample in batch:
                    fields = sample["fields"].float().unsqueeze(0)
                    params = sample.get("params")
                    bc = sample.get("bc")
                    grid_shape = infer_grid_shape(sample["fields"].float())
                    coords = make_grid_coords(grid_shape, device)
                    latent = _encode_sequence_batch(
                        encoder,
                        fields,
                        coords,
                        grid_shape,
                        field_name=field_name,
                        device=device,
                    )
                    flat_targets = []
                    for batch_idx in range(fields.shape[0]):
                        for step_idx in range(fields.shape[1]):
                            flat_targets.append(
                                _flatten_field_step(fields[batch_idx, step_idx], grid_shape)
                            )
                    targets = torch.cat(flat_targets, dim=0).to(device)
                    coords_batch = coords.expand(targets.shape[0], -1, -1)
                    decoded = decoder(coords_batch, latent, conditioning={})
                    sample_losses.append(F.mse_loss(decoded[field_name], targets))

                if not sample_losses:
                    continue
                loss = torch.stack(sample_losses).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                batches += 1
                continue

            fields = batch["fields"].float()
            params = batch.get("params")
            bc = batch.get("bc")
            grid_shape = infer_grid_shape(fields[0])
            coords = make_grid_coords(grid_shape, device)
            latent = _encode_sequence_batch(
                encoder,
                fields,
                coords,
                grid_shape,
                field_name=field_name,
                device=device,
            )
            flat_targets = []
            for batch_idx in range(fields.shape[0]):
                for step_idx in range(fields.shape[1]):
                    flat_targets.append(
                        _flatten_field_step(fields[batch_idx, step_idx], grid_shape)
                    )
            targets = torch.cat(flat_targets, dim=0).to(device)
            coords_batch = coords.expand(targets.shape[0], -1, -1)

            decoded = decoder(coords_batch, latent, conditioning={})
            loss = F.mse_loss(decoded[field_name], targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1

        epoch_time = time.time() - epoch_start
        mean_loss = epoch_loss / max(batches, 1)
        logger.log(
            epoch=epoch,
            loss=mean_loss,
            optimizer=optimizer,
            patience_counter=epochs_since_improve,
            grad_norm=None,
            epoch_time=epoch_time,
            best_loss=best_loss,
        )
        if mean_loss + 1e-6 < best_loss:
            best_loss = mean_loss
            best_state = copy.deepcopy(decoder.state_dict())
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

    decoder.load_state_dict(best_state)
    logger.close()
    decoder_path = checkpoint_dir / "decoder.pt"
    torch.save(decoder.state_dict(), decoder_path)
    print(f"Saved decoder checkpoint to {decoder_path}")
    if wandb is not None and wandb.run is not None:
        wandb.save(str(decoder_path), base_path=str(checkpoint_dir.parent))
        print("Uploaded decoder checkpoint to W&B")


def train_operator_decoded(cfg: dict, shared_run=None, global_step: int = 0) -> None:
    if not _supports_grid_codec(cfg):
        raise ValueError(
            "Decoded operator fine-tuning currently supports single-task PDEBench grid data only"
        )

    checkpoint_dir = ensure_checkpoint_dir(cfg)
    operator_path = checkpoint_dir / "operator.pt"
    encoder_path = checkpoint_dir / "encoder.pt"
    decoder_path = checkpoint_dir / "decoder.pt"
    for required in (operator_path, encoder_path, decoder_path):
        if not required.exists():
            raise FileNotFoundError(f"Decoded operator fine-tuning requires checkpoint: {required}")

    stage_cfg = cfg.get("stages", {}).get("operator_decoded", {})
    epochs = int(stage_cfg.get("epochs", 0) or 0)
    if epochs <= 0:
        print("Skipping operator_decoded stage (epochs<=0)")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    operator = make_operator(cfg)
    operator.load_state_dict(torch.load(operator_path, map_location="cpu", weights_only=False))
    operator.to(device)

    encoder = make_encoder(cfg)
    encoder.load_state_dict(torch.load(encoder_path, map_location="cpu", weights_only=False))
    encoder.to(device)
    encoder.eval()
    _freeze_initialized_parameters(encoder)

    decoder = make_decoder(cfg)
    decoder.load_state_dict(torch.load(decoder_path, map_location="cpu", weights_only=False))
    decoder.to(device)
    decoder.eval()
    _freeze_initialized_parameters(decoder)

    dataset, field_name = _raw_pdebench_dataset(cfg)
    is_multitask = isinstance(cfg.get("data", {}).get("task"), (list, tuple))
    loader = DataLoader(
        dataset,
        batch_size=cfg.get("training", {}).get("batch_size", 16),
        shuffle=True,
        collate_fn=_raw_collate if is_multitask else None,
    )
    optimizer = _create_optimizer(cfg, operator, "operator_decoded")
    scheduler = _create_scheduler(optimizer, cfg, "operator_decoded")
    patience = _get_patience(cfg, "operator_decoded")
    logger = TrainingLogger(
        cfg, stage="operator_decoded", global_step=global_step, shared_run=shared_run
    )
    dt = cfg.get("training", {}).get("dt", 0.1)
    dt_tensor = torch.tensor(dt, device=device)
    rollout_steps = int(stage_cfg.get("rollout_steps", 1) or 1)
    lambda_rollout = float(stage_cfg.get("lambda_rollout", 1.0) or 1.0)

    best_loss = float("inf")
    best_state = copy.deepcopy(operator.state_dict())
    epochs_since_improve = 0

    import time

    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        batches = 0
        for batch in loader:
            if isinstance(batch, list):
                sample_losses = []
                for sample in batch:
                    fields = sample["fields"].float().unsqueeze(0)
                    if fields.shape[1] < 2:
                        continue
                    task_name = sample.get("task_name")
                    params = sample.get("params")
                    bc = sample.get("bc")
                    grid_shape = infer_grid_shape(sample["fields"].float())
                    coords = make_grid_coords(grid_shape, device)
                    rollout_fields, rollout_start = _decoded_rollout_training_window(
                        fields, rollout_steps=rollout_steps, stage_cfg=stage_cfg
                    )
                    max_steps = rollout_fields.shape[1] - 1
                    latent = _encode_sequence_batch(
                        encoder,
                        rollout_fields[:, :1],
                        coords,
                        grid_shape,
                        field_name=field_name,
                        device=device,
                    )
                    targets = _flatten_field_batch(rollout_fields, grid_shape).to(device)
                    state = LatentState(
                        z=latent,
                        t=torch.tensor(0.0, device=device),
                        cond=_grid_structured_conditioning(
                            cfg,
                            grid_shape=grid_shape,
                            batch_size=1,
                            device=device,
                            task_name=task_name,
                            params=params,
                            bc=bc,
                            step=rollout_start,
                        ),
                    )
                    coords_batch = coords.expand(1, -1, -1)
                    decoded_losses = []
                    for step in range(1, max_steps + 1):
                        cond = _grid_structured_conditioning(
                            cfg,
                            grid_shape=grid_shape,
                            batch_size=1,
                            device=device,
                            task_name=task_name,
                            params=params,
                            bc=bc,
                            step=rollout_start + step - 1,
                        )
                        state = LatentState(z=state.z, t=state.t, cond=cond)
                        state = operator(state, dt_tensor)
                        decoded = decoder(coords_batch, state.z, conditioning={})
                        decoded_losses.append(
                            _decoded_field_loss(
                                decoded[field_name],
                                targets[:, step],
                                targets[:, step - 1],
                                stage_cfg=stage_cfg,
                            )
                        )
                    if not decoded_losses:
                        continue
                    sample_loss = _decoded_rollout_training_loss(
                        decoded_losses,
                        stage_cfg=stage_cfg,
                        lambda_rollout=lambda_rollout,
                    )
                    sample_loss = sample_loss * _task_loss_weight(stage_cfg, task_name)
                    sample_losses.append(sample_loss)

                if not sample_losses:
                    continue
                loss = torch.stack(sample_losses).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                batches += 1
                continue

            fields = batch["fields"].float()
            params = batch.get("params")
            bc = batch.get("bc")
            if fields.shape[1] < 2:
                continue
            task_name = _task_name_from_batch(batch)
            grid_shape = infer_grid_shape(fields[0])
            coords = make_grid_coords(grid_shape, device)
            rollout_fields, rollout_start = _decoded_rollout_training_window(
                fields, rollout_steps=rollout_steps, stage_cfg=stage_cfg
            )
            max_steps = rollout_fields.shape[1] - 1
            latent = _encode_sequence_batch(
                encoder,
                rollout_fields[:, :1],
                coords,
                grid_shape,
                field_name=field_name,
                device=device,
            )
            targets = _flatten_field_batch(rollout_fields, grid_shape).to(device)
            state = LatentState(
                z=latent,
                t=torch.tensor(0.0, device=device),
                cond=_grid_structured_conditioning(
                    cfg,
                    grid_shape=grid_shape,
                    batch_size=fields.shape[0],
                    device=device,
                    task_name=task_name,
                    params=params,
                    bc=bc,
                    step=rollout_start,
                ),
            )
            coords_batch = coords.expand(fields.shape[0], -1, -1)

            decoded_losses = []
            for step in range(1, max_steps + 1):
                cond = _grid_structured_conditioning(
                    cfg,
                    grid_shape=grid_shape,
                    batch_size=fields.shape[0],
                    device=device,
                    task_name=task_name,
                    params=params,
                    bc=bc,
                    step=rollout_start + step - 1,
                )
                state = LatentState(z=state.z, t=state.t, cond=cond)
                state = operator(state, dt_tensor)
                decoded = decoder(coords_batch, state.z, conditioning={})
                decoded_losses.append(
                    _decoded_field_loss(
                        decoded[field_name],
                        targets[:, step],
                        targets[:, step - 1],
                        stage_cfg=stage_cfg,
                    )
                )

            if not decoded_losses:
                continue

            loss = _decoded_rollout_training_loss(
                decoded_losses,
                stage_cfg=stage_cfg,
                lambda_rollout=lambda_rollout,
            )
            loss = loss * _task_loss_weight(stage_cfg, task_name)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1

        epoch_time = time.time() - epoch_start
        mean_loss = epoch_loss / max(batches, 1)
        logger.log(
            epoch=epoch,
            loss=mean_loss,
            optimizer=optimizer,
            patience_counter=epochs_since_improve,
            grad_norm=None,
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
    decoded_operator_path = checkpoint_dir / "operator_decoded.pt"
    torch.save(operator.state_dict(), decoded_operator_path)
    torch.save(operator.state_dict(), operator_path)
    print(f"Saved decoded-finetuned operator checkpoint to {decoded_operator_path}")
    if wandb is not None and wandb.run is not None:
        wandb.save(str(decoded_operator_path), base_path=str(checkpoint_dir.parent))
        wandb.save(str(operator_path), base_path=str(checkpoint_dir.parent))


def train_joint_codec_operator(cfg: dict, shared_run=None, global_step: int = 0) -> None:
    if not _supports_grid_codec(cfg):
        raise ValueError(
            "Joint codec/operator training currently supports single-task PDEBench grid data only"
        )

    checkpoint_dir = ensure_checkpoint_dir(cfg)
    operator_path = checkpoint_dir / "operator.pt"
    encoder_path = checkpoint_dir / "encoder.pt"
    decoder_path = checkpoint_dir / "decoder.pt"
    for required in (operator_path, encoder_path, decoder_path):
        if not required.exists():
            raise FileNotFoundError(
                f"Joint codec/operator training requires checkpoint: {required}"
            )

    stage_cfg = cfg.get("stages", {}).get("joint_codec_operator", {})
    epochs = int(stage_cfg.get("epochs", 0) or 0)
    if epochs <= 0:
        print("Skipping joint_codec_operator stage (epochs<=0)")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    operator = make_operator(cfg)
    operator.load_state_dict(torch.load(operator_path, map_location="cpu", weights_only=False))
    operator.to(device)

    encoder = make_encoder(cfg)
    encoder.load_state_dict(torch.load(encoder_path, map_location="cpu", weights_only=False))
    encoder.to(device)

    decoder = make_decoder(cfg)
    decoder.load_state_dict(torch.load(decoder_path, map_location="cpu", weights_only=False))
    decoder.to(device)

    joint_model = nn.ModuleDict({"operator": operator, "encoder": encoder, "decoder": decoder})
    dataset, field_name = _raw_pdebench_dataset(cfg)
    is_multitask = isinstance(cfg.get("data", {}).get("task"), (list, tuple))
    loader = DataLoader(
        dataset,
        batch_size=cfg.get("training", {}).get("batch_size", 16),
        shuffle=True,
        collate_fn=_raw_collate if is_multitask else None,
    )
    optimizer = _create_optimizer(cfg, joint_model, "joint_codec_operator")
    scheduler = _create_scheduler(optimizer, cfg, "joint_codec_operator")
    patience = _get_patience(cfg, "joint_codec_operator")
    logger = TrainingLogger(
        cfg, stage="joint_codec_operator", global_step=global_step, shared_run=shared_run
    )
    dt = cfg.get("training", {}).get("dt", 0.1)
    dt_tensor = torch.tensor(dt, device=device)
    rollout_steps = int(stage_cfg.get("rollout_steps", 1) or 1)
    lambda_rollout = float(stage_cfg.get("lambda_rollout", 1.0) or 1.0)
    lambda_reconstruction = float(stage_cfg.get("lambda_reconstruction", 0.0) or 0.0)

    best_loss = float("inf")
    best_states = {
        "operator": copy.deepcopy(operator.state_dict()),
        "encoder": copy.deepcopy(encoder.state_dict()),
        "decoder": copy.deepcopy(decoder.state_dict()),
    }
    epochs_since_improve = 0
    use_amp = _amp_enabled(cfg)
    scaler = _grad_scaler(use_amp)
    clip_val = _grad_clip_value(cfg, "joint_codec_operator")

    import time

    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        total_grad_norm = 0.0
        batches = 0

        for batch in loader:
            if isinstance(batch, list):
                sample_losses = []
                for sample in batch:
                    fields = sample["fields"].float().unsqueeze(0)
                    task_name = sample.get("task_name")
                    params = sample.get("params")
                    bc = sample.get("bc")
                    grid_shape = infer_grid_shape(sample["fields"].float())
                    coords = make_grid_coords(grid_shape, device)
                    rollout_fields, rollout_start = _decoded_rollout_training_window(
                        fields, rollout_steps=rollout_steps, stage_cfg=stage_cfg
                    )
                    max_steps = rollout_fields.shape[1] - 1
                    targets = _flatten_field_batch(rollout_fields, grid_shape).to(device)
                    coords_batch = coords.expand(1, -1, -1)
                    structured_cond = _grid_structured_conditioning(
                        cfg,
                        grid_shape=grid_shape,
                        batch_size=1,
                        device=device,
                        task_name=task_name,
                        params=params,
                        bc=bc,
                        step=rollout_start,
                    )
                    try:
                        with _autocast(device, use_amp):
                            latent0 = encoder(
                                {field_name: targets[:, 0]},
                                coords_batch,
                                meta={"grid_shape": grid_shape},
                            )
                            losses = []
                            if lambda_reconstruction > 0.0:
                                reconstructed = decoder(coords_batch, latent0, conditioning={})
                                losses.append(
                                    lambda_reconstruction
                                    * F.mse_loss(reconstructed[field_name], targets[:, 0])
                                )

                            state = LatentState(
                                z=latent0, t=torch.tensor(0.0, device=device), cond=structured_cond
                            )
                            decoded_losses = []
                            for step in range(1, max_steps + 1):
                                cond = _grid_structured_conditioning(
                                    cfg,
                                    grid_shape=grid_shape,
                                    batch_size=1,
                                    device=device,
                                    task_name=task_name,
                                    params=params,
                                    bc=bc,
                                    step=rollout_start + step - 1,
                                )
                                state = LatentState(z=state.z, t=state.t, cond=cond)
                                state = operator(state, dt_tensor)
                                decoded = decoder(coords_batch, state.z, conditioning={})
                                decoded_losses.append(
                                    _decoded_field_loss(
                                        decoded[field_name],
                                        targets[:, step],
                                        targets[:, step - 1],
                                        stage_cfg=stage_cfg,
                                    )
                                )

                            if decoded_losses:
                                losses.append(
                                    _decoded_rollout_training_loss(
                                        decoded_losses,
                                        stage_cfg=stage_cfg,
                                        lambda_rollout=lambda_rollout,
                                    )
                                )

                            if losses:
                                sample_loss = torch.stack(losses).sum()
                                sample_loss = sample_loss * _task_loss_weight(stage_cfg, task_name)
                                sample_losses.append(sample_loss)
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            print(
                                "Warning: OOM encountered in joint codec/operator step, skipping sample"
                            )
                            continue
                        raise

                if not sample_losses:
                    continue
                loss = torch.stack(sample_losses).mean()
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(loss).backward()
                    if clip_val is not None:
                        scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        joint_model.parameters(), float("inf") if clip_val is None else clip_val
                    )
                    total_grad_norm += float(grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        joint_model.parameters(), float("inf") if clip_val is None else clip_val
                    )
                    total_grad_norm += grad_norm.item()
                    optimizer.step()
                epoch_loss += loss.detach().item()
                batches += 1
                continue

            fields = batch["fields"].float()
            task_name = _task_name_from_batch(batch)
            params = batch.get("params")
            bc = batch.get("bc")
            grid_shape = infer_grid_shape(fields[0])
            coords = make_grid_coords(grid_shape, device)
            rollout_fields, rollout_start = _decoded_rollout_training_window(
                fields, rollout_steps=rollout_steps, stage_cfg=stage_cfg
            )
            max_steps = rollout_fields.shape[1] - 1
            targets = _flatten_field_batch(rollout_fields, grid_shape).to(device)
            coords_batch = coords.expand(fields.shape[0], -1, -1)
            structured_cond = _grid_structured_conditioning(
                cfg,
                grid_shape=grid_shape,
                batch_size=fields.shape[0],
                device=device,
                task_name=task_name,
                params=params,
                bc=bc,
                step=rollout_start,
            )

            try:
                with _autocast(device, use_amp):
                    latent0 = encoder(
                        {field_name: targets[:, 0]}, coords_batch, meta={"grid_shape": grid_shape}
                    )
                    losses = []
                    if lambda_reconstruction > 0.0:
                        reconstructed = decoder(coords_batch, latent0, conditioning={})
                        losses.append(
                            lambda_reconstruction
                            * F.mse_loss(reconstructed[field_name], targets[:, 0])
                        )

                    state = LatentState(
                        z=latent0, t=torch.tensor(0.0, device=device), cond=structured_cond
                    )
                    decoded_losses = []
                    for step in range(1, max_steps + 1):
                        cond = _grid_structured_conditioning(
                            cfg,
                            grid_shape=grid_shape,
                            batch_size=fields.shape[0],
                            device=device,
                            task_name=task_name,
                            params=params,
                            bc=bc,
                            step=rollout_start + step - 1,
                        )
                        state = LatentState(z=state.z, t=state.t, cond=cond)
                        state = operator(state, dt_tensor)
                        decoded = decoder(coords_batch, state.z, conditioning={})
                        decoded_losses.append(
                            _decoded_field_loss(
                                decoded[field_name],
                                targets[:, step],
                                targets[:, step - 1],
                                stage_cfg=stage_cfg,
                            )
                        )

                    if decoded_losses:
                        losses.append(
                            _decoded_rollout_training_loss(
                                decoded_losses,
                                stage_cfg=stage_cfg,
                                lambda_rollout=lambda_rollout,
                            )
                        )

                    if not losses:
                        continue
                    loss = torch.stack(losses).sum()
                    loss = loss * _task_loss_weight(stage_cfg, task_name)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print("Warning: OOM encountered in joint codec/operator step, skipping batch")
                    continue
                raise

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss).backward()
                if clip_val is not None:
                    scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    joint_model.parameters(), float("inf") if clip_val is None else clip_val
                )
                total_grad_norm += float(grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    joint_model.parameters(), float("inf") if clip_val is None else clip_val
                )
                total_grad_norm += grad_norm.item()
                optimizer.step()

            epoch_loss += loss.detach().item()
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
            best_states = {
                "operator": copy.deepcopy(operator.state_dict()),
                "encoder": copy.deepcopy(encoder.state_dict()),
                "decoder": copy.deepcopy(decoder.state_dict()),
            }
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

    operator.load_state_dict(best_states["operator"])
    encoder.load_state_dict(best_states["encoder"])
    decoder.load_state_dict(best_states["decoder"])
    logger.close()

    joint_operator_path = checkpoint_dir / "operator_joint.pt"
    joint_encoder_path = checkpoint_dir / "encoder_joint.pt"
    joint_decoder_path = checkpoint_dir / "decoder_joint.pt"
    torch.save(operator.state_dict(), joint_operator_path)
    torch.save(encoder.state_dict(), joint_encoder_path)
    torch.save(decoder.state_dict(), joint_decoder_path)
    torch.save(operator.state_dict(), operator_path)
    torch.save(encoder.state_dict(), encoder_path)
    torch.save(decoder.state_dict(), decoder_path)
    print(f"Saved joint codec/operator checkpoints to {checkpoint_dir}")
    if wandb is not None and wandb.run is not None:
        wandb.save(str(joint_operator_path), base_path=str(checkpoint_dir.parent))
        wandb.save(str(joint_encoder_path), base_path=str(checkpoint_dir.parent))
        wandb.save(str(joint_decoder_path), base_path=str(checkpoint_dir.parent))
        wandb.save(str(operator_path), base_path=str(checkpoint_dir.parent))
        wandb.save(str(encoder_path), base_path=str(checkpoint_dir.parent))
        wandb.save(str(decoder_path), base_path=str(checkpoint_dir.parent))


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
        operator_state = torch.load(op_path, map_location="cpu")
        operator.load_state_dict(operator_state)
    _ensure_model_on_device(operator, device)
    operator = _maybe_compile(operator, cfg, "operator_teacher")
    operator.eval()

    latent_dim = cfg.get("latent", {}).get("dim", 32)
    stage_cfg = cfg.get("stages", {}).get("diff_residual", {})
    diff = DiffusionResidual(
        DiffusionResidualConfig(latent_dim=latent_dim, hidden_dim=latent_dim * 2)
    )
    _ensure_model_on_device(diff, device)
    diff = _maybe_compile(diff, cfg, "diffusion_residual")

    optimizer = _create_optimizer(cfg, diff, "diff_residual")
    scheduler = _create_scheduler(optimizer, cfg, "diff_residual")
    patience = _get_patience(cfg, "diff_residual")
    dt = cfg.get("training", {}).get("dt", 0.1)
    epochs = stage_cfg.get("epochs", 1)
    checkpoint_interval = int(cfg.get("training", {}).get("checkpoint_interval", 0) or 0)
    logger = TrainingLogger(
        cfg, stage="diffusion_residual", global_step=global_step, shared_run=shared_run
    )
    dt_tensor = torch.tensor(dt, device=device)
    best_loss = float("inf")
    best_state = copy.deepcopy(diff.state_dict())
    # AMP + EMA setup
    use_amp = _amp_enabled(cfg)
    scaler = _grad_scaler(use_amp)
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
            z0, z1, cond = unpack_batch(batch)
            cond_device = {k: v.to(device) for k, v in cond.items()}
            state = LatentState(
                z=z0.to(device), t=torch.tensor(0.0, device=device), cond=cond_device
            )
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
            tau_tensor = torch.rand(z0.size(0), device=device)
            try:
                with _autocast(device, use_amp):
                    drift = diff(predicted, tau_tensor)
                    base = F.mse_loss(drift, residual_target)
                    extra = 0.0
                    if lam_spec > 0.0:
                        extra = extra + lam_spec * _spectral_energy_loss(
                            drift, residual_target, dim=1
                        )
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
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        diff.parameters(), float("inf") if clip_val is None else clip_val
                    )
                    total_grad_norm += float(grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        diff.parameters(), float("inf") if clip_val is None else clip_val
                    )
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
        torch.save(
            best_ema_state if best_ema_state is not None else ema_model.state_dict(),
            diffusion_ema_path,
        )
        print(f"Saved diffusion EMA checkpoint to {diffusion_ema_path}")

    # Upload checkpoint to W&B
    if wandb is not None and wandb.run is not None:
        wandb.save(str(diffusion_path), base_path=str(checkpoint_dir.parent))
        print("Uploaded diffusion checkpoint to W&B")

    # Send W&B alert
    if wandb is not None and wandb.run is not None:
        try:
            wandb.alert(
                title="✅ Diffusion Residual Training Complete",
                text=f"Final loss: {best_loss:.6f} | Ready for consistency distillation",
                level=wandb.AlertLevel.INFO,
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
    consistency_batch_size = (
        cfg_copy.get("stages", {}).get("consistency_distill", {}).get("batch_size", 8)
    )
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
        operator_state = torch.load(op_path, map_location="cpu")
        operator.load_state_dict(operator_state)
    _ensure_model_on_device(operator, device)
    operator = _maybe_compile(operator, cfg, "operator_teacher")
    operator.eval()

    # Create diffusion model and load checkpoint directly to target device
    latent_dim = cfg.get("latent", {}).get("dim", 32)
    stage_cfg = cfg.get("stages", {}).get("consistency_distill", {})
    diff = DiffusionResidual(
        DiffusionResidualConfig(latent_dim=latent_dim, hidden_dim=latent_dim * 2)
    )
    diff_path = checkpoint_dir / "diffusion_residual.pt"
    if diff_path.exists():
        diff_state = torch.load(diff_path, map_location="cpu")
        diff.load_state_dict(diff_state)
    _ensure_model_on_device(diff, device)
    diff = _maybe_compile(diff, cfg, "diffusion_residual")

    epochs = stage_cfg.get("epochs", 1)
    optimizer = _create_optimizer(cfg, diff, "consistency_distill")
    scheduler = _create_scheduler(optimizer, cfg, "consistency_distill")
    patience = _get_patience(cfg, "consistency_distill")
    logger = TrainingLogger(
        cfg, stage="consistency_distill", global_step=global_step, shared_run=shared_run
    )
    dt = cfg.get("training", {}).get("dt", 0.1)

    dt_tensor = torch.tensor(dt, device=device)

    # Teacher/student are inlined below to enable reuse and vectorized taus

    best_loss = float("inf")
    best_state = copy.deepcopy(diff.state_dict())
    use_amp = _amp_enabled(cfg)
    scaler = _grad_scaler(use_amp)
    ema_decay = _get_ema_decay(cfg, "consistency_distill")
    ema_model = _init_ema(diff) if ema_decay else None
    clip_val = _grad_clip_value(cfg, "consistency_distill")
    epochs_since_improve = 0

    # Get micro-batch size for gradient accumulation
    distill_micro = cfg.get("training", {}).get("distill_micro_batch")
    num_taus = int(cfg.get("training", {}).get("distill_num_taus", 3) or 3)

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
            optimizer.zero_grad(set_to_none=True)
            batch_loss_value = 0.0
            for start in range(0, batch_size, micro):
                end = min(start + micro, batch_size)
                chunk_weight = (end - start) / batch_size
                z_chunk = z0[start:end].to(device)
                chunk_cond = {k: v[start:end].to(device) for k, v in cond.items()}
                state = LatentState(z=z_chunk, t=torch.tensor(0.0, device=device), cond=chunk_cond)
                try:
                    with torch.no_grad():
                        teacher_state = operator(state, dt_tensor)
                    Bc, T, D = teacher_state.z.shape
                    z_tiled = (
                        teacher_state.z.unsqueeze(1)
                        .expand(Bc, num_taus, T, D)
                        .reshape(Bc * num_taus, T, D)
                        .contiguous()
                    )
                    cond_tiled = {
                        k: v.repeat_interleave(num_taus, dim=0)
                        for k, v in teacher_state.cond.items()
                    }
                    tau_flat = torch.rand(num_taus, device=device).repeat(Bc)
                    tau_flat = tau_flat.to(z_tiled.dtype)
                    tiled_state = LatentState(z=z_tiled, t=teacher_state.t, cond=cond_tiled)
                    with _autocast(device, use_amp):
                        drift = diff(tiled_state, tau_flat)
                        z_tiled_cast = z_tiled.to(drift.dtype)
                        student_z = z_tiled_cast + drift
                        teacher_z = z_tiled_cast
                        loss_chunk = torch.nn.functional.mse_loss(student_z, teacher_z)
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
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    diff.parameters(), float("inf") if clip_val is None else clip_val
                )
                total_grad_norm += float(grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    diff.parameters(), float("inf") if clip_val is None else clip_val
                )
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
        print("Uploaded updated diffusion checkpoint to W&B")

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
                level=wandb.AlertLevel.INFO,
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
    prior = SteadyPrior(
        SteadyPriorConfig(latent_dim=latent_dim, hidden_dim=latent_dim * 2, num_steps=4)
    )
    optimizer = _create_optimizer(cfg, prior, "steady_prior")
    scheduler = _create_scheduler(optimizer, cfg, "steady_prior")
    patience = _get_patience(cfg, "steady_prior")
    logger = TrainingLogger(
        cfg, stage="steady_prior", global_step=global_step, shared_run=shared_run
    )
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
            state = LatentState(
                z=z0.to(device), t=torch.tensor(0.0, device=device), cond=cond_device
            )
            refined = prior(state)
            loss = F.mse_loss(refined.z, z1.to(device))
            (loss / accum_steps).backward()
            do_step = ((i + 1) % accum_steps == 0) or ((i + 1) == num_batches)
            if do_step:
                grad_norm = torch.nn.utils.clip_grad_norm_(prior.parameters(), float("inf"))
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
        print("Uploaded steady prior checkpoint to W&B")

    # Send W&B alert
    if wandb is not None and wandb.run is not None:
        try:
            wandb.alert(
                title="🎉 All Training Stages Complete!",
                text=f"Steady prior final loss: {best_loss:.6f} | Full pipeline ready for evaluation",
                level=wandb.AlertLevel.SUCCESS,
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
            entity=wandb_cfg.get("entity"),
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
            wandb.define_metric("decoder/*", step_metric="global_step")
            wandb.define_metric("operator_decoded/*", step_metric="global_step")
            wandb.define_metric("joint_codec_operator/*", step_metric="global_step")
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
        print("\n" + "=" * 50)
        print("STAGE 1/7: Training Operator")
        print("=" * 50)
        train_operator(cfg, shared_run=shared_run, global_step=global_step)
        global_step += op_epochs
    else:
        print("\n" + "=" * 50)
        print("STAGE 1/7: Skipping Operator (epochs<=0)")
        print("=" * 50)

    # Clear GPU cache between stages
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✓ Cleared GPU cache")

    # Stage 2: Decoder
    decoder_epochs = _stage_epochs(cfg, "decoder")
    if decoder_epochs > 0:
        print("\n" + "=" * 50)
        print("STAGE 2/7: Training Decoder")
        print("=" * 50)
        train_decoder(cfg, shared_run=shared_run, global_step=global_step)
        global_step += decoder_epochs
    else:
        print("\n" + "=" * 50)
        print("STAGE 2/7: Skipping Decoder (epochs<=0)")
        print("=" * 50)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✓ Cleared GPU cache")

    # Stage 3: Decoded Operator Fine-tune
    operator_decoded_epochs = _stage_epochs(cfg, "operator_decoded")
    if operator_decoded_epochs > 0:
        print("\n" + "=" * 50)
        print("STAGE 3/7: Decoded Operator Fine-tune")
        print("=" * 50)
        train_operator_decoded(cfg, shared_run=shared_run, global_step=global_step)
        global_step += operator_decoded_epochs
    else:
        print("\n" + "=" * 50)
        print("STAGE 3/7: Skipping Decoded Operator Fine-tune (epochs<=0)")
        print("=" * 50)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✓ Cleared GPU cache")

    # Stage 4: Joint Codec/Operator Fine-tune
    joint_epochs = _stage_epochs(cfg, "joint_codec_operator")
    if joint_epochs > 0:
        print("\n" + "=" * 50)
        print("STAGE 4/7: Joint Codec/Operator Fine-tune")
        print("=" * 50)
        train_joint_codec_operator(cfg, shared_run=shared_run, global_step=global_step)
        global_step += joint_epochs
    else:
        print("\n" + "=" * 50)
        print("STAGE 4/7: Skipping Joint Codec/Operator Fine-tune (epochs<=0)")
        print("=" * 50)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✓ Cleared GPU cache")

    # Stage 5: Diffusion Residual
    diff_epochs = _stage_epochs(cfg, "diff_residual")
    if diff_epochs > 0:
        print("\n" + "=" * 50)
        print("STAGE 5/7: Training Diffusion Residual")
        print("=" * 50)
        train_diffusion(cfg, shared_run=shared_run, global_step=global_step)
        global_step += diff_epochs
    else:
        print("\n" + "=" * 50)
        print("STAGE 5/7: Skipping Diffusion Residual (epochs<=0)")
        print("=" * 50)

    # Clear GPU cache between stages
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✓ Cleared GPU cache")

    # Stage 6: Consistency Distillation
    distill_epochs = _stage_epochs(cfg, "consistency_distill")
    if distill_epochs > 0:
        print("\n" + "=" * 50)
        print("STAGE 6/7: Consistency Distillation")
        print("=" * 50)
        train_consistency(cfg, shared_run=shared_run, global_step=global_step)
        global_step += distill_epochs
    else:
        print("\n" + "=" * 50)
        print("STAGE 6/7: Skipping Consistency Distillation (epochs<=0)")
        print("=" * 50)

    # Clear GPU cache between stages
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✓ Cleared GPU cache")

    # Stage 7: Steady Prior
    steady_epochs = _stage_epochs(cfg, "steady_prior")
    if steady_epochs > 0:
        print("\n" + "=" * 50)
        print("STAGE 7/7: Training Steady Prior")
        print("=" * 50)
        train_steady_prior(cfg, shared_run=shared_run, global_step=global_step)
    else:
        print("\n" + "=" * 50)
        print("STAGE 7/7: Skipping Steady Prior (epochs<=0)")
        print("=" * 50)

    # Log final summary
    if shared_run:
        # Load final checkpoints to get model sizes
        checkpoint_dir = ensure_checkpoint_dir(cfg)
        import os

        summary = {
            "summary/total_training_complete": 1,
            "summary/operator_checkpoint_size_mb": (
                os.path.getsize(checkpoint_dir / "operator.pt") / 1e6
                if (checkpoint_dir / "operator.pt").exists()
                else 0
            ),
            "summary/decoder_checkpoint_size_mb": (
                os.path.getsize(checkpoint_dir / "decoder.pt") / 1e6
                if (checkpoint_dir / "decoder.pt").exists()
                else 0
            ),
            "summary/operator_decoded_checkpoint_size_mb": (
                os.path.getsize(checkpoint_dir / "operator_decoded.pt") / 1e6
                if (checkpoint_dir / "operator_decoded.pt").exists()
                else 0
            ),
            "summary/operator_joint_checkpoint_size_mb": (
                os.path.getsize(checkpoint_dir / "operator_joint.pt") / 1e6
                if (checkpoint_dir / "operator_joint.pt").exists()
                else 0
            ),
            "summary/diffusion_checkpoint_size_mb": (
                os.path.getsize(checkpoint_dir / "diffusion_residual.pt") / 1e6
                if (checkpoint_dir / "diffusion_residual.pt").exists()
                else 0
            ),
            "summary/steady_prior_checkpoint_size_mb": (
                os.path.getsize(checkpoint_dir / "steady_prior.pt") / 1e6
                if (checkpoint_dir / "steady_prior.pt").exists()
                else 0
            ),
        }
        shared_run.log(summary)
        shared_run.finish()

    print("\n" + "=" * 50)
    print("✅ All training stages complete!")
    print("=" * 50)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_multi_pde.yaml")
    parser.add_argument(
        "--stage",
        choices=[
            "operator",
            "decoder",
            "operator_decoded",
            "joint_codec_operator",
            "diff_residual",
            "consistency_distill",
            "steady_prior",
            "all",
        ],
        required=True,
        help="Training stage to run, or 'all' to run full pipeline",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg)
    stage = args.stage

    if stage == "all":
        train_all_stages(cfg)
    elif stage == "operator":
        train_operator(cfg)
    elif stage == "decoder":
        train_decoder(cfg)
    elif stage == "operator_decoded":
        train_operator_decoded(cfg)
    elif stage == "joint_codec_operator":
        train_joint_codec_operator(cfg)
    elif stage == "diff_residual":
        train_diffusion(cfg)
    elif stage == "consistency_distill":
        train_consistency(cfg)
    elif stage == "steady_prior":
        train_steady_prior(cfg)


if __name__ == "__main__":
    main()
