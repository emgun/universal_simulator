from __future__ import annotations

"""PDEBench evaluation helpers."""

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import torch

from ups.core.latent_state import LatentState
from ups.data.latent_pairs import (
    build_latent_pair_loader,
    infer_grid_shape,
    make_grid_coords,
    pdebench_condition_step,
    pdebench_conditioning_extras,
    unpack_batch,
)
from ups.data.pdebench import PDEBenchConfig, PDEBenchDataset, get_pdebench_spec
from ups.eval.metrics import mae, mse, nrmse, relative_rrmse, spectral_energy_error
from ups.eval.reports import MetricReport
from ups.eval.reward_models import RewardModel
from ups.inference.rollout_ttc import TTCConfig, ttc_rollout
from ups.models.diffusion_residual import DiffusionResidual
from ups.models.latent_operator import LatentOperator


@dataclass
class BaselineModel:
    forward: Callable[[torch.Tensor], torch.Tensor]


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

    data = data.contiguous().view(1, data.shape[0], H * W).transpose(1, 2)
    return data


def _encode_grid_trajectory(
    encoder: Any,
    fields: torch.Tensor,
    coords: torch.Tensor,
    grid_shape: tuple[int, int],
    *,
    field_name: str,
    device: torch.device,
) -> torch.Tensor:
    latents: List[torch.Tensor] = []
    for step in range(fields.shape[0]):
        flattened = _flatten_field_step(fields[step], grid_shape).to(device)
        latent = encoder({field_name: flattened}, coords.to(device), meta={"grid_shape": grid_shape})
        latents.append(latent.detach().cpu())
    return torch.cat(latents, dim=0)


def _aggregate_chunk_metrics(
    pred_chunks: List[torch.Tensor],
    target_chunks: List[torch.Tensor],
    *,
    eps: float = 1e-8,
) -> Dict[str, float]:
    if len(pred_chunks) != len(target_chunks) or not pred_chunks:
        raise ValueError("pred_chunks and target_chunks must be non-empty and aligned")

    total_sq = 0.0
    total_abs = 0.0
    total_target_sq = 0.0
    total_elements = 0
    spectral_total = 0.0

    for pred, target in zip(pred_chunks, target_chunks):
        diff = pred - target
        total_sq += diff.pow(2).sum().item()
        total_abs += diff.abs().sum().item()
        total_target_sq += target.pow(2).sum().item()
        total_elements += diff.numel()
        spectral_total += spectral_energy_error(pred.squeeze(-1), target.squeeze(-1)).item()

    if total_elements == 0:
        raise ValueError("Cannot aggregate empty chunk tensors")

    mse_val = total_sq / total_elements
    mae_val = total_abs / total_elements
    mean_target_sq = total_target_sq / total_elements
    nrmse_val = math.sqrt(mse_val / (mean_target_sq + eps))
    return {
        "mse": mse_val,
        "mae": mae_val,
        "nrmse": nrmse_val,
        "rrmse": nrmse_val,
        "spectral_energy_error": spectral_total / len(pred_chunks),
    }


def evaluate_pdebench(task: str, split: str = "test", root: str | None = None) -> MetricReport:
    """Identity baseline over raw PDEBench fields."""

    dataset = PDEBenchDataset(PDEBenchConfig(task=task, split=split, root=root))
    fields = torch.stack([sample["fields"].float() for sample in dataset], dim=0)
    preds = fields  # identity baseline for now
    metrics = {
        "mae": mae(preds, fields).item(),
        "mse": mse(preds, fields).item(),
        "nrmse": nrmse(preds, fields).item(),
        "rrmse": relative_rrmse(preds, fields).item(),
        "spectral_energy_error": spectral_energy_error(preds, fields).item(),
    }
    return MetricReport(metrics=metrics, extra={"task": task, "split": split, "root": root})


def evaluate_latent_operator(
    cfg: Dict[str, Any],
    operator: LatentOperator,
    *,
    diffusion: Optional[DiffusionResidual] = None,
    tau: float = 0.5,
    device: str | torch.device = "cpu",
    return_details: bool = False,
    ttc_config: Optional[TTCConfig] = None,
    reward_model: Optional[RewardModel] = None,
) -> MetricReport | tuple[MetricReport, Dict[str, Any]]:
    """Evaluate a latent operator (optionally with diffusion corrector) on PDEBench data."""

    device = torch.device(device)
    loader = build_latent_pair_loader(cfg)
    operator = operator.to(device)
    operator.eval()
    if diffusion is not None:
        diffusion = diffusion.to(device)
        diffusion.eval()

    dt = cfg.get("training", {}).get("dt", 0.1)
    dt_tensor = torch.tensor(dt, device=device)

    total_abs = 0.0
    total_sq = 0.0
    total_elements = 0
    sample_mse: list[torch.Tensor] = []
    sample_mae: list[torch.Tensor] = []
    preview: Dict[str, torch.Tensor] | None = None
    ttc_step_logs: List[Dict[str, Any]] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            z0, z1, cond = unpack_batch(batch)
            cond_device = {k: v.to(device) for k, v in cond.items()}
            state = LatentState(z=z0.to(device), t=torch.tensor(0.0, device=device), cond=cond_device)
            target = z1.to(device)

            if ttc_config is not None and reward_model is not None:
                ttc_cfg = TTCConfig(
                    steps=1,
                    dt=ttc_config.dt,
                    candidates=ttc_config.candidates,
                    beam_width=ttc_config.beam_width,
                    horizon=ttc_config.horizon,
                    tau_range=ttc_config.tau_range,
                    noise_std=ttc_config.noise_std,
                    residual_threshold=ttc_config.residual_threshold,
                    max_evaluations=ttc_config.max_evaluations,
                    early_stop_margin=ttc_config.early_stop_margin,
                    gamma=ttc_config.gamma,
                    device=device,
                )
                rollout_log, step_logs = ttc_rollout(
                    initial_state=state,
                    operator=operator,
                    reward_model=reward_model,
                    config=ttc_cfg,
                    corrector=diffusion,
                )
                pred = rollout_log.states[-1].z
                if return_details:
                    ttc_step_logs.extend(
                        [
                            {
                                "step": batch_idx,
                                "rewards": sl.rewards,
                                "totals": sl.totals,
                                "chosen": sl.chosen_index,
                                "beam_width": sl.beam_width,
                                "horizon": sl.horizon,
                            }
                            for sl in step_logs
                        ]
                    )
            else:
                predicted_state = operator(state, dt_tensor)
                pred = predicted_state.z
                if diffusion is not None:
                    tau_tensor = torch.full((pred.size(0),), tau, device=device)
                    drift = diffusion(predicted_state, tau_tensor)
                    pred = pred + drift

            diff = pred - target
            total_abs += diff.abs().sum().item()
            total_sq += diff.pow(2).sum().item()
            total_elements += diff.numel()
            if return_details:
                mse_batch = diff.pow(2).mean(dim=(1, 2))
                mae_batch = diff.abs().mean(dim=(1, 2))
                sample_mse.append(mse_batch.detach().cpu())
                sample_mae.append(mae_batch.detach().cpu())
                if preview is None:
                    preview = {
                        "predicted": pred.detach().cpu()[0].clone(),
                        "target": target.detach().cpu()[0].clone(),
                    }

    if total_elements == 0:
        raise RuntimeError("Latent evaluation received an empty dataset")

    mse_val = total_sq / total_elements
    mae_val = total_abs / total_elements
    rmse_val = math.sqrt(mse_val)
    metrics = {
        "mse": mse_val,
        "mae": mae_val,
        "rmse": rmse_val,
    }
    extra = {
        "samples": total_elements,
        "tau": tau if diffusion is not None else None,
        "ttc": bool(ttc_config and reward_model),
    }
    report = MetricReport(metrics=metrics, extra=extra)
    if not return_details:
        return report

    details: Dict[str, Any] = {}
    if sample_mse:
        mse_tensor = torch.cat(sample_mse)
        mae_tensor = torch.cat(sample_mae)
        details["per_sample_mse"] = mse_tensor.tolist()
        details["per_sample_mae"] = mae_tensor.tolist()
    else:
        details["per_sample_mse"] = []
        details["per_sample_mae"] = []
    if preview is not None:
        details["preview_predicted"] = preview.get("predicted", torch.tensor([])).tolist()
        details["preview_target"] = preview.get("target", torch.tensor([])).tolist()
    if ttc_step_logs:
        details["ttc_step_logs"] = ttc_step_logs
    return report, details


def evaluate_decoded_operator(
    cfg: Dict[str, Any],
    encoder: Any,
    operator: Any,
    decoder: Any,
    *,
    device: str | torch.device = "cpu",
    rollout_steps: Optional[int] = None,
) -> MetricReport:
    """Evaluate decoded physical-space rollout metrics for grid PDEBench tasks.

    This path is intentionally narrow: it targets grid PDEBench data and requires
    an encoder/decoder pair capable of mapping between raw fields and latent
    tokens. It provides the physical-space evaluation surface needed for
    rollout-centric research even when the main CLI still focuses on latent-only
    checkpoints.
    """

    data_cfg = cfg.get("data", {})
    task_cfg = data_cfg.get("task")
    if isinstance(task_cfg, str):
        task_names = [task_cfg]
    elif isinstance(task_cfg, (list, tuple)) and task_cfg and all(isinstance(task, str) for task in task_cfg):
        task_names = [str(task) for task in task_cfg]
    else:
        raise ValueError("Decoded operator evaluation currently requires one PDEBench task or a non-empty list of task names")

    device = torch.device(device)
    operator = operator.to(device)
    operator.eval()
    if hasattr(encoder, "to"):
        encoder = encoder.to(device)
    if hasattr(encoder, "eval"):
        encoder.eval()
    if hasattr(decoder, "to"):
        decoder = decoder.to(device)
    if hasattr(decoder, "eval"):
        decoder.eval()

    field_name = data_cfg.get("field_name", "u")
    dt = cfg.get("training", {}).get("dt", 0.1)
    dt_tensor = torch.tensor(dt, device=device)
    eval_cfg = cfg.get("evaluation", {})
    residual_alpha = float(eval_cfg.get("decoded_persistence_residual_alpha", 1.0))
    if residual_alpha < 0.0:
        raise ValueError("evaluation.decoded_persistence_residual_alpha must be non-negative")
    residual_alpha_by_task = {
        str(key): float(value)
        for key, value in (eval_cfg.get("decoded_persistence_residual_alpha_by_task") or {}).items()
    }
    residual_alpha_by_family = {
        str(key): float(value)
        for key, value in (eval_cfg.get("decoded_persistence_residual_alpha_by_family") or {}).items()
    }
    for key, value in {**residual_alpha_by_task, **residual_alpha_by_family}.items():
        if value < 0.0:
            raise ValueError(f"decoded persistence residual alpha for '{key}' must be non-negative")

    total_pred = []
    total_target = []
    horizon_pred: Dict[int, List[torch.Tensor]] = {1: [], 4: [], 16: []}
    horizon_target: Dict[int, List[torch.Tensor]] = {1: [], 4: [], 16: []}
    per_task_pred: Dict[str, List[torch.Tensor]] = {}
    per_task_target: Dict[str, List[torch.Tensor]] = {}
    per_task_step1_pred: Dict[str, List[torch.Tensor]] = {}
    per_task_step1_target: Dict[str, List[torch.Tensor]] = {}
    per_family_pred: Dict[str, List[torch.Tensor]] = {}
    per_family_target: Dict[str, List[torch.Tensor]] = {}
    per_family_step1_pred: Dict[str, List[torch.Tensor]] = {}
    per_family_step1_target: Dict[str, List[torch.Tensor]] = {}

    with torch.no_grad():
        for task_name in task_names:
            task_family = get_pdebench_spec(task_name).family
            dataset = PDEBenchDataset(
                PDEBenchConfig(
                    task=task_name,
                    split=data_cfg.get("split", "train"),
                    root=data_cfg.get("root"),
                    param_keys=tuple(data_cfg.get("param_keys", ())),
                    bc_keys=tuple(data_cfg.get("bc_keys", ())),
                    max_samples=data_cfg.get("max_samples"),
                )
            )
            if len(dataset) == 0:
                continue
            sample_fields = dataset.fields[0]
            grid_shape = infer_grid_shape(sample_fields)
            coords = make_grid_coords(grid_shape, device)
            base_cond: Dict[str, torch.Tensor] = {}
            if bool(cfg.get("training", {}).get("auto_conditioning", False)):
                extras = pdebench_conditioning_extras(task_name=task_name, grid_shape=grid_shape, task_vocab=task_names)
                base_cond = {key: value.to(device) for key, value in extras.items()}
            param_vocab = tuple(data_cfg.get("param_keys", ()))
            bc_vocab = tuple(data_cfg.get("bc_keys", ()))

            for idx in range(len(dataset)):
                sample = dataset[idx]
                fields = sample["fields"].float()
                params = sample.get("params")
                bc = sample.get("bc")
                latent_seq = _encode_grid_trajectory(
                    encoder,
                    fields,
                    coords,
                    grid_shape,
                    field_name=field_name,
                    device=device,
                )
                max_steps = latent_seq.shape[0] - 1
                if max_steps <= 0:
                    continue
                steps = max_steps if rollout_steps is None else min(max_steps, int(rollout_steps))
                initial_cond = pdebench_condition_step(
                    params,
                    bc,
                    batch_size=1,
                    step=0,
                    extras=base_cond,
                    param_vocab=param_vocab,
                    bc_vocab=bc_vocab,
                )
                state = LatentState(z=latent_seq[0:1].to(device), t=torch.tensor(0.0, device=device), cond={k: v.to(device) for k, v in initial_cond.items()})

                for step in range(steps):
                    cond = pdebench_condition_step(
                        params,
                        bc,
                        batch_size=1,
                        step=step,
                        extras=base_cond,
                        param_vocab=param_vocab,
                        bc_vocab=bc_vocab,
                    )
                    state = LatentState(z=state.z, t=state.t, cond={k: v.to(device) for k, v in cond.items()})
                    state = operator(state, dt_tensor)
                    decoded = decoder(coords, state.z, conditioning={})
                    if field_name not in decoded:
                        raise KeyError(f"Decoder did not produce requested field '{field_name}'")
                    pred_field = decoded[field_name].detach().cpu()
                    task_residual_alpha = residual_alpha_by_task.get(
                        task_name,
                        residual_alpha_by_family.get(task_family, residual_alpha),
                    )
                    if task_residual_alpha != 1.0:
                        persistence_field = _flatten_field_step(fields[step], grid_shape).cpu()
                        pred_field = persistence_field + task_residual_alpha * (pred_field - persistence_field)
                    target_field = _flatten_field_step(fields[step + 1], grid_shape).cpu()
                    total_pred.append(pred_field)
                    total_target.append(target_field)
                    per_task_pred.setdefault(task_name, []).append(pred_field)
                    per_task_target.setdefault(task_name, []).append(target_field)
                    per_family_pred.setdefault(task_family, []).append(pred_field)
                    per_family_target.setdefault(task_family, []).append(target_field)
                    horizon = step + 1
                    if horizon in horizon_pred:
                        horizon_pred[horizon].append(pred_field)
                        horizon_target[horizon].append(target_field)
                    if horizon == 1:
                        per_task_step1_pred.setdefault(task_name, []).append(pred_field)
                        per_task_step1_target.setdefault(task_name, []).append(target_field)
                        per_family_step1_pred.setdefault(task_family, []).append(pred_field)
                        per_family_step1_target.setdefault(task_family, []).append(target_field)

    if not total_pred:
        raise RuntimeError("Decoded evaluation received no valid rollout steps")

    rollout_stats = _aggregate_chunk_metrics(total_pred, total_target)
    metrics = {
        "decoded_mse": rollout_stats["mse"],
        "decoded_mae": rollout_stats["mae"],
        "decoded_nrmse": rollout_stats["nrmse"],
        "decoded_rrmse": rollout_stats["rrmse"],
        "decoded_spectral_energy_error": rollout_stats["spectral_energy_error"],
        "decoded_rollout_mse": rollout_stats["mse"],
        "decoded_rollout_mae": rollout_stats["mae"],
        "decoded_rollout_nrmse": rollout_stats["nrmse"],
        "decoded_rollout_rrmse": rollout_stats["rrmse"],
        "decoded_rollout_spectral_energy_error": rollout_stats["spectral_energy_error"],
    }
    if horizon_pred[1]:
        step1_stats = _aggregate_chunk_metrics(horizon_pred[1], horizon_target[1])
        metrics["decoded_step1_nrmse"] = step1_stats["nrmse"]
        metrics["decoded_step1_rrmse"] = step1_stats["rrmse"]
    for horizon in (4, 16):
        if horizon_pred[horizon]:
            horizon_stats = _aggregate_chunk_metrics(horizon_pred[horizon], horizon_target[horizon])
            metrics[f"decoded_h{horizon}_nrmse"] = horizon_stats["nrmse"]
            metrics[f"decoded_h{horizon}_rrmse"] = horizon_stats["rrmse"]
    for task_name, pred_chunks in per_task_pred.items():
        task_stats = _aggregate_chunk_metrics(pred_chunks, per_task_target[task_name])
        metrics[f"task_{task_name}_decoded_rollout_nrmse"] = task_stats["nrmse"]
        metrics[f"task_{task_name}_decoded_rollout_rrmse"] = task_stats["rrmse"]
        step1_pred = per_task_step1_pred.get(task_name)
        step1_target = per_task_step1_target.get(task_name)
        if step1_pred and step1_target:
            step1_stats = _aggregate_chunk_metrics(step1_pred, step1_target)
            metrics[f"task_{task_name}_decoded_step1_nrmse"] = step1_stats["nrmse"]
    for family_name, pred_chunks in per_family_pred.items():
        family_stats = _aggregate_chunk_metrics(pred_chunks, per_family_target[family_name])
        metrics[f"family_{family_name}_decoded_rollout_nrmse"] = family_stats["nrmse"]
        metrics[f"family_{family_name}_decoded_rollout_rrmse"] = family_stats["rrmse"]
        step1_pred = per_family_step1_pred.get(family_name)
        step1_target = per_family_step1_target.get(family_name)
        if step1_pred and step1_target:
            step1_stats = _aggregate_chunk_metrics(step1_pred, step1_target)
            metrics[f"family_{family_name}_decoded_step1_nrmse"] = step1_stats["nrmse"]
    task_extra: str | list[str] = task_names[0] if len(task_names) == 1 else task_names
    return MetricReport(
        metrics=metrics,
        extra={
            "task": task_extra,
            "split": data_cfg.get("split", "train"),
            "decoded_persistence_residual_alpha": residual_alpha,
            "decoded_persistence_residual_alpha_by_task": residual_alpha_by_task,
            "decoded_persistence_residual_alpha_by_family": residual_alpha_by_family,
        },
    )


def evaluate_latent_model(
    cfg: Dict[str, Any],
    model: Any,
    *,
    device: str | torch.device = "cpu",
) -> MetricReport:
    """Evaluate a generic latent model that maps (z0, cond) -> z1 prediction."""

    device = torch.device(device)
    loader = build_latent_pair_loader(cfg)
    total_abs = 0.0
    total_sq = 0.0
    total_elements = 0

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for batch in loader:
            z0, z1, cond = unpack_batch(batch)
            cond_device = {k: v.to(device) for k, v in cond.items()}
            pred = model(z0.to(device), cond_device)
            diff = pred - z1.to(device)
            total_abs += diff.abs().sum().item()
            total_sq += diff.pow(2).sum().item()
            total_elements += diff.numel()

    if total_elements == 0:
        raise RuntimeError("Evaluation received an empty dataset")

    mse_val = total_sq / total_elements
    mae_val = total_abs / total_elements
    rmse_val = math.sqrt(mse_val)
    metrics = {
        "mse": mse_val,
        "mae": mae_val,
        "rmse": rmse_val,
    }
    return MetricReport(metrics=metrics, extra={"samples": total_elements})
