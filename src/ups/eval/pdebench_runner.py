from __future__ import annotations

"""PDEBench evaluation helpers."""

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import torch

from ups.core.latent_state import LatentState
from ups.data.latent_pairs import build_latent_pair_loader, unpack_batch
from ups.data.pdebench import PDEBenchConfig, PDEBenchDataset
from ups.eval.metrics import mae, mse, nrmse, relative_rrmse, spectral_energy_error
from ups.eval.reports import MetricReport
from ups.eval.reward_models import RewardModel
from ups.inference.rollout_ttc import TTCConfig, ttc_rollout
from ups.models.diffusion_residual import DiffusionResidual
from ups.models.latent_operator import LatentOperator


@dataclass
class BaselineModel:
    forward: Callable[[torch.Tensor], torch.Tensor]


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
    # Force num_workers=0 during evaluation to avoid CUDA serialization issues with encoder
    eval_cfg = cfg.copy()
    if "training" not in eval_cfg:
        eval_cfg["training"] = {}
    eval_cfg["training"] = eval_cfg["training"].copy()
    eval_cfg["training"]["num_workers"] = 0
    loader = build_latent_pair_loader(eval_cfg)
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
    total_target_sq = 0.0  # For normalized metrics (SOTA)
    total_target_abs = 0.0  # For normalized metrics (SOTA)
    sample_mse: list[torch.Tensor] = []
    sample_mae: list[torch.Tensor] = []
    sample_rel_l2: list[torch.Tensor] = []  # For SOTA comparison
    sample_conservation: list[torch.Tensor] = []
    sample_bc_violation: list[torch.Tensor] = []
    sample_negativity: list[torch.Tensor] = []
    preview: Dict[str, torch.Tensor] | None = None
    ttc_step_logs: List[Dict[str, Any]] = []

    total_conservation_gap = 0.0
    total_bc_violation = 0.0
    total_negativity_penalty = 0.0
    physics_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            unpacked = unpack_batch(batch)
            if isinstance(unpacked, dict):
                z0 = unpacked["z0"]
                z1 = unpacked["z1"]
                cond = unpacked.get("cond", {})
            elif len(unpacked) == 4:
                z0, z1, cond, _ = unpacked
            else:
                z0, z1, cond = unpacked
            cond_device = {k: v.to(device) for k, v in cond.items()}
            state = LatentState(z=z0.to(device), t=torch.tensor(0.0, device=device), cond=cond_device)
            target = z1.to(device)

            if ttc_config is not None and reward_model is not None:
                ttc_cfg = TTCConfig(
                    steps=ttc_config.steps,  # Use configured steps instead of hardcoding to 1
                    dt=ttc_config.dt,
                    candidates=ttc_config.candidates,
                    beam_width=ttc_config.beam_width,
                    horizon=ttc_config.horizon,
                    tau_range=ttc_config.tau_range,
                    noise_std=ttc_config.noise_std,
                    noise_schedule=ttc_config.noise_schedule,
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
                                **{f"component_{k}": v for k, v in sl.reward_components.items()},
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

            # For normalized metrics (SOTA comparison)
            total_target_sq += target.pow(2).sum().item()
            total_target_abs += target.abs().sum().item()

            if return_details:
                mse_batch = diff.pow(2).mean(dim=(1, 2))
                mae_batch = diff.abs().mean(dim=(1, 2))
                sample_mse.append(mse_batch.detach().cpu())
                sample_mae.append(mae_batch.detach().cpu())

                # Per-sample normalized metrics for SOTA comparison
                target_norm_batch = target.pow(2).mean(dim=(1, 2)).sqrt()
                rel_l2_batch = mse_batch.sqrt() / (target_norm_batch + 1e-8)
                sample_rel_l2.append(rel_l2_batch.detach().cpu())

                if preview is None:
                    preview = {
                        "predicted": pred.detach().cpu()[0].clone(),
                        "target": target.detach().cpu()[0].clone(),
                    }

            # Physics-style diagnostics computed in latent coordinates
            flat_pred = pred.reshape(pred.size(0), -1)
            flat_target = target.reshape(target.size(0), -1)
            conservation_batch = (flat_pred.sum(dim=1) - flat_target.sum(dim=1)).abs()

            if pred.size(1) >= 2:
                bc_first = (pred[:, 0] - target[:, 0]).abs().mean(dim=1)
                bc_last = (pred[:, -1] - target[:, -1]).abs().mean(dim=1)
                bc_batch = 0.5 * (bc_first + bc_last)
            else:
                bc_batch = diff.abs().mean(dim=(1, 2))

            negativity_batch = torch.relu(-pred).mean(dim=(1, 2))

            total_conservation_gap += conservation_batch.sum().item()
            total_bc_violation += bc_batch.sum().item()
            total_negativity_penalty += negativity_batch.sum().item()
            physics_samples += pred.size(0)

            if return_details:
                sample_conservation.append(conservation_batch.detach().cpu())
                sample_bc_violation.append(bc_batch.detach().cpu())
                sample_negativity.append(negativity_batch.detach().cpu())

    if total_elements == 0:
        raise RuntimeError("Latent evaluation received an empty dataset")

    mse_val = total_sq / total_elements
    mae_val = total_abs / total_elements
    rmse_val = math.sqrt(mse_val)

    # Normalized metrics for SOTA comparison
    target_rms = math.sqrt(total_target_sq / total_elements)
    nrmse_val = rmse_val / (target_rms + 1e-8)  # Normalized RMSE
    rel_l2_val = rmse_val / (target_rms + 1e-8)  # Relative L2 (same as nRMSE for element-wise)

    metrics = {
        "mse": mse_val,
        "mae": mae_val,
        "rmse": rmse_val,
        "nrmse": nrmse_val,  # For SOTA comparison
        "rel_l2": rel_l2_val,  # For SOTA comparison
    }
    if physics_samples > 0:
        metrics["conservation_gap"] = total_conservation_gap / physics_samples
        metrics["bc_violation"] = total_bc_violation / physics_samples
        metrics["negativity_penalty"] = total_negativity_penalty / physics_samples
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
        rel_l2_tensor = torch.cat(sample_rel_l2)
        details["per_sample_mse"] = mse_tensor.tolist()
        details["per_sample_mae"] = mae_tensor.tolist()
        details["per_sample_rel_l2"] = rel_l2_tensor.tolist()  # SOTA comparison
        if sample_conservation:
            details["per_sample_conservation_gap"] = torch.cat(sample_conservation).tolist()
        if sample_bc_violation:
            details["per_sample_bc_violation"] = torch.cat(sample_bc_violation).tolist()
        if sample_negativity:
            details["per_sample_negativity_penalty"] = torch.cat(sample_negativity).tolist()
    else:
        details["per_sample_mse"] = []
        details["per_sample_mae"] = []
        details["per_sample_rel_l2"] = []
        details["per_sample_conservation_gap"] = []
        details["per_sample_bc_violation"] = []
        details["per_sample_negativity_penalty"] = []
    if preview is not None:
        details["preview_predicted"] = preview.get("predicted", torch.tensor([])).tolist()
        details["preview_target"] = preview.get("target", torch.tensor([])).tolist()
    if ttc_step_logs:
        details["ttc_step_logs"] = ttc_step_logs
    return report, details


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
            unpacked = unpack_batch(batch)
            if isinstance(unpacked, dict):
                z0 = unpacked["z0"]
                z1 = unpacked["z1"]
                cond = unpacked.get("cond", {})
            elif len(unpacked) == 4:
                z0, z1, cond, _ = unpacked
            else:
                z0, z1, cond = unpacked
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
