from __future__ import annotations

"""PDEBench evaluation helpers."""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

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


_DEFAULT_DECODED_HORIZONS = (1, 4, 16)
_DATA_CONDITIONED_SHIFT_FEATURES = (
    "bias",
    "horizon_norm",
    "mean",
    "std",
    "rms",
    "abs_mean",
    "max",
    "min",
)
_DATA_CONDITIONED_CONTEXT_FEATURES = {"context_shift", "context_shift_abs"}


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
    latents: list[torch.Tensor] = []
    for step in range(fields.shape[0]):
        flattened = _flatten_field_step(fields[step], grid_shape).to(device)
        latent = encoder(
            {field_name: flattened}, coords.to(device), meta={"grid_shape": grid_shape}
        )
        latents.append(latent.detach().cpu())
    return torch.cat(latents, dim=0)


def _aggregate_chunk_metrics(
    pred_chunks: list[torch.Tensor],
    target_chunks: list[torch.Tensor],
    *,
    eps: float = 1e-8,
) -> dict[str, float]:
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


def _nonnegative_alpha(value: Any, *, setting: str) -> float:
    alpha = float(value)
    if alpha < 0.0:
        raise ValueError(f"{setting} must be non-negative")
    return alpha


def _alpha_map(raw: Any, *, setting: str) -> dict[str, float]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{setting} must be a mapping")
    return {
        str(key): _nonnegative_alpha(value, setting=f"{setting}[{key!r}]")
        for key, value in raw.items()
    }


def _horizon_alpha_map(raw: Any, *, setting: str) -> dict[int, float]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{setting} must be a mapping")
    result: dict[int, float] = {}
    for key, value in raw.items():
        try:
            horizon = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{setting} horizon keys must be positive integers") from exc
        if horizon <= 0:
            raise ValueError(f"{setting} horizon keys must be positive integers")
        result[horizon] = _nonnegative_alpha(value, setting=f"{setting}[{key!r}]")
    return result


def _nested_horizon_alpha_map(raw: Any, *, setting: str) -> dict[str, dict[int, float]]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{setting} must be a mapping")
    return {
        str(key): _horizon_alpha_map(value, setting=f"{setting}[{key!r}]")
        for key, value in raw.items()
    }


def _int_map(raw: Any, *, setting: str) -> dict[str, int]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{setting} must be a mapping")
    return {str(key): int(value) for key, value in raw.items()}


def _task_root_map(raw: Any, *, setting: str) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{setting} must be a mapping")
    return {str(key): str(value) for key, value in raw.items()}


def _horizon_int_map(raw: Any, *, setting: str) -> dict[int, int]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{setting} must be a mapping")
    result: dict[int, int] = {}
    for key, value in raw.items():
        try:
            horizon = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{setting} horizon keys must be positive integers") from exc
        if horizon <= 0:
            raise ValueError(f"{setting} horizon keys must be positive integers")
        result[horizon] = int(value)
    return result


def _nested_horizon_int_map(raw: Any, *, setting: str) -> dict[str, dict[int, int]]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{setting} must be a mapping")
    return {
        str(key): _horizon_int_map(value, setting=f"{setting}[{key!r}]")
        for key, value in raw.items()
    }


def _resolve_residual_alpha(
    *,
    task_name: str,
    task_family: str,
    horizon: int,
    residual_alpha: float,
    residual_alpha_by_task: Mapping[str, float],
    residual_alpha_by_family: Mapping[str, float],
    residual_alpha_by_horizon: Mapping[int, float],
    residual_alpha_by_task_horizon: Mapping[str, Mapping[int, float]],
    residual_alpha_by_family_horizon: Mapping[str, Mapping[int, float]],
) -> float:
    task_horizon_alpha = residual_alpha_by_task_horizon.get(task_name, {})
    if horizon in task_horizon_alpha:
        return task_horizon_alpha[horizon]
    family_horizon_alpha = residual_alpha_by_family_horizon.get(task_family, {})
    if horizon in family_horizon_alpha:
        return family_horizon_alpha[horizon]
    if task_name in residual_alpha_by_task:
        return residual_alpha_by_task[task_name]
    if task_family in residual_alpha_by_family:
        return residual_alpha_by_family[task_family]
    if horizon in residual_alpha_by_horizon:
        return residual_alpha_by_horizon[horizon]
    return residual_alpha


def _resolve_roll_shift(
    *,
    task_name: str,
    task_family: str,
    horizon: int,
    shift_by_task: Mapping[str, int],
    shift_by_family: Mapping[str, int],
    shift_by_task_horizon: Mapping[str, Mapping[int, int]],
    shift_by_family_horizon: Mapping[str, Mapping[int, int]],
) -> int:
    task_horizon_shift = shift_by_task_horizon.get(task_name, {})
    if horizon in task_horizon_shift:
        return task_horizon_shift[horizon]
    family_horizon_shift = shift_by_family_horizon.get(task_family, {})
    if horizon in family_horizon_shift:
        return family_horizon_shift[horizon]
    if task_name in shift_by_task:
        return shift_by_task[task_name]
    if task_family in shift_by_family:
        return shift_by_family[task_family]
    return 0


def _roll_flattened_grid(
    field: torch.Tensor, grid_shape: tuple[int, int], *, shift_x: float
) -> torch.Tensor:
    shift_value = float(shift_x)
    if shift_value == 0.0:
        return field
    if field.dim() != 3:
        raise ValueError(
            f"Expected flattened grid field shaped (B, N, C), got {tuple(field.shape)}"
        )
    batch, nodes, channels = field.shape
    H, W = grid_shape
    if nodes != H * W:
        raise ValueError(
            f"Flattened grid has {nodes} nodes, expected {H * W} for grid shape {grid_shape}"
        )
    grid = field.transpose(1, 2).reshape(batch, channels, H, W)
    if abs(shift_value - round(shift_value)) < 1e-9:
        rolled = torch.roll(grid, shifts=int(round(shift_value)), dims=-1)
    else:
        frequencies = torch.fft.fftfreq(W, device=grid.device, dtype=grid.dtype)
        phase = torch.exp(
            -2j
            * torch.tensor(math.pi, device=grid.device, dtype=grid.dtype)
            * frequencies
            * shift_value
        )
        rolled = torch.fft.ifft(torch.fft.fft(grid, dim=-1) * phase, dim=-1).real
    return rolled.reshape(batch, channels, H * W).transpose(1, 2).contiguous()


def _roll_shift_estimator_config(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("evaluation.decoded_observed_roll_shift_estimator must be a mapping")
    cfg = dict(raw)
    candidate_shifts = cfg.get(
        "candidate_shifts",
        [-64, -48, -40, -32, -24, -16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16, 24, 32, 40, 48, 64],
    )
    if not isinstance(candidate_shifts, Sequence) or isinstance(candidate_shifts, (str, bytes)):
        raise ValueError(
            "decoded observed roll-shift estimator candidate_shifts must be a sequence of integers"
        )
    cfg["candidate_shifts"] = [int(shift) for shift in candidate_shifts]
    cfg["enabled"] = bool(cfg.get("enabled", True))
    for key in ("tasks", "families"):
        values = cfg.get(key, ())
        if values is None:
            values = ()
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, Sequence):
            raise ValueError(f"decoded observed roll-shift estimator {key} must be a sequence")
        cfg[key] = [str(value) for value in values]
    cfg["min_horizon"] = int(cfg.get("min_horizon", 2))
    if cfg["min_horizon"] <= 0:
        raise ValueError("decoded observed roll-shift estimator min_horizon must be positive")
    return cfg


def _context_roll_shift_estimator_config(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("evaluation.decoded_context_roll_shift_estimator must be a mapping")
    cfg = dict(raw)
    candidate_shifts = cfg.get(
        "candidate_shifts",
        [-64, -48, -40, -32, -24, -16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16, 24, 32, 40, 48, 64],
    )
    if not isinstance(candidate_shifts, Sequence) or isinstance(candidate_shifts, (str, bytes)):
        raise ValueError("decoded context roll-shift estimator candidate_shifts must be a sequence")
    cfg["candidate_shifts"] = [float(shift) for shift in candidate_shifts]
    cfg["enabled"] = bool(cfg.get("enabled", True))
    for key in ("tasks", "families"):
        values = cfg.get(key, ())
        if values is None:
            values = ()
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, Sequence):
            raise ValueError(f"decoded context roll-shift estimator {key} must be a sequence")
        cfg[key] = [str(value) for value in values]
    context_transitions = int(cfg.get("context_transitions", 1))
    if context_transitions <= 0:
        raise ValueError(
            "decoded context roll-shift estimator context_transitions must be positive"
        )
    cfg["context_transitions"] = context_transitions
    cfg["min_horizon"] = int(cfg.get("min_horizon", context_transitions + 1))
    if cfg["min_horizon"] <= context_transitions:
        raise ValueError(
            "decoded context roll-shift estimator min_horizon must be greater than context_transitions"
        )
    coefficients = cfg.get("coefficients", {})
    if coefficients is None:
        coefficients = {}
    if not isinstance(coefficients, Mapping):
        raise ValueError("decoded context roll-shift estimator coefficients must be a mapping")
    cfg["coefficients"] = {
        "slope": float(coefficients.get("slope", cfg.get("slope", 1.0))),
        "intercept": float(coefficients.get("intercept", cfg.get("intercept", 0.0))),
    }
    cfg["mode"] = _roll_shift_estimator_mode(cfg)
    cfg["calibration_scope"] = str(cfg.get("calibration_scope", "shared_1d_transport"))
    return cfg


def _data_conditioned_roll_shift_estimator_config(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(
            "evaluation.decoded_data_conditioned_roll_shift_estimator must be a mapping"
        )
    cfg = dict(raw)
    cfg["enabled"] = bool(cfg.get("enabled", True))
    for key in ("tasks", "families"):
        values = cfg.get(key, ())
        if values is None:
            values = ()
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, Sequence):
            raise ValueError(
                f"decoded data-conditioned roll-shift estimator {key} must be a sequence"
            )
        cfg[key] = [str(value) for value in values]
    cfg["min_horizon"] = int(cfg.get("min_horizon", 1))
    if cfg["min_horizon"] <= 0:
        raise ValueError(
            "decoded data-conditioned roll-shift estimator min_horizon must be positive"
        )
    feature_names = cfg.get("feature_names", _DATA_CONDITIONED_SHIFT_FEATURES)
    if isinstance(feature_names, str) or not isinstance(feature_names, Sequence):
        raise ValueError(
            "decoded data-conditioned roll-shift estimator feature_names must be a sequence"
        )
    cfg["feature_names"] = [str(name) for name in feature_names]
    uses_context_shift = any(
        name in _DATA_CONDITIONED_CONTEXT_FEATURES for name in cfg["feature_names"]
    )
    if uses_context_shift or "candidate_shifts" in cfg or "context_transitions" in cfg:
        candidate_shifts = cfg.get(
            "candidate_shifts",
            [
                -64,
                -48,
                -40,
                -32,
                -24,
                -16,
                -8,
                -4,
                -2,
                -1,
                0,
                1,
                2,
                4,
                8,
                16,
                24,
                32,
                40,
                48,
                64,
            ],
        )
        if not isinstance(candidate_shifts, Sequence) or isinstance(candidate_shifts, (str, bytes)):
            raise ValueError(
                "decoded data-conditioned roll-shift estimator candidate_shifts must be a sequence"
            )
        cfg["candidate_shifts"] = [float(shift) for shift in candidate_shifts]
        context_transitions = int(cfg.get("context_transitions", 1))
        if context_transitions <= 0:
            raise ValueError(
                "decoded data-conditioned roll-shift estimator context_transitions must be positive"
            )
        cfg["context_transitions"] = context_transitions
        if uses_context_shift and cfg["min_horizon"] <= context_transitions:
            raise ValueError(
                "decoded data-conditioned roll-shift estimator min_horizon must be greater than "
                "context_transitions when context_shift features are used"
            )
    coefficients = cfg.get("coefficients", {})
    if not isinstance(coefficients, Mapping):
        raise ValueError(
            "decoded data-conditioned roll-shift estimator coefficients must be a mapping"
        )
    cfg["coefficients"] = {str(key): float(value) for key, value in coefficients.items()}
    if "min_shift" in cfg:
        cfg["min_shift"] = float(cfg["min_shift"])
    if "max_shift" in cfg:
        cfg["max_shift"] = float(cfg["max_shift"])
    cfg["mode"] = _roll_shift_estimator_mode(cfg)
    cfg["calibration_scope"] = str(cfg.get("calibration_scope", "data_conditioned_train_fit"))
    return cfg


def _roll_shift_estimator_applies(
    *,
    cfg: Mapping[str, Any],
    task_name: str,
    task_family: str,
    horizon: int,
) -> bool:
    if not cfg or not bool(cfg.get("enabled", True)):
        return False
    if horizon < int(cfg.get("min_horizon", 2)):
        return False
    tasks = set(str(value) for value in cfg.get("tasks", ()))
    families = set(str(value) for value in cfg.get("families", ()))
    if tasks and task_name not in tasks:
        return False
    if families and task_family not in families:
        return False
    return True


def _roll_shift_estimator_mode(cfg: Mapping[str, Any]) -> str:
    mode = str(cfg.get("mode", "roll_prediction"))
    if mode not in {"roll_prediction", "roll_persistence"}:
        raise ValueError(
            "decoded roll-shift estimator mode must be 'roll_prediction' or 'roll_persistence'"
        )
    return mode


def _estimate_observed_roll_shift(
    *,
    previous_field: torch.Tensor,
    current_field: torch.Tensor,
    grid_shape: tuple[int, int],
    candidate_shifts: Sequence[int],
) -> int:
    if not candidate_shifts:
        return 0
    best_shift = int(candidate_shifts[0])
    best_mse = math.inf
    for shift in candidate_shifts:
        shifted = _roll_flattened_grid(previous_field, grid_shape, shift_x=int(shift))
        mse_value = float((shifted - current_field).pow(2).mean().item())
        if mse_value < best_mse:
            best_shift = int(shift)
            best_mse = mse_value
    return best_shift


def _estimate_prediction_roll_shift(
    *,
    persistence_field: torch.Tensor,
    predicted_field: torch.Tensor,
    grid_shape: tuple[int, int],
    candidate_shifts: Sequence[int],
) -> int:
    if not candidate_shifts:
        return 0
    best_shift = int(candidate_shifts[0])
    best_mse = math.inf
    for shift in candidate_shifts:
        shifted = _roll_flattened_grid(persistence_field, grid_shape, shift_x=int(shift))
        mse_value = float((shifted - predicted_field).pow(2).mean().item())
        if mse_value < best_mse:
            best_shift = int(shift)
            best_mse = mse_value
    return best_shift


def _estimate_context_roll_shift(
    *,
    fields: torch.Tensor,
    grid_shape: tuple[int, int],
    candidate_shifts: Sequence[float],
    context_transitions: int,
    coefficients: Mapping[str, float],
) -> float:
    if not candidate_shifts:
        return 0.0
    steps = min(int(context_transitions), int(fields.shape[0]) - 1)
    if steps <= 0:
        return 0.0
    best_shift = float(candidate_shifts[0])
    best_mse = math.inf
    for shift in candidate_shifts:
        total_mse = 0.0
        for step in range(steps):
            previous_field = _flatten_field_step(fields[step], grid_shape).cpu()
            current_field = _flatten_field_step(fields[step + 1], grid_shape).cpu()
            shifted = _roll_flattened_grid(previous_field, grid_shape, shift_x=float(shift))
            total_mse += float((shifted - current_field).pow(2).mean().item())
        mean_mse = total_mse / max(float(steps), 1.0)
        if mean_mse < best_mse:
            best_shift = float(shift)
            best_mse = mean_mse
    return best_shift * float(coefficients.get("slope", 1.0)) + float(
        coefficients.get("intercept", 0.0)
    )


def _data_conditioned_shift_features(
    field: torch.Tensor,
    *,
    horizon: int,
    rollout_steps: int,
    context_shift: float | None = None,
    params: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, float]:
    if field.dim() != 3:
        raise ValueError(
            f"Expected flattened grid field shaped (B, N, C), got {tuple(field.shape)}"
        )
    features = {
        "bias": 1.0,
        "horizon_norm": float(horizon) / max(float(rollout_steps), 1.0),
        "mean": float(field.mean().item()),
        "std": float(field.std(unbiased=False).item()),
        "rms": _tensor_rms(field),
        "abs_mean": float(field.abs().mean().item()),
        "max": float(field.max().item()),
        "min": float(field.min().item()),
    }
    if context_shift is not None:
        features["context_shift"] = float(context_shift)
        features["context_shift_abs"] = abs(float(context_shift))
    for name, value in (params or {}).items():
        tensor = (
            value.detach().float().reshape(-1)
            if torch.is_tensor(value)
            else torch.as_tensor(value, dtype=torch.float32).reshape(-1)
        )
        if tensor.numel() == 0:
            continue
        if tensor.numel() == 1:
            features[f"param:{name}"] = float(tensor[0].item())
            continue
        for index, scalar in enumerate(tensor):
            features[f"param:{name}:{index}"] = float(scalar.item())
    return features


def _estimate_data_conditioned_roll_shift(
    *,
    field: torch.Tensor,
    cfg: Mapping[str, Any],
    horizon: int,
    rollout_steps: int,
    context_shift: float | None = None,
    params: Mapping[str, torch.Tensor] | None = None,
) -> float:
    features = _data_conditioned_shift_features(
        field,
        horizon=horizon,
        rollout_steps=rollout_steps,
        context_shift=context_shift,
        params=params,
    )
    coefficients = cfg.get("coefficients", {})
    shift = 0.0
    for name in cfg.get("feature_names", _DATA_CONDITIONED_SHIFT_FEATURES):
        shift += float(coefficients.get(str(name), 0.0)) * float(features.get(str(name), 0.0))
    if "min_shift" in cfg:
        shift = max(float(cfg["min_shift"]), shift)
    if "max_shift" in cfg:
        shift = min(float(cfg["max_shift"]), shift)
    return float(shift)


def _logit(value: float, *, eps: float = 1e-6) -> float:
    clipped = min(max(float(value), eps), 1.0 - eps)
    return math.log(clipped / (1.0 - clipped))


def _sigmoid(score: float) -> float:
    if score >= 0.0:
        z = math.exp(-score)
        return 1.0 / (1.0 + z)
    z = math.exp(score)
    return z / (1.0 + z)


def _residual_gate_config(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("evaluation.decoded_persistence_residual_gate must be a mapping")
    cfg = dict(raw)
    min_alpha = float(cfg.get("min_alpha", 0.0))
    max_alpha = float(cfg.get("max_alpha", 1.0))
    if min_alpha < 0.0 or max_alpha > 1.0 or min_alpha > max_alpha:
        raise ValueError(
            "decoded residual gate alpha bounds must satisfy 0 <= min_alpha <= max_alpha <= 1"
        )
    for key in ("feature_weights", "task_bias", "family_bias", "horizon_bias"):
        value = cfg.get(key, {})
        if value is not None and not isinstance(value, Mapping):
            raise ValueError(f"decoded residual gate '{key}' must be a mapping")
    return cfg


def _tensor_rms(value: torch.Tensor) -> float:
    return math.sqrt(float(value.pow(2).mean().item()))


def _gate_features(
    *,
    pred_field: torch.Tensor,
    persistence_field: torch.Tensor,
    horizon: int,
    rollout_steps: int,
) -> dict[str, float]:
    residual = pred_field - persistence_field
    horizon_norm = float(horizon) / max(float(rollout_steps), 1.0)
    return {
        "horizon": float(horizon),
        "horizon_norm": horizon_norm,
        "horizon_log": math.log1p(float(horizon)),
        "residual_abs_mean": float(residual.abs().mean().item()),
        "residual_rms": _tensor_rms(residual),
        "persistence_abs_mean": float(persistence_field.abs().mean().item()),
        "persistence_rms": _tensor_rms(persistence_field),
        "prediction_abs_mean": float(pred_field.abs().mean().item()),
        "prediction_rms": _tensor_rms(pred_field),
    }


def _resolve_gate_alpha(
    *,
    static_alpha: float,
    gate_cfg: Mapping[str, Any],
    task_name: str,
    task_family: str,
    horizon: int,
    features: Mapping[str, float],
) -> float:
    if not gate_cfg:
        return static_alpha
    base_alpha = float(gate_cfg.get("base_alpha", static_alpha))
    score = _logit(base_alpha)
    score += float(gate_cfg.get("bias", 0.0))
    task_bias = gate_cfg.get("task_bias", {}) or {}
    family_bias = gate_cfg.get("family_bias", {}) or {}
    horizon_bias = gate_cfg.get("horizon_bias", {}) or {}
    score += float(task_bias.get(task_name, 0.0))
    score += float(family_bias.get(task_family, 0.0))
    score += float(horizon_bias.get(str(horizon), horizon_bias.get(horizon, 0.0)))
    for name, weight in (gate_cfg.get("feature_weights", {}) or {}).items():
        score += float(weight) * float(features.get(str(name), 0.0))
    alpha = _sigmoid(score)
    min_alpha = float(gate_cfg.get("min_alpha", 0.0))
    max_alpha = float(gate_cfg.get("max_alpha", 1.0))
    return min(max(alpha, min_alpha), max_alpha)


def _append_stat(stats: dict[str, list[float]], key: str, value: float) -> None:
    stats.setdefault(key, []).append(float(value))


def _add_alpha_stats(metrics: dict[str, float], stats: Mapping[str, list[float]]) -> None:
    for key, values in stats.items():
        if not values:
            continue
        tensor = torch.tensor(values, dtype=torch.float64)
        metrics[f"{key}_mean"] = float(tensor.mean().item())
        metrics[f"{key}_std"] = float(tensor.std(unbiased=False).item()) if len(values) > 1 else 0.0


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
    cfg: dict[str, Any],
    operator: LatentOperator,
    *,
    diffusion: DiffusionResidual | None = None,
    tau: float = 0.5,
    device: str | torch.device = "cpu",
    return_details: bool = False,
    ttc_config: TTCConfig | None = None,
    reward_model: RewardModel | None = None,
) -> MetricReport | tuple[MetricReport, dict[str, Any]]:
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
    preview: dict[str, torch.Tensor] | None = None
    ttc_step_logs: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            z0, z1, cond = unpack_batch(batch)
            cond_device = {k: v.to(device) for k, v in cond.items()}
            state = LatentState(
                z=z0.to(device), t=torch.tensor(0.0, device=device), cond=cond_device
            )
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

    details: dict[str, Any] = {}
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
    cfg: dict[str, Any],
    encoder: Any,
    operator: Any,
    decoder: Any,
    *,
    device: str | torch.device = "cpu",
    rollout_steps: int | None = None,
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
    elif (
        isinstance(task_cfg, (list, tuple))
        and task_cfg
        and all(isinstance(task, str) for task in task_cfg)
    ):
        task_names = [str(task) for task in task_cfg]
    else:
        raise ValueError(
            "Decoded operator evaluation currently requires one PDEBench task or a non-empty list of task names"
        )

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
    residual_alpha = _nonnegative_alpha(
        eval_cfg.get("decoded_persistence_residual_alpha", 1.0),
        setting="evaluation.decoded_persistence_residual_alpha",
    )
    residual_alpha_by_task = _alpha_map(
        eval_cfg.get("decoded_persistence_residual_alpha_by_task"),
        setting="evaluation.decoded_persistence_residual_alpha_by_task",
    )
    residual_alpha_by_family = _alpha_map(
        eval_cfg.get("decoded_persistence_residual_alpha_by_family"),
        setting="evaluation.decoded_persistence_residual_alpha_by_family",
    )
    residual_alpha_by_horizon = _horizon_alpha_map(
        eval_cfg.get("decoded_persistence_residual_alpha_by_horizon"),
        setting="evaluation.decoded_persistence_residual_alpha_by_horizon",
    )
    residual_alpha_by_task_horizon = _nested_horizon_alpha_map(
        eval_cfg.get("decoded_persistence_residual_alpha_by_task_horizon"),
        setting="evaluation.decoded_persistence_residual_alpha_by_task_horizon",
    )
    residual_alpha_by_family_horizon = _nested_horizon_alpha_map(
        eval_cfg.get("decoded_persistence_residual_alpha_by_family_horizon"),
        setting="evaluation.decoded_persistence_residual_alpha_by_family_horizon",
    )
    residual_gate_cfg = _residual_gate_config(eval_cfg.get("decoded_persistence_residual_gate"))
    roll_shift_by_task = _int_map(
        eval_cfg.get("decoded_roll_shift_by_task"),
        setting="evaluation.decoded_roll_shift_by_task",
    )
    roll_shift_by_family = _int_map(
        eval_cfg.get("decoded_roll_shift_by_family"),
        setting="evaluation.decoded_roll_shift_by_family",
    )
    roll_shift_by_task_horizon = _nested_horizon_int_map(
        eval_cfg.get("decoded_roll_shift_by_task_horizon"),
        setting="evaluation.decoded_roll_shift_by_task_horizon",
    )
    roll_shift_by_family_horizon = _nested_horizon_int_map(
        eval_cfg.get("decoded_roll_shift_by_family_horizon"),
        setting="evaluation.decoded_roll_shift_by_family_horizon",
    )
    observed_roll_shift_cfg = _roll_shift_estimator_config(
        eval_cfg.get("decoded_observed_roll_shift_estimator")
    )
    prediction_roll_shift_cfg = _roll_shift_estimator_config(
        eval_cfg.get("decoded_prediction_roll_shift_estimator")
    )
    context_roll_shift_cfg = _context_roll_shift_estimator_config(
        eval_cfg.get("decoded_context_roll_shift_estimator")
    )
    data_conditioned_roll_shift_cfg = _data_conditioned_roll_shift_estimator_config(
        eval_cfg.get("decoded_data_conditioned_roll_shift_estimator")
    )
    report_all_horizon_metrics = bool(eval_cfg.get("report_all_horizon_metrics", False))
    skip_missing_tasks = bool(
        eval_cfg.get("skip_missing_tasks", data_cfg.get("skip_missing_tasks", False))
    )
    task_roots = _task_root_map(data_cfg.get("task_roots"), setting="data.task_roots")

    total_pred = []
    total_target = []
    alpha_stats: dict[str, list[float]] = {}
    shift_stats: dict[str, list[float]] = {}
    horizon_pred: dict[int, list[torch.Tensor]] = {
        horizon: [] for horizon in _DEFAULT_DECODED_HORIZONS
    }
    horizon_target: dict[int, list[torch.Tensor]] = {
        horizon: [] for horizon in _DEFAULT_DECODED_HORIZONS
    }
    per_task_pred: dict[str, list[torch.Tensor]] = {}
    per_task_target: dict[str, list[torch.Tensor]] = {}
    per_task_step1_pred: dict[str, list[torch.Tensor]] = {}
    per_task_step1_target: dict[str, list[torch.Tensor]] = {}
    per_task_horizon_pred: dict[str, dict[int, list[torch.Tensor]]] = {}
    per_task_horizon_target: dict[str, dict[int, list[torch.Tensor]]] = {}
    per_family_pred: dict[str, list[torch.Tensor]] = {}
    per_family_target: dict[str, list[torch.Tensor]] = {}
    per_family_step1_pred: dict[str, list[torch.Tensor]] = {}
    per_family_step1_target: dict[str, list[torch.Tensor]] = {}
    per_family_horizon_pred: dict[str, dict[int, list[torch.Tensor]]] = {}
    per_family_horizon_target: dict[str, dict[int, list[torch.Tensor]]] = {}
    skipped_missing_tasks: list[str] = []

    with torch.no_grad():
        for task_name in task_names:
            task_family = get_pdebench_spec(task_name).family
            try:
                dataset = PDEBenchDataset(
                    PDEBenchConfig(
                        task=task_name,
                        split=data_cfg.get("split", "train"),
                        root=task_roots.get(task_name, data_cfg.get("root")),
                        param_keys=tuple(data_cfg.get("param_keys", ())),
                        bc_keys=tuple(data_cfg.get("bc_keys", ())),
                        max_samples=data_cfg.get("max_samples"),
                    )
                )
            except FileNotFoundError:
                if not skip_missing_tasks:
                    raise
                skipped_missing_tasks.append(task_name)
                continue
            if len(dataset) == 0:
                continue
            sample_fields = dataset.fields[0]
            grid_shape = infer_grid_shape(sample_fields)
            coords = make_grid_coords(grid_shape, device)
            base_cond: dict[str, torch.Tensor] = {}
            if bool(cfg.get("training", {}).get("auto_conditioning", False)):
                extras = pdebench_conditioning_extras(
                    task_name=task_name, grid_shape=grid_shape, task_vocab=task_names
                )
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
                context_roll_shift = 0.0
                if context_roll_shift_cfg and _roll_shift_estimator_applies(
                    cfg=context_roll_shift_cfg,
                    task_name=task_name,
                    task_family=task_family,
                    horizon=int(context_roll_shift_cfg["min_horizon"]),
                ):
                    context_roll_shift = _estimate_context_roll_shift(
                        fields=fields,
                        grid_shape=grid_shape,
                        candidate_shifts=context_roll_shift_cfg.get("candidate_shifts", ()),
                        context_transitions=int(context_roll_shift_cfg["context_transitions"]),
                        coefficients=context_roll_shift_cfg.get("coefficients", {}),
                    )
                data_conditioned_context_shift: float | None = None
                if (
                    data_conditioned_roll_shift_cfg
                    and "context_transitions" in data_conditioned_roll_shift_cfg
                    and _roll_shift_estimator_applies(
                        cfg=data_conditioned_roll_shift_cfg,
                        task_name=task_name,
                        task_family=task_family,
                        horizon=int(data_conditioned_roll_shift_cfg["min_horizon"]),
                    )
                ):
                    data_conditioned_context_shift = _estimate_context_roll_shift(
                        fields=fields,
                        grid_shape=grid_shape,
                        candidate_shifts=data_conditioned_roll_shift_cfg.get(
                            "candidate_shifts", ()
                        ),
                        context_transitions=int(
                            data_conditioned_roll_shift_cfg["context_transitions"]
                        ),
                        coefficients={"slope": 1.0, "intercept": 0.0},
                    )
                initial_cond = pdebench_condition_step(
                    params,
                    bc,
                    batch_size=1,
                    step=0,
                    extras=base_cond,
                    param_vocab=param_vocab,
                    bc_vocab=bc_vocab,
                )
                state = LatentState(
                    z=latent_seq[0:1].to(device),
                    t=torch.tensor(0.0, device=device),
                    cond={k: v.to(device) for k, v in initial_cond.items()},
                )

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
                    state = LatentState(
                        z=state.z, t=state.t, cond={k: v.to(device) for k, v in cond.items()}
                    )
                    state = operator(state, dt_tensor)
                    decoded = decoder(coords, state.z, conditioning={})
                    if field_name not in decoded:
                        raise KeyError(f"Decoder did not produce requested field '{field_name}'")
                    pred_field = decoded[field_name].detach().cpu()
                    horizon = step + 1
                    persistence_field = _flatten_field_step(fields[step], grid_shape).cpu()
                    raw_pred_field = pred_field
                    task_residual_alpha = _resolve_residual_alpha(
                        task_name=task_name,
                        task_family=task_family,
                        horizon=horizon,
                        residual_alpha=residual_alpha,
                        residual_alpha_by_task=residual_alpha_by_task,
                        residual_alpha_by_family=residual_alpha_by_family,
                        residual_alpha_by_horizon=residual_alpha_by_horizon,
                        residual_alpha_by_task_horizon=residual_alpha_by_task_horizon,
                        residual_alpha_by_family_horizon=residual_alpha_by_family_horizon,
                    )
                    if residual_gate_cfg:
                        gate_features = _gate_features(
                            pred_field=pred_field,
                            persistence_field=persistence_field,
                            horizon=horizon,
                            rollout_steps=steps,
                        )
                        task_residual_alpha = _resolve_gate_alpha(
                            static_alpha=task_residual_alpha,
                            gate_cfg=residual_gate_cfg,
                            task_name=task_name,
                            task_family=task_family,
                            horizon=horizon,
                            features=gate_features,
                        )
                        _append_stat(
                            alpha_stats, "decoded_residual_gate_alpha", task_residual_alpha
                        )
                        _append_stat(
                            alpha_stats,
                            f"task_{task_name}_decoded_residual_gate_alpha",
                            task_residual_alpha,
                        )
                        _append_stat(
                            alpha_stats,
                            f"family_{task_family}_decoded_residual_gate_alpha",
                            task_residual_alpha,
                        )
                        _append_stat(
                            alpha_stats,
                            f"decoded_residual_gate_h{horizon}_alpha",
                            task_residual_alpha,
                        )
                    if task_residual_alpha != 1.0:
                        pred_field = persistence_field + task_residual_alpha * (
                            pred_field - persistence_field
                        )
                    roll_shift = _resolve_roll_shift(
                        task_name=task_name,
                        task_family=task_family,
                        horizon=horizon,
                        shift_by_task=roll_shift_by_task,
                        shift_by_family=roll_shift_by_family,
                        shift_by_task_horizon=roll_shift_by_task_horizon,
                        shift_by_family_horizon=roll_shift_by_family_horizon,
                    )
                    if roll_shift == 0 and _roll_shift_estimator_applies(
                        cfg=data_conditioned_roll_shift_cfg,
                        task_name=task_name,
                        task_family=task_family,
                        horizon=horizon,
                    ):
                        roll_shift = _estimate_data_conditioned_roll_shift(
                            field=persistence_field,
                            cfg=data_conditioned_roll_shift_cfg,
                            horizon=horizon,
                            rollout_steps=steps,
                            context_shift=data_conditioned_context_shift,
                            params=params,
                        )
                        _append_stat(
                            shift_stats,
                            "decoded_data_conditioned_roll_shift",
                            float(roll_shift),
                        )
                        _append_stat(
                            shift_stats,
                            f"task_{task_name}_decoded_data_conditioned_roll_shift",
                            float(roll_shift),
                        )
                        _append_stat(
                            shift_stats,
                            f"family_{task_family}_decoded_data_conditioned_roll_shift",
                            float(roll_shift),
                        )
                        _append_stat(
                            shift_stats,
                            f"decoded_data_conditioned_roll_shift_h{horizon}",
                            float(roll_shift),
                        )
                        if (
                            _roll_shift_estimator_mode(data_conditioned_roll_shift_cfg)
                            == "roll_persistence"
                        ):
                            pred_field = persistence_field
                    if roll_shift == 0 and _roll_shift_estimator_applies(
                        cfg=observed_roll_shift_cfg,
                        task_name=task_name,
                        task_family=task_family,
                        horizon=horizon,
                    ):
                        previous_field = _flatten_field_step(fields[step - 1], grid_shape).cpu()
                        roll_shift = _estimate_observed_roll_shift(
                            previous_field=previous_field,
                            current_field=persistence_field,
                            grid_shape=grid_shape,
                            candidate_shifts=observed_roll_shift_cfg.get("candidate_shifts", ()),
                        )
                        _append_stat(shift_stats, "decoded_observed_roll_shift", float(roll_shift))
                        _append_stat(
                            shift_stats,
                            f"task_{task_name}_decoded_observed_roll_shift",
                            float(roll_shift),
                        )
                        _append_stat(
                            shift_stats,
                            f"family_{task_family}_decoded_observed_roll_shift",
                            float(roll_shift),
                        )
                        _append_stat(
                            shift_stats,
                            f"decoded_observed_roll_shift_h{horizon}",
                            float(roll_shift),
                        )
                    if roll_shift == 0 and _roll_shift_estimator_applies(
                        cfg=prediction_roll_shift_cfg,
                        task_name=task_name,
                        task_family=task_family,
                        horizon=horizon,
                    ):
                        roll_shift = _estimate_prediction_roll_shift(
                            persistence_field=persistence_field,
                            predicted_field=raw_pred_field,
                            grid_shape=grid_shape,
                            candidate_shifts=prediction_roll_shift_cfg.get("candidate_shifts", ()),
                        )
                        _append_stat(
                            shift_stats, "decoded_prediction_roll_shift", float(roll_shift)
                        )
                        _append_stat(
                            shift_stats,
                            f"task_{task_name}_decoded_prediction_roll_shift",
                            float(roll_shift),
                        )
                        _append_stat(
                            shift_stats,
                            f"family_{task_family}_decoded_prediction_roll_shift",
                            float(roll_shift),
                        )
                        _append_stat(
                            shift_stats,
                            f"decoded_prediction_roll_shift_h{horizon}",
                            float(roll_shift),
                        )
                        if (
                            _roll_shift_estimator_mode(prediction_roll_shift_cfg)
                            == "roll_persistence"
                        ):
                            pred_field = persistence_field
                    if roll_shift == 0 and _roll_shift_estimator_applies(
                        cfg=context_roll_shift_cfg,
                        task_name=task_name,
                        task_family=task_family,
                        horizon=horizon,
                    ):
                        roll_shift = context_roll_shift
                        _append_stat(shift_stats, "decoded_context_roll_shift", float(roll_shift))
                        _append_stat(
                            shift_stats,
                            f"task_{task_name}_decoded_context_roll_shift",
                            float(roll_shift),
                        )
                        _append_stat(
                            shift_stats,
                            f"family_{task_family}_decoded_context_roll_shift",
                            float(roll_shift),
                        )
                        _append_stat(
                            shift_stats,
                            f"decoded_context_roll_shift_h{horizon}",
                            float(roll_shift),
                        )
                        if _roll_shift_estimator_mode(context_roll_shift_cfg) == "roll_persistence":
                            pred_field = persistence_field
                    pred_field = _roll_flattened_grid(pred_field, grid_shape, shift_x=roll_shift)
                    target_field = _flatten_field_step(fields[step + 1], grid_shape).cpu()
                    total_pred.append(pred_field)
                    total_target.append(target_field)
                    per_task_pred.setdefault(task_name, []).append(pred_field)
                    per_task_target.setdefault(task_name, []).append(target_field)
                    per_family_pred.setdefault(task_family, []).append(pred_field)
                    per_family_target.setdefault(task_family, []).append(target_field)
                    track_horizon = (
                        report_all_horizon_metrics or horizon in _DEFAULT_DECODED_HORIZONS
                    )
                    if track_horizon:
                        horizon_pred.setdefault(horizon, []).append(pred_field)
                        horizon_target.setdefault(horizon, []).append(target_field)
                        per_task_horizon_pred.setdefault(task_name, {}).setdefault(
                            horizon, []
                        ).append(pred_field)
                        per_task_horizon_target.setdefault(task_name, {}).setdefault(
                            horizon, []
                        ).append(target_field)
                        per_family_horizon_pred.setdefault(task_family, {}).setdefault(
                            horizon, []
                        ).append(pred_field)
                        per_family_horizon_target.setdefault(task_family, {}).setdefault(
                            horizon, []
                        ).append(target_field)
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
    if horizon_pred.get(1):
        step1_stats = _aggregate_chunk_metrics(horizon_pred[1], horizon_target[1])
        metrics["decoded_step1_nrmse"] = step1_stats["nrmse"]
        metrics["decoded_step1_rrmse"] = step1_stats["rrmse"]
    for horizon in sorted(horizon_pred):
        if horizon == 1 and not report_all_horizon_metrics:
            continue
        if horizon in (4, 16) or report_all_horizon_metrics:
            if not horizon_pred[horizon]:
                continue
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
        for horizon, horizon_pred_chunks in sorted(
            per_task_horizon_pred.get(task_name, {}).items()
        ):
            if horizon == 1 and not report_all_horizon_metrics:
                continue
            if horizon in (4, 16) or report_all_horizon_metrics:
                horizon_stats = _aggregate_chunk_metrics(
                    horizon_pred_chunks,
                    per_task_horizon_target[task_name][horizon],
                )
                metrics[f"task_{task_name}_decoded_h{horizon}_nrmse"] = horizon_stats["nrmse"]
                metrics[f"task_{task_name}_decoded_h{horizon}_rrmse"] = horizon_stats["rrmse"]
    for family_name, pred_chunks in per_family_pred.items():
        family_stats = _aggregate_chunk_metrics(pred_chunks, per_family_target[family_name])
        metrics[f"family_{family_name}_decoded_rollout_nrmse"] = family_stats["nrmse"]
        metrics[f"family_{family_name}_decoded_rollout_rrmse"] = family_stats["rrmse"]
        step1_pred = per_family_step1_pred.get(family_name)
        step1_target = per_family_step1_target.get(family_name)
        if step1_pred and step1_target:
            step1_stats = _aggregate_chunk_metrics(step1_pred, step1_target)
            metrics[f"family_{family_name}_decoded_step1_nrmse"] = step1_stats["nrmse"]
        for horizon, horizon_pred_chunks in sorted(
            per_family_horizon_pred.get(family_name, {}).items()
        ):
            if horizon == 1 and not report_all_horizon_metrics:
                continue
            if horizon in (4, 16) or report_all_horizon_metrics:
                horizon_stats = _aggregate_chunk_metrics(
                    horizon_pred_chunks,
                    per_family_horizon_target[family_name][horizon],
                )
                metrics[f"family_{family_name}_decoded_h{horizon}_nrmse"] = horizon_stats["nrmse"]
                metrics[f"family_{family_name}_decoded_h{horizon}_rrmse"] = horizon_stats["rrmse"]
    _add_alpha_stats(metrics, alpha_stats)
    _add_alpha_stats(metrics, shift_stats)
    task_extra: str | list[str] = task_names[0] if len(task_names) == 1 else task_names
    return MetricReport(
        metrics=metrics,
        extra={
            "task": task_extra,
            "split": data_cfg.get("split", "train"),
            "decoded_persistence_residual_alpha": residual_alpha,
            "decoded_persistence_residual_alpha_by_task": residual_alpha_by_task,
            "decoded_persistence_residual_alpha_by_family": residual_alpha_by_family,
            "decoded_persistence_residual_alpha_by_horizon": residual_alpha_by_horizon,
            "decoded_persistence_residual_alpha_by_task_horizon": residual_alpha_by_task_horizon,
            "decoded_persistence_residual_alpha_by_family_horizon": residual_alpha_by_family_horizon,
            "decoded_persistence_residual_gate": residual_gate_cfg,
            "decoded_roll_shift_by_task": roll_shift_by_task,
            "decoded_roll_shift_by_family": roll_shift_by_family,
            "decoded_roll_shift_by_task_horizon": roll_shift_by_task_horizon,
            "decoded_roll_shift_by_family_horizon": roll_shift_by_family_horizon,
            "decoded_observed_roll_shift_estimator": observed_roll_shift_cfg,
            "decoded_prediction_roll_shift_estimator": prediction_roll_shift_cfg,
            "decoded_context_roll_shift_estimator": context_roll_shift_cfg,
            "decoded_data_conditioned_roll_shift_estimator": data_conditioned_roll_shift_cfg,
            "report_all_horizon_metrics": report_all_horizon_metrics,
            "skip_missing_tasks": skip_missing_tasks,
            "skipped_missing_tasks": skipped_missing_tasks,
            "task_roots": task_roots,
        },
    )


def evaluate_latent_model(
    cfg: dict[str, Any],
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
