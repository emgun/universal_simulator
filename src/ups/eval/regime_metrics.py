from __future__ import annotations

"""Scale-coherent metrics for comparing physical regimes within one task."""

import math
from collections.abc import Sequence

import torch


def weighted_reconstructed_nrmse(
    regime_global_scale_nrmse: Sequence[float], regime_element_counts: Sequence[int]
) -> float:
    """Reconstruct task NRMSE from global-scale regime metrics and counts."""

    if len(regime_global_scale_nrmse) != len(regime_element_counts) or not regime_element_counts:
        raise ValueError("regime metrics and element counts must be non-empty and aligned")
    weighted_sum = 0.0
    total_count = 0
    for value, count in zip(regime_global_scale_nrmse, regime_element_counts, strict=True):
        numeric = float(value)
        integer_count = int(count)
        if not math.isfinite(numeric) or numeric < 0:
            raise ValueError("regime metrics must be finite and non-negative")
        if integer_count <= 0:
            raise ValueError("regime element counts must be positive")
        weighted_sum += integer_count * numeric**2
        total_count += integer_count
    return math.sqrt(weighted_sum / total_count)


def regime_spread_ratio(regime_global_scale_nrmse: float, task_primary_nrmse: float) -> float:
    """Return the scale-coherent regime/task ratio used by strat-v1.1."""

    regime_value = float(regime_global_scale_nrmse)
    task_value = float(task_primary_nrmse)
    if not math.isfinite(regime_value) or regime_value < 0:
        raise ValueError("regime metric must be finite and non-negative")
    if not math.isfinite(task_value) or task_value <= 0:
        raise ValueError("task primary metric must be finite and positive")
    return regime_value / task_value


def aligned_element_count(
    predictions: Sequence[torch.Tensor], targets: Sequence[torch.Tensor]
) -> int:
    """Return the element count for aligned, finite prediction/target chunks."""

    if len(predictions) != len(targets) or not predictions:
        raise ValueError("predictions and targets must be non-empty and aligned")
    total = 0
    for prediction, target in zip(predictions, targets, strict=True):
        prediction_tensor = torch.as_tensor(prediction)
        target_tensor = torch.as_tensor(target)
        if prediction_tensor.shape != target_tensor.shape:
            raise ValueError("prediction and target shapes must match")
        if not bool(torch.isfinite(prediction_tensor).all()) or not bool(
            torch.isfinite(target_tensor).all()
        ):
            raise ValueError("predictions and targets must contain only finite values")
        total += int(target_tensor.numel())
    return total


def _sum_squares_and_count(chunks: Sequence[torch.Tensor]) -> tuple[float, int]:
    if not chunks:
        raise ValueError("metric chunks must be non-empty")
    total_sq = 0.0
    total_count = 0
    for chunk in chunks:
        tensor = torch.as_tensor(chunk).to(dtype=torch.float64)
        if tensor.numel() == 0:
            raise ValueError("metric chunks must not contain empty tensors")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("metric chunks must contain only finite values")
        total_sq += float(tensor.pow(2).sum().item())
        total_count += int(tensor.numel())
    return total_sq, total_count


def global_scale_regime_nrmse(
    regime_predictions: Sequence[torch.Tensor],
    regime_targets: Sequence[torch.Tensor],
    task_targets: Sequence[torch.Tensor],
    *,
    eps: float = 1e-8,
) -> float:
    """Normalize regime RMSE by the target RMS of the complete task."""

    if len(regime_predictions) != len(regime_targets) or not regime_predictions:
        raise ValueError("regime predictions and targets must be non-empty and aligned")
    if eps <= 0:
        raise ValueError("eps must be positive")

    error_chunks = []
    for prediction, target in zip(regime_predictions, regime_targets, strict=True):
        prediction_tensor = torch.as_tensor(prediction).to(dtype=torch.float64)
        target_tensor = torch.as_tensor(target).to(dtype=torch.float64)
        if prediction_tensor.shape != target_tensor.shape:
            raise ValueError("regime prediction and target shapes must match")
        error_chunks.append(prediction_tensor - target_tensor)

    error_sq, error_count = _sum_squares_and_count(error_chunks)
    target_sq, target_count = _sum_squares_and_count(task_targets)
    regime_mse = error_sq / error_count
    task_target_mean_sq = target_sq / target_count
    return math.sqrt(regime_mse / (task_target_mean_sq + eps))
