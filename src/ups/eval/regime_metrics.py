from __future__ import annotations

"""Scale-coherent metrics for comparing physical regimes within one task."""

import math
from collections.abc import Sequence

import torch


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
