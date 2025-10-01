from __future__ import annotations

"""Evaluation metrics for UPS models (general + PDEBench)."""

import torch


def nrmse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalised root mean squared error."""

    mse = torch.mean((pred - target) ** 2)
    denom = torch.mean(target ** 2) + eps
    return torch.sqrt(mse / denom)


def spectral_energy_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compare spectral energy density via FFT magnitudes."""

    pred_fft = torch.fft.rfft(pred, dim=-1)
    target_fft = torch.fft.rfft(target, dim=-1)
    pred_energy = torch.mean(pred_fft.abs() ** 2)
    target_energy = torch.mean(target_fft.abs() ** 2)
    return torch.abs(pred_energy - target_energy) / (target_energy + 1e-8)


def conservation_gap(pred: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
    """Difference in conserved quantities (e.g., mass/energy)."""

    return torch.abs(pred.sum(dim=-1) - baseline.sum(dim=-1)).mean()


def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target))


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


def relative_rrmse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.sqrt(torch.mean((pred - target) ** 2) / (torch.mean(target ** 2) + eps))
