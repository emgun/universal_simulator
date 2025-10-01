from __future__ import annotations

"""Calibration utilities (reliability diagrams and temperature scaling)."""

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn


def reliability_diagram(probabilities: torch.Tensor, targets: torch.Tensor, n_bins: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute empirical accuracy and mean confidence per bin.

    Parameters
    ----------
    probabilities:
        Tensor of predicted probabilities (after softmax/sigmoid) in ``[0, 1]``.
    targets:
        Binary targets (0/1) with the same shape as ``probabilities``.
    n_bins:
        Number of equally spaced confidence bins.
    """

    if probabilities.shape != targets.shape:
        raise ValueError("Probability and target tensors must share shape")
    probs = probabilities.detach().flatten()
    targs = targets.detach().flatten()
    bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1, device=probs.device)
    accuracies = torch.zeros(n_bins, device=probs.device)
    confidences = torch.zeros(n_bins, device=probs.device)
    for b in range(n_bins):
        lower, upper = bin_boundaries[b], bin_boundaries[b + 1]
        mask = (probs >= lower) & (probs < upper if b < n_bins - 1 else probs <= upper)
        if mask.any():
            bucket_probs = probs[mask]
            bucket_targets = targs[mask]
            confidences[b] = bucket_probs.mean()
            accuracies[b] = bucket_targets.float().mean()
        else:
            confidences[b] = 0.0
            accuracies[b] = 0.0
    return confidences, accuracies


def expected_calibration_error(probabilities: torch.Tensor, targets: torch.Tensor, n_bins: int = 10) -> torch.Tensor:
    confidences, accuracies = reliability_diagram(probabilities, targets, n_bins)
    probs = probabilities.detach().flatten()
    bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1, device=probs.device)
    weights = torch.zeros(n_bins, device=probs.device)
    for b in range(n_bins):
        lower, upper = bin_boundaries[b], bin_boundaries[b + 1]
        mask = (probs >= lower) & (probs < upper if b < n_bins - 1 else probs <= upper)
        weights[b] = mask.float().mean()
    ece = torch.sum(weights * (confidences - accuracies).abs())
    return ece


@dataclass
class TemperatureScaler:
    """Learn a single temperature parameter for logits."""

    temperature: torch.nn.Parameter = torch.nn.Parameter(torch.tensor(1.0))

    def to(self, device: torch.device | str) -> TemperatureScaler:
        device = torch.device(device)
        self.temperature = nn.Parameter(self.temperature.detach().to(device))
        return self

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp_min(1e-4)

    def fit(self, logits: torch.Tensor, targets: torch.Tensor, lr: float = 0.01, steps: int = 100) -> None:
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=steps)

        targets = targets.long()

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            scaled = self.forward(logits)
            loss = nn.CrossEntropyLoss()(scaled, targets)
            loss.backward()
            return loss

        optimizer.step(closure)

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        return self.forward(logits)

