from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch.nn import functional as F


def global_nrmse(prediction: torch.Tensor, target: torch.Tensor) -> float:
    if prediction.shape != target.shape:
        raise ValueError(f"prediction/target shape mismatch: {prediction.shape} != {target.shape}")
    numerator = (prediction.double() - target.double()).square().sum()
    denominator = target.double().square().sum()
    if denominator <= 0:
        raise ValueError("NRMSE target energy must be positive")
    return float(torch.sqrt(numerator / denominator).item())


def linear_cka(left: torch.Tensor, right: torch.Tensor) -> float:
    left_flat, right_flat = _paired_flatten(left, right)
    left_flat = left_flat.double()
    right_flat = right_flat.double()
    left_centered = left_flat - left_flat.mean(dim=0, keepdim=True)
    right_centered = right_flat - right_flat.mean(dim=0, keepdim=True)
    cross = left_centered.T @ right_centered
    numerator = cross.square().sum()
    denominator = torch.linalg.norm(left_centered.T @ left_centered) * torch.linalg.norm(
        right_centered.T @ right_centered
    )
    if denominator <= 0:
        return 0.0
    return float((numerator / denominator).item())


def effective_rank(latents: torch.Tensor) -> float:
    flattened = _flatten_samples(latents).double()
    centered = flattened - flattened.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    energy = singular_values.square()
    total = energy.sum()
    if total <= 0:
        return 0.0
    probabilities = energy / total
    nonzero = probabilities > 0
    entropy = -(probabilities[nonzero] * probabilities[nonzero].log()).sum()
    return float(torch.exp(entropy).item())


def paired_retrieval(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    left_flat, right_flat = _paired_flatten(left, right)
    left_normalized = F.normalize(left_flat.double(), dim=1)
    right_normalized = F.normalize(right_flat.double(), dim=1)
    similarities = left_normalized @ right_normalized.T
    identities = torch.arange(left_flat.shape[0], device=similarities.device)
    left_to_right = (similarities.argmax(dim=1) == identities).double().mean()
    right_to_left = (similarities.argmax(dim=0) == identities).double().mean()
    return {
        "left_to_right_top1": float(left_to_right.item()),
        "right_to_left_top1": float(right_to_left.item()),
        "symmetric_top1": float(((left_to_right + right_to_left) / 2).item()),
        "chance_top1": 1.0 / left_flat.shape[0],
    }


def paired_latent_report(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    left_flat, right_flat = _paired_flatten(left, right)
    left_standardized = F.layer_norm(left_flat, (left_flat.shape[-1],))
    right_standardized = F.layer_norm(right_flat, (right_flat.shape[-1],))
    return {
        "pair_count": left_flat.shape[0],
        "latent_shape": list(left.shape[1:]),
        "standardized_pair_rmse": float(
            torch.sqrt(F.mse_loss(left_standardized, right_standardized)).item()
        ),
        "linear_cka": linear_cka(left, right),
        "retrieval": paired_retrieval(left, right),
        "effective_rank": {
            "left": effective_rank(left),
            "right": effective_rank(right),
        },
    }


def cross_discretization_codec_report(
    predictions: Mapping[str, Mapping[str, torch.Tensor]],
    targets: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Measure every encoder-source/query-discretization combination.

    ``predictions[encoded_as][queried_as]`` must be decoded at the exact
    coordinates represented by ``targets[queried_as]``. In a canonical codec,
    one shared decoder produces all entries; source and query labels are
    evidence dimensions, not routing inputs.
    """

    representations = tuple(targets)
    if len(representations) < 2:
        raise ValueError("at least two discretizations are required")
    if set(predictions) != set(representations):
        raise ValueError("prediction encoder sources must match target representations")

    matrix: dict[str, dict[str, float]] = {}
    for encoded_as in representations:
        if set(predictions[encoded_as]) != set(representations):
            raise ValueError(f"prediction query sets for '{encoded_as}' do not match targets")
        matrix[encoded_as] = {
            queried_as: global_nrmse(predictions[encoded_as][queried_as], targets[queried_as])
            for queried_as in representations
        }

    within = [matrix[name][name] for name in representations]
    cross = [
        matrix[source][query]
        for source in representations
        for query in representations
        if source != query
    ]
    return {
        "representations": list(representations),
        "global_nrmse_matrix": matrix,
        "mean_within_nrmse": sum(within) / len(within),
        "mean_cross_nrmse": sum(cross) / len(cross),
        "cross_to_within_ratio": (
            (sum(cross) / len(cross)) / (sum(within) / len(within)) if sum(within) > 0 else None
        ),
    }


def _paired_flatten(left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    left_flat = _flatten_samples(left)
    right_flat = _flatten_samples(right)
    if left_flat.shape != right_flat.shape:
        raise ValueError(
            "paired latents must have identical batch and flattened dimensions; "
            f"got {left_flat.shape} and {right_flat.shape}"
        )
    if left_flat.shape[0] < 2:
        raise ValueError("paired latent metrics require at least two physical states")
    return left_flat, right_flat


def _flatten_samples(values: torch.Tensor) -> torch.Tensor:
    if values.dim() < 2:
        raise ValueError("values must include sample and feature dimensions")
    return values.reshape(values.shape[0], -1)
