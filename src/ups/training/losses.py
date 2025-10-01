from __future__ import annotations

"""Collection of core training losses for latent evolution."""

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass
class LossBundle:
    """Container for individual loss terms and their weighted total."""

    total: Tensor
    components: Dict[str, Tensor]


def mse(pred: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    return F.mse_loss(pred, target, reduction=reduction)


def inverse_encoding_loss(encoded: Tensor, reconstructed: Tensor, weight: float = 1.0) -> Tensor:
    loss = mse(reconstructed, encoded.detach())
    return weight * loss


def inverse_decoding_loss(pred_fields: Mapping[str, Tensor], target_fields: Mapping[str, Tensor], weight: float = 1.0) -> Tensor:
    if pred_fields.keys() != target_fields.keys():
        missing = pred_fields.keys() ^ target_fields.keys()
        raise KeyError(f"Pred/target field mismatch: {missing}")
    losses = []
    for name in pred_fields:
        losses.append(mse(pred_fields[name], target_fields[name]))
    return weight * torch.stack(losses).mean()


def one_step_loss(pred_next: Tensor, target_next: Tensor, weight: float = 1.0) -> Tensor:
    return weight * mse(pred_next, target_next)


def rollout_loss(pred_seq: Tensor, target_seq: Tensor, weight: float = 1.0) -> Tensor:
    if pred_seq.shape != target_seq.shape:
        raise ValueError("Rollout sequences must share shape")
    return weight * mse(pred_seq, target_seq)


def spectral_loss(pred: Tensor, target: Tensor, weight: float = 1.0) -> Tensor:
    pred_fft = torch.fft.rfft(pred, dim=-1)
    target_fft = torch.fft.rfft(target, dim=-1)
    loss = mse(pred_fft.abs(), target_fft.abs())
    return weight * loss


def consistency_loss(pred: Tensor, target: Tensor, weight: float = 1.0) -> Tensor:
    pred_mean = pred.mean(dim=-1)
    target_mean = target.mean(dim=-1)
    return weight * mse(pred_mean, target_mean)


def edge_total_variation(latent: Tensor, edges: Tensor, weight: float = 1.0) -> Tensor:
    if edges.numel() == 0:
        return latent.new_tensor(0.0)
    src, dst = edges[:, 0].long(), edges[:, 1].long()
    diffs = latent[:, src, :] - latent[:, dst, :]
    tv = diffs.abs().mean()
    return weight * tv


def compute_loss_bundle(
    *,
    encoded: Tensor,
    reconstructed: Tensor,
    decoded_pred: Mapping[str, Tensor],
    decoded_target: Mapping[str, Tensor],
    pred_next: Tensor,
    target_next: Tensor,
    pred_rollout: Tensor,
    target_rollout: Tensor,
    spectral_pred: Tensor,
    spectral_target: Tensor,
    consistency_pred: Tensor,
    consistency_target: Tensor,
    latent_for_tv: Tensor,
    edges: Tensor,
    weights: Optional[Mapping[str, float]] = None,
) -> LossBundle:
    weights = weights or {}
    comp = {}
    comp["L_inv_enc"] = inverse_encoding_loss(encoded, reconstructed, weights.get("L_inv_enc", 1.0))
    comp["L_inv_dec"] = inverse_decoding_loss(decoded_pred, decoded_target, weights.get("L_inv_dec", 1.0))
    comp["L_one_step"] = one_step_loss(pred_next, target_next, weights.get("L_one_step", 1.0))
    comp["L_rollout"] = rollout_loss(pred_rollout, target_rollout, weights.get("L_rollout", 1.0))
    comp["L_spec"] = spectral_loss(spectral_pred, spectral_target, weights.get("L_spec", 1.0))
    comp["L_cons"] = consistency_loss(consistency_pred, consistency_target, weights.get("L_cons", 1.0))
    comp["L_tv_edge"] = edge_total_variation(latent_for_tv, edges, weights.get("L_tv_edge", 1.0))
    total = torch.stack([c for c in comp.values() if c.numel() == 1]).sum()
    return LossBundle(total=total, components=comp)

