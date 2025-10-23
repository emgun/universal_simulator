from __future__ import annotations

"""Collection of core training losses for latent evolution."""

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class LossBundle:
    """Container for individual loss terms and their weighted total."""

    total: Tensor
    components: Dict[str, Tensor]


def mse(pred: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    return F.mse_loss(pred, target, reduction=reduction)


# ============================================================================
# Legacy loss functions (kept for backward compatibility)
# ============================================================================

def inverse_encoding_loss(encoded: Tensor, reconstructed: Tensor, weight: float = 1.0) -> Tensor:
    """Legacy inverse encoding loss (latent-to-latent reconstruction).

    Note: This is NOT the UPT inverse encoding loss. Use `upt_inverse_encoding_loss`
    for true E/A/D disentanglement.
    """
    loss = mse(reconstructed, encoded.detach())
    return weight * loss


def inverse_decoding_loss(pred_fields: Mapping[str, Tensor], target_fields: Mapping[str, Tensor], weight: float = 1.0) -> Tensor:
    """Legacy inverse decoding loss (field-to-field MSE).

    Note: This is NOT the UPT inverse decoding loss. Use `upt_inverse_decoding_loss`
    for true E/A/D disentanglement.
    """
    if pred_fields.keys() != target_fields.keys():
        missing = pred_fields.keys() ^ target_fields.keys()
        raise KeyError(f"Pred/target field mismatch: {missing}")
    losses = []
    for name in pred_fields:
        losses.append(mse(pred_fields[name], target_fields[name]))
    return weight * torch.stack(losses).mean()


# ============================================================================
# UPT Inverse Losses (True E/A/D Disentanglement)
# ============================================================================

def upt_inverse_encoding_loss(
    input_fields: Mapping[str, Tensor],
    input_coords: Tensor,
    latent: Tensor,
    decoder: nn.Module,
    meta: Optional[Mapping[str, Any]] = None,
    num_query_points: Optional[int] = None,
    weight: float = 1.0,
) -> Tensor:
    """UPT inverse encoding loss: ensures latent can reconstruct original input.

    This loss enforces that the encoder output (latent) contains sufficient
    information to reconstruct the original input fields when decoded.

    Algorithm:
        1. Encode input fields to latent: z = E(u)
        2. Decode latent back to input positions: u_recon = D(z, input_positions)
        3. Compute MSE: L = MSE(u_recon, u)

    Parameters
    ----------
    input_fields:
        Original physical fields as dict of tensors (batch, points, channels)
    input_coords:
        Spatial coordinates of input points (batch, points, coord_dim)
    latent:
        Encoded latent tokens (batch, tokens, latent_dim)
    decoder:
        Decoder module (e.g., AnyPointDecoder)
    meta:
        Optional metadata dict (e.g., grid_shape, boundary conditions)
    num_query_points:
        Number of random points to sample for loss computation (for efficiency).
        If None, uses all input points.
    weight:
        Loss weight coefficient

    Returns
    -------
    Weighted MSE loss between reconstructed and original fields
    """
    batch_size, num_points, coord_dim = input_coords.shape

    # Sample subset of query points if requested (for efficiency)
    if num_query_points is not None and num_query_points < num_points:
        indices = torch.randperm(num_points, device=input_coords.device)[:num_query_points]
        query_coords = input_coords[:, indices, :]
        target_fields = {k: v[:, indices, :] for k, v in input_fields.items()}
    else:
        query_coords = input_coords
        target_fields = input_fields

    # Decode latent at query positions
    reconstructed_fields = decoder(query_coords, latent)

    # Compute MSE for each field
    losses = []
    for name in target_fields:
        if name in reconstructed_fields:
            losses.append(mse(reconstructed_fields[name], target_fields[name]))

    if not losses:
        # No matching fields - return zero loss
        return latent.new_tensor(0.0)

    return weight * torch.stack(losses).mean()


def upt_inverse_decoding_loss(
    latent: Tensor,
    decoder: nn.Module,
    encoder: nn.Module,
    query_coords: Tensor,
    original_coords: Tensor,
    meta: Optional[Mapping[str, Any]] = None,
    num_query_points: int = 2048,
    weight: float = 1.0,
) -> Tensor:
    """UPT inverse decoding loss: ensures decoder output can be re-encoded to latent.

    This loss enforces that the decoder output contains sufficient information
    to reconstruct the latent representation when re-encoded.

    Algorithm:
        1. Decode latent to query points: u_decoded = D(z, query_positions)
        2. Re-encode decoded fields: z_recon = E(u_decoded)
        3. Compute MSE in latent space: L = MSE(z_recon, z)

    Parameters
    ----------
    latent:
        Original latent tokens (batch, tokens, latent_dim)
    decoder:
        Decoder module (e.g., AnyPointDecoder)
    encoder:
        Encoder module (e.g., GridEncoder)
    query_coords:
        Spatial coordinates for decoding (batch, num_points, coord_dim)
    original_coords:
        Full coordinate grid for re-encoding (batch, total_points, coord_dim)
    meta:
        Optional metadata dict (e.g., grid_shape for grid encoder)
    num_query_points:
        Number of random query points to sample for decoding (for efficiency)
    weight:
        Loss weight coefficient

    Returns
    -------
    Weighted MSE loss between re-encoded and original latent

    Note
    ----
    This loss requires a forward pass through both decoder and encoder,
    which can be expensive. Use `num_query_points` to control computational cost.
    """
    batch_size, total_points, coord_dim = original_coords.shape

    # Sample random query points for decoding
    if num_query_points < total_points:
        indices = torch.randperm(total_points, device=query_coords.device)[:num_query_points]
        sampled_coords = original_coords[:, indices, :]
    else:
        sampled_coords = query_coords

    # Decode latent at sampled query positions
    decoded_fields = decoder(sampled_coords, latent)

    # Re-encode decoded fields back to latent
    # Note: This assumes encoder can handle the decoded field dict format
    try:
        latent_reconstructed = encoder(decoded_fields, original_coords, meta=meta)
    except Exception as e:
        # Encoder may not support arbitrary field inputs (e.g., needs specific format)
        # Return zero loss if re-encoding fails
        print(f"Warning: Inverse decoding loss failed to re-encode: {e}")
        return latent.new_tensor(0.0)

    # MSE in latent space (detach original to avoid double gradients)
    return weight * mse(latent_reconstructed, latent.detach())


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

