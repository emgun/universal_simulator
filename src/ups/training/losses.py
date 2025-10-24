from __future__ import annotations

"""Collection of core training losses for latent evolution."""

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Callable

import torch
from torch import nn, Tensor
from torch.nn import functional as F


@dataclass
class LossBundle:
    """Container for individual loss terms and their weighted total."""

    total: Tensor
    components: Dict[str, Tensor]


def mse(pred: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    return F.mse_loss(pred, target, reduction=reduction)


def inverse_encoding_loss(
    input_fields: Mapping[str, torch.Tensor],
    latent: torch.Tensor,
    decoder: nn.Module,
    input_positions: torch.Tensor,
    weight: float = 1.0,
) -> torch.Tensor:
    """UPT Inverse Encoding Loss: Ensures latent is decodable.

    Flow: input_fields → [already encoded to latent] → decoder → reconstructed_fields
    Loss: MSE(reconstructed_fields, input_fields) in physical space

    This ensures that the encoder's output can be accurately decoded back to
    the original physical fields, which is critical for latent-space rollouts.

    Args:
        input_fields: Original physical fields dict {field_name: (B, N, C)}
        latent: Encoded latent representation (B, tokens, latent_dim)
        decoder: AnyPointDecoder instance
        input_positions: Spatial coordinates (B, N, coord_dim) where fields are defined
        weight: Loss weight multiplier

    Returns:
        Weighted MSE between reconstructed and original fields
    """
    # Decode latent back to input positions
    reconstructed = decoder(input_positions, latent)

    # Compute MSE for each field
    losses = []
    for name in input_fields:
        if name not in reconstructed:
            raise KeyError(f"Decoder did not produce field '{name}'")
        losses.append(mse(reconstructed[name], input_fields[name]))

    return weight * torch.stack(losses).mean()


def inverse_decoding_loss(
    latent: torch.Tensor,
    decoder: nn.Module,
    encoder: nn.Module,
    query_positions: torch.Tensor,
    coords: torch.Tensor,
    meta: dict,
    weight: float = 1.0,
) -> torch.Tensor:
    """UPT Inverse Decoding Loss: Ensures decoder output is re-encodable.

    Flow: latent → decoder → decoded_fields → encoder → reconstructed_latent
    Loss: MSE(reconstructed_latent, latent) in latent space

    This ensures that decoded physical fields can be re-encoded back to the
    original latent representation, completing the invertibility requirement.

    Args:
        latent: Latent representation (B, tokens, latent_dim)
        decoder: AnyPointDecoder instance
        encoder: GridEncoder or MeshParticleEncoder instance
        query_positions: Spatial coordinates (B, N, coord_dim) for decoding
        coords: Full coordinate grid (B, H*W, coord_dim) for re-encoding
        meta: Metadata dict with 'grid_shape' etc. for encoder
        weight: Loss weight multiplier

    Returns:
        Weighted MSE between reconstructed and original latent
    """
    # Decode latent to physical fields at query positions
    decoded_fields = decoder(query_positions, latent)

    # Re-encode decoded fields back to latent space
    latent_reconstructed = encoder(decoded_fields, coords, meta=meta)

    # MSE in latent space (detach original latent to avoid double backprop)
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


def compute_operator_loss_bundle(
    *,
    # For inverse encoding
    input_fields: Optional[Mapping[str, torch.Tensor]] = None,
    encoded_latent: Optional[torch.Tensor] = None,
    decoder: Optional[nn.Module] = None,
    input_positions: Optional[torch.Tensor] = None,
    # For inverse decoding
    encoder: Optional[nn.Module] = None,
    query_positions: Optional[torch.Tensor] = None,
    coords: Optional[torch.Tensor] = None,
    meta: Optional[dict] = None,
    # For forward prediction
    pred_next: Optional[torch.Tensor] = None,
    target_next: Optional[torch.Tensor] = None,
    # For rollout
    pred_rollout: Optional[torch.Tensor] = None,
    target_rollout: Optional[torch.Tensor] = None,
    # For spectral
    spectral_pred: Optional[torch.Tensor] = None,
    spectral_target: Optional[torch.Tensor] = None,
    # Weights
    weights: Optional[Mapping[str, float]] = None,
) -> LossBundle:
    """Compute full loss bundle for operator training including UPT inverse losses.

    All inputs are optional to allow flexible usage (e.g., only forward loss,
    or forward + inverse, etc.). Losses are only computed if their inputs are provided.
    """
    weights = weights or {}
    comp = {}

    # UPT Inverse Encoding Loss
    if all(x is not None for x in [input_fields, encoded_latent, decoder, input_positions]):
        comp["L_inv_enc"] = inverse_encoding_loss(
            input_fields, encoded_latent, decoder, input_positions,
            weight=weights.get("lambda_inv_enc", 0.0)
        )

    # UPT Inverse Decoding Loss
    if all(x is not None for x in [encoded_latent, decoder, encoder, query_positions, coords, meta]):
        comp["L_inv_dec"] = inverse_decoding_loss(
            encoded_latent, decoder, encoder, query_positions, coords, meta,
            weight=weights.get("lambda_inv_dec", 0.0)
        )

    # Forward prediction loss (always used)
    if pred_next is not None and target_next is not None:
        comp["L_forward"] = one_step_loss(pred_next, target_next, weight=weights.get("lambda_forward", 1.0))

    # Rollout loss (optional)
    if pred_rollout is not None and target_rollout is not None:
        comp["L_rollout"] = rollout_loss(pred_rollout, target_rollout, weight=weights.get("lambda_rollout", 0.0))

    # Spectral loss (optional)
    if spectral_pred is not None and spectral_target is not None:
        comp["L_spec"] = spectral_loss(spectral_pred, spectral_target, weight=weights.get("lambda_spectral", 0.0))

    # Sum only non-zero losses
    total = torch.stack([v for v in comp.values() if v.numel() == 1 and v.item() != 0.0]).sum()
    return LossBundle(total=total, components=comp)


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

