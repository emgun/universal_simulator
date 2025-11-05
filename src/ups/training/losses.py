from __future__ import annotations

"""Collection of core training losses for latent evolution."""

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Callable, Tuple

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
    num_queries: Optional[int] = None,
    query_strategy: str = "uniform",
    grid_shape: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """UPT Inverse Encoding Loss with optional query sampling.

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
        num_queries: Number of query points to sample (None = use all N points)
        query_strategy: "uniform" or "stratified" (only used if num_queries < N)
        grid_shape: (H, W) for stratified sampling

    Returns:
        Weighted MSE between reconstructed and original fields (at sampled queries)
    """
    # Apply query sampling if requested
    if num_queries is not None and num_queries < input_positions.shape[1]:
        from ups.training.query_sampling import apply_query_sampling

        input_fields_sampled, input_positions_sampled = apply_query_sampling(
            input_fields,
            input_positions,
            num_queries=num_queries,
            strategy=query_strategy,
            grid_shape=grid_shape,
        )
    else:
        # Use all points (no sampling)
        input_fields_sampled = input_fields
        input_positions_sampled = input_positions

    # Decode latent back to (sampled) input positions
    reconstructed = decoder(input_positions_sampled, latent)

    # Compute MSE for each field
    losses = []
    for name in input_fields_sampled:
        if name not in reconstructed:
            raise KeyError(f"Decoder did not produce field '{name}'")
        losses.append(mse(reconstructed[name], input_fields_sampled[name]))

    return weight * torch.stack(losses).mean()


def inverse_decoding_loss(
    latent: torch.Tensor,
    decoder: nn.Module,
    encoder: nn.Module,
    query_positions: torch.Tensor,
    coords: torch.Tensor,
    meta: dict,
    weight: float = 1.0,
    num_queries: Optional[int] = None,
    query_strategy: str = "uniform",
    grid_shape: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """UPT Inverse Decoding Loss with optional query sampling.

    Flow: latent → decoder → decoded_fields → encoder → reconstructed_latent
    Loss: MSE(reconstructed_latent, latent) in latent space

    This ensures that decoded physical fields can be re-encoded back to the
    original latent representation, completing the invertibility requirement.

    Note: Query sampling is NOT applied to inverse decoding loss because the
    encoder requires a full grid of values. Query sampling only applies to
    inverse encoding loss where we decode and compare in physical space.

    Args:
        latent: Latent representation (B, tokens, latent_dim)
        decoder: AnyPointDecoder instance
        encoder: GridEncoder or MeshParticleEncoder instance
        query_positions: Spatial coordinates (B, N, coord_dim) for decoding
        coords: Full coordinate grid (B, H*W, coord_dim) for re-encoding
        meta: Metadata dict with 'grid_shape' etc. for encoder
        weight: Loss weight multiplier
        num_queries: IGNORED for inverse decoding (kept for API compatibility)
        query_strategy: IGNORED for inverse decoding
        grid_shape: IGNORED for inverse decoding

    Returns:
        Weighted MSE between reconstructed and original latent
    """
    # NOTE: We intentionally ignore num_queries for inverse decoding loss
    # because the encoder requires a full grid. Query sampling only makes
    # sense for inverse encoding where we compare in physical space.

    # Always decode at ALL query positions (no sampling)
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


def latent_norm_penalty(
    latent: Tensor,
    target_norm: float = 1.0,
    norm_type: int = 2,
    weight: float = 1e-4,
) -> Tensor:
    """Regularize latent norm to prevent collapse or explosion.

    Encourages latent vectors to have a target L2 norm, preventing:
    - Collapse: All latents → 0 (no information)
    - Explosion: Latents → ∞ (numerical instability)

    Args:
        latent: Latent tensor (B, tokens, dim)
        target_norm: Target L2 norm (default: 1.0)
        norm_type: Norm type (1 = L1, 2 = L2)
        weight: Loss weight

    Returns:
        Latent norm regularization loss (scalar)
    """
    # Compute norm along latent dimension (B, tokens, dim) → (B, tokens)
    norms = latent.norm(p=norm_type, dim=-1)

    # Penalize deviation from target norm
    penalty = (norms - target_norm).abs().mean()

    return weight * penalty


def latent_diversity_penalty(
    latent: Tensor,
    weight: float = 1e-4,
) -> Tensor:
    """Encourage diversity among latent tokens (prevent collapse to same vector).

    Computes pairwise cosine similarity and penalizes high similarity.

    Args:
        latent: Latent tensor (B, tokens, dim)
        weight: Loss weight

    Returns:
        Latent diversity penalty (scalar)
    """
    B, tokens, dim = latent.shape

    # Normalize latent vectors
    latent_norm = latent / (latent.norm(dim=-1, keepdim=True) + 1e-8)

    # Compute pairwise cosine similarity (B, tokens, tokens)
    similarity = torch.bmm(latent_norm, latent_norm.transpose(1, 2))

    # Remove diagonal (self-similarity = 1)
    mask = ~torch.eye(tokens, dtype=torch.bool, device=latent.device)
    off_diagonal = similarity[:, mask].view(B, tokens, tokens - 1)

    # Penalize high similarity (encourage diversity)
    penalty = off_diagonal.abs().mean()

    return weight * penalty


def compute_inverse_loss_curriculum_weight(
    epoch: int,
    base_weight: float,
    warmup_epochs: int = 15,
    max_weight: float = 0.05,
) -> float:
    """Compute curriculum-scheduled weight for inverse losses.

    Implements gradual ramp-up to prevent gradient explosion:
    - Epochs 0-warmup_epochs: weight = 0 (pure forward training)
    - Epochs warmup_epochs to warmup_epochs*2: linear ramp from 0 to base_weight
    - Epochs > warmup_epochs*2: weight = min(base_weight, max_weight)

    Args:
        epoch: Current training epoch (0-indexed)
        base_weight: Base weight from config (e.g., 0.001)
        warmup_epochs: Number of epochs with zero inverse loss (default: 15)
        max_weight: Maximum allowed inverse loss weight (default: 0.05)

    Returns:
        Effective weight for this epoch
    """
    if epoch < warmup_epochs:
        # Phase 1: Pure forward training
        return 0.0
    elif epoch < warmup_epochs * 2:
        # Phase 2: Linear ramp-up
        progress = (epoch - warmup_epochs) / warmup_epochs
        return min(base_weight * progress, max_weight)
    else:
        # Phase 3: Full weight (but capped at max_weight)
        return min(base_weight, max_weight)


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
    # Curriculum learning
    current_epoch: Optional[int] = None,
    # Query sampling parameters
    num_queries: Optional[int] = None,
    query_strategy: str = "uniform",
    grid_shape: Optional[Tuple[int, int]] = None,
) -> LossBundle:
    """Compute full loss bundle for operator training including UPT inverse losses.

    All inputs are optional to allow flexible usage (e.g., only forward loss,
    or forward + inverse, etc.). Losses are only computed if their inputs are provided.

    If current_epoch is provided, applies curriculum learning to inverse loss weights.

    Args:
        ... (existing args) ...
        num_queries: Number of query points for inverse losses (None = use all)
        query_strategy: "uniform" or "stratified"
        grid_shape: (H, W) for stratified sampling

    Returns:
        LossBundle with total loss and individual components
    """
    weights = weights or {}
    comp = {}

    # Apply curriculum learning to inverse loss weights if epoch is provided
    lambda_inv_enc = weights.get("lambda_inv_enc", 0.0)
    lambda_inv_dec = weights.get("lambda_inv_dec", 0.0)

    if current_epoch is not None:
        warmup_epochs = weights.get("inverse_loss_warmup_epochs", 15)
        max_weight = weights.get("inverse_loss_max_weight", 0.05)

        lambda_inv_enc = compute_inverse_loss_curriculum_weight(
            current_epoch, lambda_inv_enc, warmup_epochs, max_weight
        )
        lambda_inv_dec = compute_inverse_loss_curriculum_weight(
            current_epoch, lambda_inv_dec, warmup_epochs, max_weight
        )

    # UPT Inverse Encoding Loss (with optional query sampling)
    if all(x is not None for x in [input_fields, encoded_latent, decoder, input_positions]):
        comp["L_inv_enc"] = inverse_encoding_loss(
            input_fields, encoded_latent, decoder, input_positions,
            weight=lambda_inv_enc,
            num_queries=num_queries,
            query_strategy=query_strategy,
            grid_shape=grid_shape,
        )

    # UPT Inverse Decoding Loss (with optional query sampling)
    if all(x is not None for x in [encoded_latent, decoder, encoder, query_positions, coords, meta]):
        comp["L_inv_dec"] = inverse_decoding_loss(
            encoded_latent, decoder, encoder, query_positions, coords, meta,
            weight=lambda_inv_dec,
            num_queries=num_queries,
            query_strategy=query_strategy,
            grid_shape=grid_shape,
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

    # Sum losses robustly, even if some are exactly zero or no components present
    total_tensor: Optional[Tensor] = None
    for v in comp.values():
        total_tensor = v if total_tensor is None else total_tensor + v
    if total_tensor is None:
        total_tensor = torch.tensor(0.0)
    return LossBundle(total=total_tensor, components=comp)


def compute_operator_loss_bundle_with_physics(
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
    # Curriculum learning
    current_epoch: Optional[int] = None,
    # Query sampling parameters
    num_queries: Optional[int] = None,
    query_strategy: str = "uniform",
    grid_shape: Optional[Tuple[int, int]] = None,
    # NEW: Physics prior arguments
    decoded_fields: Optional[Mapping[str, torch.Tensor]] = None,
    decoded_coords: Optional[torch.Tensor] = None,
    reference_fields: Optional[Mapping[str, torch.Tensor]] = None,
    physics_weights: Optional[Mapping[str, float]] = None,
) -> LossBundle:
    """Compute full loss bundle including physics priors.

    This extends compute_operator_loss_bundle() with physics-informed penalties.

    New Args:
        decoded_fields: Decoded physical fields {name: (B, H*W, C)} for physics checks
        decoded_coords: Spatial coordinates (B, H*W, 2)
        reference_fields: Reference fields at t=0 for conservation checks
        physics_weights: Dict of physics loss weights

    Returns:
        LossBundle with total loss and all components
    """
    # First compute standard loss bundle
    standard_bundle = compute_operator_loss_bundle(
        input_fields=input_fields,
        encoded_latent=encoded_latent,
        decoder=decoder,
        input_positions=input_positions,
        encoder=encoder,
        query_positions=query_positions,
        coords=coords,
        meta=meta,
        pred_next=pred_next,
        target_next=target_next,
        pred_rollout=pred_rollout,
        target_rollout=target_rollout,
        spectral_pred=spectral_pred,
        spectral_target=spectral_target,
        weights=weights,
        current_epoch=current_epoch,
        num_queries=num_queries,
        query_strategy=query_strategy,
        grid_shape=grid_shape,
    )

    comp = dict(standard_bundle.components)
    physics_weights = physics_weights or {}

    # Import physics loss functions
    from ups.training.physics_losses import (
        divergence_penalty_2d,
        conservation_penalty,
        boundary_condition_penalty_grid,
        positivity_penalty,
    )

    # Physics priors (only if decoded_fields provided)
    if decoded_fields is not None and grid_shape is not None:
        H, W = grid_shape
        B = list(decoded_fields.values())[0].shape[0]

        # Reshape fields to grid (B, H, W, C)
        grid_fields = {}
        for name, tensor in decoded_fields.items():
            # tensor: (B, N, C) → (B, H, W, C)
            grid_fields[name] = tensor.view(B, H, W, -1)

        # Divergence penalty (if velocity field present)
        if "u" in grid_fields and grid_fields["u"].shape[-1] == 2:
            lambda_div = physics_weights.get("lambda_divergence", 0.0)
            if lambda_div > 0:
                comp["L_divergence"] = divergence_penalty_2d(
                    grid_fields["u"],
                    grid_shape,
                    dx=1.0 / W,
                    dy=1.0 / H,
                    weight=lambda_div,
                )

        # Conservation penalty (if reference provided)
        if reference_fields is not None:
            lambda_cons = physics_weights.get("lambda_conservation", 0.0)
            if lambda_cons > 0:
                # Use first field for conservation check
                field_name = list(decoded_fields.keys())[0]
                comp["L_conservation"] = conservation_penalty(
                    decoded_fields[field_name],
                    reference_fields[field_name],
                    weight=lambda_cons,
                )

        # Boundary condition penalty (if enabled)
        lambda_bc = physics_weights.get("lambda_boundary", 0.0)
        bc_value = physics_weights.get("bc_value", 0.0)
        bc_type = physics_weights.get("bc_type", "all")
        if lambda_bc > 0:
            # Apply to first field (typically "u" for velocity)
            field_name = list(grid_fields.keys())[0]
            comp["L_boundary"] = boundary_condition_penalty_grid(
                grid_fields[field_name],
                bc_value,
                grid_shape,
                boundary=bc_type,
                weight=lambda_bc,
            )

        # Positivity penalty (if density/pressure field present)
        lambda_pos = physics_weights.get("lambda_positivity", 0.0)
        if lambda_pos > 0 and "rho" in grid_fields:
            comp["L_positivity"] = positivity_penalty(
                grid_fields["rho"],
                weight=lambda_pos,
            )

    # Latent regularization (Phase 4.3)
    if encoded_latent is not None:
        lambda_latent_norm = physics_weights.get("lambda_latent_norm", 0.0)
        if lambda_latent_norm > 0:
            comp["L_latent_norm"] = latent_norm_penalty(
                encoded_latent,
                target_norm=1.0,
                norm_type=2,
                weight=lambda_latent_norm,
            )

        lambda_latent_diversity = physics_weights.get("lambda_latent_diversity", 0.0)
        if lambda_latent_diversity > 0:
            comp["L_latent_diversity"] = latent_diversity_penalty(
                encoded_latent,
                weight=lambda_latent_diversity,
            )

    # Compute total loss
    total_tensor: Optional[Tensor] = None
    for v in comp.values():
        total_tensor = v if total_tensor is None else total_tensor + v
    if total_tensor is None:
        total_tensor = torch.tensor(0.0)
    return LossBundle(total=total_tensor, components=comp)


def compute_loss_bundle(
    *,
    pred_next: Tensor,
    target_next: Tensor,
    pred_rollout: Optional[Tensor] = None,
    target_rollout: Optional[Tensor] = None,
    spectral_pred: Optional[Tensor] = None,
    spectral_target: Optional[Tensor] = None,
    weights: Optional[Mapping[str, float]] = None,
) -> LossBundle:
    """Deprecated wrapper retained for backward compatibility in tests.

    Computes forward, rollout, and spectral losses using the new naming scheme.
    For full UPT inverse losses, use compute_operator_loss_bundle.
    """
    weights = weights or {}
    comp: Dict[str, Tensor] = {}
    comp["L_forward"] = one_step_loss(pred_next, target_next, weights.get("lambda_forward", 1.0))
    if pred_rollout is not None and target_rollout is not None:
        comp["L_rollout"] = rollout_loss(pred_rollout, target_rollout, weights.get("lambda_rollout", 0.0))
    if spectral_pred is not None and spectral_target is not None:
        comp["L_spec"] = spectral_loss(spectral_pred, spectral_target, weights.get("lambda_spectral", 0.0))
    total = None
    for v in comp.values():
        total = v if total is None else total + v
    if total is None:
        total = torch.tensor(0.0)
    return LossBundle(total=total, components=comp)
