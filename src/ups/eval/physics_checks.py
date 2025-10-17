from __future__ import annotations

"""Physics diagnostic utilities to accompany evaluation runs."""

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Tuple

import torch

from ups.models.physics_guards import (
    helmholtz_hodge_projection_grid,
    interface_flux_projection,
    positify,
)


@dataclass
class PhysicsDiagnostics:
    conservation_gap: float
    bc_violation: float
    negativity_penalty: float


def compute_conservation_gap(decoded: Mapping[str, torch.Tensor]) -> float:
    total = 0.0
    count = 0
    for tensor in decoded.values():
        if tensor.dim() >= 3:
            total += tensor.sum(dim=(-3, -2, -1), keepdim=False).abs().mean().item()
            count += 1
    return total / max(count, 1)


def compute_negativity_penalty(decoded: Mapping[str, torch.Tensor]) -> float:
    penalty = 0.0
    count = 0
    for tensor in decoded.values():
        if tensor.dim() >= 3:
            negatives = torch.clamp_min(tensor, 0.0)
            penalty += (tensor - negatives).abs().mean().item()
            count += 1
    return penalty / max(count, 1)


def compute_bc_violation(boundary_values: Iterable[torch.Tensor]) -> float:
    delta = 0.0
    count = 0
    for tensor in boundary_values:
        delta += tensor.abs().mean().item()
        count += 1
    return delta / max(count, 1)


def run_physics_diagnostics(
    decoded_fields: Mapping[str, torch.Tensor],
    *,
    apply_projection: bool = False,
    boundary_values: Optional[Iterable[torch.Tensor]] = None,
) -> PhysicsDiagnostics:
    fields = decoded_fields
    if apply_projection:
        projected: Dict[str, torch.Tensor] = {}
        for name, tensor in decoded_fields.items():
            projected[name] = tensor
            if tensor.dim() == 4 and tensor.shape[-1] == 2:
                projected[name] = helmholtz_hodge_projection_grid(tensor)
            projected[name] = positify(projected[name])
        fields = projected

    conservation = compute_conservation_gap(fields)
    negativity = compute_negativity_penalty(fields)
    bc = compute_bc_violation(boundary_values or [])
    return PhysicsDiagnostics(conservation_gap=conservation, bc_violation=bc, negativity_penalty=negativity)


def flux_match(flux_a: torch.Tensor, flux_b: torch.Tensor, *, alpha: float = 0.5) -> torch.Tensor:
    return interface_flux_projection(flux_a, flux_b, alpha)


__all__ = [
    "PhysicsDiagnostics",
    "run_physics_diagnostics",
    "compute_conservation_gap",
    "compute_bc_violation",
    "compute_negativity_penalty",
    "flux_match",
]
