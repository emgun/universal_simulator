from __future__ import annotations

"""Physics guard utilities: projections and positivity clamps."""

from dataclasses import dataclass
from typing import Optional

import torch


def helmholtz_hodge_projection_grid(field: torch.Tensor) -> torch.Tensor:
    """Project a 2D vector field onto its divergence-free component via FFT."""

    if field.dim() != 4 or field.shape[-1] != 2:
        raise ValueError("Expected tensor shaped (B, H, W, 2)")
    B, H, W, _ = field.shape
    field_fft = torch.fft.fftn(field[..., 0], dim=(-2, -1)), torch.fft.fftn(field[..., 1], dim=(-2, -1))
    kx = torch.fft.fftfreq(W, device=field.device)
    ky = torch.fft.fftfreq(H, device=field.device)
    KX, KY = torch.meshgrid(ky, kx, indexing="ij")
    k2 = KX**2 + KY**2
    k2[0, 0] = 1.0
    dot = field_fft[0] * KX + field_fft[1] * KY
    proj_u = field_fft[0] - dot * KX / k2
    proj_v = field_fft[1] - dot * KY / k2
    proj_u = torch.fft.ifftn(proj_u, dim=(-2, -1)).real
    proj_v = torch.fft.ifftn(proj_v, dim=(-2, -1)).real
    return torch.stack([proj_u, proj_v], dim=-1)


def positify(values: torch.Tensor, min_value: float = 1e-6) -> torch.Tensor:
    return torch.exp(torch.clamp(torch.log(values.clamp_min(min_value)), min=-20.0))


def interface_flux_projection(flux_a: torch.Tensor, flux_b: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    return alpha * flux_a + (1.0 - alpha) * flux_b

