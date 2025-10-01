from __future__ import annotations

"""Few-step diffusion residual corrector for latent trajectories."""

from dataclasses import dataclass
from typing import Mapping, Optional

import torch
from torch import nn

from ups.core.latent_state import LatentState


@dataclass
class DiffusionResidualConfig:
    latent_dim: int
    hidden_dim: int = 128
    cond_dim: int = 0
    residual_guidance_weight: float = 1.0


class DiffusionResidual(nn.Module):
    def __init__(self, cfg: DiffusionResidualConfig) -> None:
        super().__init__()
        self.cfg = cfg
        input_dim = cfg.latent_dim + 1 + cfg.cond_dim  # latent + tau + optional cond
        self.network = nn.Sequential(
            nn.Linear(input_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.latent_dim),
        )

    def forward(
        self,
        state: LatentState,
        tau: torch.Tensor,
        *,
        cond: Optional[Mapping[str, torch.Tensor]] = None,
        decoded_residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        z = state.z
        B, T, D = z.shape
        tau = tau.view(B, 1, 1).expand(B, T, 1)
        inputs = [z, tau]
        if cond and len(cond) > 0:
            cond_tensor = torch.cat([v.view(B, 1, -1).expand(B, T, -1) for v in cond.values()], dim=-1)
            inputs.append(cond_tensor)
        model_in = torch.cat(inputs, dim=-1)
        drift = self.network(model_in)
        if decoded_residual is not None:
            drift = drift + self.cfg.residual_guidance_weight * decoded_residual
        return drift

