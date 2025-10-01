from __future__ import annotations

"""Steady-state latent prior via conditional diffusion flow."""

from dataclasses import dataclass
from typing import Mapping, Optional

import torch
from torch import nn
from torch.nn import functional as F

from ups.core.latent_state import LatentState


@dataclass
class SteadyPriorConfig:
    latent_dim: int
    hidden_dim: int = 128
    num_steps: int = 6
    cond_dim: int = 0


class SteadyPrior(nn.Module):
    def __init__(self, cfg: SteadyPriorConfig) -> None:
        super().__init__()
        self.cfg = cfg
        input_dim = cfg.latent_dim + cfg.cond_dim + 1  # latent + time step
        self.drift = nn.Sequential(
            nn.Linear(input_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.latent_dim),
        )

    def forward(self, state: LatentState, cond: Optional[Mapping[str, torch.Tensor]] = None) -> LatentState:
        z = state.z
        B, T, D = z.shape
        current = z.clone()
        for step in range(self.cfg.num_steps):
            tau = torch.full((B, T, 1), step / max(self.cfg.num_steps - 1, 1), device=current.device)
            inputs = [current, tau]
            if cond:
                cond_tensor = torch.cat([v.view(B, 1, -1).expand(B, T, -1) for v in cond.values()], dim=-1)
                inputs.append(cond_tensor)
            drift = self.drift(torch.cat(inputs, dim=-1))
            current = current + drift
        return LatentState(z=current, t=state.t, cond=state.cond)


def steady_residual_norm(prior: SteadyPrior, state: LatentState, cond: Optional[Mapping[str, torch.Tensor]] = None) -> torch.Tensor:
    refined = prior(state, cond)
    return (refined.z - state.z).norm(dim=-1).mean()

