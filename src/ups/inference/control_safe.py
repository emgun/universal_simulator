from __future__ import annotations

"""Model predictive control (MPC) with control barrier functions in latent space."""

from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

import torch

from ups.core.latent_state import LatentState


BarrierFn = Callable[[LatentState], torch.Tensor]
DynamicsFn = Callable[[LatentState, torch.Tensor], LatentState]


@dataclass
class MPCConfig:
    horizon: int
    control_dim: int
    step_size: float = 0.1
    barrier_weight: float = 10.0
    control_limits: Tuple[float, float] = (-1.0, 1.0)


def safe_mpc(
    state: LatentState,
    dynamics: DynamicsFn,
    cost_fn: Callable[[LatentState, torch.Tensor], torch.Tensor],
    barrier_fn: BarrierFn,
    cfg: MPCConfig,
) -> torch.Tensor:
    control = torch.zeros(cfg.control_dim, requires_grad=True)
    optimizer = torch.optim.Adam([control], lr=cfg.step_size)
    for _ in range(cfg.horizon):
        optimizer.zero_grad()
        rollout_state = dynamics(state, control)
        cost = cost_fn(rollout_state, control)
        barrier = barrier_fn(rollout_state)
        loss = cost + cfg.barrier_weight * barrier.relu().pow(2).sum()
        loss.backward()
        optimizer.step()
        control.data.clamp_(*cfg.control_limits)
    return control.detach()

