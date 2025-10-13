from __future__ import annotations

"""Reward models to score candidate latent trajectories for TTC."""

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence

import torch
from torch import nn

from ups.core.latent_state import LatentState
from ups.io.decoder_anypoint import AnyPointDecoder


class RewardModel(nn.Module):
    """Abstract reward model. Higher scores indicate better candidates."""

    def score(
        self,
        prev_state: LatentState,
        next_state: LatentState,
        context: Optional[Mapping[str, float]] = None,
    ) -> torch.Tensor:
        raise NotImplementedError


def _build_grid_coords(
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    ys = torch.linspace(0.0, 1.0, height, device=device)
    xs = torch.linspace(0.0, 1.0, width, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([grid_x, grid_y], dim=-1).reshape(1, height * width, 2)
    return coords


@dataclass
class AnalyticalRewardWeights:
    mass: float = 1.0
    momentum: float = 0.0
    energy: float = 0.0
    penalty_negative: float = 0.0


class AnalyticalRewardModel(RewardModel):
    """Physics-inspired reward model using decoded mass/momentum/energy gaps."""

    def __init__(
        self,
        decoder: AnyPointDecoder,
        *,
        grid_shape: Sequence[int],
        weights: AnalyticalRewardWeights,
        mass_field: Optional[str] = None,
        momentum_field: Optional[Sequence[str]] = None,
        energy_field: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.weights = weights
        self.mass_field = mass_field
        self.momentum_field = list(momentum_field) if momentum_field else []
        self.energy_field = energy_field
        self.height, self.width = int(grid_shape[0]), int(grid_shape[1])
        self.device = device or next(decoder.parameters()).device
        self.register_buffer("query_points", _build_grid_coords(self.height, self.width, self.device))

    def _decode(
        self,
        state: LatentState,
    ) -> Dict[str, torch.Tensor]:
        tokens = state.z.to(self.device)
        cond = {k: v.to(self.device) for k, v in state.cond.items()}
        points = self.query_points.expand(tokens.size(0), -1, -1)
        outputs = self.decoder(points, tokens, conditioning=cond)
        decoded: Dict[str, torch.Tensor] = {}
        for name, values in outputs.items():
            decoded[name] = values.view(tokens.size(0), self.height, self.width, -1)
        return decoded

    def score(
        self,
        prev_state: LatentState,
        next_state: LatentState,
        context: Optional[Mapping[str, float]] = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            prev_fields = self._decode(prev_state)
            next_fields = self._decode(next_state)

        batch = prev_state.z.size(0)
        rewards = torch.zeros(batch, device=self.device)

        if self.weights.mass and self.mass_field:
            prev_mass = prev_fields[self.mass_field].sum(dim=(1, 2, 3))
            next_mass = next_fields[self.mass_field].sum(dim=(1, 2, 3))
            mass_gap = torch.abs(next_mass - prev_mass)
            rewards = rewards - self.weights.mass * mass_gap

        if self.weights.momentum and self.momentum_field:
            momentum_gaps = []
            for channel_name in self.momentum_field:
                prev_m = prev_fields[channel_name].sum(dim=(1, 2, 3))
                next_m = next_fields[channel_name].sum(dim=(1, 2, 3))
                momentum_gaps.append(torch.abs(next_m - prev_m))
            total_gap = torch.stack(momentum_gaps, dim=0).mean(dim=0)
            rewards = rewards - self.weights.momentum * total_gap

        if self.weights.energy and self.energy_field:
            prev_energy = torch.square(prev_fields[self.energy_field]).sum(dim=(1, 2, 3))
            next_energy = torch.square(next_fields[self.energy_field]).sum(dim=(1, 2, 3))
            energy_gap = torch.abs(next_energy - prev_energy)
            rewards = rewards - self.weights.energy * energy_gap

        if self.weights.penalty_negative and self.mass_field:
            negatives = torch.clamp_min(next_fields[self.mass_field], 0.0)
            # penalty is magnitude of negative mass before clamping.
            penalty = torch.abs(next_fields[self.mass_field] - negatives).sum(dim=(1, 2, 3))
            rewards = rewards - self.weights.penalty_negative * penalty

        return rewards.mean()
