from __future__ import annotations

"""Reward models to score candidate latent trajectories for TTC."""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

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


def _decode_fields(
    decoder: AnyPointDecoder,
    state: LatentState,
    *,
    query_points: torch.Tensor,
    device: torch.device,
    height: int,
    width: int,
) -> Dict[str, torch.Tensor]:
    tokens = state.z.to(device)
    cond = {k: v.to(device) for k, v in state.cond.items()}
    query = query_points.expand(tokens.size(0), -1, -1)
    outputs = decoder(query, tokens, conditioning=cond)
    decoded: Dict[str, torch.Tensor] = {}
    for name, values in outputs.items():
        decoded[name] = values.view(tokens.size(0), height, width, -1)
    return decoded


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
        return _decode_fields(
            self.decoder,
            state,
            query_points=self.query_points,
            device=self.device,
            height=self.height,
            width=self.width,
        )

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


class FeatureCriticRewardModel(RewardModel):
    """Learned critic operating on global physics features decoded from latents."""

    def __init__(
        self,
        decoder: AnyPointDecoder,
        *,
        grid_shape: Sequence[int],
        mass_field: Optional[str] = None,
        momentum_field: Optional[Iterable[str]] = None,
        energy_field: Optional[str] = None,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.mass_field = mass_field
        self.momentum_field = list(momentum_field or [])
        self.energy_field = energy_field
        if not (self.mass_field or self.momentum_field or self.energy_field):
            raise ValueError("Feature critic requires at least one decoded field")
        self.height, self.width = int(grid_shape[0]), int(grid_shape[1])
        self.device = device or next(decoder.parameters()).device
        self.register_buffer(
            "query_points",
            _build_grid_coords(self.height, self.width, self.device),
        )
        feature_dim = 0
        if self.mass_field:
            feature_dim += 2  # mass sum and negative penalty
        feature_dim += len(self.momentum_field)
        if self.energy_field:
            feature_dim += 1
        if feature_dim == 0:
            raise ValueError("Feature critic computed zero-length feature vector")
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _features(self, state: LatentState) -> torch.Tensor:
        decoded = _decode_fields(
            self.decoder,
            state,
            query_points=self.query_points,
            device=self.device,
            height=self.height,
            width=self.width,
        )
        feats: List[torch.Tensor] = []
        if self.mass_field:
            mass = decoded[self.mass_field].sum(dim=(1, 2, 3))
            feats.append(mass)
            negatives = torch.clamp_min(decoded[self.mass_field], 0.0)
            penalty = torch.abs(decoded[self.mass_field] - negatives).sum(dim=(1, 2, 3))
            feats.append(penalty)
        for channel_name in self.momentum_field:
            momentum = decoded[channel_name].sum(dim=(1, 2, 3))
            feats.append(momentum)
        if self.energy_field:
            energy = torch.square(decoded[self.energy_field]).sum(dim=(1, 2, 3))
            feats.append(energy)
        return torch.stack(feats, dim=-1)

    def score(
        self,
        prev_state: LatentState,
        next_state: LatentState,
        context: Optional[Mapping[str, float]] = None,
    ) -> torch.Tensor:
        prev_feats = self._features(prev_state)
        next_feats = self._features(next_state)
        delta = next_feats - prev_feats
        values = self.net(delta)
        return values.mean()


class CompositeRewardModel(RewardModel):
    """Weighted ensemble of multiple reward models."""

    def __init__(self, models: Sequence[RewardModel], weights: Optional[Sequence[float]] = None) -> None:
        super().__init__()
        if not models:
            raise ValueError("CompositeRewardModel requires at least one sub-model")
        self.models = nn.ModuleList(models)
        if weights is None:
            weights = [1.0] * len(models)
        if len(weights) != len(models):
            raise ValueError("weights must match number of models")
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    def score(
        self,
        prev_state: LatentState,
        next_state: LatentState,
        context: Optional[Mapping[str, float]] = None,
    ) -> torch.Tensor:
        total = None
        normaliser = 0.0
        for weight, model in zip(self.weights, self.models):
            w = float(weight.item())
            if w == 0.0:
                continue
            value = model.score(prev_state, next_state, context)
            total = value * w if total is None else total + value * w
            normaliser += abs(w)
        if total is None:
            return torch.tensor(0.0, device=prev_state.z.device)
        if normaliser > 0.0:
            total = total / normaliser
        return total
