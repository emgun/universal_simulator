from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from ups.core.latent_state import LatentState
from ups.eval.reward_models import (
    AnalyticalRewardModel,
    AnalyticalRewardWeights,
    CompositeRewardModel,
    FeatureCriticRewardModel,
    RewardModel,
)
from ups.inference.rollout_ttc import TTCConfig, ttc_rollout


class DummyDecoder(nn.Module):
    """Decoder that broadcasts latent means to query points."""

    def forward(self, points: torch.Tensor, latents: torch.Tensor, *, conditioning=None):
        mean = latents.mean(dim=(1, 2), keepdim=True)
        values = mean.expand(latents.size(0), points.size(1), 1)
        return {"rho": values}


class DummyOperator(nn.Module):
    def forward(self, state: LatentState, dt: torch.Tensor) -> LatentState:
        z = state.z + dt.view(1, 1, 1)
        return LatentState(z=z, t=state.t, cond=state.cond)


class DummyCorrector(nn.Module):
    def forward(self, state: LatentState, tau: torch.Tensor) -> torch.Tensor:
        drift = tau.view(-1, 1, 1)
        return drift.expand_as(state.z)


def make_latent(value: float) -> LatentState:
    z = torch.full((1, 1, 1), value)
    return LatentState(z=z, t=torch.tensor(0.0), cond={})


def test_analytical_reward_mass_gap() -> None:
    decoder = DummyDecoder()
    reward_model = AnalyticalRewardModel(
        decoder,
        grid_shape=(1, 1),
        weights=AnalyticalRewardWeights(mass=1.0),
        mass_field="rho",
        device=torch.device("cpu"),
    )
    prev_state = make_latent(1.0)
    next_state = make_latent(0.5)
    reward = reward_model.score(prev_state, next_state)
    assert reward.item() < 0


def test_ttc_rollout_selects_best_candidate() -> None:
    torch.manual_seed(0)
    operator = DummyOperator()
    corrector = DummyCorrector()
    decoder = DummyDecoder()
    reward_model = AnalyticalRewardModel(
        decoder,
        grid_shape=(1, 1),
        weights=AnalyticalRewardWeights(mass=1.0),
        mass_field="rho",
        device=torch.device("cpu"),
    )
    initial = make_latent(0.0)
    config = TTCConfig(
        steps=1,
        dt=0.1,
        candidates=4,
        tau_range=(0.0, 0.2),
        noise_std=0.0,
        device="cpu",
    )
    rollout_log, step_logs = ttc_rollout(
        initial_state=initial,
        operator=operator,
        reward_model=reward_model,
        config=config,
        corrector=corrector,
    )
    assert len(rollout_log.states) == 2
    chosen_rewards = step_logs[0].rewards
    best_index = step_logs[0].chosen_index
    expected_index = chosen_rewards.index(max(chosen_rewards))
    assert best_index == expected_index
    assert step_logs[0].noise_std == 0.0


class SequenceCorrector(nn.Module):
    def __init__(self, deltas: Sequence[float]) -> None:
        super().__init__()
        self.deltas = list(deltas)
        self.idx = 0

    def forward(self, state: LatentState, tau: torch.Tensor) -> torch.Tensor:
        value = self.deltas[self.idx]
        self.idx += 1
        return torch.full_like(state.z, value)


class LinearReward(RewardModel):
    def score(self, prev_state: LatentState, next_state: LatentState, context=None) -> torch.Tensor:
        return next_state.z.mean()


class ConstantReward(RewardModel):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = torch.tensor(value)

    def score(self, prev_state: LatentState, next_state: LatentState, context=None) -> torch.Tensor:
        return self.value.to(next_state.z.device)


def test_ttc_rollout_beam_lookahead_prefers_future_reward() -> None:
    operator = DummyOperator()
    corrector = SequenceCorrector([0.2, 0.6, 0.05, 0.04, 1.0, 0.9])
    reward_model = LinearReward()
    initial = make_latent(0.0)
    config = TTCConfig(
        steps=1,
        dt=0.0,
        candidates=2,
        beam_width=2,
        horizon=2,
        tau_range=(0.0, 1.0),
        noise_std=0.0,
        device="cpu",
    )
    rollout_log, step_logs = ttc_rollout(
        initial_state=initial,
        operator=operator,
        reward_model=reward_model,
        config=config,
        corrector=corrector,
    )
    assert len(rollout_log.states) == 2
    # Best immediate reward is candidate index 1, but lookahead should choose index 0 due to large future gain
    assert step_logs[0].chosen_index == 0


def test_ttc_rollout_reports_noise_schedule() -> None:
    torch.manual_seed(0)
    operator = DummyOperator()
    corrector = DummyCorrector()
    decoder = DummyDecoder()
    reward_model = AnalyticalRewardModel(
        decoder,
        grid_shape=(1, 1),
        weights=AnalyticalRewardWeights(mass=1.0),
        mass_field="rho",
        device=torch.device("cpu"),
    )
    initial = make_latent(0.0)
    config = TTCConfig(
        steps=1,
        dt=0.1,
        candidates=2,
        tau_range=(0.0, 0.5),
        noise_std=0.0,
        noise_schedule=[0.05, 0.01],
        device="cpu",
    )
    _, step_logs = ttc_rollout(
        initial_state=initial,
        operator=operator,
        reward_model=reward_model,
        config=config,
        corrector=corrector,
    )
    assert step_logs[0].noise_std == 0.05


def test_feature_critic_reward_tracks_mass_delta() -> None:
    decoder = DummyDecoder()
    critic = FeatureCriticRewardModel(
        decoder,
        grid_shape=(1, 1),
        mass_field="rho",
        hidden_dim=2,
        dropout=0.0,
        device=torch.device("cpu"),
    )
    with torch.no_grad():
        critic.net[0].weight.copy_(torch.eye(2))
        critic.net[0].bias.zero_()
        critic.net[3].weight.copy_(torch.tensor([[1.0, 0.0]]))
        critic.net[3].bias.zero_()
    prev_state = make_latent(0.5)
    next_state = make_latent(1.0)
    score = critic.score(prev_state, next_state)
    assert score.item() > 0


def test_composite_reward_weighted_average() -> None:
    reward = CompositeRewardModel(
        [ConstantReward(1.0), ConstantReward(-3.0)],
        weights=[0.25, 0.75],
    )
    prev_state = make_latent(0.0)
    next_state = make_latent(0.0)
    value = reward.score(prev_state, next_state)
    expected = (0.25 * 1.0 + 0.75 * -3.0) / (0.25 + 0.75)
    assert torch.isclose(value, torch.tensor(expected))
