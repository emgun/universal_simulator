from __future__ import annotations

"""Test-time computing (TTC) rollout utilities."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from ups.core.latent_state import LatentState
from ups.eval.reward_models import (
    AnalyticalRewardModel,
    AnalyticalRewardWeights,
    CompositeRewardModel,
    FeatureCriticRewardModel,
    RewardModel,
)
from ups.io.decoder_anypoint import AnyPointDecoder, AnyPointDecoderConfig
from ups.inference.rollout_transient import RolloutLog
from ups.logging import get_logger
from ups.models.diffusion_residual import DiffusionResidual
from ups.models.latent_operator import LatentOperator


@dataclass
class TTCConfig:
    steps: int
    dt: float
    candidates: int = 4
    beam_width: int = 1
    horizon: int = 1
    tau_range: Tuple[float, float] = (0.3, 0.7)
    noise_std: float = 0.0
    noise_schedule: Optional[Sequence[float]] = None
    residual_threshold: Optional[float] = None
    max_evaluations: Optional[int] = None
    early_stop_margin: Optional[float] = None
    gamma: float = 1.0
    device: torch.device | str = "cpu"


@dataclass
class TTCStepLog:
    rewards: List[float] = field(default_factory=list)
    totals: List[float] = field(default_factory=list)
    chosen_index: int = 0
    beam_width: int = 1
    horizon: int = 1
    noise_std: float = 0.0


def _copy_state(state: LatentState) -> LatentState:
    return LatentState(
        z=state.z.clone(),
        t=state.t.clone() if torch.is_tensor(state.t) else state.t,
        cond={k: v.clone() for k, v in state.cond.items()},
    )


@dataclass
class _EvalBudget:
    max_evaluations: Optional[int]
    used: int = 0

    def remaining(self) -> Optional[int]:
        if self.max_evaluations is None:
            return None
        return max(self.max_evaluations - self.used, 0)

    def consume(self, amount: int) -> int:
        if self.max_evaluations is None:
            self.used += amount
            return amount
        remaining = self.max_evaluations - self.used
        allowed = max(min(amount, remaining), 0)
        self.used += allowed
        return allowed


def ttc_rollout(
    *,
    initial_state: LatentState,
    operator: LatentOperator,
    reward_model: RewardModel,
    config: TTCConfig,
    corrector: Optional[DiffusionResidual] = None,
) -> Tuple[RolloutLog, List[TTCStepLog]]:
    logger = get_logger("ups.ttc")
    device = torch.device(config.device)
    state = initial_state.to(device)
    log = RolloutLog(states=[state.detach_clone()], corrections=[])
    step_logs: List[TTCStepLog] = []
    dt_tensor = torch.tensor(config.dt, device=device)
    budget = _EvalBudget(config.max_evaluations)

    def _noise_for_step(step_idx: int) -> float:
        if config.noise_schedule:
            idx = min(max(step_idx, 0), len(config.noise_schedule) - 1)
            return float(config.noise_schedule[idx])
        return float(config.noise_std)

    def sample_candidates(
        prev_state: LatentState,
        base_state: LatentState,
        step_idx: int,
    ) -> Tuple[List[LatentState], List[float]]:
        candidate_count = config.candidates
        remaining = budget.remaining()
        if remaining is not None:
            candidate_count = min(candidate_count, max(remaining, 1))
        budget.consume(candidate_count)
        candidates: List[LatentState] = []
        rewards: List[float] = []
        noise_std = _noise_for_step(step_idx)
        for _ in range(candidate_count):
            candidate = _copy_state(base_state)
            if corrector is not None:
                tau = torch.empty(candidate.z.size(0), device=device).uniform_(*config.tau_range)
                drift = corrector(candidate, tau)
                candidate = LatentState(z=candidate.z + drift, t=candidate.t, cond=candidate.cond)
            if noise_std > 0.0:
                noise = torch.randn_like(candidate.z) * noise_std
                candidate = LatentState(z=candidate.z + noise, t=candidate.t, cond=candidate.cond)
            reward_value = reward_model.score(prev_state, candidate, context={"step": float(step_idx)})
            rewards.append(float(reward_value.item()))
            candidates.append(candidate)
        return candidates, rewards

    def lookahead(candidate_state: LatentState, depth: int, step_idx: int) -> float:
        if depth <= 0:
            return 0.0
        next_base = operator(candidate_state, dt_tensor)
        candidates, rewards = sample_candidates(candidate_state, next_base, step_idx)
        if not candidates:
            return 0.0
        totals = rewards[:]
        if depth > 1:
            order = sorted(range(len(rewards)), key=lambda idx: rewards[idx], reverse=True)
            top = order[: max(config.beam_width, 1)]
            for idx in top:
                totals[idx] += config.gamma * lookahead(candidates[idx], depth - 1, step_idx + 1)
        return max(totals)

    for step in range(config.steps):
        base_prediction = operator(state, dt_tensor).to(device)

        candidates, rewards = sample_candidates(state, base_prediction, step)
        if not candidates:
            logger.warning("Budget exhausted; no candidates generated. Stopping TTC.")
            break

        totals = rewards[:]
        best_gap = None
        if len(rewards) > 1:
            sorted_rewards = sorted(rewards, reverse=True)
            best_gap = sorted_rewards[0] - sorted_rewards[1]

        need_lookahead = config.horizon > 1 and config.beam_width > 1
        if need_lookahead and candidates:
            order = sorted(range(len(rewards)), key=lambda idx: rewards[idx], reverse=True)
            if config.early_stop_margin is not None and best_gap is not None and best_gap > config.early_stop_margin:
                need_lookahead = False
            else:
                top = order[: max(config.beam_width, 1)]
                for idx in top:
                    future_reward = lookahead(candidates[idx], config.horizon - 1, step + 1)
                    totals[idx] += config.gamma * future_reward

        chosen = max(enumerate(totals), key=lambda item: item[1])[0]
        chosen_state = candidates[chosen]
        step_logs.append(
            TTCStepLog(
                rewards=rewards,
                totals=totals,
                chosen_index=chosen,
                beam_width=config.beam_width,
                horizon=config.horizon,
                noise_std=_noise_for_step(step),
            )
        )

        logger.info("step=%d rewards=%s totals=%s chosen=%d", step, rewards, totals, chosen)

        state = chosen_state
        log.corrections.append(corrector is not None)
        log.states.append(state.detach_clone())

    return log, step_logs


def build_reward_model_from_config(
    ttc_cfg: Dict[str, Any],
    latent_dim: int,
    device: torch.device,
) -> RewardModel:
    reward_cfg = ttc_cfg.get("reward", {})
    decoder_cfg = ttc_cfg.get("decoder", {})
    grid: Sequence[int] = reward_cfg.get("grid", [64, 64])
    weights_cfg = reward_cfg.get("weights", {})
    weights = AnalyticalRewardWeights(
        mass=float(weights_cfg.get("mass", 1.0)),
        momentum=float(weights_cfg.get("momentum", 0.0)),
        energy=float(weights_cfg.get("energy", 0.0)),
        penalty_negative=float(weights_cfg.get("penalty_negative", 0.0)),
    )

    decoder_config = AnyPointDecoderConfig(
        latent_dim=decoder_cfg.get("latent_dim", latent_dim),
        query_dim=decoder_cfg.get("query_dim", 2),
        hidden_dim=decoder_cfg.get("hidden_dim", 128),
        num_layers=decoder_cfg.get("num_layers", 2),
        num_heads=decoder_cfg.get("num_heads", 4),
        mlp_hidden_dim=decoder_cfg.get("mlp_hidden_dim", 128),
        frequencies=tuple(decoder_cfg.get("frequencies", (1.0, 2.0, 4.0))),
        output_channels=decoder_cfg["output_channels"],
    )
    decoder = AnyPointDecoder(decoder_config).to(device)
    momentum_field = reward_cfg.get("momentum_field", [])
    if isinstance(momentum_field, str):
        momentum_field = [momentum_field]

    models: List[RewardModel] = []
    model_weights: List[float] = []

    analytic_weight = float(reward_cfg.get("analytical_weight", 1.0))
    if analytic_weight != 0.0:
        analytic_model = AnalyticalRewardModel(
            decoder,
            grid_shape=grid,
            weights=weights,
            mass_field=reward_cfg.get("mass_field"),
            momentum_field=momentum_field,
            energy_field=reward_cfg.get("energy_field"),
            device=device,
        )
        models.append(analytic_model)
        model_weights.append(analytic_weight)

    critic_cfg = reward_cfg.get("critic")
    if critic_cfg:
        critic_weight = float(critic_cfg.get("weight", 0.0))
        if critic_weight != 0.0:
            critic_model = FeatureCriticRewardModel(
                decoder,
                grid_shape=grid,
                mass_field=reward_cfg.get("mass_field"),
                momentum_field=momentum_field,
                energy_field=reward_cfg.get("energy_field"),
                hidden_dim=int(critic_cfg.get("hidden_dim", 128)),
                dropout=float(critic_cfg.get("dropout", 0.0)),
                device=device,
            )
            checkpoint = critic_cfg.get("checkpoint")
            if checkpoint:
                state = torch.load(checkpoint, map_location=device)
                critic_model.load_state_dict(state, strict=False)
            models.append(critic_model)
            model_weights.append(critic_weight)

    if not models:
        raise ValueError("Reward configuration yielded no active reward models")
    if len(models) == 1:
        return models[0]
    return CompositeRewardModel(models, weights=model_weights)
