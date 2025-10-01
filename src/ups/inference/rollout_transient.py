from __future__ import annotations

"""Predictor–corrector rollout utilities for transient simulations."""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch

from ups.core.latent_state import LatentState
from ups.logging import get_logger
from ups.models.diffusion_residual import DiffusionResidual
from ups.models.latent_operator import LatentOperator


GateFn = Callable[[LatentState, LatentState], bool]

@dataclass
class RolloutConfig:
    steps: int
    dt: float
    correct_every: int = 1
    corrector_tau: float = 0.5
    device: torch.device | str = "cpu"


@dataclass
class RolloutLog:
    states: List[LatentState] = field(default_factory=list)
    corrections: List[bool] = field(default_factory=list)
    metrics: Dict[str, List[float]] = field(default_factory=lambda: {"residual_norm": []})


def rollout_transient(
     *,
     initial_state: LatentState,
     operator: LatentOperator,
     corrector: Optional[DiffusionResidual] = None,
     config: RolloutConfig,
     gate_fn: Optional[GateFn] = None,
 ) -> RolloutLog:
    logger = get_logger("ups.rollout")
    device = torch.device(config.device)
    state = initial_state.to(device)
    log = RolloutLog(states=[state], corrections=[])
    dt_tensor = torch.tensor(config.dt, device=device)
    tau_value = torch.tensor(config.corrector_tau, device=device)
    for step in range(config.steps):
        predicted = operator(state, dt_tensor)
        predicted = predicted.to(device)
        residual_norm = (predicted.z - state.z).norm().item()
        log.metrics.setdefault("residual_norm", []).append(residual_norm)
        should_correct = False
        if corrector is not None:
            if (step + 1) % max(config.correct_every, 1) == 0:
                should_correct = True
            if gate_fn is not None:
                should_correct = gate_fn(state, predicted)
        logger.info(f"step={step} residual_norm={residual_norm:.4f}")
        if should_correct and corrector is not None:
            logger.info(f"applying corrector at step {step}")
            tau_tensor = tau_value.expand(predicted.z.size(0))
            drift = corrector(predicted, tau_tensor)
            corrected_z = predicted.z + drift
            predicted = LatentState(z=corrected_z, t=predicted.t, cond=predicted.cond)
        log.corrections.append(should_correct)
        state = predicted
        log.states.append(state.detach_clone())

    return log
