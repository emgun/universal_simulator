from __future__ import annotations

"""Gating heuristics for predictor–corrector rollouts."""


def residual_gate(residual_norm: float, threshold: float) -> bool:
    """Trigger correction when residual exceeds threshold."""

    return residual_norm > threshold


def periodic_gate(step: int, every: int) -> bool:
    """Trigger correction at a fixed cadence (every N steps)."""

    if every <= 0:
        return False
    return (step + 1) % every == 0

