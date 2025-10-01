import torch

from ups.core.latent_state import LatentState
from ups.inference.rollout_transient import RolloutConfig, rollout_transient


class DummyOperator:
    def __init__(self, delta: float):
        self.delta = delta

    def to(self, device):
        return self

    def __call__(self, state: LatentState, dt: torch.Tensor) -> LatentState:
        return LatentState(z=state.z + self.delta, t=state.t, cond=state.cond)


class DummyCorrector:
    def __init__(self, correction: float):
        self.correction = correction

    def to(self, device):
        return self

    def __call__(self, state: LatentState, tau: torch.Tensor) -> torch.Tensor:
        return torch.full_like(state.z, self.correction)


def test_rollout_without_corrector():
    init_state = LatentState(z=torch.zeros(1, 4, 2))
    operator = DummyOperator(delta=0.1)
    config = RolloutConfig(steps=3, dt=0.1, correct_every=1)
    log = rollout_transient(initial_state=init_state, operator=operator, config=config)
    assert len(log.states) == 4
    assert all(not flag for flag in log.corrections)
    assert torch.allclose(log.states[-1].z, torch.full_like(init_state.z, 0.3))


def test_rollout_with_gate_function():
    init_state = LatentState(z=torch.zeros(1, 2, 2))
    operator = DummyOperator(delta=0.0)
    corrector = DummyCorrector(correction=1.0)

    def gate(prev_state: LatentState, predicted: LatentState) -> bool:
        return True

    config = RolloutConfig(steps=2, dt=0.1, correct_every=5)
    log = rollout_transient(
        initial_state=init_state,
        operator=operator,
        corrector=corrector,
        config=config,
        gate_fn=gate,
    )
    assert any(log.corrections)
    assert torch.allclose(log.states[-1].z, torch.full_like(init_state.z, 2.0))


def test_rollout_gate_blocks_corrections():
    init_state = LatentState(z=torch.zeros(1, 2, 2))
    operator = DummyOperator(delta=0.0)
    corrector = DummyCorrector(correction=1.0)

    def gate(prev_state: LatentState, predicted: LatentState) -> bool:
        return False

    config = RolloutConfig(steps=2, dt=0.1)
    log = rollout_transient(
        initial_state=init_state,
        operator=operator,
        corrector=corrector,
        config=config,
        gate_fn=gate,
    )
    assert not any(log.corrections)
    assert torch.allclose(log.states[-1].z, init_state.z)

