import torch

from ups.core.latent_state import LatentState
from ups.inference.control_safe import MPCConfig, safe_mpc


def dynamics(state: LatentState, control: torch.Tensor) -> LatentState:
    return LatentState(z=state.z + control.view(1, 1, -1), t=state.t, cond=state.cond)


def cost_fn(state: LatentState, control: torch.Tensor) -> torch.Tensor:
    return control.pow(2).sum()


def barrier_fn(state: LatentState) -> torch.Tensor:
    return state.z.abs().max(dim=-1).values


def test_safe_mpc_returns_control():
    state = LatentState(z=torch.zeros(1, 1, 3))
    cfg = MPCConfig(horizon=5, control_dim=3)
    control = safe_mpc(state, dynamics, cost_fn, barrier_fn, cfg)
    assert control.shape == (3,)
    assert torch.all(control.abs() <= cfg.control_limits[1])
