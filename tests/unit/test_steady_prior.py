import torch

from ups.core.latent_state import LatentState
from ups.models.steady_prior import SteadyPrior, SteadyPriorConfig, steady_residual_norm


def test_steady_prior_forward():
    cfg = SteadyPriorConfig(latent_dim=20, hidden_dim=32, num_steps=4, cond_dim=3)
    prior = SteadyPrior(cfg)
    state = LatentState(z=torch.randn(2, 6, 20))
    cond = {"params": torch.randn(2, 3)}
    refined = prior(state, cond)
    assert refined.z.shape == state.z.shape


def test_steady_residual_norm():
    cfg = SteadyPriorConfig(latent_dim=10, hidden_dim=16, num_steps=3)
    prior = SteadyPrior(cfg)
    state = LatentState(z=torch.randn(1, 4, 10))
    norm = steady_residual_norm(prior, state)
    assert torch.isfinite(norm)
    assert norm >= 0
