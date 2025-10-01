import sys

import torch

from ups.core.blocks_pdet import PDETransformerConfig
from ups.core.conditioning import ConditioningConfig
from ups.core.latent_state import LatentState
from ups.models.latent_operator import LatentOperator, LatentOperatorConfig


def make_operator(latent_dim: int = 32):
    cfg = LatentOperatorConfig(
        latent_dim=latent_dim,
        pdet=PDETransformerConfig(
            input_dim=latent_dim,
            hidden_dim=latent_dim * 2,
            depths=(1, 2, 2),
            group_size=16,
            num_heads=4,
        ),
        conditioning=ConditioningConfig(latent_dim=latent_dim, hidden_dim=latent_dim, sources={"params": 4}),
        time_embed_dim=latent_dim,
    )
    return LatentOperator(cfg)


def test_latent_operator_forward_residual_addition():
    op = make_operator()
    state = LatentState(z=torch.randn(3, 64, 32), t=torch.tensor(0.0), cond={"params": torch.randn(3, 4)})
    next_state = op(state, torch.tensor(0.25))
    assert isinstance(next_state, LatentState)
    assert next_state.z.shape == state.z.shape
    assert torch.allclose(next_state.t, torch.tensor(0.25))


def test_latent_operator_gradients():
    op = make_operator(latent_dim=16)
    state = LatentState(z=torch.randn(2, 40, 16, requires_grad=True), cond={"params": torch.randn(2, 4)})
    residual = op.step(state, torch.tensor(0.1))
    assert residual.shape == state.z.shape
    loss = residual.pow(2).mean()
    loss.backward()
    assert torch.isfinite(state.z.grad).all()
