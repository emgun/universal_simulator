import sys

import pytest
import torch

from ups.core.blocks_pdet import PDETransformerConfig
from ups.core.conditioning import ConditioningConfig
from ups.core.latent_state import LatentState
from ups.models.latent_operator import LatentOperator, LatentOperatorConfig
from ups.models.pure_transformer import PureTransformerConfig


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


def test_latent_operator_pure_transformer():
    """Test LatentOperator with pure stacked transformer architecture."""
    latent_dim = 64
    cfg = LatentOperatorConfig(
        latent_dim=latent_dim,
        architecture_type="pdet_stack",
        pdet=PureTransformerConfig(
            input_dim=latent_dim,
            hidden_dim=latent_dim * 2,
            depth=4,
            num_heads=4,
            attention_type="standard",
            drop_path=0.1,
        ),
        time_embed_dim=latent_dim,
    )
    op = LatentOperator(cfg)

    state = LatentState(z=torch.randn(2, 128, latent_dim), t=torch.tensor(0.0), cond={})
    next_state = op(state, torch.tensor(0.1))

    assert isinstance(next_state, LatentState)
    assert next_state.z.shape == state.z.shape
    assert torch.allclose(next_state.t, torch.tensor(0.1))


def test_latent_operator_backward_compatibility():
    """Test that default architecture_type is pdet_unet for backward compatibility."""
    latent_dim = 32
    cfg = LatentOperatorConfig(
        latent_dim=latent_dim,
        pdet=PDETransformerConfig(
            input_dim=latent_dim,
            hidden_dim=latent_dim * 2,
            depths=(1, 1, 1),
            group_size=16,
            num_heads=4,
        ),
        # No architecture_type specified - should default to pdet_unet
    )
    op = LatentOperator(cfg)

    # Should use PDETransformerBlock
    from ups.core.blocks_pdet import PDETransformerBlock
    assert isinstance(op.core, PDETransformerBlock)

    state = LatentState(z=torch.randn(2, 64, latent_dim), t=torch.tensor(0.0), cond={})
    next_state = op(state, torch.tensor(0.1))
    assert next_state.z.shape == state.z.shape


def test_latent_operator_invalid_architecture_type():
    """Test that invalid architecture type raises error."""
    latent_dim = 32
    cfg = LatentOperatorConfig(
        latent_dim=latent_dim,
        architecture_type="invalid_type",  # Invalid
        pdet=PDETransformerConfig(
            input_dim=latent_dim,
            hidden_dim=latent_dim * 2,
            depths=(1, 1, 1),
            group_size=16,
            num_heads=4,
        ),
    )

    with pytest.raises(ValueError, match="Unknown architecture_type"):
        LatentOperator(cfg)


def test_latent_operator_dimension_mismatch():
    """Test that dimension mismatch raises clear error."""
    latent_dim = 32
    cfg = LatentOperatorConfig(
        latent_dim=latent_dim,
        architecture_type="pdet_stack",
        pdet=PureTransformerConfig(
            input_dim=64,  # Mismatched!
            hidden_dim=128,
            depth=4,
            num_heads=4,
        ),
    )

    with pytest.raises(ValueError, match="must match latent_dim"):
        LatentOperator(cfg)
