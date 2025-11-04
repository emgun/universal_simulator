import pytest
import torch

from src.ups.models.pure_transformer import (
    PureTransformer,
    PureTransformerConfig,
    TransformerBlock,
)


def test_pure_transformer_shape():
    """Pure transformer should preserve (B, T, D) shape."""
    cfg = PureTransformerConfig(
        input_dim=64, hidden_dim=128, depth=4, num_heads=4, drop_path=0.0
    )
    model = PureTransformer(cfg)

    x = torch.randn(4, 128, 64)  # (B=4, T=128, D=64)
    out = model(x)

    assert out.shape == x.shape


def test_pure_transformer_fixed_tokens():
    """Token count should remain fixed (no pooling/unpooling)."""
    cfg = PureTransformerConfig(
        input_dim=64, hidden_dim=64, depth=8, num_heads=4, drop_path=0.1
    )
    model = PureTransformer(cfg)

    # Test various token counts
    for tokens in [32, 64, 128, 256, 512]:
        x = torch.randn(2, tokens, 64)
        out = model(x)
        assert out.shape[1] == tokens


def test_pure_transformer_standard_attention():
    """Test with standard multi-head attention."""
    cfg = PureTransformerConfig(
        input_dim=192,
        hidden_dim=192,
        depth=4,
        num_heads=6,
        attention_type="standard",
        qk_norm=True,
    )
    model = PureTransformer(cfg)

    x = torch.randn(4, 256, 192)
    out = model(x)
    assert out.shape == x.shape


def test_pure_transformer_channel_separated_attention():
    """Test with channel-separated attention."""
    cfg = PureTransformerConfig(
        input_dim=192,
        hidden_dim=192,
        depth=4,
        num_heads=6,
        attention_type="channel_separated",
        group_size=24,  # Must be divisible by num_heads
    )
    model = PureTransformer(cfg)

    x = torch.randn(4, 256, 192)
    out = model(x)
    assert out.shape == x.shape


def test_pure_transformer_drop_path():
    """Drop-path should introduce stochasticity in training."""
    cfg = PureTransformerConfig(
        input_dim=64, hidden_dim=64, depth=8, num_heads=4, drop_path=0.3
    )
    model = PureTransformer(cfg)
    model.train()

    x = torch.randn(4, 128, 64)
    out1 = model(x)
    out2 = model(x)

    # With drop-path, outputs should differ
    assert not torch.allclose(out1, out2)


def test_pure_transformer_inference_deterministic():
    """Inference should be deterministic."""
    cfg = PureTransformerConfig(
        input_dim=64, hidden_dim=64, depth=8, num_heads=4, drop_path=0.3
    )
    model = PureTransformer(cfg)
    model.eval()

    x = torch.randn(4, 128, 64)
    out1 = model(x)
    out2 = model(x)

    # In eval mode, should be deterministic
    assert torch.allclose(out1, out2)


def test_pure_transformer_depth_scaling():
    """Deeper models should have linearly increasing drop-path rates."""
    cfg = PureTransformerConfig(
        input_dim=64, hidden_dim=64, depth=12, num_heads=4, drop_path=0.2
    )
    model = PureTransformer(cfg)

    # Check drop-path rates increase linearly
    # Note: First layer has drop_prob=0, so it's Identity
    from src.ups.core.drop_path import DropPath

    drop_rates = []
    for layer in model.layers:
        if isinstance(layer.drop_path1, DropPath):
            drop_rates.append(layer.drop_path1.drop_prob)
        else:  # Identity
            drop_rates.append(0.0)

    expected = [0.2 * i / 11 for i in range(12)]

    for actual, exp in zip(drop_rates, expected):
        assert abs(actual - exp) < 1e-6


def test_transformer_block():
    """TransformerBlock should work standalone."""
    block = TransformerBlock(
        dim=192,
        num_heads=6,
        attention_type="standard",
        group_size=32,
        mlp_ratio=4.0,
        qk_norm=True,
        drop_path=0.1,
        dropout=0.0,
    )

    x = torch.randn(4, 256, 192)
    out = block(x)
    assert out.shape == x.shape


def test_pure_transformer_different_input_hidden_dims():
    """Test with different input and hidden dimensions."""
    cfg = PureTransformerConfig(
        input_dim=64, hidden_dim=192, depth=4, num_heads=6, drop_path=0.0
    )
    model = PureTransformer(cfg)

    x = torch.randn(4, 128, 64)
    out = model(x)
    assert out.shape == x.shape  # Should project back to input_dim


def test_pure_transformer_no_dropout():
    """Test with dropout disabled."""
    cfg = PureTransformerConfig(
        input_dim=64, hidden_dim=64, depth=4, num_heads=4, dropout=0.0
    )
    model = PureTransformer(cfg)
    model.eval()

    x = torch.randn(2, 64, 64)
    out = model(x)
    assert out.shape == x.shape


def test_pure_transformer_with_dropout():
    """Test with dropout enabled."""
    cfg = PureTransformerConfig(
        input_dim=64, hidden_dim=64, depth=4, num_heads=4, dropout=0.1
    )
    model = PureTransformer(cfg)
    model.train()

    x = torch.randn(2, 64, 64)
    out1 = model(x)
    out2 = model(x)

    # With dropout in training, outputs should differ
    assert not torch.allclose(out1, out2)


def test_transformer_block_invalid_attention_type():
    """Should raise error for invalid attention type."""
    with pytest.raises(ValueError, match="Unknown attention_type"):
        TransformerBlock(
            dim=64,
            num_heads=4,
            attention_type="invalid",
            group_size=32,
            mlp_ratio=4.0,
            qk_norm=False,
            drop_path=0.0,
            dropout=0.0,
        )


def test_pure_transformer_single_layer():
    """Test with single transformer layer."""
    cfg = PureTransformerConfig(
        input_dim=64, hidden_dim=64, depth=1, num_heads=4, drop_path=0.0
    )
    model = PureTransformer(cfg)

    x = torch.randn(2, 32, 64)
    out = model(x)
    assert out.shape == x.shape


def test_pure_transformer_grad_flow():
    """Test that gradients flow through the model."""
    cfg = PureTransformerConfig(
        input_dim=64, hidden_dim=64, depth=4, num_heads=4, drop_path=0.0
    )
    model = PureTransformer(cfg)

    x = torch.randn(2, 32, 64, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()

    # Check that input has gradients
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
