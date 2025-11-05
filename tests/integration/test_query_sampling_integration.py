"""Integration test for query sampling in training losses."""

import torch
import pytest
from ups.training.losses import (
    inverse_encoding_loss,
    inverse_decoding_loss,
    compute_operator_loss_bundle,
)
from ups.io.enc_grid import GridEncoder, GridEncoderConfig
from ups.io.decoder_anypoint import AnyPointDecoder, AnyPointDecoderConfig


@pytest.fixture
def setup_models():
    """Create encoder and decoder for testing."""
    latent_dim = 16
    tokens = 32

    encoder_cfg = GridEncoderConfig(
        latent_len=tokens,
        latent_dim=latent_dim,
        field_channels={"u": 1},
        patch_size=4,
    )
    encoder = GridEncoder(encoder_cfg)

    decoder_cfg = AnyPointDecoderConfig(
        latent_dim=latent_dim,
        query_dim=2,
        hidden_dim=64,
        mlp_hidden_dim=128,
        num_layers=2,
        num_heads=4,
        frequencies=(1.0, 2.0, 4.0),
        output_channels={"u": 1},
    )
    decoder = AnyPointDecoder(decoder_cfg)

    return encoder, decoder, latent_dim, tokens


def test_inverse_encoding_loss_with_query_sampling(setup_models):
    """Test inverse encoding loss with query sampling."""
    encoder, decoder, latent_dim, tokens = setup_models

    B, H, W = 2, 32, 32
    N = H * W

    # Create test data
    input_fields = {"u": torch.randn(B, N, 1)}
    latent = torch.randn(B, tokens, latent_dim)
    coords = torch.rand(B, N, 2)

    # Test with query sampling
    loss_sampled = inverse_encoding_loss(
        input_fields,
        latent,
        decoder,
        coords,
        weight=1.0,
        num_queries=256,  # Sample 256 out of 1024 points
        query_strategy="uniform",
        grid_shape=(H, W),
    )

    # Test without query sampling (baseline)
    loss_full = inverse_encoding_loss(
        input_fields,
        latent,
        decoder,
        coords,
        weight=1.0,
        num_queries=None,  # Use all points
    )

    # Both should produce valid scalar losses
    assert loss_sampled.ndim == 0, "Sampled loss should be scalar"
    assert loss_full.ndim == 0, "Full loss should be scalar"
    assert torch.isfinite(loss_sampled), "Sampled loss should be finite"
    assert torch.isfinite(loss_full), "Full loss should be finite"

    # Losses should be similar (not identical due to sampling randomness)
    # But both should be reasonable (not NaN/Inf)
    print(f"Loss with sampling (256 queries): {loss_sampled.item():.6f}")
    print(f"Loss without sampling (1024 queries): {loss_full.item():.6f}")


def test_inverse_decoding_loss_with_query_sampling(setup_models):
    """Test inverse decoding loss with query sampling."""
    encoder, decoder, latent_dim, tokens = setup_models

    B, H, W = 2, 32, 32
    N = H * W

    # Create test data
    latent = torch.randn(B, tokens, latent_dim)
    coords = torch.rand(B, N, 2)
    meta = {"grid_shape": (H, W)}

    # Test with query sampling
    loss_sampled = inverse_decoding_loss(
        latent,
        decoder,
        encoder,
        coords,  # query_positions
        coords,  # coords for encoder
        meta,
        weight=1.0,
        num_queries=256,
        query_strategy="uniform",
        grid_shape=(H, W),
    )

    # Test without query sampling
    loss_full = inverse_decoding_loss(
        latent,
        decoder,
        encoder,
        coords,
        coords,
        meta,
        weight=1.0,
        num_queries=None,
    )

    assert loss_sampled.ndim == 0
    assert loss_full.ndim == 0
    assert torch.isfinite(loss_sampled)
    assert torch.isfinite(loss_full)

    print(f"Inverse decoding loss with sampling: {loss_sampled.item():.6f}")
    print(f"Inverse decoding loss without sampling: {loss_full.item():.6f}")


def test_compute_operator_loss_bundle_with_query_sampling(setup_models):
    """Test full loss bundle computation with query sampling."""
    encoder, decoder, latent_dim, tokens = setup_models

    B, H, W = 2, 32, 32
    N = H * W

    # Create test data
    input_fields = {"u": torch.randn(B, N, 1)}
    encoded_latent = torch.randn(B, tokens, latent_dim)
    coords = torch.rand(B, N, 2)
    meta = {"grid_shape": (H, W)}
    pred_next = torch.randn(B, tokens, latent_dim)
    target_next = torch.randn(B, tokens, latent_dim)

    weights = {
        "lambda_forward": 1.0,
        "lambda_inv_enc": 0.1,
        "lambda_inv_dec": 0.1,
    }

    # Test with query sampling
    bundle_sampled = compute_operator_loss_bundle(
        input_fields=input_fields,
        encoded_latent=encoded_latent,
        decoder=decoder,
        input_positions=coords,
        encoder=encoder,
        query_positions=coords,
        coords=coords,
        meta=meta,
        pred_next=pred_next,
        target_next=target_next,
        weights=weights,
        num_queries=256,
        query_strategy="uniform",
        grid_shape=(H, W),
    )

    # Test without query sampling
    bundle_full = compute_operator_loss_bundle(
        input_fields=input_fields,
        encoded_latent=encoded_latent,
        decoder=decoder,
        input_positions=coords,
        encoder=encoder,
        query_positions=coords,
        coords=coords,
        meta=meta,
        pred_next=pred_next,
        target_next=target_next,
        weights=weights,
        num_queries=None,
    )

    # Verify both bundles have expected components
    assert "L_forward" in bundle_sampled.components
    assert "L_inv_enc" in bundle_sampled.components
    assert "L_inv_dec" in bundle_sampled.components

    assert "L_forward" in bundle_full.components
    assert "L_inv_enc" in bundle_full.components
    assert "L_inv_dec" in bundle_full.components

    # Verify total loss is finite
    assert torch.isfinite(bundle_sampled.total)
    assert torch.isfinite(bundle_full.total)

    print(f"\nLoss bundle with query sampling (256 queries):")
    print(f"  Total: {bundle_sampled.total.item():.6f}")
    for name, value in bundle_sampled.components.items():
        print(f"  {name}: {value.item():.6f}")

    print(f"\nLoss bundle without query sampling (1024 queries):")
    print(f"  Total: {bundle_full.total.item():.6f}")
    for name, value in bundle_full.components.items():
        print(f"  {name}: {value.item():.6f}")


def test_backward_compatibility():
    """Test that existing code without query sampling still works."""
    # Create simple test without encoder/decoder
    B, tokens, latent_dim = 2, 32, 16
    pred_next = torch.randn(B, tokens, latent_dim)
    target_next = torch.randn(B, tokens, latent_dim)

    # Should work without any query sampling parameters
    bundle = compute_operator_loss_bundle(
        pred_next=pred_next,
        target_next=target_next,
        weights={"lambda_forward": 1.0},
    )

    assert "L_forward" in bundle.components
    assert torch.isfinite(bundle.total)
    print(f"\nBackward compatibility test - Loss: {bundle.total.item():.6f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
