"""Simple integration test for query sampling in inverse encoding loss."""

import torch
from ups.training.losses import inverse_encoding_loss
from ups.io.decoder_anypoint import AnyPointDecoder, AnyPointDecoderConfig


def test_inverse_encoding_with_and_without_sampling():
    """Test that inverse encoding loss works with and without query sampling."""

    # Setup
    latent_dim = 16
    tokens = 32
    B, H, W = 2, 32, 32
    N = H * W

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

    # Test data
    input_fields = {"u": torch.randn(B, N, 1)}
    latent = torch.randn(B, tokens, latent_dim)
    coords = torch.rand(B, N, 2)

    # Test WITHOUT query sampling (baseline)
    loss_full = inverse_encoding_loss(
        input_fields,
        latent,
        decoder,
        coords,
        weight=1.0,
        num_queries=None,  # Use all points
    )

    # Test WITH query sampling
    loss_sampled = inverse_encoding_loss(
        input_fields,
        latent,
        decoder,
        coords,
        weight=1.0,
        num_queries=512,  # Sample 50% of points
        query_strategy="uniform",
        grid_shape=(H, W),
    )

    # Both should produce valid scalar losses
    assert loss_full.ndim == 0, "Full loss should be scalar"
    assert loss_sampled.ndim == 0, "Sampled loss should be scalar"
    assert torch.isfinite(loss_full), "Full loss should be finite"
    assert torch.isfinite(loss_sampled), "Sampled loss should be finite"
    assert loss_full.item() > 0, "Full loss should be positive"
    assert loss_sampled.item() > 0, "Sampled loss should be positive"

    print(f"✓ Loss without sampling (1024 points): {loss_full.item():.6f}")
    print(f"✓ Loss with sampling (512 points):     {loss_sampled.item():.6f}")
    print(f"✓ Relative difference: {abs(loss_full.item() - loss_sampled.item()) / loss_full.item() * 100:.1f}%")
    print("\n✅ Query sampling works correctly for inverse encoding loss!")


if __name__ == "__main__":
    test_inverse_encoding_with_and_without_sampling()
