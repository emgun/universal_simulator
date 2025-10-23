"""Integration test for UPT inverse losses data pipeline."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from ups.data.latent_pairs import LatentPair, collate_latent_pairs, unpack_batch
from ups.io.enc_grid import GridEncoder, GridEncoderConfig
from ups.io.decoder_anypoint import AnyPointDecoder, AnyPointDecoderConfig
from ups.training.losses import upt_inverse_encoding_loss, upt_inverse_decoding_loss


def test_latent_pair_with_upt_fields():
    """Test that LatentPair can hold UPT fields."""
    z0 = torch.randn(10, 32, 16)
    z1 = torch.randn(10, 32, 16)
    cond = {}
    fields_orig = {"u": torch.randn(10, 256, 1)}
    coords = torch.randn(1, 256, 2)
    meta = {"grid_shape": (16, 16)}

    pair = LatentPair(
        z0=z0,
        z1=z1,
        cond=cond,
        fields_orig=fields_orig,
        coords=coords,
        meta=meta,
    )

    assert pair.fields_orig is not None
    assert pair.coords is not None
    assert pair.meta is not None
    assert "u" in pair.fields_orig
    assert pair.meta["grid_shape"] == (16, 16)


def test_collate_with_upt_fields():
    """Test that collate_latent_pairs handles UPT fields."""
    batch_size = 2
    time_steps = 10
    num_points = 256
    latent_tokens = 32
    latent_dim = 16

    # Create mock LatentPair items
    items = []
    for _ in range(batch_size):
        items.append(
            LatentPair(
                z0=torch.randn(time_steps, latent_tokens, latent_dim),
                z1=torch.randn(time_steps, latent_tokens, latent_dim),
                cond={},
                fields_orig={"u": torch.randn(time_steps, num_points, 1)},
                coords=torch.randn(1, num_points, 2),
                meta={"grid_shape": (16, 16)},
            )
        )

    # Collate
    result = collate_latent_pairs(items)

    # Should return (z0, z1, cond, fields_orig, coords, meta) - 6 items
    assert len(result) == 6
    z0, z1, cond, fields_orig, coords, meta = result

    # Check shapes
    assert z0.shape == (batch_size * time_steps, latent_tokens, latent_dim)
    assert z1.shape == (batch_size * time_steps, latent_tokens, latent_dim)
    assert fields_orig is not None
    assert "u" in fields_orig
    assert fields_orig["u"].shape == (batch_size * time_steps, num_points, 1)
    assert coords.shape == (1, num_points, 2)
    assert meta is not None
    assert meta["grid_shape"] == (16, 16)


def test_unpack_batch_upt_format():
    """Test that unpack_batch handles new UPT format."""
    batch_size = 2
    time_steps = 10

    # Create mock batch in UPT format
    batch = (
        torch.randn(batch_size * time_steps, 32, 16),  # z0
        torch.randn(batch_size * time_steps, 32, 16),  # z1
        {},  # cond
        {"u": torch.randn(batch_size * time_steps, 256, 1)},  # fields_orig
        torch.randn(1, 256, 2),  # coords
        {"grid_shape": (16, 16)},  # meta
    )

    # Unpack
    result = unpack_batch(batch)

    # Should return 7 items (with None for future)
    assert len(result) == 7
    z0, z1, cond, future, fields_orig, coords, meta = result

    assert future is None  # No future in this batch
    assert fields_orig is not None
    assert coords is not None
    assert meta is not None


def test_upt_losses_with_real_modules():
    """Integration test: UPT losses with real encoder/decoder."""
    batch_size = 2
    num_points = 64  # Small grid for speed
    latent_tokens = 16
    latent_dim = 8
    grid_shape = (8, 8)

    # Create encoder
    encoder_cfg = GridEncoderConfig(
        latent_len=latent_tokens,
        latent_dim=latent_dim,
        field_channels={"u": 1},
        patch_size=2,
        use_fourier_features=False,
    )
    encoder = GridEncoder(encoder_cfg)

    # Create decoder
    decoder_cfg = AnyPointDecoderConfig(
        latent_dim=latent_dim,
        query_dim=2,
        hidden_dim=16,
        num_layers=2,
        num_heads=2,
        output_channels={"u": 1},
    )
    decoder = AnyPointDecoder(decoder_cfg)

    # Create mock data
    fields_orig = {"u": torch.randn(batch_size, num_points, 1)}
    coords = torch.randn(batch_size, num_points, 2)
    latent = torch.randn(batch_size, latent_tokens, latent_dim)
    meta = {"grid_shape": grid_shape}

    # Test inverse encoding loss
    loss_inv_enc = upt_inverse_encoding_loss(
        input_fields=fields_orig,
        input_coords=coords,
        latent=latent,
        decoder=decoder,
        meta=meta,
        num_query_points=32,
        weight=1.0,
    )

    assert loss_inv_enc.ndim == 0
    assert loss_inv_enc.item() >= 0
    assert not torch.isnan(loss_inv_enc)

    # Test inverse decoding loss
    loss_inv_dec = upt_inverse_decoding_loss(
        latent=latent,
        decoder=decoder,
        encoder=encoder,
        query_coords=coords,
        original_coords=coords,
        meta=meta,
        num_query_points=32,
        weight=1.0,
    )

    assert loss_inv_dec.ndim == 0
    assert loss_inv_dec.item() >= 0
    assert not torch.isnan(loss_inv_dec)

    # Test gradients flow
    loss_inv_enc.backward(retain_graph=True)
    decoder_params = list(decoder.parameters())
    assert decoder_params[0].grad is not None


def test_backward_compatibility():
    """Test that old batch format still works (backward compatibility)."""
    # Old format: (z0, z1, cond, future)
    batch_old = (
        torch.randn(10, 32, 16),
        torch.randn(10, 32, 16),
        {},
        torch.randn(10, 5, 32, 16),
    )

    result = unpack_batch(batch_old)
    assert len(result) == 4
    z0, z1, cond, future = result
    assert future is not None
    assert future.shape == (10, 5, 32, 16)

    # Old format without future: (z0, z1, cond)
    batch_old_no_future = (
        torch.randn(10, 32, 16),
        torch.randn(10, 32, 16),
        {},
    )

    result2 = unpack_batch(batch_old_no_future)
    assert len(result2) == 3
    z0, z1, cond = result2


if __name__ == "__main__":
    print("Running UPT integration tests...")

    print("✓ test_latent_pair_with_upt_fields")
    test_latent_pair_with_upt_fields()

    print("✓ test_collate_with_upt_fields")
    test_collate_with_upt_fields()

    print("✓ test_unpack_batch_upt_format")
    test_unpack_batch_upt_format()

    print("✓ test_upt_losses_with_real_modules")
    test_upt_losses_with_real_modules()

    print("✓ test_backward_compatibility")
    test_backward_compatibility()

    print("\n✅ All integration tests passed!")
