"""Unit tests for UPT inverse loss functions."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from ups.training.losses import upt_inverse_encoding_loss, upt_inverse_decoding_loss


class MockDecoder(nn.Module):
    """Mock decoder that returns random predictions matching input shape."""

    def __init__(self, output_channels: dict[str, int] | None = None):
        super().__init__()
        self.output_channels = output_channels or {"u": 1}

    def forward(self, points: torch.Tensor, latent_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            points: (B, Q, coord_dim)
            latent_tokens: (B, tokens, latent_dim)

        Returns:
            Dict of field tensors (B, Q, channels)
        """
        B, Q, _ = points.shape
        outputs = {}
        for name, channels in self.output_channels.items():
            outputs[name] = torch.randn(B, Q, channels, device=points.device, dtype=points.dtype)
        return outputs


class MockEncoder(nn.Module):
    """Mock encoder that returns random latent tokens."""

    def __init__(self, num_tokens: int = 32, latent_dim: int = 16):
        super().__init__()
        self.num_tokens = num_tokens
        self.latent_dim = latent_dim

    def forward(
        self,
        fields: dict[str, torch.Tensor],
        coords: torch.Tensor,
        meta: dict | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            fields: Dict of field tensors (B, points, channels)
            coords: (B, points, coord_dim)
            meta: Optional metadata

        Returns:
            Latent tokens (B, num_tokens, latent_dim)
        """
        # Infer batch size from fields
        first_field = next(iter(fields.values()))
        B = first_field.shape[0]
        return torch.randn(B, self.num_tokens, self.latent_dim, device=first_field.device, dtype=first_field.dtype)


class TestUPTInverseEncodingLoss:
    """Tests for upt_inverse_encoding_loss."""

    def test_basic_computation(self):
        """Test that loss computes without errors."""
        batch_size, num_points, coord_dim = 4, 64, 2
        latent_dim, num_tokens = 16, 32

        # Create mock data
        fields = {"u": torch.randn(batch_size, num_points, 1)}
        coords = torch.randn(batch_size, num_points, coord_dim)
        latent = torch.randn(batch_size, num_tokens, latent_dim)

        decoder = MockDecoder(output_channels={"u": 1})

        # Compute loss
        loss = upt_inverse_encoding_loss(fields, coords, latent, decoder, num_query_points=32, weight=1.0)

        # Assertions
        assert loss.ndim == 0, "Loss should be scalar"
        assert loss.item() >= 0, "Loss should be non-negative"
        assert not torch.isnan(loss), "Loss should not be NaN"
        assert not torch.isinf(loss), "Loss should not be Inf"

    def test_with_weight(self):
        """Test that weight parameter scales loss correctly."""
        # Set seed for deterministic decoder outputs
        torch.manual_seed(42)

        batch_size, num_points = 4, 64
        fields = {"u": torch.randn(batch_size, num_points, 1)}
        coords = torch.randn(batch_size, num_points, 2)
        latent = torch.randn(batch_size, 32, 16)

        # Use deterministic decoder
        class DeterministicDecoder(nn.Module):
            def forward(self, points, latent_tokens):
                # Return constant value to make loss deterministic
                B, Q, _ = points.shape
                return {"u": torch.ones(B, Q, 1)}

        decoder = DeterministicDecoder()

        # Compute with different weights
        loss_1 = upt_inverse_encoding_loss(fields, coords, latent, decoder, weight=1.0)
        loss_2 = upt_inverse_encoding_loss(fields, coords, latent, decoder, weight=2.0)

        # Loss should scale exactly linearly with deterministic decoder
        assert torch.isclose(loss_2, loss_1 * 2.0, rtol=1e-5), "Weight should scale loss"

    def test_query_point_sampling(self):
        """Test that num_query_points parameter works."""
        batch_size, num_points = 4, 1024
        fields = {"u": torch.randn(batch_size, num_points, 1)}
        coords = torch.randn(batch_size, num_points, 2)
        latent = torch.randn(batch_size, 32, 16)
        decoder = MockDecoder(output_channels={"u": 1})

        # Compute with different query point counts
        loss_full = upt_inverse_encoding_loss(fields, coords, latent, decoder, num_query_points=None)
        loss_sampled = upt_inverse_encoding_loss(fields, coords, latent, decoder, num_query_points=128)

        # Both should be valid losses
        assert not torch.isnan(loss_full)
        assert not torch.isnan(loss_sampled)
        # Losses will differ due to sampling, but should be same order of magnitude
        assert 0.1 < (loss_sampled / loss_full).item() < 10.0

    def test_multiple_fields(self):
        """Test with multiple field components."""
        batch_size, num_points = 4, 64
        fields = {"u": torch.randn(batch_size, num_points, 2), "p": torch.randn(batch_size, num_points, 1)}
        coords = torch.randn(batch_size, num_points, 2)
        latent = torch.randn(batch_size, 32, 16)
        decoder = MockDecoder(output_channels={"u": 2, "p": 1})

        loss = upt_inverse_encoding_loss(fields, coords, latent, decoder)

        assert not torch.isnan(loss)
        assert loss.item() >= 0

    def test_missing_fields_returns_zero(self):
        """Test that missing fields in decoder output returns zero loss."""
        batch_size, num_points = 4, 64
        fields = {"u": torch.randn(batch_size, num_points, 1)}
        coords = torch.randn(batch_size, num_points, 2)
        latent = torch.randn(batch_size, 32, 16)

        # Decoder outputs different field names
        decoder = MockDecoder(output_channels={"v": 1})

        loss = upt_inverse_encoding_loss(fields, coords, latent, decoder)

        assert loss.item() == 0.0, "Loss should be zero when no fields match"

    def test_gradient_flow(self):
        """Test that gradients flow correctly."""
        batch_size, num_points = 4, 64
        fields = {"u": torch.randn(batch_size, num_points, 1)}
        coords = torch.randn(batch_size, num_points, 2)
        latent = torch.randn(batch_size, 32, 16, requires_grad=True)

        # Use a simple learnable decoder
        class LearnableDecoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.head = nn.Linear(16, 1)

            def forward(self, points, latent_tokens):
                B, Q, _ = points.shape
                # Simple pooling + projection
                pooled = latent_tokens.mean(dim=1, keepdim=True)  # (B, 1, latent_dim)
                out = self.head(pooled).expand(B, Q, -1)  # (B, Q, 1)
                return {"u": out}

        decoder = LearnableDecoder()

        loss = upt_inverse_encoding_loss(fields, coords, latent, decoder)
        loss.backward()

        # Check gradients exist
        assert latent.grad is not None, "Latent should have gradients"
        assert not torch.isnan(latent.grad).any(), "Gradients should not be NaN"


class TestUPTInverseDecodingLoss:
    """Tests for upt_inverse_decoding_loss."""

    def test_basic_computation(self):
        """Test that loss computes without errors."""
        batch_size, num_points, coord_dim = 4, 256, 2
        latent_dim, num_tokens = 16, 32

        latent = torch.randn(batch_size, num_tokens, latent_dim)
        coords = torch.randn(batch_size, num_points, coord_dim)

        decoder = MockDecoder(output_channels={"u": 1})
        encoder = MockEncoder(num_tokens=num_tokens, latent_dim=latent_dim)

        loss = upt_inverse_decoding_loss(
            latent,
            decoder,
            encoder,
            query_coords=coords,
            original_coords=coords,
            num_query_points=128,
            weight=1.0,
        )

        assert loss.ndim == 0, "Loss should be scalar"
        assert loss.item() >= 0, "Loss should be non-negative"
        assert not torch.isnan(loss), "Loss should not be NaN"
        assert not torch.isinf(loss), "Loss should not be Inf"

    def test_with_weight(self):
        """Test that weight parameter scales loss correctly."""
        # Set seed for deterministic encoder/decoder outputs
        torch.manual_seed(42)

        batch_size, num_points = 4, 256
        latent = torch.randn(batch_size, 32, 16)
        coords = torch.randn(batch_size, num_points, 2)

        # Use deterministic decoder and encoder
        class DeterministicDecoder(nn.Module):
            def forward(self, points, latent_tokens):
                B, Q, _ = points.shape
                return {"u": torch.ones(B, Q, 1)}

        class DeterministicEncoder(nn.Module):
            def forward(self, fields, coords, meta=None, **kwargs):
                B = next(iter(fields.values())).shape[0]
                return torch.ones(B, 32, 16)

        decoder = DeterministicDecoder()
        encoder = DeterministicEncoder()

        loss_1 = upt_inverse_decoding_loss(latent, decoder, encoder, coords, coords, weight=1.0)
        loss_2 = upt_inverse_decoding_loss(latent, decoder, encoder, coords, coords, weight=2.0)

        assert torch.isclose(loss_2, loss_1 * 2.0, rtol=1e-5), "Weight should scale loss"

    def test_query_point_sampling(self):
        """Test that num_query_points parameter works."""
        batch_size, num_points = 4, 1024
        latent = torch.randn(batch_size, 32, 16)
        coords = torch.randn(batch_size, num_points, 2)
        decoder = MockDecoder()
        encoder = MockEncoder()

        loss_full = upt_inverse_decoding_loss(latent, decoder, encoder, coords, coords, num_query_points=1024)
        loss_sampled = upt_inverse_decoding_loss(latent, decoder, encoder, coords, coords, num_query_points=128)

        assert not torch.isnan(loss_full)
        assert not torch.isnan(loss_sampled)

    def test_encoder_failure_returns_zero(self):
        """Test that encoder failures return zero loss gracefully."""

        class FailingEncoder(nn.Module):
            def forward(self, fields, coords, meta=None, **kwargs):
                raise ValueError("Encoder failure")

        batch_size, num_points = 4, 256
        latent = torch.randn(batch_size, 32, 16)
        coords = torch.randn(batch_size, num_points, 2)
        decoder = MockDecoder()
        encoder = FailingEncoder()

        # Should not raise, but return zero loss
        loss = upt_inverse_decoding_loss(latent, decoder, encoder, coords, coords)
        assert loss.item() == 0.0, "Loss should be zero when encoder fails"

    def test_gradient_flow(self):
        """Test that gradients flow correctly through decoder and encoder.

        Note: upt_inverse_decoding_loss uses .detach() on the original latent
        to prevent double gradients, which is correct behavior. Gradients should
        flow through decoder and encoder, but not back to the original latent.
        """
        batch_size, num_points = 4, 256
        latent = torch.randn(batch_size, 32, 16)
        coords = torch.randn(batch_size, num_points, 2)

        # Learnable decoder and encoder
        class LearnableDecoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.head = nn.Linear(16, 1)

            def forward(self, points, latent_tokens):
                B, Q, _ = points.shape
                pooled = latent_tokens.mean(dim=1, keepdim=True)
                out = self.head(pooled).expand(B, Q, -1)
                return {"u": out}

        class LearnableEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(1, 16)

            def forward(self, fields, coords, meta=None, **kwargs):
                # Simple projection from fields to latent
                u = next(iter(fields.values()))  # Get first field
                B = u.shape[0]
                pooled = u.mean(dim=1)  # (B, channels)
                latent = self.proj(pooled).unsqueeze(1).expand(-1, 32, -1)
                return latent

        decoder = LearnableDecoder()
        encoder = LearnableEncoder()

        loss = upt_inverse_decoding_loss(latent, decoder, encoder, coords, coords, num_query_points=128)
        loss.backward()

        # Check decoder has gradients
        decoder_params = list(decoder.parameters())
        assert len(decoder_params) > 0, "Decoder should have parameters"
        assert decoder_params[0].grad is not None, "Decoder should have gradients"
        assert not torch.isnan(decoder_params[0].grad).any(), "Decoder gradients should not be NaN"

        # Check encoder has gradients
        encoder_params = list(encoder.parameters())
        assert len(encoder_params) > 0, "Encoder should have parameters"
        assert encoder_params[0].grad is not None, "Encoder should have gradients"
        assert not torch.isnan(encoder_params[0].grad).any(), "Encoder gradients should not be NaN"


@pytest.mark.parametrize("batch_size", [1, 4, 8])
@pytest.mark.parametrize("num_points", [64, 256])
def test_different_batch_sizes(batch_size, num_points):
    """Test losses work with different batch sizes."""
    fields = {"u": torch.randn(batch_size, num_points, 1)}
    coords = torch.randn(batch_size, num_points, 2)
    latent = torch.randn(batch_size, 32, 16)
    decoder = MockDecoder()

    loss = upt_inverse_encoding_loss(fields, coords, latent, decoder)
    assert not torch.isnan(loss)
    assert loss.item() >= 0


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"))])
def test_device_compatibility(device):
    """Test losses work on both CPU and CUDA."""
    device = torch.device(device)
    batch_size, num_points = 4, 64

    fields = {"u": torch.randn(batch_size, num_points, 1, device=device)}
    coords = torch.randn(batch_size, num_points, 2, device=device)
    latent = torch.randn(batch_size, 32, 16, device=device)
    decoder = MockDecoder().to(device)

    loss = upt_inverse_encoding_loss(fields, coords, latent, decoder)
    assert loss.device.type == device.type
    assert not torch.isnan(loss)
