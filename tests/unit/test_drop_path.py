import pytest
import torch
from src.ups.core.drop_path import DropPath


def test_drop_path_disabled():
    """Drop-path with prob=0.0 should return input unchanged."""
    drop = DropPath(drop_prob=0.0)
    x = torch.randn(4, 16, 32)
    out = drop(x)
    assert torch.allclose(out, x)


def test_drop_path_inference():
    """Drop-path in eval mode should return input unchanged."""
    drop = DropPath(drop_prob=0.5)
    drop.eval()
    x = torch.randn(4, 16, 32)
    out = drop(x)
    assert torch.allclose(out, x)


def test_drop_path_training():
    """Drop-path in training should randomly zero some samples."""
    drop = DropPath(drop_prob=0.5)
    drop.train()
    x = torch.ones(100, 16, 32)  # All ones

    # Run multiple times, check that some samples are zeroed
    n_dropped = 0
    for _ in range(10):
        out = drop(x)
        # Check if any batch elements are zeroed (all zeros in that sample)
        batch_norms = out.view(100, -1).norm(dim=1)
        n_dropped += (batch_norms == 0).sum().item()

    # With prob=0.5, expect ~50% dropped over 1000 samples
    drop_rate = n_dropped / 1000
    assert 0.3 < drop_rate < 0.7, f"Drop rate {drop_rate} not near 0.5"


def test_drop_path_scaling():
    """Drop-path should scale by 1/keep_prob to maintain expected value."""
    drop = DropPath(drop_prob=0.5, scale_by_keep=True)
    drop.train()
    x = torch.ones(1000, 16, 32)

    # Average over many runs should approximate input mean
    outputs = []
    for _ in range(100):
        outputs.append(drop(x))

    avg_output = torch.stack(outputs).mean(dim=0)
    expected = x  # Should average back to input

    # Check mean is close (within 10% due to randomness)
    assert torch.allclose(avg_output.mean(), expected.mean(), rtol=0.1)


def test_drop_path_shape_preservation():
    """Drop-path should preserve tensor shape."""
    drop = DropPath(drop_prob=0.2)

    # Test various shapes
    for shape in [(4, 16, 32), (2, 100, 64), (8, 256, 128)]:
        x = torch.randn(*shape)
        out = drop(x)
        assert out.shape == x.shape


def test_drop_path_extra_repr():
    """Test string representation."""
    drop = DropPath(drop_prob=0.1)
    repr_str = drop.extra_repr()
    assert "drop_prob=0.1" in repr_str
