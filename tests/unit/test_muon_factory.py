"""Unit tests for Muon factory with fallback logic."""
import pytest
import torch
import torch.nn as nn
from ups.training.muon_factory import (
    create_muon_optimizer,
    get_available_backends,
)


def test_get_available_backends():
    """Test that available backends are detected."""
    backends = get_available_backends()

    # Should return a list
    assert isinstance(backends, list)

    # Should have at least one backend (torch.optim.Muon if PyTorch 2.9+)
    # This test may fail if PyTorch < 2.9 and flash-muon not installed
    # That's expected - the test documents the requirement
    assert len(backends) > 0, "No Muon backends available. Install torch>=2.9 or flash-muon"


def test_create_muon_optimizer_auto():
    """Test creating Muon optimizer with auto backend selection."""
    params = [nn.Parameter(torch.randn(16, 32)) for _ in range(3)]

    try:
        optimizer, backend = create_muon_optimizer(
            params,
            lr=1e-3,
            weight_decay=0.01,
            backend="auto"
        )

        # Should return an optimizer instance
        assert optimizer is not None

        # Should return a backend name
        assert backend in ["flash-muon", "torch.optim.Muon"]

        # Optimizer should have standard interface
        assert hasattr(optimizer, 'step')
        assert hasattr(optimizer, 'zero_grad')

    except RuntimeError as e:
        if "No Muon optimizer implementation found" in str(e):
            pytest.skip("No Muon implementation available (expected on PyTorch < 2.9)")
        raise


def test_create_muon_optimizer_torch_backend():
    """Test creating Muon optimizer with torch backend explicitly."""
    params = [nn.Parameter(torch.randn(16, 32)) for _ in range(3)]

    try:
        optimizer, backend = create_muon_optimizer(
            params,
            lr=1e-3,
            weight_decay=0.01,
            backend="torch"
        )

        assert backend == "torch.optim.Muon"

    except (RuntimeError, ImportError) as e:
        if "No Muon optimizer" in str(e) or "torch.optim" in str(e):
            pytest.skip("torch.optim.Muon not available (PyTorch < 2.9)")
        raise


def test_create_muon_optimizer_flash_backend():
    """Test creating Muon optimizer with flash backend explicitly."""
    params = [nn.Parameter(torch.randn(16, 32)) for _ in range(3)]

    try:
        optimizer, backend = create_muon_optimizer(
            params,
            lr=1e-3,
            weight_decay=0.01,
            backend="flash"
        )

        assert backend == "flash-muon"

    except (RuntimeError, ImportError) as e:
        if "No Muon optimizer" in str(e) or "flash_muon" in str(e):
            pytest.skip("flash-muon not installed")
        raise


def test_create_muon_optimizer_invalid_backend():
    """Test that invalid backend raises error."""
    params = [nn.Parameter(torch.randn(16, 32)) for _ in range(3)]

    with pytest.raises(ValueError, match="Unknown backend"):
        create_muon_optimizer(params, lr=1e-3, backend="invalid")


def test_create_muon_optimizer_hyperparameters():
    """Test that Muon hyperparameters are passed correctly."""
    params = [nn.Parameter(torch.randn(16, 32)) for _ in range(3)]

    try:
        optimizer, backend = create_muon_optimizer(
            params,
            lr=2e-3,
            weight_decay=0.05,
            momentum=0.9,
            nesterov=True,
            ns_steps=7,
            backend="auto"
        )

        # Check that hyperparameters are set (accessed via param_groups)
        assert optimizer.param_groups[0]['lr'] == 2e-3
        assert optimizer.param_groups[0]['weight_decay'] == 0.05

    except RuntimeError as e:
        if "No Muon optimizer implementation found" in str(e):
            pytest.skip("No Muon implementation available")
        raise
