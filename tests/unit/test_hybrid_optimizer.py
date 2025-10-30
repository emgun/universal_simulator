"""Unit tests for hybrid optimizer wrapper."""
import pytest
import torch
import torch.nn as nn
from ups.training.hybrid_optimizer import HybridOptimizer


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 8)
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        return self.fc(self.linear(x))


def test_hybrid_optimizer_creation():
    """Test that HybridOptimizer can be created."""
    model = SimpleModel()

    # Split parameters manually for testing
    weights = [p for p in model.parameters() if p.ndim >= 2]
    biases = [p for p in model.parameters() if p.ndim < 2]

    opt1 = torch.optim.SGD(weights, lr=0.1)
    opt2 = torch.optim.Adam(biases, lr=0.01)

    hybrid = HybridOptimizer([opt1, opt2])

    assert len(hybrid.optimizers) == 2
    assert isinstance(hybrid.optimizers[0], torch.optim.SGD)
    assert isinstance(hybrid.optimizers[1], torch.optim.Adam)


def test_hybrid_optimizer_empty_fails():
    """Test that HybridOptimizer requires at least one optimizer."""
    with pytest.raises(ValueError, match="Must provide at least one optimizer"):
        HybridOptimizer([])


def test_hybrid_optimizer_step():
    """Test that hybrid optimizer steps all child optimizers."""
    model = SimpleModel()
    weights = [p for p in model.parameters() if p.ndim >= 2]
    biases = [p for p in model.parameters() if p.ndim < 2]

    opt1 = torch.optim.SGD(weights, lr=0.1)
    opt2 = torch.optim.Adam(biases, lr=0.01)
    hybrid = HybridOptimizer([opt1, opt2])

    # Get initial parameter values
    initial_weights = [p.clone() for p in weights]
    initial_biases = [p.clone() for p in biases]

    # Training step
    x = torch.randn(2, 4)
    loss = model(x).sum()
    loss.backward()
    hybrid.step()

    # Check that parameters were updated
    for p_before, p_after in zip(initial_weights, weights):
        assert not torch.allclose(p_before, p_after), "Weights should have changed"
    for p_before, p_after in zip(initial_biases, biases):
        assert not torch.allclose(p_before, p_after), "Biases should have changed"


def test_hybrid_optimizer_zero_grad():
    """Test that zero_grad clears gradients for all parameters."""
    model = SimpleModel()
    weights = [p for p in model.parameters() if p.ndim >= 2]
    biases = [p for p in model.parameters() if p.ndim < 2]

    opt1 = torch.optim.SGD(weights, lr=0.1)
    opt2 = torch.optim.Adam(biases, lr=0.01)
    hybrid = HybridOptimizer([opt1, opt2])

    # Create gradients
    x = torch.randn(2, 4)
    loss = model(x).sum()
    loss.backward()

    # Verify gradients exist
    for p in model.parameters():
        assert p.grad is not None

    # Zero gradients
    hybrid.zero_grad()

    # Verify gradients are None (set_to_none=True by default)
    for p in model.parameters():
        assert p.grad is None


def test_hybrid_optimizer_param_groups():
    """Test that param_groups returns flattened list from all optimizers."""
    model = SimpleModel()
    weights = [p for p in model.parameters() if p.ndim >= 2]
    biases = [p for p in model.parameters() if p.ndim < 2]

    opt1 = torch.optim.SGD(weights, lr=0.1)
    opt2 = torch.optim.Adam(biases, lr=0.01)
    hybrid = HybridOptimizer([opt1, opt2])

    # Get param_groups
    param_groups = hybrid.param_groups

    # Should have 2 param groups (one from each optimizer)
    assert len(param_groups) == 2

    # First group should be SGD (lr=0.1)
    assert param_groups[0]['lr'] == 0.1

    # Second group should be Adam (lr=0.01)
    assert param_groups[1]['lr'] == 0.01


def test_hybrid_optimizer_state_dict():
    """Test state dict saving and loading."""
    model = SimpleModel()
    weights = [p for p in model.parameters() if p.ndim >= 2]
    biases = [p for p in model.parameters() if p.ndim < 2]

    opt1 = torch.optim.SGD(weights, lr=0.1, momentum=0.9)
    opt2 = torch.optim.Adam(biases, lr=0.01)
    hybrid = HybridOptimizer([opt1, opt2])

    # Take a step to populate state
    x = torch.randn(2, 4)
    loss = model(x).sum()
    loss.backward()
    hybrid.step()

    # Save state
    state = hybrid.state_dict()

    assert "optimizer_0" in state
    assert "optimizer_1" in state

    # Create new hybrid optimizer
    opt1_new = torch.optim.SGD(weights, lr=0.1, momentum=0.9)
    opt2_new = torch.optim.Adam(biases, lr=0.01)
    hybrid_new = HybridOptimizer([opt1_new, opt2_new])

    # Load state
    hybrid_new.load_state_dict(state)

    # Verify state was loaded (check momentum buffer exists for SGD)
    assert len(opt1_new.state) > 0


def test_hybrid_optimizer_repr():
    """Test string representation."""
    model = SimpleModel()
    weights = [p for p in model.parameters() if p.ndim >= 2]
    biases = [p for p in model.parameters() if p.ndim < 2]

    opt1 = torch.optim.SGD(weights, lr=0.1)
    opt2 = torch.optim.Adam(biases, lr=0.01)
    hybrid = HybridOptimizer([opt1, opt2])

    repr_str = repr(hybrid)
    assert "HybridOptimizer" in repr_str
    assert "SGD" in repr_str
    assert "Adam" in repr_str
