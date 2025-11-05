"""Unit tests for physics-informed loss functions."""

import torch
import pytest
from ups.training.physics_losses import (
    divergence_penalty_2d,
    conservation_penalty,
    boundary_condition_penalty_grid,
    positivity_penalty,
    smoothness_penalty,
)


def test_divergence_penalty_zero():
    """Test that divergence-free flow has near-zero penalty."""
    # Create divergence-free field (circular flow)
    H, W = 32, 32
    x = torch.linspace(-1, 1, W)
    y = torch.linspace(-1, 1, H)
    X, Y = torch.meshgrid(x, y, indexing='xy')

    # Circular flow: u = -y, v = x → ∇·u = ∂(-y)/∂x + ∂(x)/∂y = 0 + 0 = 0
    u = -Y.unsqueeze(0).unsqueeze(-1)  # (1, H, W, 1)
    v = X.unsqueeze(0).unsqueeze(-1)   # (1, H, W, 1)
    velocity = torch.cat([u, v], dim=-1)  # (1, H, W, 2)

    penalty = divergence_penalty_2d(velocity, (H, W), dx=2/W, dy=2/H, weight=1.0)

    # Should be nearly zero (numerical errors from finite differences)
    assert penalty.item() < 1e-2, f"Expected divergence penalty < 0.01, got {penalty.item():.6f}"
    assert torch.isfinite(penalty), "Divergence penalty should be finite"


def test_divergence_penalty_nonzero():
    """Test that divergent flow has positive penalty."""
    # Create divergent field: u = x, v = y → ∇·u = ∂(x)/∂x + ∂(y)/∂y = 1 + 1 = 2
    H, W = 32, 32
    x = torch.linspace(-1, 1, W)
    y = torch.linspace(-1, 1, H)
    X, Y = torch.meshgrid(x, y, indexing='xy')

    u = X.unsqueeze(0).unsqueeze(-1)  # (1, H, W, 1)
    v = Y.unsqueeze(0).unsqueeze(-1)  # (1, H, W, 1)
    velocity = torch.cat([u, v], dim=-1)  # (1, H, W, 2)

    penalty = divergence_penalty_2d(velocity, (H, W), dx=2/W, dy=2/H, weight=1.0)

    # Divergence should be ≈2 everywhere, so penalty should be ≈2
    assert penalty.item() > 1.0, f"Expected divergence penalty > 1.0, got {penalty.item():.6f}"
    assert torch.isfinite(penalty), "Divergence penalty should be finite"


def test_conservation_penalty_constant():
    """Test that constant field satisfies conservation."""
    B, N, C = 4, 1024, 1
    field_ref = torch.ones(B, N, C)
    field_cur = torch.ones(B, N, C)

    penalty = conservation_penalty(field_cur, field_ref, weight=1.0)

    # Integrals should be identical
    assert penalty.item() < 1e-6, f"Expected conservation penalty ≈0, got {penalty.item():.6f}"
    assert torch.isfinite(penalty), "Conservation penalty should be finite"


def test_conservation_penalty_violation():
    """Test that non-conserved field has positive penalty."""
    B, N, C = 4, 1024, 1
    field_ref = torch.ones(B, N, C)
    field_cur = torch.ones(B, N, C) * 1.5  # 50% increase

    penalty = conservation_penalty(field_cur, field_ref, weight=1.0)

    # Integral increased by 50%, so gap should be ≈0.5
    assert penalty.item() > 0.4, f"Expected conservation penalty > 0.4, got {penalty.item():.6f}"
    assert penalty.item() < 0.6, f"Expected conservation penalty < 0.6, got {penalty.item():.6f}"
    assert torch.isfinite(penalty), "Conservation penalty should be finite"


def test_positivity_penalty_nonnegative():
    """Test that non-negative field has zero penalty."""
    field = torch.tensor([1.0, 0.0, 2.0, 0.5])
    penalty = positivity_penalty(field, weight=1.0)

    # No negative values
    assert penalty.item() == 0.0, f"Expected positivity penalty = 0, got {penalty.item():.6f}"
    assert torch.isfinite(penalty), "Positivity penalty should be finite"


def test_positivity_penalty_negative():
    """Test that negative field has positive penalty."""
    field = torch.tensor([1.0, -0.5, 2.0, -1.0])
    penalty = positivity_penalty(field, weight=1.0)

    # Average of negative magnitudes: (0.5 + 1.0) / 4 = 0.375
    expected = 0.375
    assert torch.isclose(penalty, torch.tensor(expected), atol=1e-4), \
        f"Expected positivity penalty ≈{expected}, got {penalty.item():.6f}"
    assert torch.isfinite(penalty), "Positivity penalty should be finite"


def test_boundary_condition_penalty_satisfied():
    """Test that field satisfying BC has low penalty."""
    B, H, W, C = 2, 16, 16, 1
    bc_value = 0.0

    # Create field with zero boundaries
    field = torch.randn(B, H, W, C)
    field[:, 0, :, :] = bc_value  # Top
    field[:, -1, :, :] = bc_value  # Bottom
    field[:, :, 0, :] = bc_value  # Left
    field[:, :, -1, :] = bc_value  # Right

    penalty = boundary_condition_penalty_grid(field, bc_value, (H, W), boundary="all", weight=1.0)

    # BC satisfied, penalty should be ≈0
    assert penalty.item() < 1e-6, f"Expected BC penalty ≈0, got {penalty.item():.6f}"
    assert torch.isfinite(penalty), "BC penalty should be finite"


def test_boundary_condition_penalty_violated():
    """Test that field violating BC has positive penalty."""
    B, H, W, C = 2, 16, 16, 1
    bc_value = 0.0

    # Create field with non-zero boundaries
    field = torch.ones(B, H, W, C)

    penalty = boundary_condition_penalty_grid(field, bc_value, (H, W), boundary="all", weight=1.0)

    # BC violated (boundary values = 1, target = 0), penalty should be ≈1
    assert penalty.item() > 0.8, f"Expected BC penalty > 0.8, got {penalty.item():.6f}"
    assert torch.isfinite(penalty), "BC penalty should be finite"


def test_smoothness_penalty_constant():
    """Test that constant field has zero smoothness penalty."""
    B, H, W, C = 2, 16, 16, 1
    field = torch.ones(B, H, W, C) * 2.0

    penalty = smoothness_penalty(field, (H, W), dx=1.0, dy=1.0, weight=1.0)

    # No gradients → zero total variation
    assert penalty.item() == 0.0, f"Expected smoothness penalty = 0, got {penalty.item():.6f}"
    assert torch.isfinite(penalty), "Smoothness penalty should be finite"


def test_smoothness_penalty_rough():
    """Test that rough field has positive smoothness penalty."""
    B, H, W, C = 2, 16, 16, 1

    # Create checkerboard pattern (maximum roughness)
    field = torch.zeros(B, H, W, C)
    field[:, ::2, ::2, :] = 1.0
    field[:, 1::2, 1::2, :] = 1.0

    penalty = smoothness_penalty(field, (H, W), dx=1.0, dy=1.0, weight=1.0)

    # High gradients → positive penalty
    assert penalty.item() > 0.1, f"Expected smoothness penalty > 0.1, got {penalty.item():.6f}"
    assert torch.isfinite(penalty), "Smoothness penalty should be finite"


def test_physics_losses_batched():
    """Test that all physics losses handle batched inputs correctly."""
    B, H, W = 4, 32, 32

    # Divergence
    velocity = torch.randn(B, H, W, 2)
    div_penalty = divergence_penalty_2d(velocity, (H, W), weight=1.0)
    assert div_penalty.ndim == 0, "Divergence penalty should be scalar"
    assert torch.isfinite(div_penalty), "Divergence penalty should be finite"

    # Conservation
    field_ref = torch.randn(B, H*W, 1)
    field_cur = field_ref + torch.randn_like(field_ref) * 0.1
    cons_penalty = conservation_penalty(field_cur, field_ref, weight=1.0)
    assert cons_penalty.ndim == 0, "Conservation penalty should be scalar"
    assert torch.isfinite(cons_penalty), "Conservation penalty should be finite"

    # Boundary condition
    field_grid = torch.randn(B, H, W, 1)
    bc_penalty = boundary_condition_penalty_grid(field_grid, 0.0, (H, W), weight=1.0)
    assert bc_penalty.ndim == 0, "BC penalty should be scalar"
    assert torch.isfinite(bc_penalty), "BC penalty should be finite"

    # Positivity
    pos_penalty = positivity_penalty(field_grid, weight=1.0)
    assert pos_penalty.ndim == 0, "Positivity penalty should be scalar"
    assert torch.isfinite(pos_penalty), "Positivity penalty should be finite"

    # Smoothness
    smooth_penalty = smoothness_penalty(field_grid, (H, W), weight=1.0)
    assert smooth_penalty.ndim == 0, "Smoothness penalty should be scalar"
    assert torch.isfinite(smooth_penalty), "Smoothness penalty should be finite"


def test_physics_losses_device_consistency():
    """Test that physics losses work on CPU and preserve device."""
    B, H, W = 2, 16, 16
    device = torch.device("cpu")

    # Test on CPU
    velocity = torch.randn(B, H, W, 2, device=device)
    div_penalty = divergence_penalty_2d(velocity, (H, W), weight=1.0)
    assert div_penalty.device == device, f"Expected device {device}, got {div_penalty.device}"

    # Skip CUDA test if not available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        velocity = torch.randn(B, H, W, 2, device=device)
        div_penalty = divergence_penalty_2d(velocity, (H, W), weight=1.0)
        assert div_penalty.device == device, f"Expected device {device}, got {div_penalty.device}"


def test_conservation_penalty_zero_reference():
    """Test conservation penalty handles near-zero reference values."""
    B, N, C = 4, 1024, 1

    # Near-zero reference (edge case)
    field_ref = torch.ones(B, N, C) * 1e-10
    field_cur = torch.ones(B, N, C) * 1e-9

    penalty = conservation_penalty(field_cur, field_ref, weight=1.0)

    # Should not produce NaN/Inf (due to 1e-8 epsilon)
    assert torch.isfinite(penalty), "Conservation penalty should handle near-zero reference"
