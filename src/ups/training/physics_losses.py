"""Physics-informed loss terms for conservation laws and boundary conditions.

This module provides physics-based penalties that can be added to training losses:
1. Conservation penalties (mass, momentum, energy)
2. Divergence penalties (for incompressible flows)
3. Boundary condition penalties
4. Positivity constraints
5. Smoothness penalties
"""

from __future__ import annotations

import torch
from torch import Tensor
from typing import Tuple, Optional


def divergence_penalty_2d(
    velocity_field: Tensor,
    grid_shape: Tuple[int, int],
    dx: float = 1.0,
    dy: float = 1.0,
    weight: float = 1.0,
) -> Tensor:
    """Penalize non-zero divergence for incompressible 2D flows.

    Computes ∇·u via central finite differences and penalizes deviation from zero.

    Args:
        velocity_field: (B, H, W, 2) velocity components [u, v]
        grid_shape: (H, W) grid dimensions
        dx: Grid spacing in x direction
        dy: Grid spacing in y direction
        weight: Loss weight

    Returns:
        Divergence penalty loss (scalar)
    """
    B, H, W, C = velocity_field.shape
    assert C == 2, f"Expected 2D velocity field (u, v), got {C} components"
    assert grid_shape == (H, W), f"Grid shape mismatch: expected {grid_shape}, got ({H}, {W})"

    u = velocity_field[..., 0]  # (B, H, W) - x-velocity
    v = velocity_field[..., 1]  # (B, H, W) - y-velocity

    # Compute ∂u/∂x via central differences (interior points)
    du_dx = torch.zeros_like(u)
    du_dx[:, :, 1:-1] = (u[:, :, 2:] - u[:, :, :-2]) / (2 * dx)

    # Compute ∂v/∂y via central differences
    dv_dy = torch.zeros_like(v)
    dv_dy[:, 1:-1, :] = (v[:, 2:, :] - v[:, :-2, :]) / (2 * dy)

    # Divergence: ∇·u = ∂u/∂x + ∂v/∂y
    divergence = du_dx + dv_dy

    # Penalty: mean absolute divergence (L1 norm)
    penalty = divergence.abs().mean()

    return weight * penalty


def conservation_penalty(
    field_current: Tensor,
    field_reference: Tensor,
    conserved_quantity: str = "mass",
    weight: float = 1.0,
) -> Tensor:
    """Penalize changes in conserved quantities (mass, momentum, energy).

    For conservative PDEs, global integrals should remain constant:
    - Mass: ∫ ρ dx = const
    - Momentum: ∫ ρu dx = const
    - Energy: ∫ ½ρu² dx = const

    Args:
        field_current: Predicted field (B, N, C)
        field_reference: Reference field at t=0 (B, N, C)
        conserved_quantity: "mass", "momentum", or "energy"
        weight: Loss weight

    Returns:
        Conservation penalty (scalar)
    """
    # Compute global integrals (sum over spatial dimension)
    current_integral = field_current.sum(dim=1)  # (B, C)
    ref_integral = field_reference.sum(dim=1)    # (B, C)

    # Compute relative change in integral
    gap = torch.abs(current_integral - ref_integral) / (ref_integral.abs() + 1e-8)

    # Average over batch and channels
    penalty = gap.mean()

    return weight * penalty


def boundary_condition_penalty_grid(
    field: Tensor,
    bc_value: float,
    grid_shape: Tuple[int, int],
    boundary: str = "all",  # "all", "left", "right", "top", "bottom"
    weight: float = 1.0,
) -> Tensor:
    """Penalize violations of Dirichlet boundary conditions on a grid.

    Args:
        field: Predicted field (B, H, W, C)
        bc_value: Boundary condition value (Dirichlet)
        grid_shape: (H, W) grid dimensions
        boundary: Which boundaries to enforce
        weight: Loss weight

    Returns:
        BC violation penalty (scalar)
    """
    B, H, W, C = field.shape
    assert grid_shape == (H, W), f"Grid shape mismatch: expected {grid_shape}, got ({H}, {W})"

    boundary_points = []

    if boundary in ["all", "left"]:
        boundary_points.append(field[:, :, 0, :])  # Left edge
    if boundary in ["all", "right"]:
        boundary_points.append(field[:, :, -1, :])  # Right edge
    if boundary in ["all", "top"]:
        boundary_points.append(field[:, 0, :, :])  # Top edge
    if boundary in ["all", "bottom"]:
        boundary_points.append(field[:, -1, :, :])  # Bottom edge

    if not boundary_points:
        raise ValueError(f"Unknown boundary: {boundary}")

    # Concatenate all boundary points
    all_boundary = torch.cat(boundary_points, dim=1)

    # MSE from BC value
    violation = (all_boundary - bc_value).pow(2).mean()

    return weight * violation


def positivity_penalty(
    field: Tensor,
    weight: float = 1.0,
) -> Tensor:
    """Penalize negative values in physical fields (density, pressure, etc.).

    Uses ReLU to measure magnitude of negativity violations.

    Args:
        field: Predicted field (any shape)
        weight: Loss weight

    Returns:
        Positivity violation penalty (scalar)
    """
    # Clamp negative values to 0, measure violation
    negatives = torch.clamp(field, max=0.0)
    penalty = negatives.abs().mean()

    return weight * penalty


def smoothness_penalty(
    field: Tensor,
    grid_shape: Tuple[int, int],
    dx: float = 1.0,
    dy: float = 1.0,
    weight: float = 1.0,
) -> Tensor:
    """Penalize high spatial gradients (encourage smooth solutions).

    Computes total variation (TV) norm: sum of gradient magnitudes.

    Args:
        field: Predicted field (B, H, W, C)
        grid_shape: (H, W)
        dx: Grid spacing in x
        dy: Grid spacing in y
        weight: Loss weight

    Returns:
        Smoothness penalty (scalar)
    """
    B, H, W, C = field.shape
    assert grid_shape == (H, W), f"Grid shape mismatch: expected {grid_shape}, got ({H}, {W})"

    # Compute spatial gradients via finite differences
    grad_x = (field[:, :, 1:, :] - field[:, :, :-1, :]) / dx  # (B, H, W-1, C)
    grad_y = (field[:, 1:, :, :] - field[:, :-1, :, :]) / dy  # (B, H-1, W, C)

    # Total variation: sum of absolute gradients
    tv_x = grad_x.abs().sum()
    tv_y = grad_y.abs().sum()

    # Normalize by grid size
    roughness = (tv_x + tv_y) / (H * W)

    return weight * roughness
