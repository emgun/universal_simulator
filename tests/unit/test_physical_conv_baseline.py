from __future__ import annotations

import torch

from scripts.run_physical_conv_baseline import (
    PhysicalConvBaseline,
    PhysicalFourierBaseline,
    PhysicalResidualUNetBaseline,
    build_physical_model,
    field_step_to_grid,
    grid_to_flat,
    group_key,
    train_group_model,
)


def test_field_grid_round_trip_supports_1d_pdebench_steps():
    field = torch.arange(4, dtype=torch.float32).view(4, 1)

    grid = field_step_to_grid(field, grid_shape=(1, 4))
    restored = grid_to_flat(grid)

    assert grid.shape == (1, 1, 1, 4)
    assert restored.shape == (1, 4, 1)
    assert torch.equal(restored.squeeze(0), field)


def test_physical_conv_baseline_preserves_grid_shape():
    model = PhysicalConvBaseline(channels=1, hidden_channels=4)
    current = torch.randn(2, 1, 3, 5)

    pred = model(current)

    assert pred.shape == current.shape


def test_physical_conv_baseline_initializes_to_persistence():
    model = PhysicalConvBaseline(channels=1, hidden_channels=4)
    current = torch.randn(2, 1, 3, 5)

    pred = model(current)

    assert torch.allclose(pred, current)


def test_physical_unet_baseline_preserves_shape_and_initializes_to_persistence():
    model = PhysicalResidualUNetBaseline(channels=1, hidden_channels=4)
    current = torch.randn(2, 1, 5, 17)

    pred = model(current)

    assert pred.shape == current.shape
    assert torch.allclose(pred, current)


def test_physical_fourier_baseline_preserves_shape_and_initializes_to_persistence():
    model = PhysicalFourierBaseline(channels=1, hidden_channels=4, modes=4)
    current = torch.randn(2, 1, 5, 17)

    pred = model(current)

    assert pred.shape == current.shape
    assert torch.allclose(pred, current)


def test_physical_fourier_baseline_supports_1d_grids():
    model = PhysicalFourierBaseline(channels=1, hidden_channels=4, modes=4)
    current = torch.randn(2, 1, 1, 8)

    pred = model(current)

    assert pred.shape == current.shape
    assert torch.allclose(pred, current)


def test_build_physical_model_selects_architecture():
    assert isinstance(
        build_physical_model("conv", channels=1, hidden_channels=4),
        PhysicalConvBaseline,
    )
    assert isinstance(
        build_physical_model("unet", channels=1, hidden_channels=4),
        PhysicalResidualUNetBaseline,
    )
    assert isinstance(
        build_physical_model("fourier", channels=1, hidden_channels=4),
        PhysicalFourierBaseline,
    )


def test_group_key_keeps_same_shape_tasks_separate():
    assert group_key("burgers1d", (1, 1024), 1) != group_key("advection1d", (1, 1024), 1)


def test_train_group_model_learns_simple_residual_offset():
    generator = torch.Generator().manual_seed(3)
    currents = torch.randn(12, 1, 4, 4, generator=generator)
    targets = currents + 0.25

    model, fit = train_group_model(
        currents,
        targets,
        hidden_channels=8,
        epochs=80,
        learning_rate=0.03,
        batch_size=4,
        seed=11,
        device="cpu",
    )

    with torch.no_grad():
        mse = torch.mean((model(currents) - targets) ** 2).item()
    assert fit["train_frames"] == 12
    assert mse < 0.01
