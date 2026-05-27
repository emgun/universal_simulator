from __future__ import annotations

import torch

from scripts.train_decoded_spatial_refiner import (
    SpatialDecodedResidualRefiner,
    build_spatial_refiner_input,
    evaluate_spatial_refiner_tensors,
    flat_to_grid,
    grid_to_flat,
    subsample_spatial_tensors,
    train_spatial_refiner_from_tensors,
)


def test_flat_to_grid_round_trips_scalar_fields():
    flat = torch.arange(12, dtype=torch.float32).view(1, 6, 2)

    grid = flat_to_grid(flat, grid_shape=(2, 3))
    restored = grid_to_flat(grid)

    assert grid.shape == (1, 2, 2, 3)
    assert torch.equal(restored, flat)


def test_build_spatial_refiner_input_adds_coords_horizon_and_context_channels():
    prediction = torch.ones(1, 4, 1)
    persistence = torch.zeros(1, 4, 1)
    coords = torch.tensor([[[0.0, 0.0], [0.5, 0.0], [0.0, 1.0], [0.5, 1.0]]])
    context = torch.tensor([1.0, 0.0, 0.0])

    features = build_spatial_refiner_input(
        prediction=prediction,
        persistence=persistence,
        coords=coords,
        grid_shape=(2, 2),
        horizon=2,
        rollout_steps=4,
        context_features=context,
    )

    assert features.shape == (1, 9, 2, 2)
    assert torch.allclose(features[:, 0], torch.ones(1, 2, 2))
    assert torch.allclose(features[:, 1], torch.zeros(1, 2, 2))
    assert torch.allclose(features[:, 2], torch.ones(1, 2, 2))
    assert torch.allclose(features[:, 5], torch.full((1, 2, 2), 0.5))
    assert torch.allclose(features[:, -3], torch.ones(1, 2, 2))


def test_train_spatial_refiner_from_tensors_learns_local_grid_correction():
    persistence = torch.linspace(-1.0, 1.0, 64).view(1, 1, 8, 8)
    prediction = persistence + 0.2
    coords_x = torch.linspace(0.0, 1.0, 8).view(1, 1, 1, 8).expand(1, 1, 8, 8)
    coords_y = torch.linspace(0.0, 1.0, 8).view(1, 1, 8, 1).expand(1, 1, 8, 8)
    horizon = torch.ones(1, 1, 8, 8)
    features = torch.cat(
        (
            prediction,
            persistence,
            prediction - persistence,
            coords_x,
            coords_y,
            horizon,
        ),
        dim=1,
    )
    target_delta = 0.25 * persistence + 0.1 * torch.roll(persistence, shifts=1, dims=-1)
    target = persistence + target_delta

    refiner, fit = train_spatial_refiner_from_tensors(
        features,
        target_delta,
        hidden_channels=16,
        epochs=250,
        learning_rate=0.01,
        seed=5,
    )
    metrics = evaluate_spatial_refiner_tensors(refiner, features, persistence, target)

    assert isinstance(refiner, SpatialDecodedResidualRefiner)
    assert fit["train_frames"] == 1
    assert metrics["nrmse"] < 0.1


def test_subsample_spatial_tensors_balances_shape_groups():
    tensors = {
        "features": [torch.zeros(10, 6, 2, 2), torch.ones(10, 6, 4, 4)],
        "persistence": [torch.zeros(10, 1, 2, 2), torch.ones(10, 1, 4, 4)],
        "target": [torch.zeros(10, 1, 2, 2), torch.ones(10, 1, 4, 4)],
        "target_delta": [torch.zeros(10, 1, 2, 2), torch.ones(10, 1, 4, 4)],
    }

    sampled = subsample_spatial_tensors(tensors, max_frames=5, seed=7)

    assert [group.shape[0] for group in sampled["features"]] == [3, 2]
    assert sum(group.shape[0] for group in sampled["features"]) == 5
    assert sampled["features"][0].shape[2:] == (2, 2)
    assert sampled["features"][1].shape[2:] == (4, 4)
