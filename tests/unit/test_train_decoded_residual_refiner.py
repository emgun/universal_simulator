from __future__ import annotations

import torch

from scripts.train_decoded_residual_refiner import (
    DecodedResidualRefiner,
    build_refiner_features,
    evaluate_refiner_tensors,
    subsample_refiner_tensors,
    train_refiner_from_tensors,
)


def test_build_refiner_features_includes_prediction_persistence_residual_and_coords():
    prediction = torch.tensor([[[2.0], [4.0]]])
    persistence = torch.tensor([[[1.0], [3.0]]])
    coords = torch.tensor([[[0.25, 0.0], [0.75, 0.0]]])

    features = build_refiner_features(
        prediction=prediction,
        persistence=persistence,
        coords=coords,
        horizon=2,
        rollout_steps=4,
    )

    assert features.shape == (2, 7)
    assert torch.allclose(features[:, 0], torch.tensor([2.0, 4.0]))
    assert torch.allclose(features[:, 1], torch.tensor([1.0, 3.0]))
    assert torch.allclose(features[:, 2], torch.tensor([1.0, 1.0]))
    assert torch.allclose(features[:, 3:5], coords.squeeze(0))
    assert torch.allclose(features[:, 5], torch.full((2,), 0.5))


def test_build_refiner_features_appends_context_features_to_each_node():
    prediction = torch.tensor([[[2.0], [4.0]]])
    persistence = torch.tensor([[[1.0], [3.0]]])
    coords = torch.tensor([[[0.25, 0.0], [0.75, 0.0]]])
    context = torch.tensor([1.0, 0.0, 0.0])

    features = build_refiner_features(
        prediction=prediction,
        persistence=persistence,
        coords=coords,
        horizon=1,
        rollout_steps=2,
        context_features=context,
    )

    assert features.shape == (2, 10)
    assert torch.allclose(features[:, -3:], context.expand(2, -1))


def test_train_refiner_from_tensors_learns_non_alpha_residual_correction():
    persistence = torch.linspace(-1.0, 1.0, 32).view(-1, 1)
    prediction = persistence + 0.25
    coords = torch.stack((torch.linspace(0.0, 1.0, 32), torch.zeros(32)), dim=1)
    features = torch.cat(
        (
            prediction,
            persistence,
            prediction - persistence,
            coords,
            torch.ones(32, 1),
            torch.log1p(torch.ones(32, 1)),
        ),
        dim=1,
    )
    target_delta = 0.4 * persistence.square() - 0.1

    refiner, fit = train_refiner_from_tensors(
        features,
        target_delta,
        hidden_dim=16,
        epochs=500,
        learning_rate=0.03,
        seed=3,
    )

    metrics = evaluate_refiner_tensors(refiner, features, persistence, persistence + target_delta)

    assert isinstance(refiner, DecodedResidualRefiner)
    assert fit["train_rows"] == 32
    assert metrics["nrmse"] < 0.15


def test_subsample_refiner_tensors_is_seeded_and_keeps_aligned_rows():
    tensors = {
        "features": torch.arange(30, dtype=torch.float32).view(10, 3),
        "persistence": torch.arange(10, dtype=torch.float32).view(10, 1),
        "target": torch.arange(10, dtype=torch.float32).view(10, 1) + 1.0,
        "target_delta": torch.ones(10, 1),
    }

    first = subsample_refiner_tensors(tensors, max_rows=4, seed=11)
    second = subsample_refiner_tensors(tensors, max_rows=4, seed=11)

    assert first["features"].shape[0] == 4
    assert torch.equal(first["features"], second["features"])
    assert torch.equal(first["target"] - first["persistence"], first["target_delta"])
