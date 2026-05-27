from __future__ import annotations

import json

import torch

from scripts.fit_decoded_residual_gate import (
    gate_config_override,
    least_squares_residual_alpha,
    predict_gate_alpha,
    train_logistic_gate,
)


def test_least_squares_residual_alpha_projects_target_onto_prediction_residual():
    persistence = torch.tensor([[[1.0, 1.0]]])
    prediction = torch.tensor([[[3.0, 5.0]]])

    assert (
        least_squares_residual_alpha(
            prediction=prediction,
            persistence=persistence,
            target=torch.tensor([[[2.0, 3.0]]]),
        )
        == 0.5
    )
    assert (
        least_squares_residual_alpha(
            prediction=prediction,
            persistence=persistence,
            target=torch.tensor([[[10.0, 10.0]]]),
        )
        == 1.0
    )
    assert (
        least_squares_residual_alpha(
            prediction=prediction,
            persistence=persistence,
            target=torch.tensor([[[-1.0, -1.0]]]),
        )
        == 0.0
    )


def test_train_logistic_gate_exports_eval_compatible_feature_weights():
    rows = []
    for x in (-2.0, -1.0, 0.0, 1.0, 2.0):
        rows.append(
            {
                "target_alpha": float(torch.sigmoid(torch.tensor(-0.25 + 2.0 * x))),
                "features": {"residual_rms": x},
            }
        )

    fit = train_logistic_gate(
        rows,
        feature_names=("residual_rms",),
        epochs=800,
        learning_rate=0.1,
        seed=7,
    )

    assert fit["config"]["base_alpha"] == 0.5
    assert fit["config"]["feature_weights"]["residual_rms"] > 0.0
    assert predict_gate_alpha({"residual_rms": -2.0}, fit["config"]) < 0.1
    assert predict_gate_alpha({"residual_rms": 2.0}, fit["config"]) > 0.9
    assert fit["train_mse"] < 0.01


def test_gate_config_override_round_trips_as_run_light_experiment_override():
    config = {"base_alpha": 0.5, "bias": -1.0, "feature_weights": {"horizon_norm": 2.0}}

    override = gate_config_override(config)
    key, raw_json = override.split("=", 1)

    assert key == "evaluation.decoded_persistence_residual_gate"
    assert json.loads(raw_json) == config
