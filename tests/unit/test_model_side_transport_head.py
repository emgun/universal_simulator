from __future__ import annotations

import torch

from ups.models.transport_head import ModelSideTransportHead, model_side_transport_head_config


def test_model_side_transport_head_is_default_off():
    cfg = model_side_transport_head_config(None)
    head = ModelSideTransportHead(cfg)

    assert not cfg.enabled
    assert not head.applies(task_name="advection1d", task_family="transport")


def test_model_side_transport_head_predicts_beta_horizon_bias_shift():
    cfg = model_side_transport_head_config(
        {
            "enabled": True,
            "tasks": ["advection1d"],
            "features": ["param:beta", "horizon_norm", "bias"],
            "init": {"param:beta": 10.0, "horizon_norm": 2.0, "bias": -0.5},
            "clamp": {"min_shift": -4.0, "max_shift": 4.0},
        }
    )
    head = ModelSideTransportHead(cfg)

    shift, info = head.predict_shift(
        params={"beta": torch.tensor([0.1])},
        horizon=2,
        rollout_steps=4,
    )

    assert info == {"missing_params": [], "skipped": False}
    assert shift is not None
    assert torch.isclose(shift, torch.tensor(1.5))
    assert head.trainable_parameter_count == 3
    assert head.resolved_config()["coefficients"]["param:beta"] == 10.0


def test_model_side_transport_head_skips_when_required_beta_is_missing():
    cfg = model_side_transport_head_config(
        {
            "enabled": True,
            "tasks": ["advection1d"],
            "required_params": ["beta"],
            "features": ["param:beta", "bias"],
            "init": {"param:beta": 10.0, "bias": 1.0},
            "missing_param_policy": "skip",
        }
    )
    head = ModelSideTransportHead(cfg)

    shift, info = head.predict_shift(params={}, horizon=1, rollout_steps=4)

    assert shift is None
    assert info == {"missing_params": ["beta"], "skipped": True}
