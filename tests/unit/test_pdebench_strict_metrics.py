from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch

from ups.core.latent_state import LatentState
from ups.eval.pdebench_runner import (
    DecodedMetricChunk,
    aggregate_stratified_decoded_metrics,
    conditioning_perturbation_metrics,
    evaluate_decoded_operator,
)


def _tensor(value: float, *, elements: int = 1) -> torch.Tensor:
    return torch.full((1, elements, 1), value, dtype=torch.float32)


class _IdentityEncoder(torch.nn.Module):
    def forward(self, fields, coords, *, meta=None, params=None, bc=None, geom=None):
        return fields["u"]


class _RecordingIdentityOperator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conditions: list[dict[str, torch.Tensor]] = []

    def forward(self, state: LatentState, dt):
        self.conditions.append(
            {key: value.detach().cpu().clone() for key, value in state.cond.items()}
        )
        return LatentState(
            z=state.z,
            t=dt if state.t is None else state.t + dt,
            cond=state.cond,
        )


class _IdentityDecoder(torch.nn.Module):
    def forward(self, points, latent_tokens, *, conditioning=None):
        return {"u": latent_tokens}


def _fixed_schema_data_config(root, *, tasks):
    return {
        "task": list(tasks),
        "split": "train",
        "root": str(root),
        "patch_size": 1,
        "field_name": "u",
        "task_param_keys": {
            "advection1d": ["beta"],
            "burgers1d": ["nu"],
            "darcy2d": ["beta"],
        },
        "conditioning_schema": {
            "task_vocab": ["advection1d", "burgers1d", "darcy2d"],
            "param_vocab": ["beta", "nu"],
        },
    }


def test_strict_metrics_use_equal_task_macro_and_keep_micro_evidence() -> None:
    chunks = [
        DecodedMetricChunk(
            task="advection1d",
            prediction=_tensor(0.0, elements=100),
            target=_tensor(1.0, elements=100),
            regime=0.1,
            horizon=1,
        ),
        DecodedMetricChunk(
            task="burgers1d",
            prediction=_tensor(1.0),
            target=_tensor(1.0),
            regime=0.01,
            horizon=1,
        ),
        DecodedMetricChunk(
            task="darcy2d",
            prediction=_tensor(1.0),
            target=_tensor(1.0),
            regime=1.0,
            horizon=None,
        ),
    ]

    metrics = aggregate_stratified_decoded_metrics(chunks)

    assert metrics["task_advection1d_decoded_rollout_nrmse"] == pytest.approx(1.0)
    assert metrics["task_burgers1d_decoded_rollout_nrmse"] == pytest.approx(0.0)
    assert metrics["task_darcy2d_decoded_solution_nrmse"] == pytest.approx(0.0)
    # Preserve the historical steady-task alias without confusing its semantics.
    assert metrics["task_darcy2d_decoded_rollout_nrmse"] == pytest.approx(0.0)
    assert metrics["macro_primary_nrmse"] == pytest.approx(1.0 / 3.0)
    assert metrics["decoded_rollout_nrmse"] == pytest.approx(1.0 / 3.0)
    assert metrics["micro_decoded_rollout_nrmse"] > 0.99


def test_strict_metrics_reject_synthetic_darcy_horizon() -> None:
    with pytest.raises(ValueError, match="steady task darcy2d must use horizon=None"):
        aggregate_stratified_decoded_metrics(
            [
                DecodedMetricChunk(
                    task="darcy2d",
                    prediction=_tensor(0.0),
                    target=_tensor(1.0),
                    regime=10.0,
                    horizon=1,
                )
            ]
        )


def test_strict_metrics_report_global_scale_regime_spread() -> None:
    chunks = [
        DecodedMetricChunk(
            task="darcy2d",
            prediction=_tensor(1.0),
            target=_tensor(1.0),
            regime=1.0,
        ),
        DecodedMetricChunk(
            task="darcy2d",
            prediction=_tensor(3.0),
            target=_tensor(1.0),
            regime=100.0,
        ),
    ]

    metrics = aggregate_stratified_decoded_metrics(chunks)

    primary = 2.0**0.5
    assert metrics["task_darcy2d_decoded_solution_nrmse"] == pytest.approx(primary)
    assert metrics["task_darcy2d_regime_100_decoded_solution_global_scale_nrmse"] == pytest.approx(
        2.0
    )
    assert metrics["task_darcy2d_regime_100_spread_ratio_to_task_primary"] == pytest.approx(
        2.0 / primary
    )
    assert metrics["task_darcy2d_maximum_corrected_regime_spread_ratio"] == pytest.approx(
        2.0 / primary
    )
    assert metrics["task_darcy2d_regime_100_decoded_solution_element_count"] == 1.0


def test_strict_metrics_report_per_horizon_and_temporal_macro() -> None:
    chunks = [
        DecodedMetricChunk("advection1d", _tensor(0.0), _tensor(1.0), 0.1, horizon=1),
        DecodedMetricChunk("advection1d", _tensor(0.0), _tensor(1.0), 0.1, horizon=2),
        DecodedMetricChunk("burgers1d", _tensor(1.0), _tensor(1.0), 0.01, horizon=1),
        DecodedMetricChunk("burgers1d", _tensor(1.0), _tensor(1.0), 0.01, horizon=2),
    ]

    metrics = aggregate_stratified_decoded_metrics(chunks)

    assert metrics["task_advection1d_decoded_h2_nrmse"] == pytest.approx(1.0)
    assert metrics["task_burgers1d_decoded_h2_nrmse"] == pytest.approx(0.0)
    assert metrics["temporal_macro_decoded_h2_nrmse"] == pytest.approx(0.5)


def test_conditioning_perturbation_metrics_are_inference_agnostic() -> None:
    reference = [_tensor(1.0), _tensor(1.0)]
    shuffled = [_tensor(0.0), _tensor(0.0)]
    targets = [_tensor(1.0), _tensor(1.0)]

    metrics = conditioning_perturbation_metrics(
        reference,
        shuffled,
        targets=targets,
        prefix="shuffled_parameter",
    )

    assert metrics["shuffled_parameter_relative_prediction_rms_delta"] == pytest.approx(1.0)
    assert metrics["shuffled_parameter_reference_nrmse"] == pytest.approx(0.0)
    assert metrics["shuffled_parameter_nrmse"] == pytest.approx(1.0)
    assert metrics["shuffled_parameter_nrmse_degradation_ratio"] == float("inf")


def test_decoded_evaluator_invokes_strict_metrics_for_temporal_and_steady(tmp_path) -> None:
    advection = np.ones((1, 2, 4), dtype=np.float32)
    with h5py.File(tmp_path / "advection1d_train.h5", "w") as handle:
        handle.create_dataset("data", data=advection)
        handle.create_dataset("beta", data=np.asarray([[0.1]], dtype=np.float32))

    coefficient = np.ones((1, 1, 2, 2, 1), dtype=np.float32)
    solution = np.full((1, 1, 2, 2, 1), 2.0, dtype=np.float32)
    with h5py.File(tmp_path / "darcy2d_train.h5", "w") as handle:
        handle.create_dataset("data", data=coefficient)
        handle.create_dataset("targets", data=solution)
        handle.create_dataset("beta", data=np.asarray([[100.0]], dtype=np.float32))

    cfg = {
        "training": {"auto_conditioning": True, "dt": 0.1},
        "evaluation": {"strict_stratified_metrics": True},
        "data": _fixed_schema_data_config(tmp_path, tasks=("advection1d", "darcy2d")),
    }

    report = evaluate_decoded_operator(
        cfg,
        _IdentityEncoder(),
        _RecordingIdentityOperator(),
        _IdentityDecoder(),
        rollout_steps=1,
    )

    assert report.extra["strict_stratified_metrics"] is True
    assert report.metrics["task_advection1d_decoded_rollout_nrmse"] == pytest.approx(0.0)
    assert report.metrics["task_advection1d_decoded_h1_nrmse"] == pytest.approx(0.0)
    assert report.metrics["task_darcy2d_decoded_solution_nrmse"] == pytest.approx(0.5)
    assert "task_darcy2d_decoded_h1_nrmse" not in report.metrics
    assert report.metrics["macro_primary_nrmse"] == pytest.approx(0.25)
    assert report.metrics["decoded_rollout_nrmse"] == pytest.approx(0.25)
    assert report.metrics[
        "task_darcy2d_regime_100_decoded_solution_global_scale_nrmse"
    ] == pytest.approx(0.5)


def test_parameter_index_shift_changes_params_but_preserves_fixed_task_vocab(tmp_path) -> None:
    data = np.ones((2, 2, 4), dtype=np.float32)
    with h5py.File(tmp_path / "advection1d_train.h5", "w") as handle:
        handle.create_dataset("data", data=data)
        handle.create_dataset("beta", data=np.asarray([[1.0], [2.0]], dtype=np.float32))

    cfg = {
        "training": {"auto_conditioning": True, "dt": 0.1},
        "evaluation": {
            "strict_stratified_metrics": True,
            "conditioning_parameter_index_shift": 1,
        },
        "data": _fixed_schema_data_config(tmp_path, tasks=("advection1d",)),
    }
    operator = _RecordingIdentityOperator()

    report = evaluate_decoded_operator(
        cfg,
        _IdentityEncoder(),
        operator,
        _IdentityDecoder(),
        rollout_steps=1,
    )

    assert report.extra["conditioning_parameter_index_shift"] == 1
    assert [condition["param_beta"].item() for condition in operator.conditions] == [2.0, 1.0]
    assert all(
        condition["task_id"].tolist() == [[1.0, 0.0, 0.0]] for condition in operator.conditions
    )
    assert all(
        condition["param_presence"].tolist() == [[1.0, 0.0]] for condition in operator.conditions
    )
    assert "task_advection1d_regime_1_decoded_rollout_nrmse" in report.metrics
    assert "task_advection1d_regime_2_decoded_rollout_nrmse" in report.metrics
