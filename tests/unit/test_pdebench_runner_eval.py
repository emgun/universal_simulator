from __future__ import annotations

import sys
from pathlib import Path

import h5py
import pytest
import torch
import yaml

from scripts import benchmark as benchmark_script
from scripts import evaluate as evaluate_script
from scripts import train as train_script
from scripts import train_baselines as train_baselines_script
from ups.core.latent_state import LatentState
from ups.eval import pdebench_runner
from ups.eval.pdebench_runner import evaluate_decoded_operator, evaluate_latent_operator
from ups.eval.persistence_baselines import evaluate_persistence_decoded

TRAINING_LOCK = Path(
    "docs/data/releases/strat-v1/universal/"
    "9d43d283f04f5b8d17cf6126ad189075c53307e715d7d4f61af440c2fed155c1/"
    "training.lock.json"
)


def _write_minimal_hdf5(tmp_path) -> None:
    data = torch.randn(2, 3, 4, dtype=torch.float32)
    file_path = tmp_path / "burgers1d_train.h5"
    with h5py.File(file_path, "w") as handle:
        handle.create_dataset("data", data=data.numpy())


def test_evaluate_latent_operator_runs(tmp_path):
    _write_minimal_hdf5(tmp_path)
    cfg = {
        "training": {"batch_size": 2, "dt": 0.1},
        "latent": {"dim": 8, "tokens": 4},
        "data": {
            "task": "burgers1d",
            "split": "train",
            "root": str(tmp_path),
            "patch_size": 1,
        },
    }

    operator = train_script.make_operator(cfg)
    report = evaluate_latent_operator(cfg, operator)

    assert "mse" in report.metrics
    assert report.metrics["mse"] >= 0.0


def test_evaluate_make_operator_fails_closed_on_lock_identity_mismatch(tmp_path):
    cfg = {
        "training": {"auto_conditioning": True},
        "latent": {"dim": 8, "tokens": 4},
        "data": {
            "task": "burgers1d",
            "split": "val",
            "root": str(tmp_path),
            "data_lock_path": str(TRAINING_LOCK),
            "data_lock_sha256": "f" * 64,
        },
    }

    with pytest.raises(ValueError, match="data_lock_sha256 does not match"):
        evaluate_script.make_operator(cfg)


def test_evaluate_grid_spec_fails_closed_on_unbound_normalization(tmp_path):
    cfg = {
        "data": {
            "task": "burgers1d",
            "split": "val",
            "root": str(tmp_path),
            "normalize": True,
        }
    }

    with pytest.raises(ValueError, match="normalization_path"):
        evaluate_script._pdebench_grid_spec(cfg)


def test_evaluate_latent_operator_can_skip_missing_tasks(tmp_path):
    _write_minimal_hdf5(tmp_path)
    cfg = {
        "training": {"batch_size": 1, "dt": 0.1, "num_workers": 0},
        "evaluation": {"skip_missing_tasks": True},
        "latent": {"dim": 8, "tokens": 4},
        "data": {
            "task": ["burgers1d", "darcy2d"],
            "split": "train",
            "root": str(tmp_path),
            "patch_size": 1,
        },
    }

    operator = train_script.make_operator(cfg)
    report = evaluate_latent_operator(cfg, operator)

    assert "mse" in report.metrics
    assert report.metrics["mse"] >= 0.0


def test_make_encoder_can_skip_missing_grid_spec_tasks(tmp_path):
    _write_minimal_hdf5(tmp_path)
    cfg = {
        "training": {"batch_size": 1, "dt": 0.1},
        "evaluation": {"skip_missing_tasks": True},
        "latent": {"dim": 8, "tokens": 4},
        "data": {
            "task": ["burgers1d", "darcy2d"],
            "split": "train",
            "root": str(tmp_path),
            "patch_size": 1,
        },
    }

    encoder = evaluate_script.make_encoder(cfg)

    assert encoder is not None


class _DummyEncoder(torch.nn.Module):
    def forward(self, fields, coords, *, meta=None, params=None, bc=None, geom=None):
        return fields["u"]


class _IdentityOperator(torch.nn.Module):
    def forward(self, state: LatentState, dt):
        return LatentState(z=state.z, t=dt if state.t is None else state.t + dt, cond=state.cond)


class _RecordingIdentityOperator(_IdentityOperator):
    def __init__(self) -> None:
        super().__init__()
        self.conditions = []

    def forward(self, state: LatentState, dt):
        self.conditions.append({key: value.detach().clone() for key, value in state.cond.items()})
        return super().forward(state, dt)


class _AddOperator(torch.nn.Module):
    def __init__(self, delta: float) -> None:
        super().__init__()
        self.delta = delta

    def forward(self, state: LatentState, dt):
        return LatentState(
            z=state.z + self.delta, t=dt if state.t is None else state.t + dt, cond=state.cond
        )


class _RollOperator(torch.nn.Module):
    def __init__(self, shift: int) -> None:
        super().__init__()
        self.shift = int(shift)

    def forward(self, state: LatentState, dt):
        return LatentState(
            z=torch.roll(state.z, shifts=self.shift, dims=1),
            t=dt if state.t is None else state.t + dt,
            cond=state.cond,
        )


class _DummyDecoder(torch.nn.Module):
    def forward(self, points, latent_tokens, *, conditioning=None):
        return {"u": latent_tokens}


def test_evaluate_decoded_operator_runs_on_constant_sequence(tmp_path):
    data = torch.ones(1, 3, 4, dtype=torch.float32)
    file_path = tmp_path / "burgers1d_train.h5"
    with h5py.File(file_path, "w") as handle:
        handle.create_dataset("data", data=data.numpy())

    cfg = {
        "training": {"batch_size": 1, "dt": 0.1},
        "data": {
            "task": "burgers1d",
            "split": "train",
            "root": str(tmp_path),
            "patch_size": 1,
            "field_name": "u",
        },
    }

    report = evaluate_decoded_operator(
        cfg,
        _DummyEncoder(),
        _IdentityOperator(),
        _DummyDecoder(),
    )

    assert "decoded_mse" in report.metrics
    assert report.metrics["decoded_mse"] == 0.0
    assert report.metrics["decoded_mae"] == 0.0
    assert report.metrics["decoded_rollout_nrmse"] == 0.0
    assert report.metrics["decoded_step1_nrmse"] == 0.0


def test_evaluate_decoded_operator_fails_closed_on_selection_mismatch(tmp_path):
    cfg = {
        "training": {"batch_size": 1, "dt": 0.1},
        "data": {
            "task": "burgers1d",
            "split": "val",
            "root": str(tmp_path),
            "data_lock_path": str(TRAINING_LOCK),
            "selection_sha256": "f" * 64,
        },
    }

    with pytest.raises(ValueError, match="selection_sha256 does not match"):
        evaluate_decoded_operator(cfg, _DummyEncoder(), _IdentityOperator(), _DummyDecoder())


def test_evaluate_decoded_operator_can_return_rollout_preview(tmp_path):
    data = torch.ones(1, 3, 4, dtype=torch.float32)
    file_path = tmp_path / "burgers1d_train.h5"
    with h5py.File(file_path, "w") as handle:
        handle.create_dataset("data", data=data.numpy())

    cfg = {
        "training": {"batch_size": 1, "dt": 0.1},
        "data": {
            "task": "burgers1d",
            "split": "train",
            "root": str(tmp_path),
            "patch_size": 1,
            "field_name": "u",
        },
    }

    report = evaluate_decoded_operator(
        cfg,
        _DummyEncoder(),
        _IdentityOperator(),
        _DummyDecoder(),
        preview_sample_count=1,
    )

    preview = report.extra["rollout_preview"]
    assert len(preview) == 1
    assert preview[0]["task"] == "burgers1d"
    assert tuple(preview[0]["target"].shape) == (2, 1, 4)
    assert tuple(preview[0]["prediction"].shape) == (2, 1, 4)
    assert preview[0]["time_index"].tolist() == [1.0, 2.0]


def test_flatten_field_step_handles_channel_first_scalar_2d():
    field_step = torch.randn(1, 4, 4)
    grid_shape = (4, 4)

    flattened_train = train_script._flatten_field_step(field_step, grid_shape)
    flattened_eval = pdebench_runner._flatten_field_step(field_step, grid_shape)

    assert flattened_train.shape == (1, 16, 1)
    assert flattened_eval.shape == (1, 16, 1)


def test_evaluate_decoded_operator_reports_horizon_metrics_when_available(tmp_path):
    data = torch.ones(1, 17, 4, dtype=torch.float32)
    file_path = tmp_path / "burgers1d_train.h5"
    with h5py.File(file_path, "w") as handle:
        handle.create_dataset("data", data=data.numpy())

    cfg = {
        "training": {"batch_size": 1, "dt": 0.1},
        "data": {
            "task": "burgers1d",
            "split": "train",
            "root": str(tmp_path),
            "patch_size": 1,
            "field_name": "u",
        },
    }

    report = evaluate_decoded_operator(
        cfg,
        _DummyEncoder(),
        _IdentityOperator(),
        _DummyDecoder(),
        rollout_steps=16,
    )

    assert report.metrics["decoded_h4_nrmse"] == 0.0
    assert report.metrics["decoded_h16_nrmse"] == 0.0


def test_evaluate_decoded_operator_can_blend_against_persistence_residual(tmp_path):
    data = torch.tensor([[[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]], dtype=torch.float32)
    file_path = tmp_path / "burgers1d_train.h5"
    with h5py.File(file_path, "w") as handle:
        handle.create_dataset("data", data=data.numpy())

    cfg = {
        "training": {"batch_size": 1, "dt": 0.1},
        "evaluation": {"decoded_persistence_residual_alpha": 0.5},
        "data": {
            "task": "burgers1d",
            "split": "train",
            "root": str(tmp_path),
            "patch_size": 1,
            "field_name": "u",
        },
    }

    persistence_cfg = {**cfg, "evaluation": {"decoded_persistence_residual_alpha": 0.0}}
    persistence = evaluate_persistence_decoded(persistence_cfg, rollout_steps=1)
    report = evaluate_decoded_operator(
        persistence_cfg,
        _DummyEncoder(),
        _AddOperator(delta=2.0),
        _DummyDecoder(),
        rollout_steps=1,
    )

    assert report.metrics["decoded_rollout_nrmse"] == persistence.metrics["decoded_rollout_nrmse"]
    assert report.extra["decoded_persistence_residual_alpha"] == 0.0


def test_evaluate_decoded_operator_can_apply_task_specific_residual_alpha(tmp_path):
    for task_name in ("burgers1d", "advection1d"):
        data = torch.tensor([[[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]], dtype=torch.float32)
        file_path = tmp_path / f"{task_name}_train.h5"
        with h5py.File(file_path, "w") as handle:
            handle.create_dataset("data", data=data.numpy())

    cfg = {
        "training": {"batch_size": 1, "dt": 0.1},
        "evaluation": {
            "decoded_persistence_residual_alpha": 0.0,
            "decoded_persistence_residual_alpha_by_task": {"advection1d": 1.0},
        },
        "data": {
            "task": ["burgers1d", "advection1d"],
            "split": "train",
            "root": str(tmp_path),
            "patch_size": 1,
            "field_name": "u",
        },
    }

    report = evaluate_decoded_operator(
        cfg,
        _DummyEncoder(),
        _AddOperator(delta=1.0),
        _DummyDecoder(),
        rollout_steps=1,
    )

    assert (
        report.metrics["task_burgers1d_decoded_rollout_nrmse"]
        > report.metrics["task_advection1d_decoded_rollout_nrmse"]
    )
    assert report.metrics["task_advection1d_decoded_rollout_nrmse"] == 0.0
    assert report.extra["decoded_persistence_residual_alpha_by_task"] == {"advection1d": 1.0}


def test_evaluate_decoded_operator_can_skip_missing_tasks(tmp_path):
    data = torch.tensor([[[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]], dtype=torch.float32)
    file_path = tmp_path / "burgers1d_train.h5"
    with h5py.File(file_path, "w") as handle:
        handle.create_dataset("data", data=data.numpy())

    cfg = {
        "training": {"batch_size": 1, "dt": 0.1},
        "evaluation": {"skip_missing_tasks": True},
        "data": {
            "task": ["burgers1d", "darcy2d"],
            "split": "train",
            "root": str(tmp_path),
            "patch_size": 1,
            "field_name": "u",
        },
    }

    report = evaluate_decoded_operator(
        cfg,
        _DummyEncoder(),
        _IdentityOperator(),
        _DummyDecoder(),
        rollout_steps=1,
    )

    assert "task_burgers1d_decoded_rollout_nrmse" in report.metrics
    assert "task_darcy2d_decoded_rollout_nrmse" not in report.metrics
    assert report.extra["skipped_missing_tasks"] == ["darcy2d"]


def test_evaluate_decoded_operator_can_apply_task_horizon_residual_alpha(tmp_path):
    data = torch.tensor(
        [[[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]]], dtype=torch.float32
    )
    file_path = tmp_path / "advection1d_train.h5"
    with h5py.File(file_path, "w") as handle:
        handle.create_dataset("data", data=data.numpy())

    cfg = {
        "training": {"batch_size": 1, "dt": 0.1},
        "evaluation": {
            "decoded_persistence_residual_alpha": 0.0,
            "decoded_persistence_residual_alpha_by_task_horizon": {
                "advection1d": {"1": 0.5, 2: 1.0 / 3.0}
            },
            "report_all_horizon_metrics": True,
        },
        "data": {
            "task": "advection1d",
            "split": "train",
            "root": str(tmp_path),
            "patch_size": 1,
            "field_name": "u",
        },
    }

    baseline_report = evaluate_decoded_operator(
        {
            **cfg,
            "evaluation": {
                "decoded_persistence_residual_alpha": 0.0,
                "report_all_horizon_metrics": True,
            },
        },
        _DummyEncoder(),
        _AddOperator(delta=2.0),
        _DummyDecoder(),
        rollout_steps=2,
    )
    report = evaluate_decoded_operator(
        cfg,
        _DummyEncoder(),
        _AddOperator(delta=2.0),
        _DummyDecoder(),
        rollout_steps=2,
    )

    assert (
        report.metrics["task_advection1d_decoded_h2_nrmse"]
        < baseline_report.metrics["task_advection1d_decoded_h2_nrmse"]
    )
    assert (
        report.metrics["family_transport_decoded_h2_nrmse"]
        < baseline_report.metrics["family_transport_decoded_h2_nrmse"]
    )
    assert report.extra["decoded_persistence_residual_alpha_by_task_horizon"] == {
        "advection1d": {1: 0.5, 2: 1.0 / 3.0}
    }


def test_evaluate_decoded_operator_can_apply_bounded_residual_gate(tmp_path):
    data = torch.tensor(
        [[[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]]], dtype=torch.float32
    )
    file_path = tmp_path / "advection1d_train.h5"
    with h5py.File(file_path, "w") as handle:
        handle.create_dataset("data", data=data.numpy())

    cfg = {
        "training": {"batch_size": 1, "dt": 0.1},
        "evaluation": {
            "decoded_persistence_residual_alpha": 0.0,
            "decoded_persistence_residual_gate": {
                "base_alpha": 0.5,
                "min_alpha": 0.1,
                "max_alpha": 0.9,
                "task_bias": {"advection1d": 1.0},
                "horizon_bias": {"2": -2.0},
                "feature_weights": {"residual_rms": 0.0},
            },
            "report_all_horizon_metrics": True,
        },
        "data": {
            "task": "advection1d",
            "split": "train",
            "root": str(tmp_path),
            "patch_size": 1,
            "field_name": "u",
        },
    }

    report = evaluate_decoded_operator(
        cfg,
        _DummyEncoder(),
        _AddOperator(delta=2.0),
        _DummyDecoder(),
        rollout_steps=2,
    )

    assert 0.1 <= report.metrics["decoded_residual_gate_alpha_mean"] <= 0.9
    assert (
        report.metrics["decoded_residual_gate_h1_alpha_mean"]
        > report.metrics["decoded_residual_gate_h2_alpha_mean"]
    )
    assert (
        report.metrics["task_advection1d_decoded_residual_gate_alpha_mean"]
        == report.metrics["decoded_residual_gate_alpha_mean"]
    )
    assert (
        report.metrics["family_transport_decoded_residual_gate_alpha_mean"]
        == report.metrics["decoded_residual_gate_alpha_mean"]
    )
    assert report.extra["decoded_persistence_residual_gate"]["base_alpha"] == 0.5


def test_evaluate_decoded_operator_can_apply_task_roll_shift(tmp_path):
    data = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]], dtype=torch.float32)
    file_path = tmp_path / "advection1d_train.h5"
    with h5py.File(file_path, "w") as handle:
        handle.create_dataset("data", data=data.numpy())

    base_cfg = {
        "training": {"batch_size": 1, "dt": 0.1},
        "evaluation": {"decoded_persistence_residual_alpha": 0.0},
        "data": {
            "task": "advection1d",
            "split": "train",
            "root": str(tmp_path),
            "patch_size": 1,
            "field_name": "u",
        },
    }
    shifted_cfg = {
        **base_cfg,
        "evaluation": {
            "decoded_persistence_residual_alpha": 0.0,
            "decoded_roll_shift_by_task": {"advection1d": 1},
        },
    }

    baseline_report = evaluate_decoded_operator(
        base_cfg,
        _DummyEncoder(),
        _IdentityOperator(),
        _DummyDecoder(),
    )
    shifted_report = evaluate_decoded_operator(
        shifted_cfg,
        _DummyEncoder(),
        _IdentityOperator(),
        _DummyDecoder(),
    )

    assert shifted_report.metrics["task_advection1d_decoded_rollout_nrmse"] == 0.0
    assert (
        shifted_report.metrics["decoded_rollout_nrmse"]
        < baseline_report.metrics["decoded_rollout_nrmse"]
    )
    assert shifted_report.extra["decoded_roll_shift_by_task"] == {"advection1d": 1}


def test_evaluate_decoded_operator_can_estimate_observed_roll_shift(tmp_path):
    data = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    file_path = tmp_path / "advection1d_train.h5"
    with h5py.File(file_path, "w") as handle:
        handle.create_dataset("data", data=data.numpy())

    base_cfg = {
        "training": {"batch_size": 1, "dt": 0.1},
        "evaluation": {
            "decoded_persistence_residual_alpha": 0.0,
            "report_all_horizon_metrics": True,
        },
        "data": {
            "task": "advection1d",
            "split": "train",
            "root": str(tmp_path),
            "patch_size": 1,
            "field_name": "u",
        },
    }
    estimator_cfg = {
        **base_cfg,
        "evaluation": {
            "decoded_persistence_residual_alpha": 0.0,
            "decoded_observed_roll_shift_estimator": {
                "candidate_shifts": [-1, 0, 1],
                "tasks": ["advection1d"],
            },
            "report_all_horizon_metrics": True,
        },
    }

    baseline_report = evaluate_decoded_operator(
        base_cfg,
        _DummyEncoder(),
        _IdentityOperator(),
        _DummyDecoder(),
        rollout_steps=2,
    )
    estimator_report = evaluate_decoded_operator(
        estimator_cfg,
        _DummyEncoder(),
        _IdentityOperator(),
        _DummyDecoder(),
        rollout_steps=2,
    )

    assert estimator_report.metrics["task_advection1d_decoded_h2_nrmse"] == 0.0
    assert (
        estimator_report.metrics["decoded_rollout_nrmse"]
        < baseline_report.metrics["decoded_rollout_nrmse"]
    )
    assert estimator_report.metrics["decoded_observed_roll_shift_mean"] == 1.0
    assert estimator_report.extra["decoded_observed_roll_shift_estimator"]["tasks"] == [
        "advection1d"
    ]


def test_evaluate_decoded_operator_can_estimate_prediction_roll_shift(tmp_path):
    data = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]], dtype=torch.float32)
    file_path = tmp_path / "advection1d_train.h5"
    with h5py.File(file_path, "w") as handle:
        handle.create_dataset("data", data=data.numpy())

    cfg = {
        "training": {"batch_size": 1, "dt": 0.1},
        "evaluation": {
            "decoded_persistence_residual_alpha": 0.0,
            "decoded_prediction_roll_shift_estimator": {
                "candidate_shifts": [-1, 0, 1],
                "tasks": ["advection1d"],
                "min_horizon": 1,
                "mode": "roll_persistence",
            },
        },
        "data": {
            "task": "advection1d",
            "split": "train",
            "root": str(tmp_path),
            "patch_size": 1,
            "field_name": "u",
        },
    }

    report = evaluate_decoded_operator(
        cfg,
        _DummyEncoder(),
        _RollOperator(shift=1),
        _DummyDecoder(),
    )

    assert report.metrics["task_advection1d_decoded_rollout_nrmse"] == 0.0
    assert report.metrics["decoded_prediction_roll_shift_mean"] == 1.0
    assert report.extra["decoded_prediction_roll_shift_estimator"]["mode"] == "roll_persistence"


def test_evaluate_decoded_operator_can_apply_data_conditioned_roll_shift(tmp_path):
    data = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]], dtype=torch.float32)
    file_path = tmp_path / "advection1d_train.h5"
    with h5py.File(file_path, "w") as handle:
        handle.create_dataset("data", data=data.numpy())

    cfg = {
        "training": {"batch_size": 1, "dt": 0.1},
        "evaluation": {
            "decoded_persistence_residual_alpha": 0.0,
            "decoded_data_conditioned_roll_shift_estimator": {
                "coefficients": {"bias": 1.0},
                "feature_names": ["bias"],
                "tasks": ["advection1d"],
                "min_horizon": 1,
                "mode": "roll_persistence",
            },
        },
        "data": {
            "task": "advection1d",
            "split": "train",
            "root": str(tmp_path),
            "patch_size": 1,
            "field_name": "u",
        },
    }

    report = evaluate_decoded_operator(
        cfg,
        _DummyEncoder(),
        _IdentityOperator(),
        _DummyDecoder(),
    )

    assert report.metrics["task_advection1d_decoded_rollout_nrmse"] == 0.0
    assert abs(report.metrics["decoded_data_conditioned_roll_shift_mean"] - 1.0) < 1e-6
    assert report.extra["decoded_data_conditioned_roll_shift_estimator"]["feature_names"] == [
        "bias"
    ]


def test_evaluate_decoded_operator_can_apply_parameter_conditioned_roll_shift(tmp_path):
    data = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]], dtype=torch.float32)
    file_path = tmp_path / "advection1d_train.h5"
    with h5py.File(file_path, "w") as handle:
        handle.create_dataset("data", data=data.numpy())
        handle.create_dataset("source_file_index", data=torch.tensor([0]).numpy())
        handle.attrs["source_paths"] = ["1D/Advection/Train/1D_Advection_Sols_beta0.1.hdf5"]

    cfg = {
        "training": {"batch_size": 1, "dt": 0.1},
        "evaluation": {
            "decoded_persistence_residual_alpha": 0.0,
            "decoded_data_conditioned_roll_shift_estimator": {
                "coefficients": {"param:beta": 10.0},
                "feature_names": ["param:beta"],
                "tasks": ["advection1d"],
                "min_horizon": 1,
                "mode": "roll_persistence",
            },
        },
        "data": {
            "task": "advection1d",
            "split": "train",
            "root": str(tmp_path),
            "patch_size": 1,
            "field_name": "u",
            "param_keys": ["beta"],
        },
    }

    report = evaluate_decoded_operator(
        cfg,
        _DummyEncoder(),
        _IdentityOperator(),
        _DummyDecoder(),
    )

    assert report.metrics["task_advection1d_decoded_rollout_nrmse"] < 1e-6
    assert abs(report.metrics["decoded_data_conditioned_roll_shift_mean"] - 1.0) < 1e-6
    assert report.extra["decoded_data_conditioned_roll_shift_estimator"]["feature_names"] == [
        "param:beta"
    ]


def test_evaluate_decoded_operator_can_use_task_specific_roots_for_parameter_sidecar(tmp_path):
    base_root = tmp_path / "base"
    advection_root = tmp_path / "advection"
    base_root.mkdir()
    advection_root.mkdir()

    burgers = torch.full((1, 2, 4), 2.0, dtype=torch.float32)
    with h5py.File(base_root / "burgers1d_train.h5", "w") as handle:
        handle.create_dataset("data", data=burgers.numpy())

    advection = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]])
    with h5py.File(advection_root / "advection1d_train.h5", "w") as handle:
        handle.create_dataset("data", data=advection.numpy())
        handle.create_dataset("source_file_index", data=torch.tensor([0]).numpy())
        handle.attrs["source_paths"] = ["1D/Advection/Train/1D_Advection_Sols_beta0.1.hdf5"]

    cfg = {
        "training": {"batch_size": 1, "dt": 0.1},
        "evaluation": {
            "decoded_persistence_residual_alpha": 0.0,
            "decoded_data_conditioned_roll_shift_estimator": {
                "coefficients": {"param:beta": 10.0},
                "feature_names": ["param:beta"],
                "tasks": ["advection1d"],
                "min_horizon": 1,
                "mode": "roll_persistence",
            },
        },
        "data": {
            "task": ["burgers1d", "advection1d"],
            "split": "train",
            "root": str(base_root),
            "task_roots": {"advection1d": str(advection_root)},
            "patch_size": 1,
            "field_name": "u",
            "param_keys": ["beta"],
        },
    }

    report = evaluate_decoded_operator(
        cfg,
        _DummyEncoder(),
        _IdentityOperator(),
        _DummyDecoder(),
    )

    assert report.extra["task_roots"] == {"advection1d": str(advection_root)}
    assert report.extra["skipped_missing_tasks"] == []
    assert report.metrics["task_burgers1d_decoded_rollout_nrmse"] == 0.0
    assert report.metrics["task_advection1d_decoded_rollout_nrmse"] < 1e-6
    assert abs(report.metrics["decoded_data_conditioned_roll_shift_mean"] - 1.0) < 1e-6


def test_evaluate_decoded_operator_applies_task_parameter_transform(tmp_path):
    data = torch.ones(1, 2, 4, dtype=torch.float32)
    with h5py.File(tmp_path / "advection1d_train.h5", "w") as handle:
        handle.create_dataset("data", data=data.numpy())
        handle.create_dataset("beta", data=torch.tensor([[10.0]]).numpy())

    cfg = {
        "training": {"batch_size": 1, "dt": 0.1, "auto_conditioning": True},
        "evaluation": {"decoded_persistence_residual_alpha": 0.0},
        "data": {
            "task": "advection1d",
            "split": "train",
            "root": str(tmp_path),
            "param_keys": ["beta"],
            "parameter_transforms": {
                "advection1d": {
                    "beta": {
                        "kind": "log10_zscore",
                        "mean": 0.5,
                        "std": 2.0,
                        "count": 1,
                        "source_sha256": "a" * 64,
                    }
                }
            },
        },
    }
    operator = _RecordingIdentityOperator()
    evaluate_decoded_operator(cfg, _DummyEncoder(), operator, _DummyDecoder())

    assert operator.conditions
    assert operator.conditions[0]["param_beta"].item() == pytest.approx(0.25)


def test_evaluate_decoded_operator_data_conditioned_roll_shift_is_default_off(tmp_path):
    data = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]], dtype=torch.float32)
    file_path = tmp_path / "advection1d_train.h5"
    with h5py.File(file_path, "w") as handle:
        handle.create_dataset("data", data=data.numpy())

    cfg = {
        "training": {"batch_size": 1, "dt": 0.1},
        "evaluation": {"decoded_persistence_residual_alpha": 0.0},
        "data": {
            "task": "advection1d",
            "split": "train",
            "root": str(tmp_path),
            "patch_size": 1,
            "field_name": "u",
        },
    }

    report = evaluate_decoded_operator(
        cfg,
        _DummyEncoder(),
        _IdentityOperator(),
        _DummyDecoder(),
    )

    assert "decoded_data_conditioned_roll_shift_mean" not in report.metrics
    assert report.extra["decoded_data_conditioned_roll_shift_estimator"] == {}
    assert report.extra["model_side_transport_head"] == {}


def test_evaluate_decoded_operator_can_apply_model_side_beta_transport_head(tmp_path):
    data = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]], dtype=torch.float32)
    file_path = tmp_path / "advection1d_train.h5"
    with h5py.File(file_path, "w") as handle:
        handle.create_dataset("data", data=data.numpy())
        handle.create_dataset("source_file_index", data=torch.tensor([0]).numpy())
        handle.attrs["source_paths"] = ["1D/Advection/Train/1D_Advection_Sols_beta0.1.hdf5"]

    cfg = {
        "training": {"batch_size": 1, "dt": 0.1},
        "model_side_transport_head": {
            "enabled": True,
            "tasks": ["advection1d"],
            "required_params": ["beta"],
            "features": ["param:beta", "bias"],
            "init": {"param:beta": 10.0, "bias": 0.0},
            "mode": "periodic_roll",
            "apply_at": "decoded_rollout",
            "missing_param_policy": "skip",
        },
        "evaluation": {"decoded_persistence_residual_alpha": 0.0},
        "data": {
            "task": "advection1d",
            "split": "train",
            "root": str(tmp_path),
            "patch_size": 1,
            "field_name": "u",
            "param_keys": ["beta"],
        },
    }

    report = evaluate_decoded_operator(
        cfg,
        _DummyEncoder(),
        _IdentityOperator(),
        _DummyDecoder(),
    )

    assert report.metrics["task_advection1d_decoded_rollout_nrmse"] < 1e-6
    assert abs(report.metrics["model_side_transport_head_shift_mean"] - 1.0) < 1e-6
    assert report.extra["model_side_transport_head"]["enabled"] is True
    assert report.extra["model_side_transport_head"]["trainable_parameter_count"] == 2
    assert report.extra["model_side_transport_head_metrics"] == {
        "applied_sample_count": 1,
        "skipped_sample_count": 0,
        "beta_missing_count": 0,
    }


def test_evaluate_decoded_operator_model_side_transport_head_skips_missing_beta(tmp_path):
    data = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]], dtype=torch.float32)
    file_path = tmp_path / "advection1d_train.h5"
    with h5py.File(file_path, "w") as handle:
        handle.create_dataset("data", data=data.numpy())

    cfg = {
        "training": {"batch_size": 1, "dt": 0.1},
        "model_side_transport_head": {
            "enabled": True,
            "tasks": ["advection1d"],
            "required_params": ["beta"],
            "features": ["param:beta"],
            "init": {"param:beta": 10.0},
            "missing_param_policy": "skip",
        },
        "evaluation": {"decoded_persistence_residual_alpha": 0.0},
        "data": {
            "task": "advection1d",
            "split": "train",
            "root": str(tmp_path),
            "patch_size": 1,
            "field_name": "u",
        },
    }

    report = evaluate_decoded_operator(
        cfg,
        _DummyEncoder(),
        _IdentityOperator(),
        _DummyDecoder(),
    )

    assert report.metrics["task_advection1d_decoded_rollout_nrmse"] > 0.0
    assert "model_side_transport_head_shift_mean" not in report.metrics
    assert report.extra["model_side_transport_head_metrics"] == {
        "applied_sample_count": 0,
        "skipped_sample_count": 1,
        "beta_missing_count": 1,
    }


def test_evaluate_decoded_operator_data_conditioned_roll_shift_can_use_context_feature(
    tmp_path,
):
    data = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ],
        dtype=torch.float32,
    )
    file_path = tmp_path / "advection1d_train.h5"
    with h5py.File(file_path, "w") as handle:
        handle.create_dataset("data", data=data.numpy())

    cfg = {
        "training": {"batch_size": 1, "dt": 0.1},
        "evaluation": {
            "decoded_persistence_residual_alpha": 0.0,
            "decoded_data_conditioned_roll_shift_estimator": {
                "candidate_shifts": [-1, 0, 1],
                "context_transitions": 1,
                "coefficients": {"context_shift": 1.0},
                "feature_names": ["context_shift"],
                "min_horizon": 2,
                "mode": "roll_persistence",
                "tasks": ["advection1d"],
            },
            "report_all_horizon_metrics": True,
        },
        "data": {
            "task": "advection1d",
            "split": "train",
            "root": str(tmp_path),
            "patch_size": 1,
            "field_name": "u",
        },
    }

    report = evaluate_decoded_operator(
        cfg,
        _DummyEncoder(),
        _IdentityOperator(),
        _DummyDecoder(),
        rollout_steps=3,
    )

    assert report.metrics["task_advection1d_decoded_h2_nrmse"] == 0.0
    assert report.metrics["task_advection1d_decoded_h3_nrmse"] == 0.0
    assert report.metrics["decoded_data_conditioned_roll_shift_mean"] == 1.0
    assert report.extra["decoded_data_conditioned_roll_shift_estimator"]["context_transitions"] == 1


def test_evaluate_decoded_operator_can_apply_context_calibrated_roll_shift(tmp_path):
    data = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ],
        dtype=torch.float32,
    )
    file_path = tmp_path / "advection1d_train.h5"
    with h5py.File(file_path, "w") as handle:
        handle.create_dataset("data", data=data.numpy())

    cfg = {
        "training": {"batch_size": 1, "dt": 0.1},
        "evaluation": {
            "decoded_persistence_residual_alpha": 0.0,
            "decoded_context_roll_shift_estimator": {
                "candidate_shifts": [-1, 0, 1],
                "context_transitions": 1,
                "coefficients": {"slope": 1.0, "intercept": 0.0},
                "tasks": ["advection1d"],
                "mode": "roll_persistence",
            },
            "report_all_horizon_metrics": True,
        },
        "data": {
            "task": "advection1d",
            "split": "train",
            "root": str(tmp_path),
            "patch_size": 1,
            "field_name": "u",
        },
    }

    report = evaluate_decoded_operator(
        cfg,
        _DummyEncoder(),
        _IdentityOperator(),
        _DummyDecoder(),
        rollout_steps=3,
    )

    assert report.metrics["task_advection1d_decoded_h2_nrmse"] == 0.0
    assert report.metrics["task_advection1d_decoded_h3_nrmse"] == 0.0
    assert report.metrics["decoded_context_roll_shift_mean"] == 1.0
    assert report.extra["decoded_context_roll_shift_estimator"]["calibration_scope"] == (
        "shared_1d_transport"
    )


def test_evaluate_decoded_operator_reports_multitask_metrics(tmp_path):
    for task_name in ("burgers1d", "advection1d"):
        data = torch.ones(1, 3, 4, dtype=torch.float32)
        file_path = tmp_path / f"{task_name}_train.h5"
        with h5py.File(file_path, "w") as handle:
            handle.create_dataset("data", data=data.numpy())

    cfg = {
        "training": {"batch_size": 1, "dt": 0.1, "auto_conditioning": True},
        "data": {
            "task": ["burgers1d", "advection1d"],
            "split": "train",
            "root": str(tmp_path),
            "patch_size": 1,
            "field_name": "u",
        },
    }

    report = evaluate_decoded_operator(
        cfg,
        _DummyEncoder(),
        _IdentityOperator(),
        _DummyDecoder(),
    )

    assert report.metrics["task_burgers1d_decoded_rollout_nrmse"] == 0.0
    assert report.metrics["task_advection1d_decoded_rollout_nrmse"] == 0.0
    assert report.metrics["family_conservation_decoded_rollout_nrmse"] == 0.0
    assert report.metrics["family_transport_decoded_rollout_nrmse"] == 0.0


def test_evaluate_decoded_operator_reports_heterogeneous_multitask_metrics(tmp_path):
    burgers = torch.ones(1, 3, 4, dtype=torch.float32)
    advection = torch.ones(1, 3, 4, dtype=torch.float32)
    darcy_coefficient = torch.ones(1, 1, 4, 4, 1, dtype=torch.float32)
    darcy_solution = torch.ones(1, 1, 4, 4, 1, dtype=torch.float32)

    with h5py.File(tmp_path / "burgers1d_train.h5", "w") as handle:
        handle.create_dataset("data", data=burgers.numpy())
    with h5py.File(tmp_path / "advection1d_train.h5", "w") as handle:
        handle.create_dataset("data", data=advection.numpy())
    with h5py.File(tmp_path / "darcy2d_train.h5", "w") as handle:
        handle.create_dataset("data", data=darcy_coefficient.numpy())
        handle.create_dataset("targets", data=darcy_solution.numpy())

    cfg = {
        "training": {"batch_size": 1, "dt": 0.1, "auto_conditioning": True},
        "data": {
            "task": ["burgers1d", "advection1d", "darcy2d"],
            "split": "train",
            "root": str(tmp_path),
            "patch_size": 1,
            "field_name": "u",
        },
    }

    report = evaluate_decoded_operator(
        cfg,
        _DummyEncoder(),
        _IdentityOperator(),
        _DummyDecoder(),
    )

    assert report.metrics["decoded_rollout_nrmse"] == 0.0
    assert report.metrics["task_burgers1d_decoded_rollout_nrmse"] == 0.0
    assert report.metrics["task_advection1d_decoded_rollout_nrmse"] == 0.0
    assert report.metrics["task_darcy2d_decoded_rollout_nrmse"] == 0.0
    assert report.metrics["family_elliptic_decoded_rollout_nrmse"] == 0.0


def test_evaluate_cli_main(tmp_path, monkeypatch, capsys):
    _write_minimal_hdf5(tmp_path)
    cfg = {
        "training": {"batch_size": 2, "dt": 0.1},
        "latent": {"dim": 8, "tokens": 4},
        "data": {
            "task": "burgers1d",
            "split": "train",
            "root": str(tmp_path),
            "patch_size": 1,
        },
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    operator = evaluate_script.make_operator(cfg)
    operator_path = tmp_path / "operator.pt"
    torch.save(operator.state_dict(), operator_path)

    output_prefix = tmp_path / "eval_run"
    args = [
        "evaluate",
        "--config",
        str(cfg_path),
        "--operator",
        str(operator_path),
        "--output-prefix",
        str(output_prefix),
        "--log-path",
        str(tmp_path / "eval_log.jsonl"),
        "--print-json",
    ]
    monkeypatch.setattr(sys, "argv", args)

    evaluate_script.main()
    output = capsys.readouterr().out
    assert '"metrics"' in output
    assert output_prefix.with_suffix(".json").exists()
    assert output_prefix.with_suffix(".csv").exists()
    assert output_prefix.with_suffix(".html").exists()
    assert output_prefix.with_suffix(".config.yaml").exists()
    assert (output_prefix.parent / f"{output_prefix.name}_metrics.png").exists()
    assert (output_prefix.parent / f"{output_prefix.name}_mse_hist.png").exists()
    assert (output_prefix.parent / f"{output_prefix.name}_mae_hist.png").exists()
    assert (tmp_path / "eval_log.jsonl").exists()


def test_evaluate_cli_main_with_decoded_metrics(tmp_path, monkeypatch, capsys):
    _write_minimal_hdf5(tmp_path)
    cfg = {
        "training": {"batch_size": 2, "dt": 0.1},
        "latent": {"dim": 8, "tokens": 4},
        "decoder": {"hidden_dim": 16, "mlp_hidden_dim": 16, "num_heads": 4, "num_layers": 1},
        "data": {
            "task": "burgers1d",
            "split": "train",
            "root": str(tmp_path),
            "patch_size": 1,
        },
    }
    cfg_path = tmp_path / "cfg_decoded.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    operator = evaluate_script.make_operator(cfg)
    encoder = evaluate_script.make_encoder(cfg)
    decoder = evaluate_script.make_decoder(cfg)
    dataset = evaluate_script.PDEBenchDataset(
        evaluate_script.PDEBenchConfig(task="burgers1d", split="train", root=str(tmp_path))
    )
    grid_shape = evaluate_script.infer_grid_shape(dataset.fields[0])
    coords = train_script.make_grid_coords(grid_shape, torch.device("cpu"))
    field_step = dataset[0]["fields"][0]
    flattened = train_script._flatten_field_step(field_step, grid_shape)
    with torch.no_grad():
        encoder({"u": flattened}, coords, meta={"grid_shape": grid_shape})
    operator_path = tmp_path / "operator.pt"
    encoder_path = tmp_path / "encoder.pt"
    decoder_path = tmp_path / "decoder.pt"
    torch.save(operator.state_dict(), operator_path)
    torch.save(encoder.state_dict(), encoder_path)
    torch.save(decoder.state_dict(), decoder_path)

    output_prefix = tmp_path / "eval_decoded"
    args = [
        "evaluate",
        "--config",
        str(cfg_path),
        "--operator",
        str(operator_path),
        "--encoder",
        str(encoder_path),
        "--decoder",
        str(decoder_path),
        "--decoded",
        "--decoded-rollout-steps",
        "1",
        "--output-prefix",
        str(output_prefix),
        "--log-path",
        str(tmp_path / "eval_decoded_log.jsonl"),
        "--print-json",
    ]
    monkeypatch.setattr(sys, "argv", args)

    evaluate_script.main()
    output = capsys.readouterr().out
    assert '"decoded_mse"' in output
    assert '"decoded_rollout_nrmse"' in output
    assert '"decoded_h4_nrmse"' not in output


def test_evaluate_cli_main_with_transfer_tasks(tmp_path, monkeypatch, capsys):
    for task_name in ("burgers1d", "advection1d"):
        data = torch.ones(1, 3, 4, dtype=torch.float32)
        file_path = tmp_path / f"{task_name}_train.h5"
        with h5py.File(file_path, "w") as handle:
            handle.create_dataset("data", data=data.numpy())

    cfg = {
        "training": {"batch_size": 1, "dt": 0.1, "auto_conditioning": True},
        "latent": {"dim": 8, "tokens": 4},
        "decoder": {"hidden_dim": 16, "mlp_hidden_dim": 16, "num_heads": 4, "num_layers": 1},
        "data": {
            "task": "burgers1d",
            "split": "train",
            "root": str(tmp_path),
            "patch_size": 1,
        },
    }
    cfg_path = tmp_path / "cfg_transfer.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    operator = evaluate_script.make_operator(cfg)
    encoder = evaluate_script.make_encoder(cfg)
    decoder = evaluate_script.make_decoder(cfg)
    dataset = evaluate_script.PDEBenchDataset(
        evaluate_script.PDEBenchConfig(task="burgers1d", split="train", root=str(tmp_path))
    )
    grid_shape = evaluate_script.infer_grid_shape(dataset.fields[0])
    coords = train_script.make_grid_coords(grid_shape, torch.device("cpu"))
    field_step = dataset[0]["fields"][0]
    flattened = train_script._flatten_field_step(field_step, grid_shape)
    with torch.no_grad():
        encoder({"u": flattened}, coords, meta={"grid_shape": grid_shape})
    operator_path = tmp_path / "operator_transfer.pt"
    encoder_path = tmp_path / "encoder_transfer.pt"
    decoder_path = tmp_path / "decoder_transfer.pt"
    torch.save(operator.state_dict(), operator_path)
    torch.save(encoder.state_dict(), encoder_path)
    torch.save(decoder.state_dict(), decoder_path)

    output_prefix = tmp_path / "eval_transfer"
    args = [
        "evaluate",
        "--config",
        str(cfg_path),
        "--operator",
        str(operator_path),
        "--encoder",
        str(encoder_path),
        "--decoder",
        str(decoder_path),
        "--decoded",
        "--decoded-rollout-steps",
        "1",
        "--transfer-tasks",
        "advection1d",
        "--output-prefix",
        str(output_prefix),
        "--log-path",
        str(tmp_path / "eval_transfer_log.jsonl"),
        "--print-json",
    ]
    monkeypatch.setattr(sys, "argv", args)

    evaluate_script.main()
    output = capsys.readouterr().out
    assert '"transfer_mse"' in output
    assert '"transfer_decoded_rollout_nrmse"' in output
    assert '"transfer_task_advection1d_decoded_rollout_nrmse"' in output


def test_evaluate_cli_main_with_promotion_rules(tmp_path, monkeypatch, capsys):
    _write_minimal_hdf5(tmp_path)
    cfg = {
        "training": {"batch_size": 2, "dt": 0.1},
        "latent": {"dim": 8, "tokens": 4},
        "data": {
            "task": "burgers1d",
            "split": "train",
            "root": str(tmp_path),
            "patch_size": 1,
        },
        "evaluation": {"promotion": {"rules": ["mse>=0.0"]}},
    }
    cfg_path = tmp_path / "cfg_promotion.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    operator = evaluate_script.make_operator(cfg)
    operator_path = tmp_path / "operator_promotion.pt"
    torch.save(operator.state_dict(), operator_path)

    args = [
        "evaluate",
        "--config",
        str(cfg_path),
        "--operator",
        str(operator_path),
        "--promotion-rule",
        "rmse>=0.0",
        "--print-json",
    ]
    monkeypatch.setattr(sys, "argv", args)

    evaluate_script.main()
    output = capsys.readouterr().out
    assert '"promotion_passed": true' in output


def test_evaluate_cli_main_with_wildcard_family_promotion_rule(tmp_path, monkeypatch, capsys):
    for task_name in ("burgers1d", "advection1d"):
        data = torch.ones(1, 3, 4, dtype=torch.float32)
        file_path = tmp_path / f"{task_name}_train.h5"
        with h5py.File(file_path, "w") as handle:
            handle.create_dataset("data", data=data.numpy())

    cfg = {
        "training": {"batch_size": 1, "dt": 0.1, "auto_conditioning": True},
        "latent": {"dim": 8, "tokens": 4},
        "decoder": {"hidden_dim": 16, "mlp_hidden_dim": 16, "num_heads": 4, "num_layers": 1},
        "data": {
            "task": ["burgers1d", "advection1d"],
            "split": "train",
            "root": str(tmp_path),
            "patch_size": 1,
        },
    }
    cfg_path = tmp_path / "cfg_family_promotion.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    operator = evaluate_script.make_operator(cfg)
    operator_path = tmp_path / "operator_family_promotion.pt"
    torch.save(operator.state_dict(), operator_path)

    encoder = _DummyEncoder()
    decoder = _DummyDecoder()
    monkeypatch.setattr(evaluate_script, "make_encoder", lambda _cfg: encoder)
    monkeypatch.setattr(evaluate_script, "make_decoder", lambda _cfg: decoder)
    real_load_state_dict = evaluate_script._load_state_dict_compat
    monkeypatch.setattr(
        evaluate_script,
        "_load_state_dict_compat",
        lambda model, *args, **kwargs: (
            real_load_state_dict(model, *args, **kwargs)
            if isinstance(model, evaluate_script.LatentOperator)
            else None
        ),
    )

    output_prefix = tmp_path / "eval_family_promotion"
    args = [
        "evaluate",
        "--config",
        str(cfg_path),
        "--operator",
        str(operator_path),
        "--encoder",
        str(operator_path),
        "--decoder",
        str(operator_path),
        "--decoded",
        "--decoded-rollout-steps",
        "1",
        "--promotion-rule",
        "max:family_*_decoded_rollout_nrmse>=0.0",
        "--output-prefix",
        str(output_prefix),
        "--print-json",
    ]
    monkeypatch.setattr(sys, "argv", args)

    evaluate_script.main()
    output = capsys.readouterr().out
    assert '"promotion_passed": true' in output


def test_evaluate_cli_main_fails_on_promotion_failure(tmp_path, monkeypatch):
    _write_minimal_hdf5(tmp_path)
    cfg = {
        "training": {"batch_size": 2, "dt": 0.1},
        "latent": {"dim": 8, "tokens": 4},
        "data": {
            "task": "burgers1d",
            "split": "train",
            "root": str(tmp_path),
            "patch_size": 1,
        },
    }
    cfg_path = tmp_path / "cfg_promotion_fail.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    operator = evaluate_script.make_operator(cfg)
    operator_path = tmp_path / "operator_promotion_fail.pt"
    torch.save(operator.state_dict(), operator_path)

    args = [
        "evaluate",
        "--config",
        str(cfg_path),
        "--operator",
        str(operator_path),
        "--promotion-rule",
        "mse<0.0",
        "--fail-on-promotion",
    ]
    monkeypatch.setattr(sys, "argv", args)

    try:
        evaluate_script.main()
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("Expected promotion failure to exit with code 2")


def test_benchmark_cli(tmp_path, monkeypatch, capsys):
    _write_minimal_hdf5(tmp_path)
    cfg = {
        "latent": {"dim": 8, "tokens": 4},
        "training": {"batch_size": 2, "dt": 0.1},
        "data": {
            "task": "burgers1d",
            "split": "train",
            "root": str(tmp_path),
            "patch_size": 1,
        },
        "checkpoint": {"dir": str(tmp_path / "ckpts")},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    operator = train_script.make_operator(cfg)
    operator_path = tmp_path / "ckpts" / "operator.pt"
    operator_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(operator.state_dict(), operator_path)

    log_path = tmp_path / "benchmark_log.jsonl"
    args = [
        "benchmark",
        "--config",
        str(cfg_path),
        "--operator",
        str(operator_path),
        "--baseline",
        "identity",
        "--output",
        str(tmp_path / "benchmark.json"),
        "--log-path",
        str(log_path),
    ]
    monkeypatch.setattr(sys, "argv", args)

    benchmark_script.main()
    captured = capsys.readouterr().out
    assert "Benchmark results" in captured
    out_path = tmp_path / "benchmark.json"
    assert out_path.exists()
    assert log_path.exists()


def test_train_baseline_cli(tmp_path, monkeypatch, capsys):
    _write_minimal_hdf5(tmp_path)
    cfg = {
        "latent": {"dim": 8, "tokens": 4},
        "training": {"batch_size": 2, "dt": 0.1},
        "baseline": {"epochs": 1, "log_path": str(tmp_path / "baseline_log.jsonl")},
        "data": {
            "task": "burgers1d",
            "split": "train",
            "root": str(tmp_path),
            "patch_size": 1,
        },
        "checkpoint": {"dir": str(tmp_path / "ckpts")},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    log_path = tmp_path / "baseline_log.jsonl"
    args = [
        "train_baselines",
        "--config",
        str(cfg_path),
        "--baseline",
        "identity",
        "--seed",
        "0",
    ]
    monkeypatch.setattr(sys, "argv", args)

    train_baselines_script.main()
    out = capsys.readouterr().out
    assert "Saved baseline checkpoint" in out
    ckpt = tmp_path / "ckpts" / "baseline_identity.pt"
    assert ckpt.exists()
    assert log_path.exists()
