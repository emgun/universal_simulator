from __future__ import annotations

import json
from argparse import Namespace

import h5py
import numpy as np

from scripts.run_inferred_transport_transfer_scorecard import run_scorecard
from scripts.run_source_conditioned_transport_shift_gate import _periodic_shift


def _write_trajectory_split(
    root,
    *,
    task: str,
    split: str,
    shift: float,
    samples: int = 2,
    steps: int = 5,
    width: int = 32,
) -> None:
    rows = []
    x = np.arange(width, dtype=np.float32)
    for sample_idx in range(samples):
        trajectory = np.zeros((steps, width), dtype=np.float32)
        trajectory[0] = np.exp(-0.5 * ((x - (8.0 + sample_idx)) / 2.0) ** 2)
        for step_idx in range(1, steps):
            trajectory[step_idx] = _periodic_shift(trajectory[step_idx - 1 : step_idx], shift)[0]
        rows.append(trajectory)
    with h5py.File(root / f"{task}_{split}.h5", "w") as handle:
        handle.create_dataset("data", data=np.asarray(rows, dtype=np.float32)[..., None])


def _write_static_split(root, *, task: str, split: str, samples: int = 2) -> None:
    with h5py.File(root / f"{task}_{split}.h5", "w") as handle:
        handle.create_dataset("data", data=np.zeros((samples, 16, 16, 1), dtype=np.float32))


def test_transfer_scorecard_evaluates_supported_1d_tasks_and_skips_unsupported(tmp_path):
    _write_trajectory_split(tmp_path, task="advection1d", split="train", shift=1.0)
    _write_trajectory_split(tmp_path, task="advection1d", split="val", shift=1.0)
    _write_trajectory_split(tmp_path, task="burgers1d", split="train", shift=2.0)
    _write_trajectory_split(tmp_path, task="burgers1d", split="val", shift=2.0)
    _write_static_split(tmp_path, task="darcy2d", split="val")

    record = run_scorecard(
        Namespace(
            data_root=tmp_path,
            tasks=["advection1d", "burgers1d", "darcy2d"],
            train_split="train",
            val_split="val",
            max_samples=None,
            context_transitions=1,
            rollout_steps=3,
            shift=[0, 1, 2],
            metric="nrmse",
            fit_kind="linear",
            fit_intercept=False,
            refine_radius=0,
            fractional_refine_step=0.0,
            reference_metric_value=1.0,
            val_min_relative_improvement=0.0,
            shared_calibrator=False,
            output_dir=tmp_path / "out",
        )
    )

    assert record["status"] == "partial_transfer_validated"
    assert record["evaluated_task_count"] == 2
    assert record["skipped_task_count"] == 1
    assert record["tasks"]["advection1d"]["status"] == "validated"
    assert record["tasks"]["advection1d"]["validation_nrmse"] < 1e-5
    assert record["tasks"]["burgers1d"]["status"] == "validated"
    assert record["tasks"]["burgers1d"]["validation_nrmse"] < 1e-5
    assert record["tasks"]["darcy2d"]["status"] == "skipped"
    assert "missing train split" in record["tasks"]["darcy2d"]["reason"]
    assert (tmp_path / "out" / "advection1d_transfer_gate.json").exists()
    assert (
        json.loads((tmp_path / "out" / "scorecard.json").read_text())["status"] == record["status"]
    )


def test_transfer_scorecard_skips_non_1d_tasks_even_when_splits_exist(tmp_path):
    _write_static_split(tmp_path, task="darcy2d", split="train")
    _write_static_split(tmp_path, task="darcy2d", split="val")

    record = run_scorecard(
        Namespace(
            data_root=tmp_path,
            tasks=["darcy2d"],
            train_split="train",
            val_split="val",
            max_samples=None,
            context_transitions=1,
            rollout_steps=3,
            shift=[0, 1, 2],
            metric="nrmse",
            fit_kind="linear",
            fit_intercept=False,
            refine_radius=0,
            fractional_refine_step=0.0,
            reference_metric_value=1.0,
            val_min_relative_improvement=0.0,
            shared_calibrator=False,
            output_dir=tmp_path / "out",
        )
    )

    assert record["status"] == "blocked_no_supported_transfer_tasks"
    assert record["evaluated_task_count"] == 0
    assert record["skipped_task_count"] == 1
    assert record["tasks"]["darcy2d"]["status"] == "skipped"
    assert record["tasks"]["darcy2d"]["reason"] == "unsupported task for 1D transport gate: darcy2d"


def test_transfer_scorecard_can_use_one_shared_1d_transport_calibrator(tmp_path):
    _write_trajectory_split(tmp_path, task="advection1d", split="train", shift=1.0)
    _write_trajectory_split(tmp_path, task="advection1d", split="val", shift=1.0)
    _write_trajectory_split(tmp_path, task="burgers1d", split="train", shift=2.0)
    _write_trajectory_split(tmp_path, task="burgers1d", split="val", shift=2.0)

    record = run_scorecard(
        Namespace(
            data_root=tmp_path,
            tasks=["advection1d", "burgers1d"],
            train_split="train",
            val_split="val",
            max_samples=None,
            context_transitions=1,
            rollout_steps=3,
            shift=[0, 1, 2],
            metric="nrmse",
            fit_kind="linear",
            fit_intercept=False,
            refine_radius=0,
            fractional_refine_step=0.0,
            reference_metric_value=1.0,
            val_min_relative_improvement=0.0,
            shared_calibrator=True,
            output_dir=tmp_path / "out",
        )
    )

    assert record["status"] == "transfer_validated"
    assert record["calibration_scope"] == "shared_1d_transport"
    assert record["shared_fit"]["train_task_count"] == 2
    assert record["shared_fit"]["coefficients"]["slope"] == 1.0
    assert record["tasks"]["advection1d"]["calibration_scope"] == "shared_1d_transport"
    assert record["tasks"]["advection1d"]["validation_nrmse"] < 1e-5
    assert record["tasks"]["burgers1d"]["calibration_scope"] == "shared_1d_transport"
    assert record["tasks"]["burgers1d"]["validation_nrmse"] < 1e-5
    assert (tmp_path / "out" / "shared_1d_transport_calibrator.json").exists()
