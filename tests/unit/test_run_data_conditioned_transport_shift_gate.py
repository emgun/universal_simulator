from __future__ import annotations

import argparse

import h5py
import numpy as np
import pytest

from scripts.run_data_conditioned_transport_shift_gate import run_gate


def _trajectory(*, width: int, steps: int, shift: int, amplitude: float) -> np.ndarray:
    frame = np.zeros(width, dtype=np.float32)
    frame[0] = float(amplitude)
    rows = []
    for step in range(steps):
        rows.append(np.roll(frame, shift * step))
    return np.stack(rows, axis=0)


def _write_split(path, rows: list[np.ndarray]) -> None:
    data = np.stack(rows, axis=0).astype(np.float32)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=data)


def _args(tmp_path, **overrides):
    values = {
        "data_root": str(tmp_path),
        "task": "advection1d",
        "train_split": "train",
        "val_split": "val",
        "max_samples": 4,
        "train_max_samples": None,
        "val_max_samples": None,
        "rollout_steps": 3,
        "shift": [0, 1, 2],
        "feature": ["bias", "rms"],
        "metric": "nrmse",
        "ridge": 1e-6,
        "min_horizon": 1,
        "min_shift": 0.0,
        "max_shift": 2.0,
        "no_normalize": True,
        "reference_metric_value": 1.0,
        "val_min_relative_improvement": 0.0,
        "output_json": str(tmp_path / "gate.json"),
        "context_transitions": 0,
        "refine_radius": 0,
        "fractional_refine_step": 0.0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_run_gate_fits_train_only_data_conditioned_shift(tmp_path):
    _write_split(
        tmp_path / "advection1d_train.h5",
        [
            _trajectory(width=8, steps=4, shift=1, amplitude=1.0),
            _trajectory(width=8, steps=4, shift=2, amplitude=2.0),
        ],
    )
    _write_split(
        tmp_path / "advection1d_val.h5",
        [
            _trajectory(width=8, steps=4, shift=1, amplitude=1.0),
            _trajectory(width=8, steps=4, shift=2, amplitude=2.0),
        ],
    )

    record = run_gate(_args(tmp_path))

    assert record["held_out_test_used"] is False
    assert record["held_out_test_data_read"] is False
    assert record["fit"]["selected_validation"]["nrmse"] < 0.01
    estimator = record["selected_override"][
        "evaluation.decoded_data_conditioned_roll_shift_estimator"
    ]
    assert estimator["mode"] == "roll_persistence"
    assert estimator["tasks"] == ["advection1d"]
    assert estimator["feature_names"] == ["bias", "rms"]


def test_run_gate_rejects_same_split(tmp_path):
    with pytest.raises(ValueError, match="must differ"):
        run_gate(_args(tmp_path, val_split="train"))


def test_run_gate_can_fit_context_shift_feature_across_regimes(tmp_path):
    _write_split(
        tmp_path / "advection1d_train.h5",
        [
            _trajectory(width=8, steps=4, shift=1, amplitude=1.0),
            _trajectory(width=8, steps=4, shift=1, amplitude=2.0),
        ],
    )
    _write_split(
        tmp_path / "advection1d_val.h5",
        [
            _trajectory(width=8, steps=4, shift=2, amplitude=1.0),
            _trajectory(width=8, steps=4, shift=2, amplitude=2.0),
        ],
    )

    record = run_gate(
        _args(
            tmp_path,
            context_transitions=1,
            feature=["context_shift"],
            min_horizon=2,
        )
    )

    assert record["fit"]["selected_validation"]["nrmse"] < 0.01
    estimator = record["selected_override"][
        "evaluation.decoded_data_conditioned_roll_shift_estimator"
    ]
    assert estimator["context_transitions"] == 1
    assert estimator["coefficients"]["context_shift"] == pytest.approx(1.0)
