from __future__ import annotations

import argparse

import h5py
import numpy as np

from scripts.run_data_conditioned_ablation_matrix import run_matrix


def _trajectory(*, width: int, steps: int, shift: int) -> np.ndarray:
    frame = np.zeros(width, dtype=np.float32)
    frame[0] = 1.0
    return np.stack([np.roll(frame, shift * step) for step in range(steps)], axis=0)


def _write_split(path, rows: list[np.ndarray]) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=np.stack(rows, axis=0).astype(np.float32))


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
        "metric": "nrmse",
        "ridge": 1e-6,
        "context_transitions": 1,
        "full_shift_min": 0,
        "full_shift_max": 3,
        "weak_shift_min": 0,
        "weak_shift_max": 1,
        "no_data_shift_min": 0,
        "no_data_shift_max": 3,
        "min_shift": 0.0,
        "max_shift": 3.0,
        "no_normalize": True,
        "reference_metric_value": 1.0,
        "val_min_relative_improvement": 0.0,
        "output_dir": str(tmp_path / "ablation"),
        "output_json": str(tmp_path / "ablation" / "matrix.json"),
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_run_matrix_records_required_validation_only_ablation_variants(tmp_path):
    _write_split(
        tmp_path / "advection1d_train.h5",
        [_trajectory(width=16, steps=4, shift=1) for _ in range(2)],
    )
    _write_split(
        tmp_path / "advection1d_val.h5",
        [_trajectory(width=16, steps=4, shift=3) for _ in range(2)],
    )

    record = run_matrix(_args(tmp_path))

    assert record["held_out_test_used"] is False
    assert record["held_out_test_data_read"] is False
    assert record["split"] == "val"
    assert set(record["variants"]) == {
        "full_context_shift",
        "weaker_context_shift",
        "no_data_conditioning",
    }
    assert record["variants"]["full_context_shift"]["feature_names"] == ["context_shift"]
    assert record["variants"]["weaker_context_shift"]["candidate_shift_max_abs"] < (
        record["variants"]["full_context_shift"]["candidate_shift_max_abs"]
    )
    assert "context_shift" not in record["variants"]["no_data_conditioning"]["feature_names"]
    assert record["deltas_vs_full_context_shift"]["full_context_shift"]["absolute"] == 0.0
    assert (
        record["deltas_vs_full_context_shift"]["no_data_conditioning"]["absolute"]
        > record["deltas_vs_full_context_shift"]["full_context_shift"]["absolute"]
    )
    assert record["context_dependency_interpretation"]["full_context_shift_is_best"] is True
    assert record["test_ledger_writes"] == []
