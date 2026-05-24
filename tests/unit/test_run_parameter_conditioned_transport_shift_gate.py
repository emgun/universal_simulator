from __future__ import annotations

import json
from argparse import Namespace

import h5py
import numpy as np
import pytest

from scripts.run_parameter_conditioned_transport_shift_gate import (
    _beta_from_source_path,
    run_gate,
)
from scripts.run_source_conditioned_transport_shift_gate import _periodic_shift


def _write_beta_shifted_split(
    root,
    *,
    split: str,
    beta_shifts: list[tuple[float, float]],
    samples_per_source: int = 2,
    steps: int = 5,
    width: int = 32,
) -> None:
    source_paths = [
        f"1D/Advection/Train/1D_Advection_Sols_beta{beta}.hdf5" for beta, _shift in beta_shifts
    ]
    rows = []
    source_file_index = []
    source_sample_index = []
    x = np.arange(width, dtype=np.float32)
    for source_index, (_beta, shift) in enumerate(beta_shifts):
        for sample_idx in range(samples_per_source):
            center = 8.0 + source_index * 2.0 + sample_idx
            trajectory = np.zeros((steps, width), dtype=np.float32)
            trajectory[0] = np.exp(-0.5 * ((x - center) / 2.0) ** 2)
            for step_idx in range(1, steps):
                trajectory[step_idx] = _periodic_shift(trajectory[step_idx - 1 : step_idx], shift)[
                    0
                ]
            rows.append(trajectory)
            source_file_index.append(source_index)
            source_sample_index.append(sample_idx)
    with h5py.File(root / f"advection1d_{split}.h5", "w") as handle:
        handle.create_dataset("data", data=np.asarray(rows, dtype=np.float32)[..., None])
        handle.create_dataset("source_file_index", data=source_file_index)
        handle.create_dataset("source_sample_index", data=source_sample_index)
        handle.attrs["source_paths"] = source_paths


def _args(tmp_path, *, reference: float = 1.0) -> Namespace:
    return Namespace(
        data_root=tmp_path,
        task="advection1d",
        train_split="train",
        val_split="val",
        test_split="",
        max_samples=None,
        train_max_samples=None,
        val_max_samples=None,
        test_max_samples=None,
        rollout_steps=4,
        shift=[-4, 0, 4],
        metric="nrmse",
        fit_kind="linear",
        fit_intercept=True,
        refine_radius=2,
        fractional_refine_step=0.5,
        reference_metric_value=reference,
        val_min_relative_improvement=0.0,
        test_ledger_json=None,
        allow_repeat_test=False,
    )


def test_beta_from_source_path_parses_official_advection_names():
    assert _beta_from_source_path("1D/Advection/Train/1D_Advection_Sols_beta0.1.hdf5") == 0.1
    assert _beta_from_source_path("1D_Advection_Sols_beta7.0.hdf5") == 7.0


def test_parameter_conditioned_gate_fits_beta_shift_function_without_source_identity(tmp_path):
    beta_shifts = [(0.1, 1.0), (0.2, 2.0), (0.4, 4.0)]
    _write_beta_shifted_split(tmp_path, split="train", beta_shifts=beta_shifts)
    _write_beta_shifted_split(tmp_path, split="val", beta_shifts=beta_shifts)

    record = run_gate(_args(tmp_path))

    assert record["estimator"]["name"] == "parameter_conditioned_periodic_shift"
    assert record["estimator"]["fit_uses"] == "train_split beta parameters and trajectories only"
    assert record["fit"]["coefficients"]["slope"] == pytest.approx(10.0, abs=1e-6)
    assert record["fit"]["coefficients"]["intercept"] == pytest.approx(0.0, abs=1e-6)
    assert record["fit"]["selected_validation"]["nrmse"] < 1e-5
    assert record["validation_guard"]["passed"] is True
    assert record["test_eligible"] is True
    assert record["test"] is None


def test_parameter_conditioned_gate_measures_test_once_after_validation_passes(tmp_path):
    beta_shifts = [(0.1, 1.0), (0.2, 2.0), (0.4, 4.0)]
    _write_beta_shifted_split(tmp_path, split="train", beta_shifts=beta_shifts)
    _write_beta_shifted_split(tmp_path, split="val", beta_shifts=beta_shifts)
    _write_beta_shifted_split(tmp_path, split="test", beta_shifts=beta_shifts)
    ledger_path = tmp_path / "parameter-test-ledger.json"
    args = _args(tmp_path)
    args.test_split = "test"
    args.test_ledger_json = str(ledger_path)

    record = run_gate(args)

    assert record["test_eligible"] is True
    assert record["test"]["selected_test"]["nrmse"] < 1e-5
    assert record["held_out_test_policy"]["recorded"] is True
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert len(ledger["measurements"]) == 1

    with pytest.raises(RuntimeError, match="held-out test measurement already recorded"):
        run_gate(args)
