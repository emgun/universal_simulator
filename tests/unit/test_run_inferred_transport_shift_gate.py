from __future__ import annotations

import json
from argparse import Namespace

import h5py
import numpy as np
import pytest

from scripts.run_inferred_transport_shift_gate import run_gate
from scripts.run_source_conditioned_transport_shift_gate import _periodic_shift


def _write_context_shifted_split(
    root,
    *,
    split: str,
    warmup_shift: float,
    rollout_shift: float,
    samples: int = 3,
    steps: int = 6,
    width: int = 32,
) -> None:
    rows = []
    x = np.arange(width, dtype=np.float32)
    for sample_idx in range(samples):
        trajectory = np.zeros((steps, width), dtype=np.float32)
        trajectory[0] = np.exp(-0.5 * ((x - (8.0 + sample_idx)) / 2.0) ** 2)
        trajectory[1] = _periodic_shift(trajectory[0:1], warmup_shift)[0]
        for step_idx in range(2, steps):
            trajectory[step_idx] = _periodic_shift(
                trajectory[step_idx - 1 : step_idx], rollout_shift
            )[0]
        rows.append(trajectory)
    with h5py.File(root / f"advection1d_{split}.h5", "w") as handle:
        handle.create_dataset("data", data=np.asarray(rows, dtype=np.float32)[..., None])


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
        context_transitions=1,
        rollout_steps=4,
        shift=[0, 1, 2],
        metric="nrmse",
        fit_kind="linear",
        fit_intercept=False,
        refine_radius=0,
        fractional_refine_step=0.0,
        reference_metric_value=reference,
        val_min_relative_improvement=0.0,
        test_ledger_json=None,
        allow_repeat_test=False,
    )


def test_inferred_gate_calibrates_rollout_shift_from_context_without_metadata(tmp_path):
    _write_context_shifted_split(tmp_path, split="train", warmup_shift=1, rollout_shift=2)
    _write_context_shifted_split(tmp_path, split="val", warmup_shift=1, rollout_shift=2)

    record = run_gate(_args(tmp_path))

    assert record["estimator"]["name"] == "inferred_context_periodic_shift"
    assert record["estimator"]["fit_uses"] == "train_split context-inferred shifts only"
    assert record["fit"]["coefficients"]["slope"] == pytest.approx(2.0)
    assert record["fit"]["coefficients"]["intercept"] == 0.0
    assert record["fit"]["selected_validation"]["nrmse"] < 1e-5
    assert record["validation_guard"]["passed"] is True
    assert record["test_eligible"] is True
    assert record["test"] is None


def test_inferred_gate_measures_test_once_after_validation_passes(tmp_path):
    _write_context_shifted_split(tmp_path, split="train", warmup_shift=1, rollout_shift=2)
    _write_context_shifted_split(tmp_path, split="val", warmup_shift=1, rollout_shift=2)
    _write_context_shifted_split(tmp_path, split="test", warmup_shift=1, rollout_shift=2)
    ledger_path = tmp_path / "inferred-test-ledger.json"
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
