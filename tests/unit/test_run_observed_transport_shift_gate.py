from __future__ import annotations

import json
from argparse import Namespace

import h5py
import numpy as np
import pytest

from scripts.run_observed_transport_shift_gate import run_gate


def _write_split(
    root, *, split: str, shift: int, samples: int = 2, steps: int = 5, width: int = 8
) -> None:
    path = root / f"advection1d_{split}.h5"
    base = np.zeros((samples, width), dtype=np.float32)
    base[:, 1] = 1.0
    data = []
    for sample_idx in range(samples):
        trajectory = [np.roll(base[sample_idx], shift * step_idx) for step_idx in range(steps)]
        data.append(trajectory)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=np.asarray(data, dtype=np.float32)[..., None])


def _args(tmp_path, *, reference: float = 1.0):
    return Namespace(
        data_root=tmp_path,
        task="advection1d",
        train_split="train",
        val_split="val",
        test_split="",
        max_samples=2,
        train_max_samples=None,
        val_max_samples=None,
        test_max_samples=None,
        rollout_steps=3,
        shift=[-1, 0, 1, 2],
        metric="nrmse",
        reference_metric_value=reference,
        val_min_relative_improvement=0.0,
        test_ledger_json=None,
        allow_repeat_test=False,
    )


def test_observed_transport_gate_measures_test_after_validation_passes(tmp_path):
    _write_split(tmp_path, split="train", shift=1)
    _write_split(tmp_path, split="val", shift=2)
    _write_split(tmp_path, split="test", shift=-1)

    args = _args(tmp_path, reference=1.0)
    args.test_split = "test"
    record = run_gate(args)

    assert record["validation_guard"]["passed"] is True
    assert record["test_eligible"] is True
    assert record["train"]["shift_mean"] == 1.0
    assert record["validation"]["shift_mean"] == 2.0
    assert record["test"]["shift_mean"] == -1.0
    assert record["test"]["nrmse"] == 0.0
    assert record["data_sources"]["train"]["exists"] is True
    assert record["data_sources"]["val"]["sha256"]
    assert record["data_sources"]["test"]["bytes"] > 0
    assert record["next_action"] == "held-out test measured"


def test_observed_transport_gate_records_and_blocks_repeat_test_measurement(tmp_path):
    _write_split(tmp_path, split="train", shift=1)
    _write_split(tmp_path, split="val", shift=2)
    _write_split(tmp_path, split="test", shift=-1)

    ledger_path = tmp_path / "test-ledger.json"
    args = _args(tmp_path, reference=1.0)
    args.test_split = "test"
    args.test_ledger_json = str(ledger_path)

    record = run_gate(args)

    assert record["held_out_test_policy"]["recorded"] is True
    assert record["held_out_test_policy"]["measurement_key"]
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert len(ledger["measurements"]) == 1

    with pytest.raises(RuntimeError, match="held-out test measurement already recorded"):
        run_gate(args)


def test_observed_transport_gate_can_explicitly_allow_repeat_test_measurement(tmp_path):
    _write_split(tmp_path, split="train", shift=1)
    _write_split(tmp_path, split="val", shift=2)
    _write_split(tmp_path, split="test", shift=-1)

    args = _args(tmp_path, reference=1.0)
    args.test_split = "test"
    args.test_ledger_json = str(tmp_path / "test-ledger.json")
    run_gate(args)

    args.allow_repeat_test = True
    repeat = run_gate(args)

    assert repeat["test"]["nrmse"] == 0.0
    assert repeat["held_out_test_policy"]["allow_repeat_test"] is True
    ledger = json.loads((tmp_path / "test-ledger.json").read_text(encoding="utf-8"))
    assert len(ledger["measurements"]) == 1


def test_observed_transport_gate_blocks_failed_validation_guard(tmp_path):
    _write_split(tmp_path, split="train", shift=1)
    _write_split(tmp_path, split="val", shift=1)

    args = _args(tmp_path, reference=0.1)
    args.shift = [0]
    record = run_gate(args)

    assert record["validation_guard"]["passed"] is False
    assert record["test_eligible"] is False
    assert record["test"] is None
