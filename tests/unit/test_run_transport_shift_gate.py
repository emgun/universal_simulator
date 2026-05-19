from __future__ import annotations

import argparse

import h5py
import torch

from scripts.run_transport_shift_gate import run_gate


def _write_shifted_split(path, *, split: str, shift: int) -> None:
    data = torch.zeros(2, 4, 8, 1, dtype=torch.float32)
    data[:, 0, 1, 0] = 1.0
    for step in range(1, data.shape[1]):
        data[:, step, :, 0] = torch.roll(data[:, step - 1, :, 0], shifts=shift, dims=-1)
    with h5py.File(path / f"advection1d_{split}.h5", "w") as handle:
        handle.create_dataset("data", data=data.numpy())


def _args(tmp_path, *, reference: float = 1.0):
    return argparse.Namespace(
        data_root=str(tmp_path),
        task="advection1d",
        train_split="train",
        val_split="val",
        test_split="",
        max_samples=2,
        test_max_samples=None,
        rollout_steps=3,
        shift=[0, 1, 2],
        metric="nrmse",
        kind="task",
        key="advection1d",
        reference_metric_value=reference,
        val_min_relative_improvement=0.0,
        top_k=2,
    )


def test_transport_shift_gate_allows_test_when_train_val_match_and_guard_passes(tmp_path):
    _write_shifted_split(tmp_path, split="train", shift=1)
    _write_shifted_split(tmp_path, split="val", shift=1)

    record = run_gate(_args(tmp_path, reference=1.0))

    assert record["test_eligible"] is True
    assert record["blockers"] == []
    assert record["fit"]["selected_train_shift"] == 1
    assert record["data_sources"]["train"]["exists"] is True
    assert len(record["data_sources"]["train"]["sha256"]) == 64
    assert record["data_sources"]["val"]["bytes"] > 0


def test_transport_shift_gate_blocks_split_mismatch(tmp_path):
    _write_shifted_split(tmp_path, split="train", shift=0)
    _write_shifted_split(tmp_path, split="val", shift=2)

    record = run_gate(_args(tmp_path, reference=1.0))

    assert record["test_eligible"] is False
    assert any("best transport shifts differ" in blocker for blocker in record["blockers"])


def test_transport_shift_gate_blocks_failed_validation_guard(tmp_path):
    _write_shifted_split(tmp_path, split="train", shift=1)
    _write_shifted_split(tmp_path, split="val", shift=1)

    record = run_gate(_args(tmp_path, reference=-1.0))

    assert record["test_eligible"] is False
    assert any("SOTA guard" in blocker for blocker in record["blockers"])


def test_transport_shift_gate_measures_test_only_after_gate_passes(tmp_path):
    _write_shifted_split(tmp_path, split="train", shift=1)
    _write_shifted_split(tmp_path, split="val", shift=1)
    _write_shifted_split(tmp_path, split="test", shift=1)

    args = _args(tmp_path, reference=1.0)
    args.test_split = "test"
    record = run_gate(args)

    assert record["test_eligible"] is True
    assert record["test"]["selected_shift"] == 1
    assert record["test"]["selected_test"]["nrmse"] == 0.0
    assert record["data_sources"]["test"]["exists"] is True
    assert record["next_action"] == "held-out test measured"
