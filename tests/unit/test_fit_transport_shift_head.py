from __future__ import annotations

import argparse

import h5py
import pytest
import torch

from scripts.fit_transport_shift_head import fit_and_validate


def _write_shifted_split(path, *, split: str, shift: int) -> None:
    base = torch.zeros(2, 4, 8, 1, dtype=torch.float32)
    base[:, 0, 1, 0] = 1.0
    for step in range(1, base.shape[1]):
        base[:, step, :, 0] = torch.roll(base[:, step - 1, :, 0], shifts=shift, dims=-1)
    with h5py.File(path / f"advection1d_{split}.h5", "w") as handle:
        handle.create_dataset("data", data=base.numpy())


def _args(tmp_path, **overrides):
    values = {
        "data_root": str(tmp_path),
        "task": "advection1d",
        "train_split": "train",
        "val_split": "val",
        "max_samples": 2,
        "rollout_steps": 3,
        "shift": [-1, 0, 1, 2],
        "metric": "nrmse",
        "kind": "task",
        "key": "advection1d",
        "reference_metric_value": None,
        "val_min_relative_improvement": None,
        "allow_same_split_smoke": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_fit_transport_shift_head_uses_train_shift_and_validates(tmp_path):
    _write_shifted_split(tmp_path, split="train", shift=1)
    _write_shifted_split(tmp_path, split="val", shift=1)

    record = fit_and_validate(_args(tmp_path))

    assert record["selected_train_shift"] == 1
    assert record["selected_override"] == 'evaluation.decoded_roll_shift_by_task={"advection1d":1}'
    assert record["selected_validation"]["nrmse"] == 0.0
    assert record["oracle_validation"]["shift"] == 1


def test_fit_transport_shift_head_reports_train_val_mismatch(tmp_path):
    _write_shifted_split(tmp_path, split="train", shift=1)
    _write_shifted_split(tmp_path, split="val", shift=2)

    record = fit_and_validate(_args(tmp_path))

    assert record["selected_train_shift"] == 1
    assert record["oracle_validation"]["shift"] == 2
    assert record["selected_validation"]["nrmse"] > record["oracle_validation"]["nrmse"]


def test_fit_transport_shift_head_refuses_same_split_without_smoke_flag(tmp_path):
    _write_shifted_split(tmp_path, split="train", shift=1)

    with pytest.raises(ValueError, match="must differ"):
        fit_and_validate(_args(tmp_path, val_split="train"))
