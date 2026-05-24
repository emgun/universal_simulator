from __future__ import annotations

import argparse

import h5py
import torch

from scripts.diagnose_transport_shift_splits import diagnose


def _write_shifted_split(path, *, split: str, shift: int) -> None:
    data = torch.zeros(2, 4, 8, 1, dtype=torch.float32)
    data[:, 0, 1, 0] = 1.0
    for step in range(1, data.shape[1]):
        data[:, step, :, 0] = torch.roll(data[:, step - 1, :, 0], shifts=shift, dims=-1)
    with h5py.File(path / f"advection1d_{split}.h5", "w") as handle:
        handle.create_dataset("data", data=data.numpy())


def test_diagnose_transport_shift_splits_reports_consistent_regime(tmp_path):
    _write_shifted_split(tmp_path, split="train", shift=1)
    _write_shifted_split(tmp_path, split="val", shift=1)

    record = diagnose(
        argparse.Namespace(
            data_root=str(tmp_path),
            task="advection1d",
            splits="train,val",
            max_samples=2,
            rollout_steps=3,
            shift=[0, 1, 2],
            top_k=2,
        )
    )

    assert record["best_shifts"] == {"train": 1, "val": 1}
    assert record["consistent_best_shift"] is True


def test_diagnose_transport_shift_splits_reports_split_mismatch(tmp_path):
    _write_shifted_split(tmp_path, split="train", shift=0)
    _write_shifted_split(tmp_path, split="val", shift=2)

    record = diagnose(
        argparse.Namespace(
            data_root=str(tmp_path),
            task="advection1d",
            splits="train,val",
            max_samples=2,
            rollout_steps=3,
            shift=[0, 1, 2],
            top_k=2,
        )
    )

    assert record["best_shifts"] == {"train": 0, "val": 2}
    assert record["consistent_best_shift"] is False
