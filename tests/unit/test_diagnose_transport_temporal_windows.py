from __future__ import annotations

from argparse import Namespace

import h5py
import torch

from scripts.diagnose_transport_temporal_windows import diagnose


def _write_piecewise_split(
    root, *, split: str, first_shift: int, second_shift: int, samples: int = 2
) -> None:
    data = torch.zeros(samples, 7, 16, 1)
    for sample_idx in range(samples):
        data[sample_idx, 0, sample_idx, 0] = 1.0
        for step in range(1, data.shape[1]):
            shift = first_shift if step <= 3 else second_shift
            data[sample_idx, step, :, 0] = torch.roll(
                data[sample_idx, step - 1, :, 0], shifts=shift, dims=-1
            )
    with h5py.File(root / f"advection1d_{split}.h5", "w") as handle:
        handle.create_dataset("data", data=data.numpy())


def _args(tmp_path):
    return Namespace(
        data_root=str(tmp_path),
        task="advection1d",
        train_split="train",
        val_split="val",
        max_samples=2,
        val_max_samples=None,
        rollout_steps=2,
        start_step=0,
        stride=3,
        max_windows=None,
        shift=[0, 1, 2],
        metric="nrmse",
        top_k=2,
        output_json=str(tmp_path / "diagnostic.json"),
    )


def test_temporal_window_diagnostic_finds_common_later_shift(tmp_path):
    _write_piecewise_split(tmp_path, split="train", first_shift=0, second_shift=2)
    _write_piecewise_split(tmp_path, split="val", first_shift=1, second_shift=2)

    record = diagnose(_args(tmp_path))

    assert record["train_shift_histogram"] == {"0": 1, "2": 1}
    assert record["val_shift_histogram"] == {"1": 1, "2": 1}
    assert record["common_temporal_best_shifts"] == [2]
    assert record["conclusion"] == "temporal_train_val_shift_support_found"
    assert (
        record["notes"][1]
        == "Scans temporal start windows; does not read or evaluate held-out test."
    )


def test_temporal_window_diagnostic_reports_no_common_support(tmp_path):
    _write_piecewise_split(tmp_path, split="train", first_shift=0, second_shift=0)
    _write_piecewise_split(tmp_path, split="val", first_shift=1, second_shift=1)

    record = diagnose(_args(tmp_path))

    assert record["common_temporal_best_shifts"] == []
    assert record["conclusion"] == "blocked_no_temporal_common_shift"
