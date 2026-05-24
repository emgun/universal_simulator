from __future__ import annotations

from argparse import Namespace

import h5py
import torch

from scripts.diagnose_train_only_transport_features import diagnose


def _write_split(root, *, split: str, shifts: list[int], width: int = 16, steps: int = 5) -> None:
    data = torch.zeros(len(shifts), steps, width, 1)
    for sample_idx, shift in enumerate(shifts):
        data[sample_idx, 0, sample_idx % width, 0] = 1.0
        for step in range(1, steps):
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
        max_samples=4,
        val_max_samples=None,
        rollout_steps=4,
        shift=[0, 1, 2],
        metric="nrmse",
        output_json=str(tmp_path / "diagnostic.json"),
    )


def test_train_only_feature_diagnostic_flags_unsupported_validation_shift(tmp_path):
    _write_split(tmp_path, split="train", shifts=[0, 0, 1, 1])
    _write_split(tmp_path, split="val", shifts=[2, 2])

    record = diagnose(_args(tmp_path))

    assert record["train_shift_histogram"] == {"0": 2, "1": 2}
    assert record["val_shift_histogram"] == {"2": 2}
    assert record["val_best_margin_summary"]["min"] >= 0.0
    assert record["unsupported_val_shifts"] == [2]
    assert record["conclusion"] == "blocked_no_train_support_for_validation_shift"


def test_train_only_feature_diagnostic_uses_no_test_split(tmp_path):
    _write_split(tmp_path, split="train", shifts=[0, 0])
    _write_split(tmp_path, split="val", shifts=[0, 0])

    record = diagnose(_args(tmp_path))

    assert record["notes"][-1] == "Does not read or evaluate the held-out test split."
    assert record["unsupported_val_shifts"] == []


def test_train_only_feature_diagnostic_allows_distinct_val_cap(tmp_path):
    _write_split(tmp_path, split="train", shifts=[0, 0])
    _write_split(tmp_path, split="val", shifts=[0, 0, 0, 0])
    args = _args(tmp_path)
    args.max_samples = 2
    args.val_max_samples = 3

    record = diagnose(args)

    assert record["max_samples"] == 2
    assert record["val_max_samples"] == 3
    assert record["val_shift_histogram"] == {"0": 3}


def test_train_only_feature_diagnostic_supports_full_split_caps(tmp_path):
    _write_split(tmp_path, split="train", shifts=[0, 0, 0])
    _write_split(tmp_path, split="val", shifts=[0, 0])
    args = _args(tmp_path)
    args.max_samples = -1
    args.val_max_samples = -1

    record = diagnose(args)

    assert record["max_samples"] is None
    assert record["val_max_samples"] is None
    assert record["train_shift_histogram"] == {"0": 3}
    assert record["val_shift_histogram"] == {"0": 2}
