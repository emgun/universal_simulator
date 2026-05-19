from __future__ import annotations

from argparse import Namespace

import h5py
import numpy as np

from scripts.audit_train_only_transport_identifiability import audit_identifiability


def _write_split(root, *, split: str, shift: int, samples: int = 2, steps: int = 5, width: int = 8) -> None:
    path = root / f"advection1d_{split}.h5"
    base = np.zeros((samples, width), dtype=np.float32)
    base[:, 1] = 1.0
    data = []
    for sample_idx in range(samples):
        trajectory = [np.roll(base[sample_idx], shift * step_idx) for step_idx in range(steps)]
        data.append(trajectory)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=np.asarray(data, dtype=np.float32)[..., None])


def _args(tmp_path):
    return Namespace(
        data_root=tmp_path,
        task="advection1d",
        train_split="train",
        val_split="val",
        max_samples=2,
        val_max_samples=None,
        rollout_steps=3,
        shift=[0, 1, 2],
        metric="nrmse",
        top_k=2,
        output_json=str(tmp_path / "audit.json"),
    )


def test_identifiability_audit_blocks_single_train_regime_with_unseen_validation_shift(tmp_path):
    _write_split(tmp_path, split="train", shift=0)
    _write_split(tmp_path, split="val", shift=2)

    record = audit_identifiability(_args(tmp_path))

    assert record["status"] == "blocked_underidentified_train_only_shift"
    assert record["train_shift_histogram"] == {"0": 2}
    assert record["val_shift_histogram"] == {"2": 2}
    assert record["unsupported_val_shifts"] == [2]
    assert record["val_requires_unseen_regime"] is True


def test_identifiability_audit_allows_supported_validation_shift(tmp_path):
    _write_split(tmp_path, split="train", shift=1)
    _write_split(tmp_path, split="val", shift=1)

    record = audit_identifiability(_args(tmp_path))

    assert record["status"] == "train_shift_support_covers_validation"
    assert record["blockers"] == []
    assert record["unsupported_val_shifts"] == []
