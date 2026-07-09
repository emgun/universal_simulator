from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from scripts.audit_split_integrity import audit_task, main


def _write(path, arrays):
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=np.stack(arrays).astype(np.float32))


def _traj(seed, steps=4, width=16):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(steps, width, 1))


def test_audit_detects_shared_initial_conditions(tmp_path):
    shared = _traj(1)
    _write(tmp_path / "burgers1d_train.h5", [shared, _traj(2), _traj(3)])
    perturbed = shared.copy()
    perturbed[1:] += 1e-4  # same IC, diverged later frames
    _write(tmp_path / "burgers1d_val.h5", [perturbed])

    record = audit_task(tmp_path, "burgers1d", ["train", "val"])

    assert record["overlaps"]["train<->val"] == 1
    assert record["split_sizes"] == {"train": 3, "val": 1}


def test_audit_reports_clean_disjoint_splits(tmp_path):
    _write(tmp_path / "darcy2d_train.h5", [_traj(10, steps=8), _traj(11, steps=8)])
    _write(tmp_path / "darcy2d_val.h5", [_traj(12, steps=8)])

    record = audit_task(tmp_path, "darcy2d", ["train", "val"])

    assert record["overlaps"]["train<->val"] == 0


def test_cli_refuses_test_without_flag(tmp_path):
    with pytest.raises(SystemExit, match="--include-test"):
        main(["--root", str(tmp_path), "--splits", "train", "test"])


def test_cli_writes_honest_record(tmp_path):
    _write(tmp_path / "burgers1d_train.h5", [_traj(1)])
    _write(tmp_path / "burgers1d_val.h5", [_traj(2)])
    out = tmp_path / "out.json"

    assert (
        main(
            [
                "--root",
                str(tmp_path),
                "--tasks",
                "burgers1d",
                "--splits",
                "train",
                "val",
                "--output-json",
                str(out),
            ]
        )
        == 0
    )

    record = json.loads(out.read_text())
    assert record["candidate_scored"] is False
    assert record["held_out_test_data_read"] is False
    assert record["tasks"]["burgers1d"]["overlaps"]["train<->val"] == 0
