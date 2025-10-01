from __future__ import annotations

import json

import h5py
import torch
from scripts import train as train_script


def _write_minimal_hdf5(tmp_path) -> None:
    data = torch.randn(2, 3, 4, dtype=torch.float32)
    file_path = tmp_path / "burgers1d_train.h5"
    with h5py.File(file_path, "w") as handle:
        handle.create_dataset("data", data=data.numpy())


def test_training_logs_written_and_early_stop(tmp_path):
    _write_minimal_hdf5(tmp_path)
    log_path = tmp_path / "training_log.jsonl"
    cfg = {
        "seed": 123,
        "training": {
            "batch_size": 2,
            "dt": 0.1,
            "patience": 0,
            "log_path": str(log_path),
        },
        "latent": {"dim": 8, "tokens": 4},
        "stages": {
            "operator": {"epochs": 3},
        },
        "optimizer": {"lr": 1e-3},
        "data": {
            "task": "burgers1d",
            "split": "train",
            "root": str(tmp_path),
            "patch_size": 1,
        },
        "checkpoint": {"dir": str(tmp_path / "ckpts")},
    }

    train_script.train_operator(cfg)

    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines
    entries = [json.loads(line) for line in lines]
    loss_entries = [entry for entry in entries if "loss" in entry]
    assert loss_entries
    first_entry = loss_entries[0]
    assert first_entry["stage"] == "operator"
