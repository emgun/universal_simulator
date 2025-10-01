from __future__ import annotations

import h5py
import torch

from scripts import train as train_script


def _write_minimal_pdebench(tmp_path) -> None:
    data = torch.randn(2, 3, 4, dtype=torch.float32)
    file_path = tmp_path / "burgers1d_train.h5"
    with h5py.File(file_path, "w") as handle:
        handle.create_dataset("data", data=data.numpy())


def test_train_operator_runs_with_pdebench_loader(tmp_path):
    _write_minimal_pdebench(tmp_path)
    cfg = {
        "training": {"batch_size": 2, "dt": 0.1},
        "latent": {"dim": 8, "tokens": 4},
        "stages": {"operator": {"epochs": 1}},
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
    ckpt_path = tmp_path / "ckpts" / "operator.pt"
    assert ckpt_path.exists()


def test_train_diffusion_runs_with_pdebench_loader(tmp_path):
    _write_minimal_pdebench(tmp_path)
    cfg = {
        "training": {"batch_size": 2, "dt": 0.1},
        "latent": {"dim": 8, "tokens": 4},
        "stages": {"diff_residual": {"epochs": 1}},
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
    train_script.train_diffusion(cfg)
    ckpt_path = tmp_path / "ckpts" / "diffusion_residual.pt"
    assert ckpt_path.exists()
