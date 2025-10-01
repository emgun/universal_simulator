import h5py
import torch

from ups.data.pdebench import PDEBenchConfig, PDEBenchDataset
from ups.eval.pdebench_runner import evaluate_pdebench


def test_pdebench_dataset_hdf5(tmp_path):
    path = tmp_path / "burgers1d_train.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=torch.randn(5, 4, 4).numpy())
    cfg = PDEBenchConfig(task="burgers1d", split="train", root=tmp_path)
    ds = PDEBenchDataset(cfg)
    sample = ds[0]
    assert "fields" in sample


def test_evaluate_pdebench(tmp_path):
    path = tmp_path / "burgers1d_val.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=torch.randn(3, 4, 4).numpy())
    report = evaluate_pdebench("burgers1d", "val", root=tmp_path)
    assert "mae" in report.metrics
