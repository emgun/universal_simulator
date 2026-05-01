import h5py
import torch

from ups.data.pdebench import PDEBenchConfig, PDEBenchDataset, pdebench_equation_signature, pdebench_task_semantics
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


def test_pdebench_task_semantics_exposes_family_and_traits():
    semantics = pdebench_task_semantics(
        "burgers1d",
        task_vocab=("burgers1d", "darcy2d"),
    )

    assert semantics["task_id"].shape == (2,)
    assert semantics["task_family"].shape == (2,)
    assert semantics["equation_traits"].shape[0] >= 4
    assert semantics["task_id"].sum().item() == 1.0
    assert semantics["task_family"].sum().item() == 1.0


def test_pdebench_equation_signature_is_nonempty_and_stable():
    signature = pdebench_equation_signature("burgers1d")

    assert signature.dim() == 1
    assert signature.numel() > 4
    assert signature.sum().item() >= 2.0
