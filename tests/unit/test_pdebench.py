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


def test_pdebench_dataset_respects_max_samples_across_shards(tmp_path):
    first = torch.arange(3 * 4 * 4, dtype=torch.float32).view(3, 4, 4)
    second = torch.arange(4 * 4 * 4, dtype=torch.float32).view(4, 4, 4) + 1000.0
    with h5py.File(tmp_path / "burgers1d_train_000.h5", "w") as f:
        f.create_dataset("data", data=first.numpy())
        f.create_dataset("beta", data=torch.zeros(3, 1).numpy())
    with h5py.File(tmp_path / "burgers1d_train_001.h5", "w") as f:
        f.create_dataset("data", data=second.numpy())
        f.create_dataset("beta", data=torch.ones(4, 1).numpy())

    cfg = PDEBenchConfig(
        task="burgers1d",
        split="train",
        root=tmp_path,
        normalize=False,
        param_keys=("beta",),
        max_samples=5,
    )
    ds = PDEBenchDataset(cfg)

    assert len(ds) == 5
    assert torch.equal(ds.fields[:3], first)
    assert torch.equal(ds.fields[3:], second[:2])
    assert torch.equal(ds.params["beta"].squeeze(-1), torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0]))


def test_pdebench_dataset_rejects_nonpositive_max_samples(tmp_path):
    cfg = PDEBenchConfig(task="burgers1d", split="train", root=tmp_path, max_samples=0)

    try:
        PDEBenchDataset(cfg)
    except ValueError as exc:
        assert "max_samples" in str(exc)
    else:
        raise AssertionError("expected max_samples validation error")


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
