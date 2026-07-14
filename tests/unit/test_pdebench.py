import hashlib
import pickle

import h5py
import pytest
import torch

from ups.data.manifests import (
    ProtocolManifest,
    SourceManifest,
    resolve_data_lock,
    write_data_lock,
)
from ups.data.normalization import NormalizationStats
from ups.data.pdebench import (
    PDEBenchConfig,
    PDEBenchDataset,
    pdebench_equation_signature,
    pdebench_task_semantics,
)
from ups.eval.pdebench_runner import evaluate_pdebench


def _write_data_lock(tmp_path, file_path, *, role="train"):
    digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
    source = SourceManifest.from_dict(
        {
            "schema_version": 1,
            "dataset_id": "pdebench",
            "provider": "fixture",
            "revision": "sha256:0123456789abcdef",
            "native_format": "hdf5",
            "license": "CC-BY-4.0",
            "citation": "fixture",
            "objects": [
                {
                    "object_id": f"burgers-{role}",
                    "path": file_path.name,
                    "size_bytes": file_path.stat().st_size,
                    "checksums": {"sha256": digest},
                    "uris": [file_path.as_uri()],
                    "declared_roles": [role],
                }
            ],
        }
    )
    protocol = ProtocolManifest.from_dict(
        {
            "schema_version": 1,
            "protocol_id": "fixture-v1",
            "dataset_id": "pdebench",
            "source_revision": source.revision,
            "adapter": "pdebench_hdf5",
            "adapter_revision": "1.0.0",
            "split_authority": "fixture",
            "splits": {role: [f"burgers-{role}"]},
            "identity_fields": ["source_file_id", "source_sample_index"],
            "selection": {"algorithm": "identity_hash", "seed": 0},
            "normalization": {"fit_role": "train", "method": "zscore"},
            "test_access": "measurement_contract_required",
        }
    )
    purpose = "measurement" if role == "test" else "training"
    lock = resolve_data_lock(
        source,
        protocol,
        requested_roles=(role,),
        purpose=purpose,
        measurement_contract_id="fixture-heldout" if role == "test" else None,
    )
    lock_path = tmp_path / f"{role}.lock.json"
    write_data_lock(lock_path, lock)
    return lock_path, lock


def test_pdebench_dataset_hdf5(tmp_path):
    path = tmp_path / "burgers1d_train.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=torch.randn(5, 4, 4).numpy())
    cfg = PDEBenchConfig(task="burgers1d", split="train", root=tmp_path)
    ds = PDEBenchDataset(cfg)
    assert ds._handles == {}
    sample = ds[0]
    assert "fields" in sample
    assert len(ds._handles) == 1


def test_pdebench_darcy_loads_official_coefficient_to_solution_contract(tmp_path):
    coefficient = torch.randint(0, 2, (2, 1, 128, 128, 1), dtype=torch.int64).float()
    solution = torch.randn(2, 1, 128, 128, 1)
    with h5py.File(tmp_path / "darcy2d_train.h5", "w") as handle:
        handle.create_dataset("data", data=coefficient.numpy())
        handle.create_dataset("targets", data=solution.numpy())

    dataset = PDEBenchDataset(PDEBenchConfig(task="darcy2d", root=tmp_path))
    sample = dataset[1]

    assert torch.equal(sample["fields"], coefficient[1])
    assert torch.equal(sample["targets"], solution[1])
    assert sample["fields"].shape == (1, 128, 128, 1)


def test_pdebench_darcy_rejects_solution_only_or_coefficient_only_legacy_shard(tmp_path):
    with h5py.File(tmp_path / "darcy2d_train.h5", "w") as handle:
        handle.create_dataset("data", data=torch.zeros(2, 1, 8, 8, 1).numpy())

    with pytest.raises(ValueError, match="missing required target dataset"):
        PDEBenchDataset(PDEBenchConfig(task="darcy2d", root=tmp_path))


def test_pdebench_dataset_pickle_drops_process_local_handles(tmp_path):
    path = tmp_path / "burgers1d_train.h5"
    expected = torch.arange(24, dtype=torch.float32).view(2, 3, 4)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=expected.numpy())
    dataset = PDEBenchDataset(PDEBenchConfig(task="burgers1d", root=tmp_path))
    assert torch.equal(dataset[0]["fields"], expected[0])
    assert dataset._handles

    restored = pickle.loads(pickle.dumps(dataset))

    assert restored._handles == {}
    assert torch.equal(restored[1]["fields"], expected[1])


def test_pdebench_dataset_requires_and_applies_bound_normalization(tmp_path):
    path = tmp_path / "burgers1d_train.h5"
    values = torch.tensor([[[1.0, 12.0], [3.0, 16.0]]])
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=values.numpy())
    stats_path = tmp_path / "stats.json"
    NormalizationStats(
        mean=(2.0, 14.0),
        std=(1.0, 2.0),
        count=2,
        data_lock_sha256="lock",
        selection_sha256="selection",
    ).save(stats_path)

    dataset = PDEBenchDataset(
        PDEBenchConfig(
            task="burgers1d",
            root=tmp_path,
            normalize=True,
            normalization_path=str(stats_path),
            data_lock_sha256="lock",
            selection_sha256="selection",
        )
    )

    assert torch.equal(dataset[0]["fields"], torch.tensor([[-1.0, -1.0], [1.0, 1.0]]))


def test_pdebench_dataset_rejects_unbound_normalization(tmp_path):
    with h5py.File(tmp_path / "burgers1d_train.h5", "w") as handle:
        handle.create_dataset("data", data=torch.zeros(1, 2, 2).numpy())

    try:
        PDEBenchDataset(PDEBenchConfig(task="burgers1d", root=tmp_path, normalize=True))
    except ValueError as exc:
        assert "normalization_path" in str(exc)
    else:
        raise AssertionError("expected unbound normalization to fail closed")


def test_pdebench_dataset_lock_authorizes_role_and_exact_file(tmp_path):
    path = tmp_path / "burgers1d_train.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=torch.zeros(1, 2, 2).numpy())
    lock_path, _ = _write_data_lock(tmp_path, path)

    dataset = PDEBenchDataset(
        PDEBenchConfig(task="burgers1d", root=tmp_path, data_lock_path=str(lock_path))
    )
    assert len(dataset) == 1

    with h5py.File(tmp_path / "burgers1d_test.h5", "w") as handle:
        handle.create_dataset("data", data=torch.zeros(1, 2, 2).numpy())
    try:
        PDEBenchDataset(
            PDEBenchConfig(
                task="burgers1d",
                split="test",
                root=tmp_path,
                data_lock_path=str(lock_path),
            )
        )
    except PermissionError as exc:
        assert "does not authorize" in str(exc)
    else:
        raise AssertionError("expected training lock to deny test data")


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


def test_pdebench_dataset_derives_advection_beta_from_source_paths(tmp_path):
    data = torch.randn(3, 4, 8, dtype=torch.float32)
    with h5py.File(tmp_path / "advection1d_val.h5", "w") as f:
        f.create_dataset("data", data=data.numpy())
        f.create_dataset("source_file_index", data=torch.tensor([0, 1, 1]).numpy())
        f.attrs["source_paths"] = [
            "1D/Advection/Train/1D_Advection_Sols_beta0.1.hdf5",
            "1D/Advection/Train/1D_Advection_Sols_beta0.7.hdf5",
        ]

    cfg = PDEBenchConfig(
        task="advection1d",
        split="val",
        root=tmp_path,
        param_keys=("beta",),
        normalize=False,
    )
    ds = PDEBenchDataset(cfg)

    assert torch.allclose(ds.params["beta"].squeeze(-1), torch.tensor([0.1, 0.7, 0.7]))


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


def test_evaluate_pdebench_darcy_scores_coefficient_against_solution(tmp_path):
    with h5py.File(tmp_path / "darcy2d_val.h5", "w") as handle:
        handle.create_dataset("data", data=torch.zeros(2, 1, 4, 4, 1).numpy())
        handle.create_dataset("targets", data=torch.ones(2, 1, 4, 4, 1).numpy())

    report = evaluate_pdebench("darcy2d", "val", root=tmp_path)

    assert report.metrics["mae"] == 1.0


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
