from __future__ import annotations

import hashlib
from pathlib import Path

import h5py
import torch
import yaml
from torch.utils.data import DataLoader

from ups.data.cli import verify_lock_cache
from ups.data.manifests import (
    canonical_sha256,
    load_data_lock,
    load_protocol_manifest,
    load_source_manifest,
    resolve_data_lock,
    write_data_lock,
)
from ups.data.normalization import NormalizationStats, fit_normalization_stats
from ups.data.pdebench import PDEBenchConfig, PDEBenchDataset
from ups.data.staging import plan_staging, stage_objects, staging_objects_from_lock


def test_locked_data_pipeline_from_manifest_to_worker_read(tmp_path: Path, monkeypatch) -> None:
    canonical_dir = tmp_path / "canonical"
    canonical_dir.mkdir()
    canonical_path = canonical_dir / "burgers1d_train.h5"
    fields = torch.tensor(
        [
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
            [[4.0, 40.0], [5.0, 50.0], [6.0, 60.0]],
            [[7.0, 70.0], [8.0, 80.0], [9.0, 90.0]],
            [[10.0, 100.0], [11.0, 110.0], [12.0, 120.0]],
        ],
        dtype=torch.float32,
    )
    with h5py.File(canonical_path, "w") as handle:
        handle.create_dataset("data", data=fields.numpy(), chunks=(1, 3, 2))

    source_bytes = canonical_path.read_bytes()
    source_digest = hashlib.sha256(source_bytes).hexdigest()
    source_path = tmp_path / "source.yaml"
    protocol_path = tmp_path / "protocol.yaml"
    source_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "dataset_id": "pdebench-integration-fixture",
                "provider": "generated-fixture",
                "revision": f"sha256:{source_digest}",
                "native_format": "hdf5",
                "license": "CC0-1.0",
                "citation": "Generated integration fixture",
                "objects": [
                    {
                        "object_id": "burgers1d-train-000",
                        "path": "burgers1d_train.h5",
                        "size_bytes": len(source_bytes),
                        "checksums": {"sha256": source_digest},
                        "uris": [canonical_path.as_uri()],
                        "declared_roles": ["train"],
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    protocol_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "protocol_id": "integration-training-v1",
                "dataset_id": "pdebench-integration-fixture",
                "source_revision": f"sha256:{source_digest}",
                "adapter": "pdebench_hdf5",
                "adapter_revision": "1.0.0",
                "split_authority": "generated-fixture",
                "splits": {"train": ["burgers1d-train-000"]},
                "identity_fields": ["source_file_id", "source_sample_index"],
                "selection": {"algorithm": "sha256_identity_rank", "seed": 17},
                "normalization": {"fit_role": "train", "method": "zscore"},
                "test_access": "measurement_contract_required",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    source = load_source_manifest(source_path)
    protocol = load_protocol_manifest(protocol_path)
    lock = resolve_data_lock(source, protocol, requested_roles=("train",))
    repeated_lock = resolve_data_lock(source, protocol, requested_roles=("train",))
    assert repeated_lock.to_dict() == lock.to_dict()

    lock_path = tmp_path / "training.data.lock.json"
    write_data_lock(lock_path, lock)
    loaded_lock = load_data_lock(lock_path)
    assert loaded_lock == lock

    objects = staging_objects_from_lock(loaded_lock)
    cache_dir = tmp_path / "cache"
    run_dir = tmp_path / "run"
    plan = plan_staging(objects, cache_dir, allowed_roles={"train"})
    assert plan["object_count"] == 1
    assert plan["missing_bytes"] == len(source_bytes)

    report = stage_objects(
        objects,
        cache_dir,
        run_dir=run_dir,
        allowed_roles={"train"},
    )
    assert report["status"] == "complete"
    assert report["bytes_transferred"] == len(source_bytes)
    verification = verify_lock_cache(loaded_lock, cache_dir)
    assert verification["status"] == "verified"
    assert verification["lock_sha256"] == lock.lock_sha256
    assert (run_dir / "burgers1d_train.h5").read_bytes() == source_bytes

    monkeypatch.setenv("DATA_LOCK", str(lock_path))
    unnormalized = PDEBenchDataset(
        PDEBenchConfig(task="burgers1d", split="train", root=str(run_dir))
    )
    assert unnormalized._handles == {}
    selection_sha256 = canonical_sha256(lock.selection)
    stats = fit_normalization_stats(
        (unnormalized[index]["fields"] for index in range(len(unnormalized))),
        channel_axis=-1,
        data_lock_sha256=lock.lock_sha256,
        selection_sha256=selection_sha256,
    )
    stats_path = tmp_path / "normalization.json"
    stats.save(stats_path)
    loaded_stats = NormalizationStats.load(
        stats_path,
        expected_data_lock_sha256=lock.lock_sha256,
        expected_selection_sha256=selection_sha256,
    )
    assert loaded_stats == stats

    normalized = PDEBenchDataset(
        PDEBenchConfig(
            task="burgers1d",
            split="train",
            root=str(run_dir),
            normalize=True,
            normalization_path=str(stats_path),
        )
    )
    loader = DataLoader(normalized, batch_size=2, num_workers=1, shuffle=False)
    iterator = iter(loader)
    batch = next(iterator)
    del iterator

    assert batch["fields"].shape == (2, 3, 2)
    assert torch.isfinite(batch["fields"]).all()
    assert torch.equal(batch["fields"], batch["targets"])
    expected = loaded_stats.apply(fields[:2])
    assert torch.allclose(batch["fields"], expected)
