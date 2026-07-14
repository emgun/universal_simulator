from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts.run_physical_conv_baseline import _dataset, collect_training_pairs
from ups.data.baseline_runtime import STRAT_V1_TASKS, load_strat_v1_baseline_runtime
from ups.data.manifests import (
    ProtocolManifest,
    SourceManifest,
    resolve_data_lock,
    write_data_lock,
)


def _write_shard(path: Path, task: str, role: str) -> None:
    semantics = STRAT_V1_TASKS[task]
    path.parent.mkdir(parents=True, exist_ok=True)
    regimes = np.asarray(semantics.expected_regimes, dtype=np.float64)
    values = np.repeat(regimes, 2)
    samples = len(values)
    with h5py.File(path, "w") as handle:
        handle.attrs.update(
            {
                "task": task,
                "selection_protocol": "strat-v1",
                "field_kind": semantics.field_kind,
                "time_axis": semantics.time_axis,
                "regime_dataset": semantics.regime_dataset,
            }
        )
        if semantics.mapping_kind == "steady_operator":
            data = np.zeros((samples, 1, 4, 4, 1), dtype=np.float32)
            handle.create_dataset("targets", data=np.ones_like(data))
        else:
            data = np.zeros((samples, 3, 4, 1), dtype=np.float32)
        handle.create_dataset("data", data=data)
        handle.create_dataset(semantics.regime_dataset, data=values)
        handle.create_dataset("source_file_index", data=np.arange(samples, dtype=np.int32))
        offset = 0 if role == "train" else 10_000
        handle.create_dataset(
            "source_sample_index", data=np.arange(offset, offset + samples, dtype=np.int64)
        )


def _fixture_lock(tmp_path: Path):
    run_root = tmp_path / "run"
    objects = []
    splits = {"train": [], "valid": [], "test": []}
    for task in STRAT_V1_TASKS:
        task_root = run_root / task
        for role in ("train", "valid"):
            filename = f"{task}_{'val' if role == 'valid' else 'train'}.h5"
            path = task_root / filename
            _write_shard(path, task, role)
            object_id = f"{task}-{role}"
            data = path.read_bytes()
            objects.append(
                {
                    "object_id": object_id,
                    "path": f"{task}/{filename}",
                    "size_bytes": len(data),
                    "checksums": {"sha256": hashlib.sha256(data).hexdigest()},
                    "uris": [path.as_uri()],
                    "declared_roles": [role],
                }
            )
            splits[role].append(object_id)
    source = SourceManifest.from_dict(
        {
            "schema_version": 1,
            "dataset_id": "pdebench-strat-v1-universal",
            "provider": "fixture",
            "revision": "sha256:" + "a" * 64,
            "native_format": "HDF5",
            "license": "fixture",
            "citation": "fixture",
            "objects": objects,
        }
    )
    protocol = ProtocolManifest.from_dict(
        {
            "schema_version": 1,
            "protocol_id": "pdebench-strat-v1-universal",
            "dataset_id": source.dataset_id,
            "source_revision": source.revision,
            "adapter": "pdebench_hdf5",
            "adapter_revision": "1.0.0",
            "split_authority": "fixture",
            "splits": splits,
            "identity_fields": ["source_file_index", "source_sample_index"],
            "selection": {
                "algorithm": "sha256-protocol-seed-provenance-v1",
                "protocol": "strat-v1",
                "seed": 0,
            },
            "normalization": {"fit_role": "train", "method": "zscore"},
            "test_access": "measurement_contract_required",
        }
    )
    lock = resolve_data_lock(source, protocol, requested_roles=("train", "valid"))
    lock_path = tmp_path / "training.lock.json"
    write_data_lock(lock_path, lock)
    return lock, lock_path, run_root, source, protocol


def test_runtime_exposes_task_roots_semantics_regimes_and_shared_runner_config(tmp_path):
    lock, lock_path, run_root, _source, _protocol = _fixture_lock(tmp_path)

    runtime = load_strat_v1_baseline_runtime(
        lock_path, run_root, expected_lock_sha256=lock.lock_sha256
    )

    assert set(runtime.tasks) == set(STRAT_V1_TASKS)
    assert runtime.tasks["burgers1d"].semantics.regime_dataset == "nu"
    assert runtime.tasks["darcy2d"].semantics.mapping_kind == "steady_operator"
    assert {item.count for item in runtime.tasks["advection1d"].valid.regimes} == {2}
    cfg = runtime.apply_to_runner_config({"data": {"root": "must-not-be-used"}})
    dataset = _dataset(
        cfg,
        task="burgers1d",
        split="train",
        data_root=None,
        max_samples=12,
    )
    assert dataset.cfg.root == str(run_root / "burgers1d")
    assert dataset.cfg.data_lock_sha256 == lock.lock_sha256
    assert "params" not in dataset[0]
    assert cfg["data"]["physical_parameter_conditioning"] is False
    assert cfg["data"]["runtime_identity"] == {
        "lock_sha256": lock.lock_sha256,
        "source_manifest_sha256": lock.source_manifest_sha256,
        "protocol_manifest_sha256": lock.protocol_manifest_sha256,
        "selection_sha256": runtime.selection_sha256,
    }
    conditioned = _dataset(
        runtime.apply_to_runner_config({}, condition_on_regime=True),
        task="burgers1d",
        split="train",
        data_root=None,
        max_samples=12,
    )
    assert conditioned[0]["params"]["nu"].item() == pytest.approx(0.001)
    balanced = _dataset(
        runtime.apply_to_runner_config({}, condition_on_regime=True),
        task="burgers1d",
        split="train",
        data_root=None,
        max_samples=len(STRAT_V1_TASKS["burgers1d"].expected_regimes),
    )
    observed = sorted(balanced[index]["params"]["nu"].item() for index in range(len(balanced)))
    assert observed == pytest.approx(STRAT_V1_TASKS["burgers1d"].expected_regimes)
    with pytest.raises(ValueError, match="would omit at least one"):
        _dataset(
            runtime.apply_to_runner_config({}),
            task="burgers1d",
            split="train",
            data_root=None,
            max_samples=11,
        )
    with pytest.raises(PermissionError, match="authorized staged task root"):
        _dataset(
            cfg,
            task="burgers1d",
            split="train",
            data_root=str(tmp_path / "other-root"),
            max_samples=1,
        )


def test_shared_pair_collector_includes_steady_darcy_and_rejects_regime_undercoverage(tmp_path):
    lock, lock_path, run_root, _source, _protocol = _fixture_lock(tmp_path)
    runtime = load_strat_v1_baseline_runtime(
        lock_path, run_root, expected_lock_sha256=lock.lock_sha256
    )
    cfg = runtime.apply_to_runner_config({})

    pairs = collect_training_pairs(
        cfg,
        tasks=["darcy2d"],
        split="train",
        data_root=None,
        max_samples=5,
        max_pairs_per_task=5,
        rollout_steps=16,
        stride=1,
    )

    assert sum(len(currents) for currents, _targets in pairs.values()) == 5
    with pytest.raises(ValueError, match="would omit a darcy2d regime"):
        collect_training_pairs(
            cfg,
            tasks=["darcy2d"],
            split="train",
            data_root=None,
            max_samples=5,
            max_pairs_per_task=4,
            rollout_steps=16,
            stride=1,
        )
    with pytest.raises(PermissionError, match="train and validation only"):
        runtime.dataset_config("burgers1d", "test")


def test_runtime_rejects_measurement_lock_before_inspecting_run_view(tmp_path, monkeypatch):
    _lock, _lock_path, run_root, source, protocol = _fixture_lock(tmp_path)
    measurement = resolve_data_lock(
        source,
        protocol,
        requested_roles=("test",),
        purpose="measurement",
        measurement_contract_id="baseline-heldout-v1",
    )
    measurement_path = tmp_path / "measurement.lock.json"
    write_data_lock(measurement_path, measurement)
    monkeypatch.setattr(
        "ups.data.baseline_runtime.h5py.File",
        lambda *_args, **_kwargs: pytest.fail("measurement lock must not open HDF5"),
    )

    with pytest.raises(PermissionError, match="measurement locks"):
        load_strat_v1_baseline_runtime(
            measurement_path, run_root, expected_lock_sha256=measurement.lock_sha256
        )


def test_runtime_rejects_heldout_file_in_training_run_view_without_opening_it(
    tmp_path, monkeypatch
):
    lock, lock_path, run_root, _source, _protocol = _fixture_lock(tmp_path)
    heldout = run_root / "burgers1d" / "burgers1d_test.h5"
    heldout.write_bytes(b"reserved bytes that must not be opened")
    real_file = h5py.File

    def guarded_file(path, *args, **kwargs):
        assert Path(path) != heldout
        return real_file(path, *args, **kwargs)

    monkeypatch.setattr("ups.data.baseline_runtime.h5py.File", guarded_file)
    with pytest.raises(PermissionError, match="must not contain held-out"):
        load_strat_v1_baseline_runtime(lock_path, run_root, expected_lock_sha256=lock.lock_sha256)


def test_runtime_rejects_regime_imbalance_even_with_a_self_consistent_lock(tmp_path):
    lock, lock_path, run_root, _source, _protocol = _fixture_lock(tmp_path)
    path = run_root / "advection1d" / "advection1d_val.h5"
    with h5py.File(path, "r+") as handle:
        handle["beta"][-1] = handle["beta"][0]

    # The bytes no longer match first, so update the fixture lock through a new
    # source/protocol resolution to reach the semantic balance gate.
    raw_source = asdict(_source)
    raw_source["objects"] = list(raw_source["objects"])
    target = next(
        item for item in raw_source["objects"] if item["object_id"] == "advection1d-valid"
    )
    data = path.read_bytes()
    target["size_bytes"] = len(data)
    target["checksums"]["sha256"] = hashlib.sha256(data).hexdigest()
    source = SourceManifest.from_dict(raw_source)
    updated = resolve_data_lock(source, _protocol, requested_roles=("train", "valid"))
    write_data_lock(lock_path, updated)

    with pytest.raises(ValueError, match="not balanced"):
        load_strat_v1_baseline_runtime(
            lock_path, run_root, expected_lock_sha256=updated.lock_sha256
        )
