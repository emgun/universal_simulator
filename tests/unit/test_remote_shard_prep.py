from __future__ import annotations

import os
import subprocess

import h5py
import numpy as np
import pytest
import yaml


def test_remote_shard_prep_dry_run_supports_source_key_overrides():
    proc = subprocess.run(
        [
            "bash",
            "scripts/run_remote_shard_prep_b2.sh",
        ],
        check=True,
        capture_output=True,
        env={
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "DRY_RUN": "1",
            "TASKS": "burgers1d",
            "BURGERS1D_SOURCE_SPLITS": "train",
            "BURGERS1D_TRAIN_SOURCE_KEYS": "burgers1d/burgers1d_train_000.h5",
        },
        text=True,
    )

    assert "source_splits=train" in proc.stdout
    assert "fetch full/burgers1d/burgers1d_train_000.h5" in proc.stdout
    assert "burgers1d_val.h5" not in proc.stdout


def test_remote_shard_prep_dry_run_reports_protocol_settings():
    proc = subprocess.run(
        [
            "bash",
            "scripts/run_remote_shard_prep_b2.sh",
        ],
        check=True,
        capture_output=True,
        env={
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "DRY_RUN": "1",
            "TASKS": "advection1d",
            "ADVECTION1D_SOURCE_SPLITS": "val",
            "ADVECTION1D_PROVENANCE_DATASETS": "source_file_index,source_sample_index",
            "ADVECTION1D_REGIME_DATASET": "beta",
            "ADVECTION1D_FIELD_KIND": "temporal",
            "ADVECTION1D_TIME_AXIS": "1",
        },
        text=True,
    )

    assert "source_splits=val" in proc.stdout
    assert (
        "provenance=source_file_index,source_sample_index regime=beta field_kind=temporal"
        in proc.stdout
    )
    assert "fetch full/advection1d/advection1d_val.h5" in proc.stdout
    assert "advection1d_train.h5" not in proc.stdout


def test_remote_shard_prep_accepts_cli_assignments():
    proc = subprocess.run(
        [
            "bash",
            "scripts/run_remote_shard_prep_b2.sh",
            "DRY_RUN=1",
            "TASKS=darcy2d",
            "REMOTE_PREFIX=strat-v1",
        ],
        check=True,
        capture_output=True,
        env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"},
        text=True,
    )

    assert "would build strat-v1 shards for tasks: darcy2d" in proc.stdout
    assert "task=darcy2d source_splits=train" in proc.stdout


@pytest.mark.parametrize("version", ["smoke-v1", "light-v1", "medium-v1"])
def test_remote_shard_prep_rejects_legacy_version_labels(version):
    proc = subprocess.run(
        [
            "bash",
            "scripts/run_remote_shard_prep_b2.sh",
        ],
        capture_output=True,
        env={
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "DRY_RUN": "1",
            "VERSION": version,
            "REMOTE_PREFIX": "strat-v1-test",
            "TASKS": "burgers1d",
        },
        text=True,
    )

    assert proc.returncode == 2
    assert "reserved for immutable legacy artifacts" in proc.stderr


def test_remote_shard_prep_rejects_legacy_remote_prefix_override():
    proc = subprocess.run(
        ["bash", "scripts/run_remote_shard_prep_b2.sh"],
        capture_output=True,
        env={
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "DRY_RUN": "1",
            "VERSION": "strat-v1",
            "REMOTE_PREFIX": "light-v1",
        },
        text=True,
    )

    assert proc.returncode == 2
    assert "frozen legacy remote prefix" in proc.stderr


def test_smoke_shard_prep_wrapper_blocks_until_all_protocol_roots_exist():
    proc = subprocess.run(
        [
            "bash",
            "scripts/run_smoke_shard_prep_b2.sh",
        ],
        capture_output=True,
        env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"},
        text=True,
    )

    assert proc.returncode == 2
    assert "requires canonical Burgers and Darcy provenance roots" in proc.stderr


def test_remote_shard_prep_can_use_existing_sources_without_fetch_or_publish(tmp_path):
    data_root = tmp_path / "source"
    out_root = tmp_path / "out"
    manifest = tmp_path / "manifest.yaml"
    data_root.mkdir()
    with h5py.File(data_root / "advection1d_val.h5", "w") as handle:
        handle.create_dataset("data", data=np.arange(8 * 2 * 2, dtype=np.float32).reshape(8, 2, 2))
        handle.create_dataset("source_file_index", data=np.repeat(np.arange(2), 4))
        handle.create_dataset("source_sample_index", data=np.tile(np.arange(4), 2))
        handle.create_dataset("beta", data=np.repeat(np.asarray([0.1, 0.2]), 4))

    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "0",
            "FETCH_DATA": "0",
            "PUBLISH_SHARDS": "0",
            "CLEAN_SOURCE": "0",
            "TASKS": "advection1d",
            "ADVECTION1D_SOURCE_SPLITS": "val",
            "ADVECTION1D_PROVENANCE_DATASETS": "source_file_index,source_sample_index",
            "ADVECTION1D_REGIME_DATASET": "beta",
            "ADVECTION1D_FIELD_KIND": "temporal",
            "ADVECTION1D_TIME_AXIS": "1",
            "DATA_ROOT": str(data_root),
            "OUT_ROOT": str(out_root),
            "MANIFEST": str(manifest),
            "TRAIN_COUNT": "4",
            "VAL_COUNT": "2",
            "TEST_COUNT": "2",
        }
    )

    proc = subprocess.run(
        [
            "bash",
            "scripts/run_remote_shard_prep_b2.sh",
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert "Skipping fetch for advection1d/advection1d_val.h5" in proc.stdout
    assert "PUBLISH_SHARDS=0: skipping B2 publish." in proc.stdout
    payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    assert payload["protocol_mode"] == "strat-v1"
    assert payload["protocol_gates"]["advection1d"]["status"] == "passed"
    assert [record["source_split"] for record in payload["records"]] == ["val", "val", "val"]
    assert [record["derived_from_source_split"] for record in payload["records"]] == [
        True,
        False,
        True,
    ]
    with h5py.File(out_root / "advection1d" / "advection1d_train.h5", "r") as handle:
        assert handle["data"].shape == (4, 2, 2)


def test_remote_shard_prep_fails_early_when_required_disk_is_unavailable(tmp_path):
    data_root = tmp_path / "source"
    data_root.mkdir()

    proc = subprocess.run(
        [
            "bash",
            "scripts/run_remote_shard_prep_b2.sh",
        ],
        capture_output=True,
        env={
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "DRY_RUN": "0",
            "FETCH_DATA": "0",
            "PUBLISH_SHARDS": "0",
            "TASKS": "advection1d",
            "DATA_ROOT": str(data_root),
            "OUT_ROOT": str(tmp_path / "out"),
            "MANIFEST": str(tmp_path / "manifest.yaml"),
            "REQUIRED_GB": "999999",
        },
        text=True,
    )

    assert proc.returncode == 1
    assert "Insufficient free disk for shard prep" in proc.stderr
