from __future__ import annotations

import os
import subprocess

import h5py
import numpy as np
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


def test_remote_shard_prep_dry_run_supports_split_source_mapping():
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
            "ADVECTION1D_SPLIT_SOURCES": "train=val,val=val,test=val",
        },
        text=True,
    )

    assert "source_splits=val" in proc.stdout
    assert "split_source_args=--split-source train=val --split-source val=val --split-source test=val" in proc.stdout
    assert "fetch full/advection1d/advection1d_val.h5" in proc.stdout
    assert "advection1d_train.h5" not in proc.stdout


def test_remote_shard_prep_accepts_cli_assignments():
    proc = subprocess.run(
        [
            "bash",
            "scripts/run_remote_shard_prep_b2.sh",
            "DRY_RUN=1",
            "TASKS=darcy2d",
            "REMOTE_PREFIX=light-v1",
        ],
        check=True,
        capture_output=True,
        env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"},
        text=True,
    )

    assert "would build light-v1 shards for tasks: darcy2d" in proc.stdout
    assert "task=darcy2d source_splits=train test" in proc.stdout


def test_smoke_shard_prep_wrapper_uses_smoke_defaults():
    proc = subprocess.run(
        [
            "bash",
            "scripts/run_smoke_shard_prep_b2.sh",
        ],
        check=True,
        capture_output=True,
        env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"},
        text=True,
    )

    assert "would build smoke-v1 shards" in proc.stdout
    assert "fetch full/burgers1d/burgers1d_train_000.h5" in proc.stdout
    assert "fetch full/advection1d/advection1d_val.h5" in proc.stdout
    assert "fetch full/darcy2d/darcy2d_test.h5" in proc.stdout
    assert "publish data/pdebench_smoke/*.h5 and docs/demo_smoke_data_manifest.yaml to prefix smoke-v1" in proc.stdout


def test_remote_shard_prep_can_use_existing_sources_without_fetch_or_publish(tmp_path):
    data_root = tmp_path / "source"
    out_root = tmp_path / "out"
    manifest = tmp_path / "manifest.yaml"
    data_root.mkdir()
    with h5py.File(data_root / "advection1d_val.h5", "w") as handle:
        handle.create_dataset("data", data=np.arange(20 * 2, dtype=np.float32).reshape(20, 2))

    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "0",
            "FETCH_DATA": "0",
            "PUBLISH_SHARDS": "0",
            "CLEAN_SOURCE": "0",
            "TASKS": "advection1d",
            "ADVECTION1D_SOURCE_SPLITS": "val",
            "ADVECTION1D_SPLIT_SOURCES": "train=val,val=val,test=val",
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
    assert [record["source_split"] for record in payload["records"]] == ["val", "val", "val"]
    assert [record["derived_from_source_split"] for record in payload["records"]] == [True, False, True]
    with h5py.File(out_root / "advection1d" / "advection1d_train.h5", "r") as handle:
        assert handle["data"].shape == (4, 2)


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
