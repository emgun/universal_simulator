from __future__ import annotations

from argparse import Namespace
import hashlib
import json

import h5py
import numpy as np

from scripts.hydrate_official_advection_source_sequential import hydrate_sequential


def test_sequential_hydration_dry_run_records_one_file_at_a_time(tmp_path):
    plan = {
        "raw_out": str(tmp_path / "raw"),
        "hydrated_source_root": str(tmp_path / "hydrated"),
        "samples_per_file": 4,
        "remote_entries": [
            {"path": "1D/Advection/Train/a.hdf5", "size_bytes": 10},
            {"path": "1D/Advection/Train/b.hdf5", "size_bytes": 20},
        ],
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    record = hydrate_sequential(
        Namespace(
            plan_json=str(plan_path),
            raw_out=None,
            hydrated_source_root=None,
            samples_per_file=None,
            execute=False,
            execute_downloads=False,
            use_existing_raw=False,
            cleanup_raw=True,
            overwrite=False,
        )
    )

    assert record["status"] == "dry_run"
    assert record["samples_per_file"] == 4
    assert record["disk_strategy"] == "download one official file, append sampled rows, optionally remove raw file"
    assert [item["logical_path"] for item in record["records"]] == [
        "1D/Advection/Train/a.hdf5",
        "1D/Advection/Train/b.hdf5",
    ]
    assert all(not item["download_executed"] for item in record["records"])


def test_sequential_hydration_initializes_source_paths_before_download(monkeypatch, tmp_path):
    plan = {
        "raw_out": str(tmp_path / "raw"),
        "hydrated_source_root": str(tmp_path / "hydrated"),
        "samples_per_file": 2,
        "remote_entries": [
            {"path": "1D/Advection/Train/a.hdf5", "size_bytes": 10},
        ],
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    source_file = tmp_path / "raw" / "1D/Advection/Train/a.hdf5"

    def fake_run(command, check):
        source_file.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(source_file, "w") as handle:
            handle.create_dataset("u", data=np.ones((3, 4, 8), dtype=np.float32))

        class Completed:
            returncode = 0

        return Completed()

    monkeypatch.setattr("scripts.hydrate_official_advection_source_sequential.subprocess.run", fake_run)

    record = hydrate_sequential(
        Namespace(
            plan_json=str(plan_path),
            raw_out=None,
            hydrated_source_root=None,
            samples_per_file=None,
            execute=True,
            execute_downloads=True,
            use_existing_raw=False,
            cleanup_raw=False,
            overwrite=True,
        )
    )

    assert record["status"] == "executed"
    assert record["source_paths_initialized_before_download"] is True
    with h5py.File(tmp_path / "hydrated" / "advection1d_train.h5", "r") as handle:
        assert list(handle.attrs["source_paths"]) == ["1D/Advection/Train/a.hdf5"]
        assert bool(handle.attrs["sequential_hydration_complete"]) is True
        assert handle["source_file_index"][:].tolist() == [0, 0]


def test_sequential_hydration_can_use_existing_raw_without_download(monkeypatch, tmp_path):
    source_file = tmp_path / "raw" / "1D/Advection/Train/a.hdf5"
    source_file.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(source_file, "w") as handle:
        handle.create_dataset("u", data=np.ones((3, 4, 8), dtype=np.float32))
    plan = {
        "raw_out": str(tmp_path / "raw"),
        "hydrated_source_root": str(tmp_path / "hydrated"),
        "samples_per_file": 2,
        "remote_entries": [
            {"path": "1D/Advection/Train/a.hdf5", "size_bytes": source_file.stat().st_size},
        ],
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    def fail_run(command, check):
        raise AssertionError("download subprocess should not run when use_existing_raw=True")

    monkeypatch.setattr("scripts.hydrate_official_advection_source_sequential.subprocess.run", fail_run)

    record = hydrate_sequential(
        Namespace(
            plan_json=str(plan_path),
            raw_out=None,
            hydrated_source_root=None,
            samples_per_file=None,
            execute=True,
            execute_downloads=False,
            use_existing_raw=True,
            cleanup_raw=False,
            overwrite=True,
        )
    )

    assert record["status"] == "executed"
    assert record["use_existing_raw"] is True
    assert record["records"][0]["used_existing_raw"] is True
    assert record["records"][0]["download_executed"] is False
    with h5py.File(tmp_path / "hydrated" / "advection1d_train.h5", "r") as handle:
        assert handle["data"].shape[0] == 2


def test_sequential_hydration_resume_skips_completed_sources(monkeypatch, tmp_path):
    source_file = tmp_path / "raw" / "1D/Advection/Train/b.hdf5"
    source_file.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(source_file, "w") as handle:
        handle.create_dataset("u", data=np.full((3, 4, 8), 2, dtype=np.float32))
    out_path = tmp_path / "hydrated" / "advection1d_train.h5"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as handle:
        handle.create_dataset("data", data=np.ones((2, 4, 8, 1), dtype=np.float32), maxshape=(None, 4, 8, 1))
        handle.create_dataset("source_file_index", data=np.asarray([0, 0], dtype=np.int32), maxshape=(None,))
        handle.create_dataset("source_sample_index", data=np.asarray([0, 1], dtype=np.int64), maxshape=(None,))
    plan = {
        "raw_out": str(tmp_path / "raw"),
        "hydrated_source_root": str(tmp_path / "hydrated"),
        "samples_per_file": 2,
        "remote_entries": [
            {"path": "1D/Advection/Train/a.hdf5", "size_bytes": 10},
            {"path": "1D/Advection/Train/b.hdf5", "size_bytes": source_file.stat().st_size},
        ],
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    def fake_run(command, check):
        assert command[2] == "1D/Advection/Train/b.hdf5"

        class Completed:
            returncode = 0

        return Completed()

    monkeypatch.setattr("scripts.hydrate_official_advection_source_sequential.subprocess.run", fake_run)

    record = hydrate_sequential(
        Namespace(
            plan_json=str(plan_path),
            raw_out=None,
            hydrated_source_root=None,
            samples_per_file=None,
            execute=True,
            execute_downloads=True,
            use_existing_raw=False,
            cleanup_raw=False,
            overwrite=False,
            resume=True,
        )
    )

    assert record["status"] == "executed"
    assert record["records"][0]["resume_skipped"] is True
    assert record["records"][1]["append_executed"] is True
    with h5py.File(out_path, "r") as handle:
        assert handle["source_file_index"][:].tolist() == [0, 0, 1, 1]


def test_sequential_hydration_blocks_when_existing_raw_is_missing(tmp_path):
    plan = {
        "raw_out": str(tmp_path / "raw"),
        "hydrated_source_root": str(tmp_path / "hydrated"),
        "samples_per_file": 2,
        "remote_entries": [
            {"path": "1D/Advection/Train/a.hdf5", "size_bytes": 10},
        ],
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    record = hydrate_sequential(
        Namespace(
            plan_json=str(plan_path),
            raw_out=None,
            hydrated_source_root=None,
            samples_per_file=None,
            execute=True,
            execute_downloads=False,
            use_existing_raw=True,
            cleanup_raw=False,
            overwrite=True,
        )
    )

    assert record["status"] == "blocked"
    assert any("existing raw file is missing" in blocker for blocker in record["blockers"])


def test_sequential_hydration_blocks_when_existing_raw_checksum_mismatches(tmp_path):
    plan = {
        "raw_out": str(tmp_path / "raw"),
        "hydrated_source_root": str(tmp_path / "hydrated"),
        "samples_per_file": 2,
        "remote_entries": [
            {
                "path": "1D/Advection/Train/a.hdf5",
                "size_bytes": 3,
                "checksum": hashlib.md5(b"good").hexdigest(),
                "checksum_type": "MD5",
            },
        ],
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    raw_path = tmp_path / "raw" / "1D/Advection/Train/a.hdf5"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(b"bad")

    record = hydrate_sequential(
        Namespace(
            plan_json=str(plan_path),
            raw_out=None,
            hydrated_source_root=None,
            samples_per_file=None,
            execute=True,
            execute_downloads=False,
            use_existing_raw=True,
            cleanup_raw=False,
            overwrite=True,
        )
    )

    assert record["status"] == "blocked"
    assert any("checksum mismatch" in blocker for blocker in record["blockers"])
