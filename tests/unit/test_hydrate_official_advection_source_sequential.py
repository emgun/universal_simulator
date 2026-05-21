from __future__ import annotations

from argparse import Namespace
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
