from __future__ import annotations

from argparse import Namespace
import json

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
