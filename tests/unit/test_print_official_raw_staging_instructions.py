from __future__ import annotations

from argparse import Namespace
import hashlib
import json

from scripts.print_official_raw_staging_instructions import build_staging_instructions


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_build_staging_instructions_lists_required_files_and_command(tmp_path):
    raw_root = tmp_path / "raw"
    plan = _write_json(
        tmp_path / "plan.json",
        {
            "raw_out": str(raw_root),
            "remote_entries": [
                {
                    "path": "1D/Advection/Train/a.hdf5",
                    "file_id": 1,
                    "size_bytes": 3,
                    "checksum": hashlib.md5(b"aaa").hexdigest(),
                    "checksum_type": "MD5",
                },
                {
                    "path": "1D/Advection/Train/b.hdf5",
                    "file_id": 2,
                    "size_bytes": 4,
                    "checksum": hashlib.md5(b"bbbb").hexdigest(),
                    "checksum_type": "MD5",
                },
            ],
        },
    )
    readiness = _write_json(
        tmp_path / "readiness.json",
        {
            "status": "blocked",
            "disk": {"local_sequential_required_bytes": 4},
            "route_blockers": {"local_sequential_hydration": ["official data host(s) do not resolve: darus"]},
            "staged_raw": {
                "complete_file_count": 0,
                "selected_file_count": 2,
                "files": [
                    {
                        "logical_path": "1D/Advection/Train/a.hdf5",
                        "local_path": str(raw_root / "1D/Advection/Train/a.hdf5"),
                        "exists": False,
                        "complete": False,
                        "expected_size_bytes": 3,
                        "expected_checksum": hashlib.md5(b"aaa").hexdigest(),
                        "checksum_type": "md5",
                    },
                    {
                        "logical_path": "1D/Advection/Train/b.hdf5",
                        "local_path": str(raw_root / "1D/Advection/Train/b.hdf5"),
                        "exists": False,
                        "complete": False,
                        "expected_size_bytes": 4,
                        "expected_checksum": hashlib.md5(b"bbbb").hexdigest(),
                        "checksum_type": "md5",
                    },
                ],
            },
        },
    )

    record = build_staging_instructions(
        Namespace(
            plan_json=str(plan),
            readiness_json=str(readiness),
            output_json=None,
            raw_out=None,
            run_json="reports/research/sota_loop/official_advection_sequential_hydration_run.json",
        )
    )

    assert record["status"] == "needs_staging"
    assert record["raw_root"] == str(raw_root)
    assert record["complete_file_count"] == 0
    assert record["selected_file_count"] == 2
    assert record["files"][0]["local_path"] == str(raw_root / "1D/Advection/Train/a.hdf5")
    assert record["files"][0]["expected_size_bytes"] == 3
    assert record["files"][0]["expected_checksum"] == hashlib.md5(b"aaa").hexdigest()
    assert record["files"][0]["source_url"] == "https://darus.uni-stuttgart.de/api/access/datafile/1?format=original"
    assert "curl -L --fail --continue-at -" in record["files"][0]["download_command"]
    assert str(raw_root / "1D/Advection/Train/a.hdf5") in record["files"][0]["download_command"]
    assert "SEQUENTIAL_USE_EXISTING_RAW=1" in record["next_command"]
    assert "--use-existing-raw" not in record["next_command"]
    assert "scripts/run_remote_official_hydration.sh" in record["next_command"]


def test_build_staging_instructions_marks_ready_when_all_files_complete(tmp_path):
    raw_root = tmp_path / "raw"
    entry = {
        "path": "1D/Advection/Train/a.hdf5",
        "file_id": 1,
        "size_bytes": 3,
        "checksum": hashlib.md5(b"aaa").hexdigest(),
        "checksum_type": "MD5",
    }
    plan = _write_json(tmp_path / "plan.json", {"raw_out": str(raw_root), "remote_entries": [entry]})
    readiness = _write_json(
        tmp_path / "readiness.json",
        {
            "status": "ready",
            "local_sequential_hydration_ready": True,
            "staged_raw": {
                "complete_file_count": 1,
                "selected_file_count": 1,
                "files": [
                    {
                        "logical_path": entry["path"],
                        "local_path": str(raw_root / entry["path"]),
                        "exists": True,
                        "complete": True,
                        "expected_size_bytes": 3,
                        "actual_size_bytes": 3,
                        "expected_checksum": entry["checksum"],
                        "actual_checksum": entry["checksum"],
                        "checksum_matches": True,
                        "checksum_type": "md5",
                    }
                ],
            },
        },
    )

    record = build_staging_instructions(
        Namespace(
            plan_json=str(plan),
            readiness_json=str(readiness),
            output_json=None,
            raw_out=None,
            run_json="reports/research/sota_loop/official_advection_sequential_hydration_run.json",
        )
    )

    assert record["status"] == "ready_for_existing_raw_hydration"
    assert record["files"][0]["complete"] is True
    assert "SEQUENTIAL_USE_EXISTING_RAW=1" in record["next_command"]


def test_build_staging_instructions_uses_manifest_url_for_download_command(tmp_path):
    plan = _write_json(
        tmp_path / "plan.json",
        {
            "raw_out": str(tmp_path / "raw"),
            "remote_entries": [
                {
                    "path": "1D/Advection/Train/a.hdf5",
                    "file_id": 1,
                    "size_bytes": 3,
                    "url": "https://mirror.example/a.hdf5",
                }
            ],
        },
    )
    readiness = _write_json(tmp_path / "readiness.json", {"staged_raw": {"files": []}})

    record = build_staging_instructions(
        Namespace(
            plan_json=str(plan),
            readiness_json=str(readiness),
            output_json=None,
            raw_out=None,
            run_json="reports/research/sota_loop/official_advection_sequential_hydration_run.json",
        )
    )

    assert record["files"][0]["source_url"] == "https://mirror.example/a.hdf5"
    assert "https://mirror.example/a.hdf5" in record["files"][0]["download_command"]
