from __future__ import annotations

import hashlib
import json
from argparse import Namespace

from scripts.check_official_execution_readiness import check_readiness


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _args(tmp_path):
    plan = _write_json(
        tmp_path / "plan.json",
        {
            "estimated_download_bytes": 300,
            "held_out_test_policy": {"test_split_downloaded": False},
            "remote_entries": [
                {"path": "a.hdf5", "size_bytes": 100},
                {"path": "b.hdf5", "size_bytes": 200},
            ],
        },
    )
    remote_plan = _write_json(tmp_path / "remote.json", {"required_disk_gb": 32})
    return Namespace(
        plan_json=str(plan),
        remote_plan_json=str(remote_plan),
        local_disk_root=str(tmp_path),
        local_safety_factor=1.0,
        remote_api_host="console.vast.ai",
        official_data_host="darus.uni-stuttgart.de",
        output_json=str(tmp_path / "readiness.json"),
    )


def test_readiness_blocks_when_dns_fails(monkeypatch, tmp_path):
    import scripts.check_official_execution_readiness as module

    def fake_dns(host):
        return {"host": host, "resolves": False, "error": "dns failed", "addresses": []}

    monkeypatch.setattr(module, "_dns_record", fake_dns)

    record = check_readiness(_args(tmp_path))

    assert record["status"] == "blocked"
    assert record["remote_launch_ready"] is False
    assert record["local_sequential_hydration_ready"] is False
    assert any("does not resolve" in blocker for blocker in record["blockers"])


def test_readiness_next_action_points_to_staging_when_disk_is_ready_but_dns_fails(
    monkeypatch, tmp_path
):
    import scripts.check_official_execution_readiness as module

    def fake_dns(host):
        return {"host": host, "resolves": False, "error": "dns failed", "addresses": []}

    monkeypatch.setattr(module, "_dns_record", fake_dns)

    record = check_readiness(_args(tmp_path))

    assert record["next_action"] == "stage official raw files or restore official data DNS"


def test_readiness_allows_remote_when_vast_dns_resolves(monkeypatch, tmp_path):
    import scripts.check_official_execution_readiness as module

    def fake_dns(host):
        resolves = host == "console.vast.ai"
        return {
            "host": host,
            "resolves": resolves,
            "error": None if resolves else "dns failed",
            "addresses": ["127.0.0.1"] if resolves else [],
        }

    monkeypatch.setattr(module, "_dns_record", fake_dns)

    record = check_readiness(_args(tmp_path))

    assert record["status"] == "ready"
    assert record["remote_launch_ready"] is True
    assert record["next_action"] == "run pinned remote actual_launcher"
    assert any(
        "official data host" in blocker
        for blocker in record["route_blockers"]["local_sequential_hydration"]
    )


def test_readiness_allows_local_when_data_dns_and_disk_are_ready(monkeypatch, tmp_path):
    import scripts.check_official_execution_readiness as module

    def fake_dns(host):
        resolves = host == "darus.uni-stuttgart.de"
        return {
            "host": host,
            "resolves": resolves,
            "error": None if resolves else "dns failed",
            "addresses": ["127.0.0.1"] if resolves else [],
        }

    monkeypatch.setattr(module, "_dns_record", fake_dns)

    record = check_readiness(_args(tmp_path))

    assert record["status"] == "ready"
    assert record["local_sequential_hydration_ready"] is True
    assert record["next_action"] == "run local sequential hydration"
    assert any(
        "remote API host" in blocker for blocker in record["route_blockers"]["remote_launch"]
    )


def test_readiness_uses_manifest_url_hosts_for_local_data_route(monkeypatch, tmp_path):
    import scripts.check_official_execution_readiness as module

    args = _args(tmp_path)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "estimated_download_bytes": 100,
                "held_out_test_policy": {"test_split_downloaded": False},
                "remote_entries": [
                    {
                        "path": "a.hdf5",
                        "file_id": 1,
                        "size_bytes": 100,
                        "url": "https://mirror.example/a.hdf5",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    remote_plan = _write_json(tmp_path / "remote.json", {"required_disk_gb": 32})
    args.plan_json = str(plan_path)
    args.remote_plan_json = str(remote_plan)

    def fake_dns(host):
        resolves = host == "mirror.example"
        return {
            "host": host,
            "resolves": resolves,
            "error": None if resolves else "dns failed",
            "addresses": ["127.0.0.1"] if resolves else [],
        }

    monkeypatch.setattr(module, "_dns_record", fake_dns)

    record = check_readiness(args)

    assert record["status"] == "ready"
    assert record["local_sequential_hydration_ready"] is True
    assert record["dns"]["official_data"]["hosts"] == ["mirror.example"]
    assert record["next_action"] == "run local sequential hydration"


def test_readiness_uses_datafile_url_template_hosts(monkeypatch, tmp_path):
    import scripts.check_official_execution_readiness as module

    monkeypatch.setenv("PDEBENCH_DATAFILE_URL_TEMPLATE", "https://objects.example/{file_id}/{path}")

    def fake_dns(host):
        resolves = host == "objects.example"
        return {
            "host": host,
            "resolves": resolves,
            "error": None if resolves else "dns failed",
            "addresses": ["127.0.0.1"] if resolves else [],
        }

    monkeypatch.setattr(module, "_dns_record", fake_dns)

    record = check_readiness(_args(tmp_path))

    assert record["status"] == "ready"
    assert record["local_sequential_hydration_ready"] is True
    assert record["dns"]["official_data"]["hosts"] == ["objects.example"]


def test_readiness_allows_local_when_all_raw_files_are_staged(monkeypatch, tmp_path):
    import scripts.check_official_execution_readiness as module

    args = _args(tmp_path)
    raw_root = tmp_path / "raw"
    plan_path = tmp_path / "plan.json"
    entries = [
        {
            "path": "1D/Advection/Train/a.hdf5",
            "file_id": 1,
            "size_bytes": 3,
            "checksum": hashlib.md5(b"xxx").hexdigest(),
            "checksum_type": "MD5",
        },
        {
            "path": "1D/Advection/Train/b.hdf5",
            "file_id": 2,
            "size_bytes": 4,
            "checksum": hashlib.md5(b"xxxx").hexdigest(),
            "checksum_type": "MD5",
        },
    ]
    for entry in entries:
        path = raw_root / entry["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * int(entry["size_bytes"]))
    plan_path.write_text(
        json.dumps(
            {
                "raw_out": str(raw_root),
                "estimated_download_bytes": 7,
                "held_out_test_policy": {"test_split_downloaded": False},
                "remote_entries": entries,
            }
        ),
        encoding="utf-8",
    )
    args.plan_json = str(plan_path)

    def fake_dns(host):
        return {"host": host, "resolves": False, "error": "dns failed", "addresses": []}

    monkeypatch.setattr(module, "_dns_record", fake_dns)

    record = check_readiness(args)

    assert record["status"] == "ready"
    assert record["local_sequential_hydration_ready"] is True
    assert record["staged_raw"]["all_present"] is True
    assert record["staged_raw"]["complete_file_count"] == 2
    assert all(row["checksum_matches"] for row in record["staged_raw"]["files"])
    assert record["route_blockers"]["local_sequential_hydration"] == []


def test_readiness_blocks_when_staged_raw_size_is_incomplete_and_dns_fails(monkeypatch, tmp_path):
    import scripts.check_official_execution_readiness as module

    args = _args(tmp_path)
    raw_root = tmp_path / "raw"
    plan_path = tmp_path / "plan.json"
    entry = {"path": "1D/Advection/Train/a.hdf5", "file_id": 1, "size_bytes": 3}
    path = raw_root / entry["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    plan_path.write_text(
        json.dumps(
            {
                "raw_out": str(raw_root),
                "estimated_download_bytes": 3,
                "held_out_test_policy": {"test_split_downloaded": False},
                "remote_entries": [entry],
            }
        ),
        encoding="utf-8",
    )
    args.plan_json = str(plan_path)

    def fake_dns(host):
        return {"host": host, "resolves": False, "error": "dns failed", "addresses": []}

    monkeypatch.setattr(module, "_dns_record", fake_dns)

    record = check_readiness(args)

    assert record["status"] == "blocked"
    assert record["staged_raw"]["all_present"] is False
    assert record["staged_raw"]["complete_file_count"] == 0
    assert any(
        "official data host" in blocker
        for blocker in record["route_blockers"]["local_sequential_hydration"]
    )


def test_readiness_blocks_when_staged_raw_checksum_mismatches(monkeypatch, tmp_path):
    import scripts.check_official_execution_readiness as module

    args = _args(tmp_path)
    raw_root = tmp_path / "raw"
    plan_path = tmp_path / "plan.json"
    entry = {
        "path": "1D/Advection/Train/a.hdf5",
        "file_id": 1,
        "size_bytes": 3,
        "checksum": hashlib.md5(b"good").hexdigest(),
        "checksum_type": "MD5",
    }
    path = raw_root / entry["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"bad")
    plan_path.write_text(
        json.dumps(
            {
                "raw_out": str(raw_root),
                "estimated_download_bytes": 3,
                "held_out_test_policy": {"test_split_downloaded": False},
                "remote_entries": [entry],
            }
        ),
        encoding="utf-8",
    )
    args.plan_json = str(plan_path)

    def fake_dns(host):
        return {"host": host, "resolves": False, "error": "dns failed", "addresses": []}

    monkeypatch.setattr(module, "_dns_record", fake_dns)

    record = check_readiness(args)

    assert record["status"] == "blocked"
    assert record["staged_raw"]["all_present"] is False
    assert record["staged_raw"]["files"][0]["checksum_matches"] is False
    assert record["staged_raw"]["files"][0]["complete"] is False
