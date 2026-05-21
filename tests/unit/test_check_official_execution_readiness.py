from __future__ import annotations

from argparse import Namespace
import json

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


def test_readiness_allows_remote_when_vast_dns_resolves(monkeypatch, tmp_path):
    import scripts.check_official_execution_readiness as module

    def fake_dns(host):
        resolves = host == "console.vast.ai"
        return {"host": host, "resolves": resolves, "error": None if resolves else "dns failed", "addresses": ["127.0.0.1"] if resolves else []}

    monkeypatch.setattr(module, "_dns_record", fake_dns)

    record = check_readiness(_args(tmp_path))

    assert record["status"] == "ready"
    assert record["remote_launch_ready"] is True
    assert record["next_action"] == "run pinned remote actual_launcher"
    assert any("official data host" in blocker for blocker in record["route_blockers"]["local_sequential_hydration"])


def test_readiness_allows_local_when_data_dns_and_disk_are_ready(monkeypatch, tmp_path):
    import scripts.check_official_execution_readiness as module

    def fake_dns(host):
        resolves = host == "darus.uni-stuttgart.de"
        return {"host": host, "resolves": resolves, "error": None if resolves else "dns failed", "addresses": ["127.0.0.1"] if resolves else []}

    monkeypatch.setattr(module, "_dns_record", fake_dns)

    record = check_readiness(_args(tmp_path))

    assert record["status"] == "ready"
    assert record["local_sequential_hydration_ready"] is True
    assert record["next_action"] == "run local sequential hydration"
    assert any("remote API host" in blocker for blocker in record["route_blockers"]["remote_launch"])
