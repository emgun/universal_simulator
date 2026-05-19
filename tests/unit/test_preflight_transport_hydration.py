from __future__ import annotations

from argparse import Namespace
import json

from scripts.plan_transport_official_hydration import create_plan
from scripts.preflight_transport_hydration import preflight
from tests.unit.test_plan_transport_official_hydration import _args as plan_args


def _write_plan(tmp_path, plan):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


def _args(path, tmp_path, *, safety_factor: float = 1.0):
    return Namespace(
        plan_json=str(path),
        raw_out=str(tmp_path / "raw"),
        disk_root=str(tmp_path),
        safety_factor=safety_factor,
        output_json=str(tmp_path / "preflight.json"),
    )


def test_preflight_reports_ready_when_raw_files_are_present(tmp_path):
    plan = create_plan(plan_args(tmp_path))
    raw_out = tmp_path / "raw"
    for entry in plan["remote_entries"]:
        path = raw_out / entry["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * int(entry["size_bytes"]))
    plan["raw_out"] = str(raw_out)
    plan_path = _write_plan(tmp_path, plan)

    record = preflight(_args(plan_path, tmp_path))

    assert record["status"] == "ready_raw_files_present"
    assert record["remaining_download_bytes"] == 0
    assert record["complete_file_count"] == 2


def test_preflight_reports_remaining_bytes_for_missing_files(tmp_path):
    plan = create_plan(plan_args(tmp_path))
    plan["raw_out"] = str(tmp_path / "raw")
    plan_path = _write_plan(tmp_path, plan)

    record = preflight(_args(plan_path, tmp_path))

    assert record["status"] == "ready_for_download"
    assert record["remaining_download_bytes"] == 300
    assert record["complete_file_count"] == 0


def test_preflight_blocks_when_safety_factor_exceeds_free_space(tmp_path):
    plan = create_plan(plan_args(tmp_path))
    plan["raw_out"] = str(tmp_path / "raw")
    plan_path = _write_plan(tmp_path, plan)

    record = preflight(_args(plan_path, tmp_path, safety_factor=10**18))

    assert record["status"] == "blocked_insufficient_disk"
    assert record["blockers"]
