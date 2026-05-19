from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path

from scripts.plan_remote_official_hydration import create_remote_plan


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _args(tmp_path):
    hydration = _write_json(
        tmp_path / "hydration.json",
        {
            "estimated_download_gib": 61.34,
            "held_out_test_policy": {"test_split_downloaded": False, "test_split_sharded": False},
        },
    )
    preflight = _write_json(tmp_path / "preflight.json", {"status": "blocked_insufficient_disk"})
    storage = _write_json(tmp_path / "storage.json", {"status": "external_or_freed_space_required"})
    return Namespace(
        hydration_plan_json=str(hydration),
        preflight_json=str(preflight),
        storage_json=str(storage),
        remote_plan_json="reports/plan.json",
        remote_validation_json="reports/validation.json",
        remote_run_json="reports/run.json",
        min_disk_gb=120,
        disk_multiplier=1.3,
        disk_padding_gb=40,
        output_json=str(tmp_path / "remote.json"),
    )


def test_remote_plan_is_ready_when_local_disk_is_blocked(tmp_path):
    record = create_remote_plan(_args(tmp_path))

    assert record["status"] == "ready_for_remote_hydration"
    assert record["required_disk_gb"] >= 120
    assert "DRY_RUN=1" in record["commands"]["dry_run_launcher"]
    assert "DRY_RUN=0" in record["commands"]["actual_launcher"]
    assert "REMOTE_SCRIPT=scripts/run_remote_official_hydration.sh" in record["commands"]["actual_launcher"]
    assert "EXECUTE_DOWNLOADS=1" in record["commands"]["actual_launcher"]
    assert record["held_out_test_policy"]["test_split_downloaded"] is False


def test_remote_plan_notes_when_remote_not_needed(tmp_path):
    args = _args(tmp_path)
    _write_json(Path(args.preflight_json), {"status": "ready_for_download"})

    record = create_remote_plan(args)

    assert record["status"] == "blocked_remote_plan_not_needed"
    assert record["blockers"]
