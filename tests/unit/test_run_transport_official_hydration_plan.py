from __future__ import annotations

from argparse import Namespace
import json

from scripts.plan_transport_official_hydration import create_plan
from scripts.run_transport_official_hydration_plan import run_plan
from tests.unit.test_plan_transport_official_hydration import _args as plan_args


def _write_plan(tmp_path, plan):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


def _args(path, *, execute: bool = False, execute_downloads: bool = False, stages=None):
    return Namespace(
        plan_json=str(path),
        validation_json=str(path.parent / "validation.json"),
        min_download_bytes=1,
        stage=stages,
        execute=execute,
        execute_downloads=execute_downloads,
        output_json=str(path.parent / "run.json"),
    )


def test_hydration_runner_dry_run_lists_all_stages(tmp_path):
    path = _write_plan(tmp_path, create_plan(plan_args(tmp_path)))

    record = run_plan(_args(path))

    assert record["status"] == "dry_run"
    assert record["blockers"] == ["download stage requires --execute-downloads"]
    assert [row["stage"] for row in record["executed"]][:2] == ["download", "download"]
    assert all(row["executed"] is False for row in record["executed"])


def test_hydration_runner_can_preview_non_download_stages_without_blocker(tmp_path):
    path = _write_plan(tmp_path, create_plan(plan_args(tmp_path)))

    record = run_plan(_args(path, stages=["convert", "shard", "validate"]))

    assert record["status"] == "dry_run"
    assert record["blockers"] == []
    assert [row["stage"] for row in record["executed"]] == ["convert", "shard", "validate"]


def test_hydration_runner_refuses_invalid_plan(tmp_path):
    plan = create_plan(plan_args(tmp_path))
    plan["held_out_test_policy"]["test_split_downloaded"] = True
    path = _write_plan(tmp_path, plan)

    record = run_plan(_args(path))

    assert record["status"] == "invalid_plan"
    assert any("must not download test split" in blocker for blocker in record["blockers"])
