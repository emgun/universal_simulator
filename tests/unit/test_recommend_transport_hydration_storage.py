from __future__ import annotations

from argparse import Namespace
import json

from scripts.plan_transport_official_hydration import create_plan
from scripts.recommend_transport_hydration_storage import recommend_storage
from tests.unit.test_plan_transport_official_hydration import _args as plan_args


def _write_plan(tmp_path, plan):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


def test_storage_recommendation_finds_viable_tmp_root(tmp_path):
    plan = create_plan(plan_args(tmp_path))
    plan["estimated_download_bytes"] = 1
    path = _write_plan(tmp_path, plan)

    record = recommend_storage(
        Namespace(
            plan_json=str(path),
            candidate_root=[str(tmp_path)],
            safety_factor=1.0,
            mode="all",
            output_json=str(tmp_path / "storage.json"),
        )
    )

    assert record["status"] == "storage_root_available"
    assert record["recommended_root"] == str(tmp_path)


def test_storage_recommendation_blocks_when_no_candidate_has_space(tmp_path):
    plan = create_plan(plan_args(tmp_path))
    plan["estimated_download_bytes"] = 10**30
    path = _write_plan(tmp_path, plan)

    record = recommend_storage(
        Namespace(
            plan_json=str(path),
            candidate_root=[str(tmp_path)],
            safety_factor=1.0,
            mode="all",
            output_json=str(tmp_path / "storage.json"),
        )
    )

    assert record["status"] == "external_or_freed_space_required"
    assert record["recommended_root"] is None
    assert record["blockers"]


def test_storage_recommendation_sequential_mode_uses_largest_file(tmp_path):
    plan = create_plan(plan_args(tmp_path))
    plan["estimated_download_bytes"] = 10**30
    plan["remote_entries"] = [{"path": "a", "size_bytes": 1}]
    path = _write_plan(tmp_path, plan)

    record = recommend_storage(
        Namespace(
            plan_json=str(path),
            candidate_root=[str(tmp_path)],
            safety_factor=1.0,
            mode="sequential",
            output_json=str(tmp_path / "storage.json"),
        )
    )

    assert record["status"] == "storage_root_available"
    assert record["required_download_bytes"] == 1
    assert record["mode"] == "sequential"
