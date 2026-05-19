from __future__ import annotations

from argparse import Namespace
import json

from scripts.plan_transport_official_hydration import create_plan
from scripts.validate_transport_hydration_plan import validate_plan
from tests.unit.test_plan_transport_official_hydration import _args as plan_args


def _write_plan(tmp_path, plan):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


def _args(path):
    return Namespace(
        plan_json=str(path),
        min_download_bytes=1,
        output_json=str(path.parent / "validation.json"),
    )


def test_validate_hydration_plan_accepts_train_only_plan(tmp_path):
    plan = create_plan(plan_args(tmp_path))
    path = _write_plan(tmp_path, plan)

    record = validate_plan(_args(path))

    assert record["status"] == "valid"
    assert record["blockers"] == []
    assert record["selected_file_count"] == 2
    assert record["held_out_test_policy"]["test_split_downloaded"] is False


def test_validate_hydration_plan_rejects_test_sharding(tmp_path):
    plan = create_plan(plan_args(tmp_path))
    plan["commands"]["build_light_train_val_shards"] = plan["commands"]["build_light_train_val_shards"].replace(
        "--test-count 0", "--test-count 4"
    )
    path = _write_plan(tmp_path, plan)

    record = validate_plan(_args(path))

    assert record["status"] == "invalid"
    assert any("--test-count 0" in blocker for blocker in record["blockers"])


def test_validate_hydration_plan_rejects_download_path_mismatch(tmp_path):
    plan = create_plan(plan_args(tmp_path))
    plan["commands"]["download_official_train_files"][0] = (
        "python scripts/download_pdebench_file.py '1D/Advection/Test/test.hdf5' --out data/pdebench/raw"
    )
    path = _write_plan(tmp_path, plan)

    record = validate_plan(_args(path))

    assert record["status"] == "invalid"
    assert any("download command paths" in blocker for blocker in record["blockers"])
