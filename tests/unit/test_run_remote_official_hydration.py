from __future__ import annotations

import json
import os
import subprocess


def test_remote_official_hydration_wrapper_rejects_positional_args():
    proc = subprocess.run(
        ["bash", "scripts/run_remote_official_hydration.sh", "not-an-assignment"],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 2
    assert "Pass KEY=VALUE assignments" in proc.stderr


def test_remote_official_hydration_wrapper_can_dry_run_invalid_missing_plan(tmp_path):
    env = os.environ.copy()
    env["EXECUTE"] = "0"
    env["EXECUTE_DOWNLOADS"] = "0"
    env["PLAN_JSON"] = str(tmp_path / "missing.json")
    env["VALIDATION_JSON"] = str(tmp_path / "validation.json")
    env["RUN_JSON"] = str(tmp_path / "run.json")
    proc = subprocess.run(
        ["bash", "scripts/run_remote_official_hydration.sh"],
        capture_output=True,
        env=env,
        text=True,
    )

    assert proc.returncode != 0
    assert "missing.json" in proc.stderr or "missing.json" in proc.stdout


def test_remote_official_hydration_wrapper_can_chain_guarded_post_validation_test(tmp_path):
    env = os.environ.copy()
    env["EXECUTE"] = "0"
    env["EXECUTE_DOWNLOADS"] = "0"
    env["RUN_POST_VALIDATION_TEST"] = "1"
    env["EXECUTE_TEST"] = "0"
    env["PLAN_JSON"] = "reports/research/sota_loop/official_advection_hydration_plan.json"
    env["VALIDATION_JSON"] = str(tmp_path / "validation.json")
    env["RUN_JSON"] = str(tmp_path / "run.json")
    env["OBJECTIVE_STATUS_JSON"] = str(tmp_path / "objective.json")
    env["POST_VALIDATION_TEST_JSON"] = str(tmp_path / "post_validation_test.json")
    (tmp_path / "objective.json").write_text('{"status":"literal_blocked"}', encoding="utf-8")

    proc = subprocess.run(
        ["bash", "scripts/run_remote_official_hydration.sh"],
        capture_output=True,
        env=env,
        text=True,
    )

    assert proc.returncode == 0
    assert "expected literal_test_ready" in proc.stdout
    post_validation = json.loads((tmp_path / "post_validation_test.json").read_text(encoding="utf-8"))
    assert post_validation["objective_status"] == "literal_blocked"
    assert post_validation["held_out_test_policy"]["requires_literal_test_ready"] is True
