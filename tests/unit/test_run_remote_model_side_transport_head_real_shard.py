from __future__ import annotations

import os
import subprocess


def test_model_side_remote_wrapper_rejects_positional_args():
    proc = subprocess.run(
        ["bash", "scripts/run_remote_model_side_transport_head_real_shard.sh", "bad-arg"],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 2
    assert "Pass KEY=VALUE assignments" in proc.stderr


def test_model_side_remote_wrapper_generates_missing_hydration_plan(tmp_path):
    plan_json = tmp_path / "generated_plan.json"
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["PUBLISH_ARTIFACTS"] = "0"
    env["HYDRATION_PLAN_JSON"] = str(plan_json)
    env["SEQUENTIAL_HYDRATION_JSON"] = str(tmp_path / "sequential.json")
    env["HYDRATION_VALIDATION_JSON"] = str(tmp_path / "validation.json")
    env["HYDRATION_RUN_JSON"] = str(tmp_path / "run.json")
    env["FULL_TASK_ROOT"] = str(tmp_path / "full_task_root")
    env["FULL_ROOT_MANIFEST_JSON"] = str(tmp_path / "manifest.json")
    env["CHECKPOINT_SOURCE"] = str(tmp_path / "checkpoint_source")
    env["OUTPUT_ROOT"] = str(tmp_path / "output")

    proc = subprocess.run(
        ["bash", "scripts/run_remote_model_side_transport_head_real_shard.sh"],
        capture_output=True,
        env=env,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    assert plan_json.exists()
    assert "Hydration plan missing; generating" in proc.stdout
    assert "DRY_RUN command:" in proc.stdout
