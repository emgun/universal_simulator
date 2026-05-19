from __future__ import annotations

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
