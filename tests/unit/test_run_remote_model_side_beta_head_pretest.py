from __future__ import annotations

import os
import subprocess
from pathlib import Path

SCRIPT = Path("scripts/run_remote_model_side_beta_head_pretest.sh")
LAUNCHER = Path("scripts/launch_remote_model_side_beta_head_pretest_vast.sh")


def test_pretest_remote_wrapper_rejects_positional_args():
    proc = subprocess.run(
        ["bash", str(SCRIPT), "bad-arg"],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 2
    assert "Pass KEY=VALUE assignments" in proc.stderr


def test_pretest_remote_wrapper_defaults_are_fail_closed():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "DRY_RUN=${DRY_RUN:-1}" in text
    assert "ALLOW_HELDOUT_PRETEST=${ALLOW_HELDOUT_PRETEST:-0}" in text
    assert "Refusing held-out pretest execution without ALLOW_HELDOUT_PRETEST=1" in text
    assert "scripts/validate_p2_model_side_beta_head_pretest_contract.py" in text
    assert "scripts/build_p2_model_side_beta_head_pretest_root.py" in text
    assert "--allow-heldout-pretest-root" in text
    assert "heldout_command_from_contract" in text
    assert 'run_shell_or_echo "$heldout_cmd"' in text


def test_pretest_remote_wrapper_wires_val_test_roots_and_artifacts():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "burgers1d/burgers1d_val.h5" in text
    assert "burgers1d/burgers1d_test.h5" in text
    assert "darcy2d/darcy2d_val.h5" in text
    assert "darcy2d/darcy2d_test.h5" in text
    assert "scripts/make_light_hdf5_shards.py" in text
    assert '--split-block-offset test="$ADVECTION_TEST_BLOCK_OFFSET"' in text
    assert '--test-count "$ADVECTION_TEST_COUNT"' in text
    assert "model_side_beta_head_pretest_" in text
    assert "remote-runs/model-side-beta-head-pretest" in text


def test_pretest_launcher_is_dry_run_first_and_uses_pretest_wrapper():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "DRY_RUN=${DRY_RUN:-1}" in text
    assert "ALLOW_HELDOUT_PRETEST=${ALLOW_HELDOUT_PRETEST:-0}" in text
    assert "scripts/run_remote_model_side_beta_head_pretest.sh" in text
    assert "ALLOW_HELDOUT_PRETEST=${ALLOW_HELDOUT_PRETEST}" in text


def test_pretest_remote_wrapper_dry_run_preview_uses_temp_outputs(tmp_path):
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["PUBLISH_ARTIFACTS"] = "0"
    env["DATA_ROOT"] = str(tmp_path / "data")
    env["HYDRATION_PLAN_JSON"] = "reports/research/sota_loop/official_advection_hydration_plan.json"
    env["SEQUENTIAL_HYDRATION_JSON"] = str(tmp_path / "sequential.json")
    env["HYDRATION_VALIDATION_JSON"] = str(tmp_path / "validation.json")
    env["OFFICIAL_SOURCE_ROOT"] = str(tmp_path / "official_source")
    env["OFFICIAL_LIGHT_ROOT"] = str(tmp_path / "official_light")
    env["OFFICIAL_SHARD_MANIFEST"] = str(tmp_path / "advection_manifest.yaml")
    env["FULL_TASK_ROOT"] = str(tmp_path / "full_task_root")
    env["FULL_ROOT_MANIFEST_JSON"] = str(tmp_path / "root_manifest.json")
    env["CHECKPOINT_SOURCE"] = str(tmp_path / "checkpoint_source")
    env["OUTPUT_ROOT"] = str(tmp_path / "output")

    proc = subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        env=env,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    assert "DRY_RUN command:" in proc.stdout
    assert "DRY_RUN shell command:" in proc.stdout
    assert "Pretest hydration plan shape validated." in proc.stdout
