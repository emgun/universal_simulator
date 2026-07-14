from __future__ import annotations

import json
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
    assert "Archived legacy workflow" in proc.stderr


def test_pretest_remote_wrapper_defaults_are_fail_closed():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "ARCHIVED_LEGACY_WORKFLOW=1" in text
    assert "must not be rerun" in text


def test_pretest_remote_wrapper_wires_val_test_roots_and_artifacts():
    text = SCRIPT.read_text(encoding="utf-8")

    assert text.index("exit 2") < text.index("scripts/make_light_hdf5_shards.py")


def test_pretest_launcher_is_dry_run_first_and_uses_pretest_wrapper():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "ARCHIVED_LEGACY_WORKFLOW=1" in text
    assert "must not be relaunched" in text


def test_pretest_remote_wrapper_dry_run_preview_uses_temp_outputs(tmp_path):
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["PUBLISH_ARTIFACTS"] = "0"
    env["DATA_ROOT"] = str(tmp_path / "data")
    plan_path = tmp_path / "hydration_plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "samples_per_file": 48,
                "train_count": 256,
                "val_count": 64,
                "reserved_test_count": 64,
                "stratified_split_policy": {
                    "val_block_offset": 32,
                    "test_block_offset": 40,
                },
                "held_out_test_policy": {
                    "test_split_downloaded": False,
                    "test_split_sharded": False,
                },
            }
        ),
        encoding="utf-8",
    )
    env["HYDRATION_PLAN_JSON"] = str(plan_path)
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

    assert proc.returncode == 2
    assert "Archived legacy workflow" in proc.stderr
