from __future__ import annotations

import os
import subprocess


def test_remote_medium_pipeline_dry_run_emits_medium_confirmation_commands(tmp_path):
    env = os.environ.copy()
    env.update(
        {
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "DRY_RUN": "1",
            "CHECK_B2": "0",
            "PREP_SHARDS": "1",
            "RUN_CANDIDATE": "1",
            "RUN_PERSISTENCE": "1",
            "PIPELINE_ROOT": str(tmp_path / "pipeline"),
            "OUTPUT_ROOT": str(tmp_path / "medium_runs"),
        }
    )

    proc = subprocess.run(
        ["bash", "scripts/run_remote_medium_confirmation.sh"],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert "Checking medium-v1 shard readiness" in proc.stdout
    assert "VERSION=medium-v1" in proc.stdout
    assert "TRAIN_COUNT=512" in proc.stdout
    assert "VAL_COUNT=128" in proc.stdout
    assert "TEST_COUNT=128" in proc.stdout
    assert "REMOTE_B2_PREFIX=medium-v1" in proc.stdout
    assert "RUN_NAME=ups_medium_shared_context_transport" in proc.stdout
    assert "evaluation.decoded_context_roll_shift_estimator" in proc.stdout
    assert "run_persistence_baseline.py" in proc.stdout
    assert "persistence_medium_v1_test" in proc.stdout
    assert "data/pdebench_light" not in proc.stdout
