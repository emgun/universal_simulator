from __future__ import annotations

import os
import stat
import subprocess
import tarfile


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


def test_remote_medium_pipeline_fetches_checkpoint_archive(tmp_path):
    checkpoint_parent = tmp_path / "checkpoints"
    checkpoint_source = checkpoint_parent / "source_ckpt"
    archive_source_root = tmp_path / "archive_src" / "source_ckpt"
    archive_source_root.mkdir(parents=True)
    (archive_source_root / "marker.txt").write_text("checkpoint", encoding="utf-8")
    archive_path = tmp_path / "checkpoint.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(archive_source_root, arcname="source_ckpt")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    rclone_log = tmp_path / "rclone.log"
    fake_rclone = fake_bin / "rclone"
    fake_rclone.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                'echo "$@" >> "$RCLONE_LOG"',
                'if [ "$1" = copyto ]; then',
                '  cp "$FAKE_CHECKPOINT_ARCHIVE" "$3"',
                "fi",
            ]
        ),
        encoding="utf-8",
    )
    fake_rclone.chmod(fake_rclone.stat().st_mode | stat.S_IXUSR)

    python_log = tmp_path / "python.log"
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                'echo "$@" >> "$PYTHON_LOG"',
            ]
        ),
        encoding="utf-8",
    )
    fake_python.chmod(fake_python.stat().st_mode | stat.S_IXUSR)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
            "RCLONE_LOG": str(rclone_log),
            "FAKE_CHECKPOINT_ARCHIVE": str(archive_path),
            "PYTHON_LOG": str(python_log),
            "B2_KEY_ID": "key-id",
            "B2_APP_KEY": "app-key",
            "B2_BUCKET": "bucket",
            "B2_S3_ENDPOINT": "https://example.invalid",
            "B2_S3_REGION": "us-west-000",
            "DRY_RUN": "0",
            "CHECK_B2": "0",
            "ALLOW_UNCHECKED_LIVE_RUNS": "1",
            "PREP_SHARDS": "0",
            "FETCH_DATA": "0",
            "RUN_CANDIDATE": "1",
            "RUN_PERSISTENCE": "0",
            "SKIP_TRAINING": "1",
            "CHECKPOINT_SOURCE": str(checkpoint_source),
            "CHECKPOINT_SOURCE_B2_KEY": "remote-runs/checkpoints/checkpoint.tar.gz",
            "PIPELINE_ROOT": str(tmp_path / "pipeline"),
            "OUTPUT_ROOT": str(tmp_path / "medium_runs"),
            "DATA_ROOT": str(tmp_path / "data"),
        }
    )

    proc = subprocess.run(
        ["bash", "scripts/run_remote_medium_confirmation.sh"],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert (
        "Hydrating checkpoint source from b2://bucket/remote-runs/checkpoints/checkpoint.tar.gz"
        in proc.stdout
    )
    assert (checkpoint_source / "marker.txt").read_text(encoding="utf-8") == "checkpoint"
    assert "copyto UPSB2:bucket/remote-runs/checkpoints/checkpoint.tar.gz" in rclone_log.read_text(
        encoding="utf-8"
    )
    assert f"--checkpoint-source {checkpoint_source}" in python_log.read_text(encoding="utf-8")


def test_remote_medium_pipeline_publishes_artifacts_when_requested(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    rclone_log = tmp_path / "rclone.log"
    fake_rclone = fake_bin / "rclone"
    fake_rclone.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                'echo "$@" >> "$RCLONE_LOG"',
            ]
        ),
        encoding="utf-8",
    )
    fake_rclone.chmod(fake_rclone.stat().st_mode | stat.S_IXUSR)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
            "RCLONE_LOG": str(rclone_log),
            "B2_KEY_ID": "key-id",
            "B2_APP_KEY": "app-key",
            "B2_BUCKET": "bucket",
            "DRY_RUN": "0",
            "CHECK_B2": "0",
            "ALLOW_UNCHECKED_LIVE_RUNS": "1",
            "PREP_SHARDS": "0",
            "FETCH_DATA": "0",
            "RUN_CANDIDATE": "0",
            "RUN_PERSISTENCE": "0",
            "PUBLISH_MEDIUM_ARTIFACTS": "1",
            "MEDIUM_ARTIFACT_PREFIX": "remote-runs/medium",
            "PIPELINE_ROOT": str(tmp_path / "pipeline"),
            "OUTPUT_ROOT": str(tmp_path / "medium_runs"),
            "DATA_ROOT": str(tmp_path / "data"),
        }
    )

    proc = subprocess.run(
        ["bash", "scripts/run_remote_medium_confirmation.sh"],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert "Published medium artifacts: b2://bucket/remote-runs/medium/" in proc.stdout
    rclone_output = rclone_log.read_text(encoding="utf-8")
    assert "copyto /tmp/" in rclone_output
    assert "UPSB2:bucket/remote-runs/medium/" in rclone_output
