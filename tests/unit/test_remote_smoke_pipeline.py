from __future__ import annotations

import os
import subprocess


def test_remote_smoke_pipeline_generates_queue_without_b2_or_training(tmp_path):
    env = os.environ.copy()
    env.update(
        {
            "CHECK_B2": "0",
            "PREP_SHARDS": "0",
            "RUN_EXPERIMENTS": "0",
            "DRY_RUN": "1",
            "PIPELINE_ROOT": str(tmp_path / "pipeline"),
            "QUEUE_VARIANTS": "current_best no_conditioning",
        }
    )

    proc = subprocess.run(
        ["bash", "scripts/run_remote_smoke_pipeline.sh"],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    queue_dir = tmp_path / "pipeline" / "queue"
    assert "PREP_SHARDS=0: leaving smoke shards missing." in proc.stdout
    assert "RUN_EXPERIMENTS=0: generated queue only." in proc.stdout
    assert (tmp_path / "pipeline" / "readiness_before.json").exists()
    assert (queue_dir / "smoke_queue.jsonl").exists()
    assert (queue_dir / "smoke_queue.tsv").exists()
    assert (queue_dir / "run_smoke_queue.sh").exists()
    assert "ups_smoke_current_best" in (queue_dir / "run_smoke_queue.sh").read_text(
        encoding="utf-8"
    )
    assert "ups_smoke_no_conditioning" in (queue_dir / "run_smoke_queue.sh").read_text(
        encoding="utf-8"
    )


def test_remote_smoke_pipeline_accepts_cli_assignments(tmp_path):
    proc = subprocess.run(
        [
            "bash",
            "scripts/run_remote_smoke_pipeline.sh",
            "CHECK_B2=0",
            "PREP_SHARDS=0",
            "RUN_EXPERIMENTS=0",
            "DRY_RUN=1",
            f"PIPELINE_ROOT={tmp_path / 'pipeline'}",
            "QUEUE_VARIANTS=current_best",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "RUN_EXPERIMENTS=0: generated queue only." in proc.stdout
    assert (tmp_path / "pipeline" / "queue" / "run_smoke_queue.sh").exists()


def test_remote_smoke_pipeline_keeps_queue_dry_run_when_prep_is_live(tmp_path):
    env = os.environ.copy()
    env.update(
        {
            "CHECK_B2": "0",
            "PREP_SHARDS": "0",
            "RUN_EXPERIMENTS": "0",
            "DRY_RUN": "0",
            "PIPELINE_ROOT": str(tmp_path / "pipeline"),
            "QUEUE_VARIANTS": "current_best",
        }
    )

    subprocess.run(
        ["bash", "scripts/run_remote_smoke_pipeline.sh"],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    queue_script = (tmp_path / "pipeline" / "queue" / "run_smoke_queue.sh").read_text(
        encoding="utf-8"
    )
    assert "DRY_RUN=1" in queue_script
    assert "RUN_NAME=ups_smoke_current_best" in queue_script


def test_remote_smoke_pipeline_refuses_live_queue_when_b2_shards_are_missing(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_rclone = fake_bin / "rclone"
    fake_rclone.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [ "$1" = "size" ]; then
  echo '{"count":0,"bytes":0}'
  exit 0
fi
echo "unexpected rclone command: $*" >&2
exit 2
""",
        encoding="utf-8",
    )
    fake_rclone.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "B2_KEY_ID": "key",
            "B2_APP_KEY": "app",
            "B2_BUCKET": "bucket",
            "CHECK_B2": "1",
            "PREP_SHARDS": "0",
            "RUN_EXPERIMENTS": "1",
            "QUEUE_DRY_RUN": "0",
            "DRY_RUN": "0",
            "PIPELINE_ROOT": str(tmp_path / "pipeline"),
            "QUEUE_VARIANTS": "current_best",
        }
    )

    proc = subprocess.run(
        ["bash", "scripts/run_remote_smoke_pipeline.sh"],
        capture_output=True,
        env=env,
        text=True,
    )

    assert proc.returncode == 1
    assert "Refusing live smoke experiments because smoke shards are not ready" in proc.stderr
    assert not (tmp_path / "pipeline" / "queue" / "run_smoke_queue.sh").exists()


def test_remote_smoke_pipeline_refuses_unchecked_live_queue(tmp_path):
    env = os.environ.copy()
    env.update(
        {
            "CHECK_B2": "0",
            "PREP_SHARDS": "0",
            "RUN_EXPERIMENTS": "1",
            "QUEUE_DRY_RUN": "0",
            "DRY_RUN": "0",
            "PIPELINE_ROOT": str(tmp_path / "pipeline"),
            "QUEUE_VARIANTS": "current_best",
        }
    )

    proc = subprocess.run(
        ["bash", "scripts/run_remote_smoke_pipeline.sh"],
        capture_output=True,
        env=env,
        text=True,
    )

    assert proc.returncode == 1
    assert "Refusing live smoke experiments without CHECK_B2=1" in proc.stderr
    assert not (tmp_path / "pipeline" / "queue" / "run_smoke_queue.sh").exists()
