from __future__ import annotations

import os
import stat
import subprocess
import tarfile


def test_run_remote_light_promotion_fetches_checkpoint_archive(tmp_path):
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
            "WORKDIR": os.getcwd(),
            "FETCH_DATA": "0",
            "CHECK_DATA": "0",
            "REMOTE_B2_PREFIX": "light-v1",
            "DRY_RUN": "0",
            "SKIP_TRAINING": "1",
            "CHECKPOINT_SOURCE": str(checkpoint_source),
            "CHECKPOINT_SOURCE_B2_KEY": "remote-runs/checkpoints/checkpoint.tar.gz",
            "OUTPUT_ROOT": str(tmp_path / "out"),
            "RUN_NAME": "checkpoint_fetch_test",
        }
    )

    proc = subprocess.run(
        ["bash", "scripts/run_remote_light_promotion.sh"],
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
