from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _write_fake_rclone(path: Path, *, empty_first: bool) -> None:
    if empty_first:
        body = """#!/usr/bin/env bash
set -euo pipefail
if [ "$1" = "lsjson" ]; then
  echo '[]'
  exit 0
fi
echo "unexpected rclone command: $*" >&2
exit 2
"""
    else:
        body = """#!/usr/bin/env bash
set -euo pipefail
if [ "$1" = "lsjson" ]; then
  echo '[{"Name":"dataset.h5"}]'
  exit 0
fi
if [ "$1" = "copy" ]; then
  echo "COPY $2 $3"
  exit 0
fi
echo "unexpected rclone command: $*" >&2
exit 2
"""
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def _env(tmp_path: Path, fake_bin: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "B2_KEY_ID": "key",
            "B2_APP_KEY": "app",
            "B2_BUCKET": "bucket",
            "B2_PREFIX": "full",
            "DATA_ROOT": str(tmp_path / "data"),
            "CLEAN_OLD_SPLITS": "0",
            "DRY_RUN": "0",
        }
    )
    return env


def test_fetch_datasets_rejects_empty_lsjson_success(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_fake_rclone(fake_bin / "rclone", empty_first=True)

    proc = subprocess.run(
        ["bash", "scripts/fetch_datasets_b2.sh", "missing/file.h5"],
        capture_output=True,
        env=_env(tmp_path, fake_bin),
        text=True,
    )

    assert proc.returncode == 1
    assert "could not locate dataset" in proc.stderr
    assert "Copying" not in proc.stdout


def test_fetch_datasets_accepts_non_empty_lsjson(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_fake_rclone(fake_bin / "rclone", empty_first=False)

    proc = subprocess.run(
        ["bash", "scripts/fetch_datasets_b2.sh", "present/file.h5"],
        check=True,
        capture_output=True,
        env=_env(tmp_path, fake_bin),
        text=True,
    )

    assert "Copying bucket/full/present/file.h5" in proc.stdout
    assert "COPY UPSB2:bucket/full/present/file.h5" in proc.stdout
