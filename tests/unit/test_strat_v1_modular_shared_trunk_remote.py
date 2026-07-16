from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

from scripts import finalize_d5_presigned_transfer as finalizer
from tests.unit.test_d5_presigned_io import _manifest
from tests.unit.test_finalize_d5_presigned_transfer import FakeS3, _receipt

ROOT = Path(__file__).resolve().parents[2]
REMOTE = ROOT / "scripts/run_remote_strat_v1_modular_shared_trunk.sh"
LAUNCHER = ROOT / "scripts/launch_strat_v1_modular_shared_trunk_vast.sh"


def test_d6_remote_dry_run_is_non_mutating_and_uses_d6_commands(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "CACHE": str(tmp_path / "cache"),
            "DATA_ROOT": str(tmp_path / "data"),
            "OUTPUT_DIR": str(tmp_path / "output"),
            "RESULT": str(tmp_path / "result.json"),
            "STAGE_REPORT": str(tmp_path / "stage.json"),
        }
    )
    result = subprocess.run(
        ["bash", str(REMOTE)], cwd=ROOT, env=env, check=True, capture_output=True, text=True
    )

    assert "ups.data.cli stage" in result.stdout
    assert "run_strat_v1_modular_shared_trunk.py" in result.stdout
    assert "materialize_strat_v1_modular_shared_trunk.py" in result.stdout
    assert not (tmp_path / "cache").exists()
    assert not (tmp_path / "data").exists()
    assert not (tmp_path / "output").exists()


def test_d6_remote_real_override_fails_before_staging_without_capability(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "CACHE": str(tmp_path / "cache"),
            "DATA_ROOT": str(tmp_path / "data"),
            "OUTPUT_DIR": str(tmp_path / "output"),
        }
    )
    result = subprocess.run(
        ["bash", str(REMOTE), "DRY_RUN=0", "ARTIFACT_PREFIX=test-prefix"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Set TRANSFER_MANIFEST_URL_B64" in result.stderr
    assert "ups.data.cli stage" not in result.stdout
    assert not (tmp_path / "cache").exists()
    assert not (tmp_path / "data").exists()
    assert not (tmp_path / "output").exists()


def test_d6_remote_uses_only_sealed_signed_transport() -> None:
    source = REMOTE.read_text(encoding="utf-8")
    assert "scripts/d5_presigned_io.py fetch-manifest" in source
    assert "scripts/d5_presigned_io.py fetch-resume" in source
    assert "scripts/d5_presigned_io.py preserve" in source
    assert "scripts/d5_presigned_io.py publish" in source
    assert "UPS_B2_PRESIGNED_URLS_FILE" in source
    assert "$PYTHON -m pip install -e . --no-deps" in source
    assert 'heldout_access") != "forbidden"' in source
    assert 'measurement_lock_access") != "forbidden"' in source
    assert "B2_KEY_ID" not in source
    assert "B2_APP_KEY" not in source
    assert "RCLONE_CONFIG" not in source
    assert "rclone " not in source
    assert "measurement.lock" not in source
    assert "_test.h5" not in source
    assert '2>&1 | tee "$RUN_LOG"' in source
    assert '"$PLAN" "$CONFIG" "$STAGE_REPORT" "$OUTPUT_DIR" "$RESULT" "$RUN_LOG"' in source
    assert "Uploaded verified D6 ingress artifact:" in source


def test_d6_vast_dry_run_is_managed_bounded_and_credential_free(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    args_log = tmp_path / "args.log"
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        'if [ "$1" = "scripts/vast_launch.py" ]; then\n'
        '  printf \'%s\\n\' "$@" > "$ARGS_LOG"\n'
        "else\n"
        f'  exec "{sys.executable}" "$@"\n'
        "fi\n"
    )
    fake_python.chmod(fake_python.stat().st_mode | stat.S_IXUSR)
    env = os.environ.copy()
    for key in ("B2_KEY_ID", "B2_APP_KEY", "B2_BUCKET", "B2_S3_ENDPOINT", "B2_S3_REGION"):
        env.pop(key, None)
    env.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
            "ARGS_LOG": str(args_log),
            "DRY_RUN": "1",
            "GIT_REF": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "ENV_FILE": str(tmp_path / "absent.env"),
        }
    )

    subprocess.run(["bash", str(LAUNCHER)], cwd=ROOT, env=env, check=True)
    args = args_log.read_text(encoding="utf-8").splitlines()

    assert args[:2] == ["scripts/vast_launch.py", "launch"]
    assert "--dry-run" in args
    assert "--managed" in args
    assert "--auto-shutdown" in args
    assert "--skip-prefetch" in args
    assert "--skip-rclone-install" in args
    assert args[args.index("--remote-script") + 1].endswith(
        "run_remote_strat_v1_modular_shared_trunk.sh"
    )
    assert args[args.index("--success-marker") + 1] == "Uploaded verified D6 ingress artifact:"
    assert args[args.index("--launch-retries") + 1] == "0"
    assert not any("b2-key" in arg or "b2-app" in arg for arg in args)


def test_finalizer_supports_d6_archive_identity_without_changing_d5_default(
    tmp_path: Path, monkeypatch
) -> None:
    manifest_path = _manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    artifact = b"verified D6 archive"
    digest = hashlib.sha256(artifact).hexdigest()
    client = FakeS3(manifest, artifact, digest)
    monkeypatch.setattr("scripts.d5_presigned_io.time.time", lambda: 1500)

    handle = finalizer.finalize(
        manifest_path,
        env_file=tmp_path / "absent.env",
        client=client,
        bucket="pdebench",
        receipt_path=_receipt(tmp_path),
        archive_stem="strat_v1_modular_shared_trunk",
        workflow_label="D6",
    )

    assert handle.endswith(f"/strat_v1_modular_shared_trunk_{manifest['launch_id']}.tar.gz")


def test_d6_shells_parse() -> None:
    subprocess.run(["bash", "-n", str(REMOTE)], cwd=ROOT, check=True)
    subprocess.run(["bash", "-n", str(LAUNCHER)], cwd=ROOT, check=True)
