from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REMOTE = ROOT / "scripts/run_remote_reference_recipe_adequacy.sh"
LAUNCHER = ROOT / "scripts/launch_reference_recipe_adequacy_vast.sh"


def _run(
    script: Path, *, env: dict[str, str], check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(script)],
        cwd=ROOT,
        env=env,
        check=check,
        capture_output=True,
        text=True,
    )


def test_remote_wrapper_defaults_to_non_mutating_validation_only_preview(tmp_path):
    scratch = tmp_path / "scratch"
    pipeline = tmp_path / "pipeline"
    output = tmp_path / "output"
    env = os.environ.copy()
    env.update(
        {
            "SCRATCH_ROOT": str(scratch),
            "PIPELINE_ROOT": str(pipeline),
            "OUTPUT_ROOT": str(output),
        }
    )

    result = _run(REMOTE, env=env)

    assert "DRY_RUN=1: validation-only preview" in result.stdout
    assert "ups.data.cli stage" in result.stdout
    assert "training.lock.json" in result.stdout
    assert "--run-set discovery" in result.stdout
    assert "stop cleanly" in result.stdout
    assert "immutable/sha256/<sha256>" in result.stdout
    assert not scratch.exists()
    assert not pipeline.exists()
    assert not output.exists()


def test_remote_wrapper_has_selection_bound_confirmation_and_finalizer_contract():
    source = REMOTE.read_text(encoding="utf-8")

    assert "NEURALOPERATOR_VERSION=${NEURALOPERATOR_VERSION:-2.0.0}" in source
    assert '--selection-artifact "$SELECTION_ARTIFACT"' in source
    assert '--discovery-plan "$DISCOVERY_PLAN"' in source
    assert "--run-set confirmation" in source
    assert "s29_confirmation_val/summary.json" in source
    assert "s43_confirmation_val/summary.json" in source
    assert "scripts/finalize_reference_recipe_adequacy.py" in source
    assert '--discovery-summary "$discovery_summary"' in source
    assert source.count("--confirmation-summary") == 2
    assert 'remote_key="${ARTIFACT_PREFIX%/}/immutable/sha256/${digest}/${archive_name}"' in source
    assert 'tar -czf "$archive_path" "$PIPELINE_ROOT" "$OUTPUT_ROOT"' in source
    assert "measurement.lock" not in source
    assert "_test.h5" not in source


def test_vast_launcher_preview_is_single_gpu_bounded_and_auto_shutdown(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    args_log = tmp_path / "args.log"
    fake_python = fake_bin / "python"
    fake_python.write_text(
        '#!/usr/bin/env bash\nprintf \'%s\\n\' "$@" > "$ARGS_LOG"\n',
        encoding="utf-8",
    )
    fake_python.chmod(fake_python.stat().st_mode | stat.S_IXUSR)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
            "ARGS_LOG": str(args_log),
            "GIT_REF": "codex/reference-recipe-adequacy",
            "ENV_FILE": str(tmp_path / "absent.env"),
        }
    )

    _run(LAUNCHER, env=env)
    args = args_log.read_text(encoding="utf-8").splitlines()

    assert args[:2] == ["scripts/vast_launch.py", "launch"]
    assert args[args.index("--num-gpus") + 1] == "1"
    assert args[args.index("--disk") + 1] == "64"
    assert args[args.index("--remote-script") + 1].endswith(
        "run_remote_reference_recipe_adequacy.sh"
    )
    assert "--dry-run" in args
    assert "--auto-shutdown" in args
    assert "--skip-prefetch" in args
    assert "tracked-script" in args
    # The preview records the paid-run ceiling without contacting Vast.
    # (The fake Python process captures launcher arguments, so inspect source.)
    launcher_source = LAUNCHER.read_text(encoding="utf-8")
    assert "MAX_DPH=${MAX_DPH:-0.45}" in launcher_source
    assert "dph_total<=${MAX_DPH}" in launcher_source
    assert 'git cat-file -e "${remote_commit}:${REMOTE_SCRIPT}"' in launcher_source


def test_vast_launcher_refuses_paid_run_without_b2_credentials(tmp_path):
    env = os.environ.copy()
    for key in ("B2_KEY_ID", "B2_APP_KEY", "B2_BUCKET"):
        env.pop(key, None)
    env.update(
        {
            "DRY_RUN": "0",
            "GIT_REF": "codex/reference-recipe-adequacy",
            "ENV_FILE": str(tmp_path / "absent.env"),
        }
    )

    result = _run(LAUNCHER, env=env, check=False)

    assert result.returncode == 2
    assert "missing B2 credentials" in result.stderr


def test_vast_launcher_refuses_unbounded_disk_before_launch(tmp_path):
    env = os.environ.copy()
    env.update(
        {
            "DISK_GB": "120",
            "MAX_DISK_GB": "96",
            "ENV_FILE": str(tmp_path / "absent.env"),
        }
    )

    result = _run(LAUNCHER, env=env, check=False)

    assert result.returncode == 2
    assert "bounded maximum is 96 GB" in result.stderr


def test_vast_launcher_refuses_non_positive_disk_before_launch(tmp_path):
    env = os.environ.copy()
    env.update({"DISK_GB": "0", "ENV_FILE": str(tmp_path / "absent.env")})

    result = _run(LAUNCHER, env=env, check=False)

    assert result.returncode == 2
    assert "DISK_GB must be a positive integer" in result.stderr
