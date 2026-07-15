from __future__ import annotations

import importlib.util
import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

from ups.data.manifests import canonical_sha256

ROOT = Path(__file__).resolve().parents[2]
REMOTE = ROOT / "scripts/run_remote_strat_v1_shared_tier_b.sh"
LAUNCHER = ROOT / "scripts/launch_strat_v1_shared_tier_b_vast.sh"


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_accepts_only_self_hashed_sealed_plan(tmp_path: Path) -> None:
    runner = load_script("run_strat_v1_shared_tier_b.py")
    plan = {
        "schema_version": 1,
        "mode": "validation_only",
        "heldout_access": "forbidden",
        "measurement_lock_access": "forbidden",
        "command": ["python", "scripts/run_strat_v1_shared_tier_b.py"],
    }
    plan["command_sha256"] = canonical_sha256(plan["command"])
    plan["plan_sha256"] = canonical_sha256(plan)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")

    assert runner._checked_plan(path)["plan_sha256"] == plan["plan_sha256"]

    plan["measurement_lock_access"] = "allowed"
    plan["plan_sha256"] = canonical_sha256(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(PermissionError, match="measurement-lock"):
        runner._checked_plan(path)


def test_planner_covers_transitive_sources_and_only_training_objects() -> None:
    planner = load_script("plan_strat_v1_shared_tier_b.py")
    required = {
        "configs/d5_strat_v1_shared_tier_b.yaml",
        "scripts/run_strat_v1_shared_tier_b.py",
        "scripts/plan_strat_v1_shared_tier_b.py",
        "scripts/materialize_strat_v1_shared_tier_b.py",
        "scripts/run_remote_strat_v1_shared_tier_b.sh",
        "scripts/launch_strat_v1_shared_tier_b_vast.sh",
        "scripts/d5_presigned_io.py",
        "scripts/generate_b2_presigned_bundle.py",
        "scripts/finalize_d5_presigned_transfer.py",
        "scripts/run_light_experiment.py",
        "scripts/train.py",
        "scripts/evaluate.py",
        "src/ups/data/latent_pairs.py",
        "src/ups/data/parameter_conditioning.py",
        "src/ups/eval/pdebench_runner.py",
    }
    assert required.issubset(planner.source_paths())
    assert set(planner.OBJECTS) == {
        "advection1d-train",
        "burgers1d-train",
        "darcy2d-train",
        "advection1d-valid",
        "burgers1d-valid",
        "darcy2d-valid",
    }
    assert all("test" not in object_id for object_id in planner.OBJECTS)


def test_remote_dry_run_is_non_mutating_and_validation_only(tmp_path: Path) -> None:
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
        ["bash", str(REMOTE)],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "ups.data.cli stage" in result.stdout
    assert "run_strat_v1_shared_tier_b.py" in result.stdout
    assert "materialize_strat_v1_shared_tier_b.py" in result.stdout
    assert not (tmp_path / "cache").exists()
    assert not (tmp_path / "data").exists()
    assert not (tmp_path / "output").exists()


def test_remote_positional_dry_run_override_reaches_real_preflight(tmp_path: Path) -> None:
    env = os.environ.copy()
    for key in ("B2_KEY_ID", "B2_APP_KEY", "B2_BUCKET", "B2_S3_ENDPOINT", "B2_S3_REGION"):
        env.pop(key, None)
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


def test_remote_pipeline_uses_only_sealed_presigned_transfer_slots() -> None:
    remote = REMOTE.read_text(encoding="utf-8")
    assert "$PYTHON -m pip install -e . --no-deps" in remote
    assert "$PYTHON -m pip install -e .\n" not in remote
    assert 'heldout_access") != "forbidden"' in remote
    assert 'measurement_lock_access") != "forbidden"' in remote
    assert "scripts/d5_presigned_io.py fetch-manifest" in remote
    assert "scripts/d5_presigned_io.py fetch-resume" in remote
    assert "scripts/d5_presigned_io.py preserve" in remote
    assert "scripts/d5_presigned_io.py publish" in remote
    assert "UPS_B2_PRESIGNED_URLS_FILE" in remote
    assert "B2_KEY_ID" not in remote
    assert "B2_APP_KEY" not in remote
    assert "RCLONE_CONFIG" not in remote
    assert "rclone " not in remote
    assert "RUN_LOG=${RUN_LOG:-reports/research/strat_v1_shared_tier_b.remote.log}" in remote
    assert '2>&1 | tee "$RUN_LOG"' in remote
    assert '"$RESULT" "$RUN_LOG"' in remote
    assert "trusted local finalization is required" in remote
    assert "measurement.lock" not in remote
    assert "_test.h5" not in remote


def test_vast_dry_run_is_managed_bounded_and_auto_shutdown(tmp_path: Path) -> None:
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
    for key in ("B2_KEY_ID", "B2_APP_KEY", "B2_BUCKET", "B2_S3_ENDPOINT", "B2_S3_REGION"):
        env.pop(key, None)
    env.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
            "ARGS_LOG": str(args_log),
            "DRY_RUN": "1",
            "GIT_REF": "a" * 40,
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
    assert args[args.index("--num-gpus") + 1] == "1"
    assert args[args.index("--max-runtime-minutes") + 1] == "600"
    assert args[args.index("--success-marker") + 1] == "Uploaded verified D5 ingress artifact:"
    assert args[args.index("--launch-retries") + 1] == "0"
    assert not any("b2-key" in arg or "b2-app" in arg for arg in args)


def test_shells_parse() -> None:
    subprocess.run(["bash", "-n", str(REMOTE)], cwd=ROOT, check=True)
    subprocess.run(["bash", "-n", str(LAUNCHER)], cwd=ROOT, check=True)
