from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from ups.data.manifests import canonical_sha256

ROOT = Path(__file__).resolve().parents[2]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_planner_binds_frozen_d2_control_and_d3_sources() -> None:
    planner = load_script("plan_darcy_fno_regime_balanced_objective.py")
    assert planner.D2_CONTROL_PRIMARY == 0.12244374883447355
    assert "scripts/run_darcy_fno_regime_balanced_objective.py" in planner.SOURCE_FILES
    assert "scripts/run_darcy_fno_affine_head_ablation.py" in planner.SOURCE_FILES
    assert "scripts/materialize_darcy_fno_regime_balanced_objective.py" in planner.SOURCE_FILES


def test_materializer_accepts_only_self_hashed_validation_plan(tmp_path: Path) -> None:
    materializer = load_script("materialize_darcy_fno_regime_balanced_objective.py")
    plan = {
        "schema_version": 2,
        "mode": "validation_only",
        "heldout_access": "forbidden",
        "measurement_lock_access": "forbidden",
        "command": ["python", "scripts/run_darcy_fno_regime_balanced_objective.py"],
    }
    plan["command_sha256"] = canonical_sha256(plan["command"])
    plan["plan_sha256"] = canonical_sha256(plan)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    assert materializer.load_and_validate_plan(path)["plan_sha256"] == plan["plan_sha256"]
    plan["heldout_access"] = "allowed"
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(PermissionError):
        materializer.load_and_validate_plan(path)


def test_materializer_handles_float32_beta_labels_and_candidate_plateau() -> None:
    materializer = load_script("materialize_darcy_fno_regime_balanced_objective.py")
    assert materializer.regime_values_match(
        [0.009999999776482582, 0.10000000149011612, 1.0, 10.0, 100.0]
    )
    history = [
        {"epoch": 96, "primary_value": 1.0},
        {"epoch": 192, "primary_value": 0.995},
        {"epoch": 384, "primary_value": 0.994},
    ]
    assert materializer.plateau_epoch(history, 0.01, 2) == 384


def test_remote_pipeline_is_sealed_and_plan_keyed() -> None:
    remote = (ROOT / "scripts/run_remote_darcy_fno_regime_balanced_objective.sh").read_text()
    launcher = (ROOT / "scripts/launch_darcy_fno_regime_balanced_objective_vast.sh").read_text()
    assert 'heldout_access") != "forbidden"' in remote
    assert 'measurement_lock_access") != "forbidden"' in remote
    assert 'temporary_prefix="${ARTIFACT_PREFIX%/}/resumable/${plan_sha}"' in remote
    assert "remote_digest=$(rclone cat" in remote
    assert '"$remote_digest" = "$digest"' in remote
    assert "Published immutable Darcy D3 artifact:" in remote
    assert "Published immutable Darcy D3 artifact:" in launcher
    assert "--auto-shutdown" in launcher and "--managed" in launcher
