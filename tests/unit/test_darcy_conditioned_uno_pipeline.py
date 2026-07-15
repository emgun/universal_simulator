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


def test_planner_binds_exact_d3_baseline_and_transitive_sources() -> None:
    planner = load_script("plan_darcy_conditioned_uno_ablation.py")
    assert planner.D3_BASELINE == {
        "selected_epoch": 384,
        "primary_value": 0.11694165553982801,
        "beta100_global_scale_nrmse": 0.25762147974587746,
        "maximum_corrected_spread_ratio": 2.2029915564017024,
        "plateau_epoch": 384,
    }
    assert planner.D3_ARTIFACT_SHA256 == (
        "25d29a00d69233acfd2e05789f4dd6bd5488a8011cc584a7a11ba8d384d1ee1a"
    )
    required = {
        "scripts/run_darcy_conditioned_uno_ablation.py",
        "scripts/run_darcy_fno_regime_balanced_objective.py",
        "scripts/run_darcy_fno_affine_head_ablation.py",
        "scripts/run_external_neuraloperator_uno_baseline.py",
        "src/ups/training/resumable_checkpoint.py",
        "scripts/materialize_darcy_conditioned_uno_ablation.py",
    }
    assert required.issubset(planner.SOURCE_FILES)


def test_planner_accepts_only_exact_self_hashed_d3_result(tmp_path: Path) -> None:
    planner = load_script("plan_darcy_conditioned_uno_ablation.py")
    source = (
        ROOT / "docs/research/artifacts/strat_v1_darcy_fno_regime_balanced_objective_result.json"
    )
    assert planner.checked_d3(source)["artifact_sha256"] == planner.D3_ARTIFACT_SHA256
    payload = json.loads(source.read_text())
    payload["arms"]["R-mean"]["primary_value"] += 1e-12
    payload["artifact_sha256"] = canonical_sha256(
        {k: v for k, v in payload.items() if k != "artifact_sha256"}
    )
    tampered = tmp_path / "d3.json"
    tampered.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="frozen artifact"):
        planner.checked_d3(tampered)


def test_materializer_accepts_only_self_hashed_validation_plan(tmp_path: Path) -> None:
    materializer = load_script("materialize_darcy_conditioned_uno_ablation.py")
    plan = {
        "schema_version": 2,
        "mode": "validation_only",
        "heldout_access": "forbidden",
        "measurement_lock_access": "forbidden",
        "command": ["python", "scripts/run_darcy_conditioned_uno_ablation.py"],
    }
    plan["command_sha256"] = canonical_sha256(plan["command"])
    plan["plan_sha256"] = canonical_sha256(plan)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    assert materializer.load_and_validate_plan(path)["plan_sha256"] == plan["plan_sha256"]
    plan["heldout_access"] = "allowed"
    plan["plan_sha256"] = canonical_sha256({k: v for k, v in plan.items() if k != "plan_sha256"})
    path.write_text(json.dumps(plan))
    with pytest.raises(PermissionError):
        materializer.load_and_validate_plan(path)


def test_materializer_handles_float32_regimes_and_plateau() -> None:
    materializer = load_script("materialize_darcy_conditioned_uno_ablation.py")
    assert materializer.regime_values_match(
        [0.009999999776482582, 0.10000000149011612, 1.0, 10.0, 100.0]
    )
    history = [
        {"epoch": 96, "primary_value": 1.0},
        {"epoch": 192, "primary_value": 0.995},
        {"epoch": 384, "primary_value": 0.994},
    ]
    assert materializer.plateau_epoch(history, 0.01, 2) == 384


def test_remote_pipeline_is_sealed_plan_keyed_and_has_no_repair_escape() -> None:
    remote = (ROOT / "scripts/run_remote_darcy_conditioned_uno_ablation.sh").read_text()
    launcher = (ROOT / "scripts/launch_darcy_conditioned_uno_ablation_vast.sh").read_text()
    assert 'heldout_access") != "forbidden"' in remote
    assert 'measurement_lock_access") != "forbidden"' in remote
    assert 'temporary_prefix="${ARTIFACT_PREFIX%/}/resumable/${plan_sha}"' in remote
    assert "remote_digest=$(rclone cat" in remote
    assert '"$remote_digest" = "$digest"' in remote
    assert "rclone purge" in remote
    assert "repair" not in remote.lower()
    assert "Published immutable Darcy D4 artifact:" in remote
    assert "Published immutable Darcy D4 artifact:" in launcher
    assert "--auto-shutdown" in launcher and "--managed" in launcher
    assert "--launch-retries 0" in launcher
