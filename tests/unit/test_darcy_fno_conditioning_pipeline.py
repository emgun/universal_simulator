from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from scripts import materialize_darcy_fno_conditioning_ablation as materialize
from ups.data.manifests import canonical_sha256

REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, *, shuffled: float = 0.8) -> tuple[Path, Path, Path]:
    runner = REPO_ROOT / "scripts/run_darcy_fno_conditioning_ablation.py"
    materializer_path = REPO_ROOT / "scripts/materialize_darcy_fno_conditioning_ablation.py"
    current_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    object_hashes = {
        "darcy2d-train": "a" * 64,
        "darcy2d-valid": "b" * 64,
    }
    plan = {
        "schema_version": 1,
        "mode": "validation_only",
        "heldout_access": "forbidden",
        "bindings": {
            "source": {
                "implementation_commit": current_commit,
                "files": {
                    str(runner.relative_to(REPO_ROOT)): _sha(runner),
                },
            },
            "runner": {"path": str(runner.relative_to(REPO_ROOT)), "file_sha256": _sha(runner)},
            "materializer": {
                "path": str(materializer_path.relative_to(REPO_ROOT)),
                "file_sha256": _sha(materializer_path),
            },
            "training_lock": {"lock_sha256": "c" * 64, "darcy_objects": object_hashes},
        },
        "dependencies": {"neuraloperator": "2.0.0"},
        "design": {"arms": ["U", "K"], "seed": 17, "epoch_rungs": [3, 6, 12, 24]},
        "gates": {
            "conditioned_primary_relative_improvement_minimum": 0.10,
            "conditioned_max_corrected_regime_spread_maximum": 1.5,
            "conditioned_counterfactual_relative_prediction_rms_minimum_exclusive": 1e-6,
            "shuffled_beta_primary_relative_degradation_minimum": 0.05,
            "plateau": {
                "best_so_far_relative_improvement_threshold": 0.01,
                "must_occur_by_epoch": 24,
            },
        },
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))

    arms = {}
    for arm, values in {"U": [1.0, 0.9, 0.9, 0.9], "K": [0.7, 0.6, 0.6, 0.6]}.items():
        checkpoints = {}
        history = []
        for epoch, value in zip([3, 6, 12, 24], values, strict=True):
            checkpoint = tmp_path / f"{arm}-{epoch}.pt"
            checkpoint.write_bytes(f"{arm}-{epoch}".encode())
            checkpoints[str(epoch)] = {
                "path": str(checkpoint),
                "sha256": _sha(checkpoint),
                "epoch": epoch,
            }
            history.append(
                {
                    "epoch": epoch,
                    "primary_value": value,
                    "maximum_corrected_spread_ratio": 1.2 if arm == "K" else 1.9,
                    "per_beta": [
                        {"beta": beta, "global_scale_nrmse": value, "spread_ratio_to_primary": 1.0}
                        for beta in (0.01, 0.1, 1.0, 10.0, 100.0)
                    ],
                }
            )
        winner = min(history, key=lambda row: (row["primary_value"], row["epoch"]))
        arms[arm] = {
            "validation_history": history,
            "selection": {
                "selected_epoch": winner["epoch"],
                "selected_value": winner["primary_value"],
            },
            "checkpoints": {"rungs": checkpoints, "selected": checkpoints[str(winner["epoch"])]},
            "compute": {
                "duration_sec": 1.0,
                "total_parameter_count": 10,
                "trainable_parameter_count": 10,
                "optimizer_steps": 100,
                "examples_seen": 1000,
            },
        }
    summary = {
        "schema_version": 1,
        "status": "complete_validation_only",
        "held_out_reads": 0,
        "source": {"git_commit": current_commit},
        "task": "darcy2d",
        "architecture": {"dependency": {"available": True, "version": "2.0.0"}},
        "training_lock": {
            "lock_sha256": "c" * 64,
            "darcy_objects": {key: {"sha256": value} for key, value in object_hashes.items()},
        },
        "matched_design": {
            "arms": ["U", "K"],
            "seed": 17,
            "rungs": [3, 6, 12, 24],
            "same_data_order_updates": True,
        },
        "arms": arms,
        "diagnostics": {
            "counterfactual_beta_sensitivity": {
                "K": {"relative_prediction_rms_from_first_beta": 0.1}
            },
            "deterministic_shuffled_beta": {"primary_value": shuffled},
        },
    }
    summary["artifact_sha256"] = canonical_sha256(summary)
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary))
    return plan_path, summary_path, tmp_path / "result.json"


def test_materializer_accepts_complete_causal_evidence(monkeypatch, tmp_path):
    plan, summary, output = _fixture(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["materialize", "--plan", str(plan), "--summary", str(summary), "--output", str(output)],
    )
    materialize.main()
    result = json.loads(output.read_text())
    assert result["all_gates_passed"] is True
    assert result["effect"]["conditioned_primary_relative_improvement"] > 0.1
    assert result["artifact_sha256"] == canonical_sha256(
        {k: v for k, v in result.items() if k != "artifact_sha256"}
    )


def test_materializer_records_failed_shuffled_beta_gate(monkeypatch, tmp_path):
    plan, summary, output = _fixture(tmp_path, shuffled=0.61)
    monkeypatch.setattr(
        sys,
        "argv",
        ["materialize", "--plan", str(plan), "--summary", str(summary), "--output", str(output)],
    )
    materialize.main()
    result = json.loads(output.read_text())
    assert result["gate_checks"]["shuffled_beta_degradation"] is False
    assert result["all_gates_passed"] is False


def test_remote_wrapper_has_no_measurement_or_test_surface():
    wrapper = (REPO_ROOT / "scripts/run_remote_darcy_fno_conditioning_ablation.sh").read_text()
    assert "measurement.lock" not in wrapper
    assert "_test.h5" not in wrapper
    assert "--test" not in wrapper
    assert (
        "--auto-shutdown"
        in (REPO_ROOT / "scripts/launch_darcy_fno_conditioning_ablation_vast.sh").read_text()
    )
