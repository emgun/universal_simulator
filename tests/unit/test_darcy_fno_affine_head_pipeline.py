from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from scripts import materialize_darcy_fno_affine_head_ablation as materialize
from scripts import plan_darcy_fno_affine_head_ablation as planner
from ups.data.manifests import canonical_sha256
from ups.training.resumable_checkpoint import (
    CheckpointBindings,
    TrainingProgress,
    save_training_checkpoint,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNGS = [3, 6, 12, 24, 48, 96, 192]


def _fixture(tmp_path: Path, *, candidate_primary: float = 0.15) -> tuple[Path, Path, Path]:
    current = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    materializer_path = REPO_ROOT / "scripts/materialize_darcy_fno_affine_head_ablation.py"
    lock_objects = {"darcy2d-train": "a" * 64, "darcy2d-valid": "b" * 64}
    plan = {
        "schema_version": 2,
        "mode": "validation_only",
        "heldout_access": "forbidden",
        "measurement_lock_access": "forbidden",
        "bindings": {
            "source": {
                "implementation_commit": current,
                "files": {
                    str(materializer_path.relative_to(REPO_ROOT)): materialize.sha(
                        materializer_path
                    )
                },
            },
            "training_lock": {
                "lock_sha256": "c" * 64,
                "file_sha256": "d" * 64,
                "darcy_objects": lock_objects,
            },
        },
        "dependencies": {"neuraloperator": "2.0.0"},
        "design": {"arms": ["K-long", "A-affine"], "seed": 17, "epoch_rungs": RUNGS},
        "gates": {
            "candidate_max_corrected_regime_spread_maximum": 1.5,
            "candidate_primary_maximum": 0.18947497984876743,
            "candidate_shuffled_beta_relative_degradation_minimum": 0.05,
            "candidate_counterfactual_relative_prediction_rms_minimum_exclusive": 1e-6,
            "plateau": {
                "best_so_far_relative_improvement_threshold": 0.01,
                "consecutive_transitions_required": 2,
                "must_occur_by_epoch": 192,
            },
            "heldout_reads": 0,
        },
        "command": ["python", "scripts/run_darcy_fno_affine_head_ablation.py"],
    }
    plan["command_sha256"] = canonical_sha256(plan["command"])
    plan["plan_sha256"] = canonical_sha256(plan)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    identities = {
        "plan_fingerprint": plan["plan_sha256"],
        "data_fingerprint": "data",
        "source_fingerprint": "source",
        "runtime_fingerprint": "runtime",
    }
    arms = {}
    for arm, final in (("K-long", 0.18), ("A-affine", candidate_primary)):
        values = [final * 1.5, final * 1.2, final, final, final, final, final]
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters())
        generator = torch.Generator().manual_seed(17)
        bindings = CheckpointBindings(
            model_spec={"arm": arm},
            optimizer_spec={"objective": "raw_solution_mse"},
            normalizer_spec={},
            **identities,
        )
        parent = None
        checkpoints, history = {}, []
        for epoch, value in zip(RUNGS, values, strict=True):
            path = tmp_path / arm / f"epoch-{epoch}.pt"
            record = save_training_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                progress=TrainingProgress(epoch, epoch, epoch, [value]),
                sampler_generator=generator,
                bindings=bindings,
                parent_checkpoint_sha256=parent,
            )
            checkpoints[str(epoch)] = {
                "path": str(path),
                "sha256": record.checkpoint_sha256,
                "parent_checkpoint_sha256": parent,
            }
            parent = record.checkpoint_sha256
            history.append(
                {
                    "epoch": epoch,
                    "primary_value": value,
                    "maximum_corrected_spread_ratio": 1.1,
                    "per_beta": [
                        {
                            "beta": beta,
                            "slice_normalized_nrmse": value,
                            "global_scale_nrmse": value * (1.1 if beta == 100 else 1),
                            "spread_ratio_to_primary": 1.1 if beta == 100 else 1,
                            "element_count": 64,
                        }
                        for beta in (0.01, 0.1, 1.0, 10.0, 100.0)
                    ],
                }
            )
        winner = min(history, key=lambda x: (x["primary_value"], x["epoch"]))
        arms[arm] = {
            "validation_history": history,
            "selection": {
                "selected_epoch": winner["epoch"],
                "selected_value": winner["primary_value"],
            },
            "checkpoints": {"rungs": checkpoints},
        }
    summary = {
        "schema_version": 1,
        "status": "complete_validation_only",
        "task": "darcy2d",
        "held_out_reads": 0,
        "source": {"git_commit": current},
        "integrity_bindings": identities,
        "training_lock": {
            "lock_sha256": "c" * 64,
            "lock_file_sha256": "d" * 64,
            "darcy_objects": {key: {"sha256": value} for key, value in lock_objects.items()},
        },
        "matched_design": {
            "arms": ["K-long", "A-affine"],
            "seed": 17,
            "rungs": RUNGS,
            "same_optimizer_and_raw_solution_mse": True,
        },
        "architecture": {"dependency": {"available": True, "version": "2.0.0"}},
        "arms": arms,
        "diagnostics": {
            "deterministic_shuffled_beta": {
                "arms": {"A-affine": {"relative_degradation_vs_true_beta": 0.2}}
            },
            "counterfactual_beta_sensitivity": {
                "A-affine": {"relative_prediction_rms_from_first_beta": 0.1}
            },
        },
    }
    summary["artifact_sha256"] = canonical_sha256(summary)
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return plan_path, summary_path, tmp_path / "result.json"


def test_materializer_verifies_lineage_bindings_and_passes_gates(monkeypatch, tmp_path):
    plan, summary, output = _fixture(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["materialize", "--plan", str(plan), "--summary", str(summary), "--output", str(output)],
    )
    materialize.main()
    result = json.loads(output.read_text())
    assert result["all_gates_passed"] is True
    assert result["arms"]["A-affine"]["plateau_epoch"] == 48


def test_materializer_rejects_checkpoint_lineage_tampering(monkeypatch, tmp_path):
    plan, summary, output = _fixture(tmp_path)
    payload = json.loads(summary.read_text())
    payload["arms"]["A-affine"]["checkpoints"]["rungs"]["6"]["parent_checkpoint_sha256"] = "0" * 64
    payload["artifact_sha256"] = canonical_sha256(
        {k: v for k, v in payload.items() if k != "artifact_sha256"}
    )
    summary.write_text(json.dumps(payload))
    monkeypatch.setattr(
        sys,
        "argv",
        ["materialize", "--plan", str(plan), "--summary", str(summary), "--output", str(output)],
    )
    with pytest.raises(ValueError, match="lineage"):
        materialize.main()


def test_planner_rejects_live_source_bytes_that_differ_from_commit(monkeypatch):
    relative = "scripts/materialize_darcy_fno_affine_head_ablation.py"
    monkeypatch.setattr(planner, "source_paths", lambda: (relative,))

    def fake_check_output(command, **kwargs):
        if command[:3] == ["git", "status", "--porcelain"]:
            return ""
        if command[:2] == ["git", "show"]:
            return b"different committed bytes"
        raise AssertionError(command)

    monkeypatch.setattr(planner.subprocess, "check_output", fake_check_output)
    with pytest.raises(ValueError, match="differ from implementation commit"):
        planner.source_manifest("0" * 40)


def test_plan_validation_rejects_modified_plan_and_command(tmp_path):
    plan, _, _ = _fixture(tmp_path)
    payload = json.loads(plan.read_text())
    payload["gates"]["heldout_reads"] = 1
    plan.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="self hash"):
        materialize.load_and_validate_plan(plan)

    payload["plan_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "plan_sha256"}
    )
    payload["command"].append("--resume")
    payload["plan_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "plan_sha256"}
    )
    plan.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="command hash"):
        materialize.load_and_validate_plan(plan)


def test_remote_wrapper_rejects_plan_and_command_tampering_before_credentials(tmp_path):
    plan, _, _ = _fixture(tmp_path)
    wrapper = REPO_ROOT / "scripts/run_remote_darcy_fno_affine_head_ablation.sh"
    payload = json.loads(plan.read_text())
    payload["gates"]["heldout_reads"] = 1
    plan.write_text(json.dumps(payload))
    result = subprocess.run(
        ["bash", str(wrapper)],
        cwd=REPO_ROOT,
        env={
            "PATH": str(Path(sys.executable).parent) + ":/usr/bin:/bin",
            "DRY_RUN": "0",
            "PLAN": str(plan),
            "RESULT": str(tmp_path / "result.json"),
            "PYTHON": sys.executable,
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "invalid canonical plan self hash" in result.stderr
    assert "Set B2_KEY_ID" not in result.stderr

    payload["plan_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "plan_sha256"}
    )
    payload["command"].append("--resume")
    payload["plan_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "plan_sha256"}
    )
    plan.write_text(json.dumps(payload))
    result = subprocess.run(
        ["bash", str(wrapper)],
        cwd=REPO_ROOT,
        env={
            "PATH": str(Path(sys.executable).parent) + ":/usr/bin:/bin",
            "DRY_RUN": "0",
            "PLAN": str(plan),
            "RESULT": str(tmp_path / "result.json"),
            "PYTHON": sys.executable,
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "invalid canonical plan command hash" in result.stderr
    assert "Set B2_KEY_ID" not in result.stderr


def test_remote_surfaces_are_validation_only_resumable_and_bounded():
    wrapper = (REPO_ROOT / "scripts/run_remote_darcy_fno_affine_head_ablation.sh").read_text()
    launcher = (REPO_ROOT / "scripts/launch_darcy_fno_affine_head_ablation_vast.sh").read_text()
    for text in (wrapper, launcher):
        assert "measurement.lock" not in text
        assert "_test.h5" not in text
        assert "--test" not in text
    assert "resumable/${plan_sha}" in wrapper
    assert 'tee "$RUNNER_LOG"' in wrapper
    assert 'cp "$RUNNER_LOG" "$OUTPUT_DIR/remote_runner.log"' in wrapper
    assert "D2 plan bytes changed after validation" in wrapper
    assert 'if [ -n "$resume_listing" ]' in wrapper
    assert 'test -f "$OUTPUT_DIR/run_identity.json"' in wrapper
    assert "rclone purge" in wrapper
    assert "--managed" in launcher and "MAX_RUNTIME_MINUTES=${MAX_RUNTIME_MINUTES:-600}" in launcher
    assert "--launch-retries 0" in launcher
