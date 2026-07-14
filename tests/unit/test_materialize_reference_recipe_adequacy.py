from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.materialize_reference_recipe_adequacy import build_selection
from ups.data.manifests import canonical_sha256

TASKS = ("advection1d", "burgers1d", "darcy2d")
RUNGS = (3, 6, 12, 24, 48)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metrics(value: float, *, bad_spread: bool = False) -> dict[str, float]:
    result = {}
    for task, count in zip(TASKS, (8, 12, 5), strict=True):
        primary = "decoded_solution_nrmse" if task == "darcy2d" else "decoded_rollout_nrmse"
        result[f"task_{task}_{primary}"] = value
        global_name = primary.replace("_nrmse", "_global_scale_nrmse")
        for regime in range(count):
            multiplier = 1.6 if bad_spread and task == "darcy2d" and regime == 0 else 1.0
            result[f"task_{task}_regime_{regime}_{global_name}"] = value * multiplier
    result["decoded_rollout_nrmse"] = value
    return result


def _fixture(tmp_path: Path) -> tuple[dict, list[Path]]:
    binding = {
        "lock_sha256": "a" * 64,
        "source_revision": "sha256:" + "b" * 64,
        "source_manifest_sha256": "c" * 64,
        "protocol_manifest_sha256": "d" * 64,
        "selection_sha256": "e" * 64,
    }
    runner_identity = {}
    runs = []
    summary_paths = []
    curves = {
        "fno": (1.0, 0.9, 0.895, 0.894, 0.893),
        "uno": (1.0, 0.8, 0.79, 0.789, 0.788),
    }
    for architecture in ("fno", "uno"):
        runner_path = Path(f"scripts/run_external_neuraloperator_{architecture}_baseline.py")
        real_runner = Path(__file__).resolve().parents[2] / runner_path
        (tmp_path / runner_path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / runner_path).write_bytes(real_runner.read_bytes())
        runner_identity[architecture] = {
            "path": str(runner_path),
            "file_sha256": _sha(tmp_path / runner_path),
            "external_package": "neuraloperator==2.0.0",
        }
        run_id = f"r0_strat_v1_1_{architecture}_all_e48_s17_discovery_val"
        summary_rel = Path("reports") / run_id / "summary.json"
        command = [
            "python",
            str(runner_path),
            "--name",
            run_id,
            "--eval-split",
            "val",
            "--epochs",
            "48",
            "--seed",
            "17",
            "--tasks",
            *TASKS,
            "--validation-rungs",
            *(str(value) for value in RUNGS),
        ]
        runs.append(
            {
                "run_id": run_id,
                "phase": "discovery",
                "architecture": architecture,
                "tasks": list(TASKS),
                "epochs": 48,
                "seed": 17,
                "expected_summary": str(summary_rel),
                "command": command,
                "command_sha256": canonical_sha256(command),
            }
        )
        checkpoints = {}
        history = []
        for epoch, value in zip(RUNGS, curves[architecture], strict=True):
            checkpoint_rel = Path("reports") / run_id / f"models_epoch_{epoch}.pt"
            checkpoint_path = tmp_path / checkpoint_rel
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_bytes(f"{architecture}-{epoch}".encode())
            checkpoints[str(epoch)] = {
                "epoch": epoch,
                "path": str(checkpoint_rel),
                "sha256": _sha(checkpoint_path),
            }
            history.append(
                {
                    "epoch": epoch,
                    "metric_name": "decoded_rollout_nrmse",
                    "metric_value": value,
                    "metrics": _metrics(value),
                    "duration_sec": 1.0,
                }
            )
        summary = {
            "status": "complete",
            "run_name": run_id,
            "split": "val",
            "data_provenance": {
                **binding,
                "purpose": "training",
                "requested_roles": ["train", "valid"],
                "objects": [
                    {"object_id": "train", "role": "train", "sha256": "1" * 64},
                    {"object_id": "valid", "role": "valid", "sha256": "2" * 64},
                ],
            },
            "extra": {
                "baseline": f"external_neuraloperator_{architecture}",
                "task": list(TASKS),
                "seed": 17,
                "epochs": 48,
                "split": "val",
                "allow_held_out_test_eval": False,
                "command": command,
            },
            "held_out_test_policy": {"enabled": False, "recorded": False},
            "details": {"validation_history": history},
            "checkpoints": {"rungs": checkpoints, "selected": checkpoints["48"]},
            "metrics": history[-1]["metrics"],
            "recipe_adequacy": {
                "validation_rungs": list(RUNGS),
                "selection_metric": "decoded_rollout_nrmse",
                "selected_epoch": 48,
                "selected_metric_value": curves[architecture][-1],
                "selection_rule": "minimum_finite_validation_metric_earliest_tie",
            },
            "compute": {
                "total_parameter_count": 100,
                "trainable_parameter_count": 100,
                "optimizer_steps": 50,
                "examples_seen": 1000,
                "duration_sec": 10.0,
                "device": "cpu",
            },
        }
        summary_path = tmp_path / summary_rel
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        summary_paths.append(summary_path)

    payload = {
        "schema_version": 1,
        "plan_id": "strat-v1.1-fno-uno-reference-recipe-adequacy-v1",
        "mode": "validation_only",
        "execution_policy": "plan_only_no_runner_invocation",
        "heldout_access": "forbidden",
        "measurement_lock_access": "forbidden",
        "bindings": {"training_lock": binding},
        "runner_identity": runner_identity,
        "discovery": {
            "architectures": ["fno", "uno"],
            "tasks": list(TASKS),
            "epoch_rungs": list(RUNGS),
            "maximum_epochs": 48,
            "seed": 17,
            "runs": runs,
        },
        "plateau_criterion": {
            "consecutive_transitions_required": 2,
            "relative_improvement_threshold": 0.01,
            "operator": "strictly_less_than",
            "adequate_label": "adequate",
            "maximum_rung_without_plateau_label": "budget-capped",
            "non_finite_label": "invalid",
        },
        "selection": {
            "secondary_eligibility_gate": {
                "operator": "less_than_or_equal",
                "maximum": 1.5,
            }
        },
    }
    return {**payload, "plan_sha256": canonical_sha256(payload)}, summary_paths


def test_selects_lowest_adequate_macro_and_hashes_evidence(tmp_path: Path) -> None:
    plan, summaries = _fixture(tmp_path)

    artifact = build_selection(plan, summary_paths=summaries, repo_root=tmp_path)

    assert artifact["selection"]["architecture"] == "uno"
    assert artifact["selection"]["epoch"] == 48
    assert artifact["architectures"]["fno"]["plateau_epoch"] == 24
    assert artifact["architectures"]["fno"]["chosen_epoch"] == 24
    assert artifact["held_out_measurements"] == 0
    payload = {key: value for key, value in artifact.items() if key != "selection_sha256"}
    assert artifact["selection_sha256"] == canonical_sha256(payload)
    assert all(row["summary"]["sha256"] for row in artifact["architectures"].values())


def test_rejects_tampered_plan_hash(tmp_path: Path) -> None:
    plan, summaries = _fixture(tmp_path)
    plan["discovery"]["seed"] = 29
    with pytest.raises(ValueError, match="plan SHA-256"):
        build_selection(plan, summary_paths=summaries, repo_root=tmp_path)


def test_rejects_heldout_or_wrong_lock_summary(tmp_path: Path) -> None:
    plan, summaries = _fixture(tmp_path)
    summary = json.loads(summaries[0].read_text())
    summary["held_out_test_policy"]["enabled"] = True
    summaries[0].write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="held-out access"):
        build_selection(plan, summary_paths=summaries, repo_root=tmp_path)


def test_regime_gate_makes_architecture_ineligible(tmp_path: Path) -> None:
    plan, summaries = _fixture(tmp_path)
    uno = json.loads(summaries[1].read_text())
    for row in uno["details"]["validation_history"]:
        row["metrics"] = _metrics(row["metric_value"], bad_spread=True)
    uno["metrics"] = uno["details"]["validation_history"][-1]["metrics"]
    summaries[1].write_text(json.dumps(uno))

    artifact = build_selection(plan, summary_paths=summaries, repo_root=tmp_path)

    assert artifact["architectures"]["uno"]["eligible"] is False
    assert artifact["selection"]["architecture"] == "fno"


def test_no_plateau_is_budget_capped_and_not_selectable(tmp_path: Path) -> None:
    plan, summaries = _fixture(tmp_path)
    for path in summaries:
        summary = json.loads(path.read_text())
        for row, value in zip(
            summary["details"]["validation_history"], (1.0, 0.8, 0.6, 0.4, 0.2), strict=True
        ):
            row["metric_value"] = value
            row["metrics"] = _metrics(value)
        summary["metrics"] = summary["details"]["validation_history"][-1]["metrics"]
        summary["recipe_adequacy"]["selected_epoch"] = 48
        summary["recipe_adequacy"]["selected_metric_value"] = 0.2
        path.write_text(json.dumps(summary))

    artifact = build_selection(plan, summary_paths=summaries, repo_root=tmp_path)

    assert artifact["selection"] is None
    assert artifact["no_eligible_architecture"] is True
    assert {row["label"] for row in artifact["architectures"].values()} == {"budget-capped"}


def test_non_finite_discovery_is_labeled_invalid(tmp_path: Path) -> None:
    plan, summaries = _fixture(tmp_path)
    fno = json.loads(summaries[0].read_text())
    fno["details"]["validation_history"][0]["metric_value"] = float("nan")
    summaries[0].write_text(json.dumps(fno))

    artifact = build_selection(plan, summary_paths=summaries, repo_root=tmp_path)

    assert artifact["architectures"]["fno"]["label"] == "invalid"
    assert artifact["architectures"]["fno"]["eligible"] is False
    assert artifact["selection"]["architecture"] == "uno"
