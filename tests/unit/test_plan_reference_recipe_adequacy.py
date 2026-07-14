from __future__ import annotations

import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from scripts.plan_reference_recipe_adequacy import build_plan
from ups.data.manifests import canonical_sha256

RELEASE = Path(
    "docs/data/releases/strat-v1/universal/"
    "9d43d283f04f5b8d17cf6126ad189075c53307e715d7d4f61af440c2fed155c1"
)


def _args(tmp_path: Path, **overrides) -> Namespace:
    values = {
        "metric_addendum": "docs/data/protocols/strat_v1_1_metric_addendum.yaml",
        "training_lock": str(RELEASE / "training.lock.json"),
        "config": "configs/a4_strat_v1_baselines.yaml",
        "data_root": str(tmp_path / "training-cache"),
        "output_root": str(tmp_path / "adequacy"),
        "device": "cuda",
        "neuraloperator_version": "2.0.0",
        "discovery_plan": None,
        "selection_artifact": None,
    }
    values.update(overrides)
    return Namespace(**values)


def _selection_evidence(tmp_path: Path) -> tuple[Path, Path]:
    discovery = build_plan(_args(tmp_path))
    discovery_path = tmp_path / "discovery-plan.json"
    discovery_path.write_text(json.dumps(discovery), encoding="utf-8")
    checkpoint = {"epoch": 24, "path": "checkpoints/uno-e24.pt", "sha256": "c" * 64}
    selected = {
        "architecture": "uno",
        "epoch": 24,
        "macro_primary_nrmse": 0.5,
        "checkpoint": checkpoint,
    }
    payload = {
        "schema_version": 1,
        "selection_id": "strat-v1.1-reference-recipe-adequacy-selection-v1",
        "status": "complete_validation_only",
        "plan_sha256": discovery["plan_sha256"],
        "architectures": {
            "fno": {"eligible": False},
            "uno": {
                "eligible": True,
                "chosen_epoch": 24,
                "chosen_macro_primary_nrmse": 0.5,
                "chosen_checkpoint": checkpoint,
            },
        },
        "selection": selected,
        "no_eligible_architecture": False,
        "held_out_measurements": 0,
    }
    artifact = {**payload, "selection_sha256": canonical_sha256(payload)}
    artifact_path = tmp_path / "selection.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    return discovery_path, artifact_path


def test_discovery_plan_is_bounded_and_binds_frozen_strat_v1_1(tmp_path: Path) -> None:
    plan = build_plan(_args(tmp_path))

    assert plan["mode"] == "validation_only"
    assert plan["heldout_access"] == "forbidden"
    assert plan["measurement_lock_access"] == "forbidden"
    assert plan["bindings"]["metric_addendum"]["self_sha256"] == (
        "2fedaaf445d093a40571a475d5793567842582b5a457d7039ab21db525f50ad0"
    )
    assert plan["bindings"]["training_lock"]["lock_sha256"] == (
        "5666fa8bb646b270d23077d1936c8ee80fa1cfeaf8f5870c55103fc58dd80afd"
    )
    assert plan["bindings"]["training_lock"]["roles"] == ["train", "valid"]
    assert plan["discovery"]["epoch_rungs"] == [3, 6, 12, 24, 48]
    assert plan["discovery"]["maximum_epochs"] == 48
    assert plan["discovery"]["independent_rung_restarts"] is False
    assert len(plan["discovery"]["runs"]) == 2
    assert plan["confirmation"]["runs"] == []


def test_plan_and_run_commands_are_deterministic(tmp_path: Path) -> None:
    first = build_plan(_args(tmp_path))
    second = build_plan(_args(tmp_path))

    assert first == second
    assert first["plan_sha256"] == second["plan_sha256"]
    assert [run["command_sha256"] for run in first["discovery"]["runs"]] == [
        run["command_sha256"] for run in second["discovery"]["runs"]
    ]


def test_every_discovery_command_is_train_valid_only_and_full_shard(tmp_path: Path) -> None:
    plan = build_plan(_args(tmp_path))
    for run in plan["discovery"]["runs"]:
        command = run["command"]
        assert run["evaluation_role"] == "valid"
        assert command[command.index("--train-split") + 1] == "train"
        assert command[command.index("--eval-split") + 1] == "val"
        assert command[command.index("--max-train-samples") + 1] == "288"
        assert command[command.index("--max-eval-samples") + 1] == "72"
        assert command[command.index("--max-pairs-per-task") + 1] == "1152"
        rung_index = command.index("--validation-rungs")
        assert command[rung_index + 1 : rung_index + 6] == ["3", "6", "12", "24", "48"]
        assert len(run["command_sha256"]) == 64
        assert "test" not in command
        assert "--allow-held-out-test-eval" not in command
        assert "--allow-repeat-test" not in command
        assert "--refuse-overwrite" in command


def test_plateau_and_secondary_regime_gate_are_machine_readable(tmp_path: Path) -> None:
    plan = build_plan(_args(tmp_path))
    plateau = plan["plateau_criterion"]
    gate = plan["selection"]["secondary_eligibility_gate"]

    assert plateau["consecutive_transitions_required"] == 2
    assert plateau["relative_improvement_threshold"] == 0.01
    assert plateau["operator"] == "strictly_less_than"
    assert plateau["maximum_rung_without_plateau_label"] == "budget-capped"
    assert gate["metric"] == "maximum_global_scale_regime_nrmse_to_task_primary_nrmse"
    assert gate["maximum"] == 1.5
    assert gate["required_scope"] == "every_task"


def test_selected_architecture_emits_only_two_new_all_task_seed_runs(tmp_path: Path) -> None:
    discovery_path, artifact_path = _selection_evidence(tmp_path)
    plan = build_plan(
        _args(
            tmp_path,
            discovery_plan=str(discovery_path),
            selection_artifact=str(artifact_path),
        )
    )
    confirmation = plan["confirmation"]

    assert confirmation["required_seeds"] == [17, 29, 43]
    assert confirmation["reuse_discovery_seed_17"] is True
    assert len(confirmation["runs"]) == 2
    assert {run["architecture"] for run in confirmation["runs"]} == {"uno"}
    assert {run["seed"] for run in confirmation["runs"]} == {29, 43}
    assert {run["epochs"] for run in confirmation["runs"]} == {24}
    assert all(
        run["tasks"] == ["advection1d", "burgers1d", "darcy2d"] for run in confirmation["runs"]
    )
    evidence = confirmation["evidence_binding"]
    assert (
        evidence["discovery_plan"]["plan_sha256"]
        == json.loads(discovery_path.read_text())["plan_sha256"]
    )
    assert (
        evidence["selection_artifact"]["selection_sha256"]
        == json.loads(artifact_path.read_text())["selection_sha256"]
    )
    assert len(evidence["discovery_plan"]["file_sha256"]) == 64
    assert len(evidence["selection_artifact"]["file_sha256"]) == 64


@pytest.mark.parametrize(
    ("discovery_plan", "selection_artifact"),
    [("discovery.json", None), (None, "selection.json")],
)
def test_confirmation_evidence_arguments_must_be_paired(
    tmp_path: Path, discovery_plan: str | None, selection_artifact: str | None
) -> None:
    with pytest.raises(ValueError, match="provided together"):
        build_plan(
            _args(
                tmp_path,
                discovery_plan=discovery_plan,
                selection_artifact=selection_artifact,
            )
        )


def test_confirmation_rejects_tampered_selection_self_hash(tmp_path: Path) -> None:
    discovery_path, artifact_path = _selection_evidence(tmp_path)
    artifact = json.loads(artifact_path.read_text())
    artifact["selection"]["architecture"] = "fno"
    artifact_path.write_text(json.dumps(artifact))

    with pytest.raises(ValueError, match="selection_sha256 is invalid"):
        build_plan(
            _args(
                tmp_path,
                discovery_plan=str(discovery_path),
                selection_artifact=str(artifact_path),
            )
        )


def test_confirmation_rejects_selection_bound_to_other_plan(tmp_path: Path) -> None:
    discovery_path, artifact_path = _selection_evidence(tmp_path)
    artifact = json.loads(artifact_path.read_text())
    artifact["plan_sha256"] = "f" * 64
    payload = {key: value for key, value in artifact.items() if key != "selection_sha256"}
    artifact["selection_sha256"] = canonical_sha256(payload)
    artifact_path.write_text(json.dumps(artifact))

    with pytest.raises(ValueError, match="not bound to the supplied discovery plan"):
        build_plan(
            _args(
                tmp_path,
                discovery_plan=str(discovery_path),
                selection_artifact=str(artifact_path),
            )
        )


@pytest.mark.parametrize("mutation", ("no_selection", "ineligible"))
def test_confirmation_rejects_no_selection_or_ineligible_artifact(
    tmp_path: Path, mutation: str
) -> None:
    discovery_path, artifact_path = _selection_evidence(tmp_path)
    artifact = json.loads(artifact_path.read_text())
    if mutation == "no_selection":
        artifact["selection"] = None
        artifact["no_eligible_architecture"] = True
    else:
        artifact["architectures"]["uno"]["eligible"] = False
    payload = {key: value for key, value in artifact.items() if key != "selection_sha256"}
    artifact["selection_sha256"] = canonical_sha256(payload)
    artifact_path.write_text(json.dumps(artifact))

    message = "no eligible architecture" if mutation == "no_selection" else "not eligible"
    with pytest.raises(ValueError, match=message):
        build_plan(
            _args(
                tmp_path,
                discovery_plan=str(discovery_path),
                selection_artifact=str(artifact_path),
            )
        )


def test_cli_is_plan_only_and_creates_no_training_output(tmp_path: Path) -> None:
    output_plan = tmp_path / "plan.json"
    output_root = tmp_path / "training-output"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/plan_reference_recipe_adequacy.py",
            "--output-plan",
            str(output_plan),
            "--output-root",
            str(output_root),
        ],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    assert output_plan.is_file()
    assert not output_root.exists()
    payload = json.loads(output_plan.read_text(encoding="utf-8"))
    assert len(payload["discovery"]["runs"]) == 2
    assert payload["execution_policy"] == "plan_only_no_runner_invocation"
    assert payload["runner_identity"]["fno"]["external_package"] == "neuraloperator==2.0.0"


def test_cli_confirmation_derives_selection_from_bound_artifacts(tmp_path: Path) -> None:
    discovery_path, artifact_path = _selection_evidence(tmp_path)
    output_plan = tmp_path / "confirmation-plan.json"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/plan_reference_recipe_adequacy.py",
            "--output-plan",
            str(output_plan),
            "--output-root",
            str(tmp_path / "adequacy"),
            "--data-root",
            str(tmp_path / "training-cache"),
            "--discovery-plan",
            str(discovery_path),
            "--selection-artifact",
            str(artifact_path),
        ],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    plan = json.loads(output_plan.read_text())
    assert plan["confirmation"]["selected_architecture"] == "uno"
    assert plan["confirmation"]["selected_epochs"] == 24
    assert len(plan["confirmation"]["runs"]) == 2
