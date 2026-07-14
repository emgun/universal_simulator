#!/usr/bin/env python
"""Build the bounded, validation-only FNO/UNO recipe-adequacy plan.

This planner is intentionally non-executing.  Each architecture is trained once
across all three tasks, with validation metrics and checkpoints captured at the
frozen epoch rungs.  No test or measurement lock is accepted, and confirmation
commands are emitted only after the caller names the validation-selected
architecture and epoch rung.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ups.data.manifests import canonical_sha256, load_data_lock  # noqa: E402

TASKS = ("advection1d", "burgers1d", "darcy2d")
ARCHITECTURES = ("fno", "uno")
EPOCH_RUNGS = (3, 6, 12, 24, 48)
DISCOVERY_SEED = 17
CONFIRMATION_SEEDS = (17, 29, 43)
TRAINING_LOCK_SHA256 = "5666fa8bb646b270d23077d1936c8ee80fa1cfeaf8f5870c55103fc58dd80afd"
ADDENDUM_SELF_SHA256 = "2fedaaf445d093a40571a475d5793567842582b5a457d7039ab21db525f50ad0"
RUNNERS = {
    "fno": "scripts/run_external_neuraloperator_fno_baseline.py",
    "uno": "scripts/run_external_neuraloperator_uno_baseline.py",
}
MODEL_ARGS = {
    "fno": {
        "hidden_channels": 16,
        "fourier_modes": 16,
        "n_layers": 4,
    },
    "uno": {
        "hidden_channels": 16,
        "fourier_modes": 16,
        "n_layers": 4,
        "lifting_channels": 32,
        "projection_channels": 32,
        "channel_mlp_skip": "linear",
    },
}
DEFAULT_RELEASE = Path(
    "docs/data/releases/strat-v1/universal/"
    "9d43d283f04f5b8d17cf6126ad189075c53307e715d7d4f61af440c2fed155c1"
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _load_and_validate_bindings(addendum_path: Path, training_lock_path: Path) -> dict[str, Any]:
    addendum = yaml.safe_load(addendum_path.read_text(encoding="utf-8"))
    if not isinstance(addendum, dict):
        raise ValueError("strat-v1.1 addendum must be a mapping")
    self_hash = addendum.get("self_hash")
    if not isinstance(self_hash, dict):
        raise ValueError("strat-v1.1 addendum lacks self_hash")
    recorded_self_hash = self_hash.get("value")
    unhashed = json.loads(json.dumps(addendum))
    del unhashed["self_hash"]["value"]
    if (
        recorded_self_hash != canonical_sha256(unhashed)
        or recorded_self_hash != ADDENDUM_SELF_SHA256
    ):
        raise ValueError("strat-v1.1 addendum self hash does not match the frozen identity")

    base = addendum.get("base_protocol")
    access = addendum.get("freeze_access")
    if not isinstance(base, dict) or not isinstance(access, dict):
        raise ValueError("strat-v1.1 addendum lacks frozen protocol/access bindings")
    if access.get("allowed_roles") != ["valid"] or access.get("forbidden_roles") != ["test"]:
        raise ValueError("strat-v1.1 addendum must remain validation-only")
    if (
        access.get("measurement_lock_access") != "forbidden"
        or access.get("heldout_reads") != "forbidden"
    ):
        raise ValueError("strat-v1.1 addendum permits held-out access")

    lock = load_data_lock(training_lock_path)
    if lock.lock_sha256 != TRAINING_LOCK_SHA256 or lock.lock_sha256 != base.get(
        "training_lock_sha256"
    ):
        raise ValueError("recipe adequacy requires the frozen universal training lock")
    if _file_sha256(training_lock_path) != base.get("training_lock_file_sha256"):
        raise ValueError("training lock file hash does not match strat-v1.1")
    if lock.purpose != "training" or set(lock.requested_roles) != {"train", "valid"}:
        raise ValueError("recipe adequacy requires exactly train and valid roles")
    if lock.measurement_contract_id is not None or any(
        item.role == "test" for item in lock.objects
    ):
        raise ValueError("recipe adequacy forbids measurement locks and test objects")
    if lock.protocol_manifest_sha256 != base.get("protocol_manifest_sha256"):
        raise ValueError("training lock protocol hash does not match strat-v1.1")
    if lock.source_revision != base.get("source_revision"):
        raise ValueError("training lock source revision does not match strat-v1.1")

    return {
        "metric_addendum": {
            "path": str(addendum_path),
            "file_sha256": _file_sha256(addendum_path),
            "self_sha256": recorded_self_hash,
            "addendum_id": addendum.get("addendum_id"),
        },
        "training_lock": {
            "path": str(training_lock_path),
            "file_sha256": _file_sha256(training_lock_path),
            "lock_sha256": lock.lock_sha256,
            "protocol_manifest_sha256": lock.protocol_manifest_sha256,
            "source_manifest_sha256": lock.source_manifest_sha256,
            "source_revision": lock.source_revision,
            "selection_sha256": canonical_sha256(lock.selection),
            "roles": sorted(set(lock.requested_roles)),
            "object_ids": [item.object_id for item in lock.objects],
        },
    }


def _command(
    *,
    architecture: str,
    epochs: int,
    seed: int,
    phase: str,
    args: argparse.Namespace,
    lock_sha256: str,
) -> tuple[str, list[str]]:
    run_id = f"r0_strat_v1_1_{architecture}_all_e{epochs}_s{seed}_{phase}_val"
    command = [
        sys.executable,
        RUNNERS[architecture],
        "--config",
        str(args.config),
        "--name",
        run_id,
        "--output-root",
        str(args.output_root),
        "--data-root",
        str(args.data_root),
        "--tasks",
        *TASKS,
        "--data-lock",
        str(args.training_lock),
        "--expected-data-lock-sha256",
        lock_sha256,
        "--train-split",
        "train",
        "--eval-split",
        "val",
        "--max-train-samples",
        "288",
        "--max-eval-samples",
        "72",
        "--rollout-steps",
        "16",
        "--max-pairs-per-task",
        "1152",
        "--train-stride",
        "4",
        "--epochs",
        str(epochs),
        "--validation-rungs",
        *(str(rung) for rung in (EPOCH_RUNGS if phase == "discovery" else (epochs,))),
        "--learning-rate",
        "0.001",
        "--weight-decay",
        "0.0001",
        "--batch-size",
        "8",
        "--seed",
        str(seed),
        "--device",
        str(args.device),
        "--metric",
        "decoded_rollout_nrmse",
        "--strict-contract",
        "--refuse-overwrite",
    ]
    for key, value in MODEL_ARGS[architecture].items():
        command.extend((f"--{key.replace('_', '-')}", str(value)))
    forbidden = {"test", "--allow-held-out-test-eval", "--allow-repeat-test"}
    if forbidden.intersection(command):
        raise AssertionError("recipe-adequacy command attempted held-out access")
    return run_id, command


def _run_record(
    *,
    architecture: str,
    epochs: int,
    seed: int,
    phase: str,
    args: argparse.Namespace,
    lock_sha256: str,
) -> dict[str, Any]:
    run_id, command = _command(
        architecture=architecture,
        epochs=epochs,
        seed=seed,
        phase=phase,
        args=args,
        lock_sha256=lock_sha256,
    )
    return {
        "run_id": run_id,
        "phase": phase,
        "architecture": architecture,
        "tasks": list(TASKS),
        "epochs": epochs,
        "seed": seed,
        "train_role": "train",
        "evaluation_role": "valid",
        "expected_summary": str(Path(args.output_root) / run_id / "summary.json"),
        "command": command,
        "command_display": shlex.join(command),
        "command_sha256": canonical_sha256(command),
    }


def _canonical_artifact(path: Path, *, hash_field: str, label: str) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    recorded = payload.get(hash_field)
    unhashed = {key: value for key, value in payload.items() if key != hash_field}
    if not isinstance(recorded, str) or recorded != canonical_sha256(unhashed):
        raise ValueError(f"{label} canonical {hash_field} is invalid")
    return payload, recorded


def _confirmation_evidence(
    args: argparse.Namespace,
    *,
    bindings: dict[str, Any],
    config: dict[str, Any],
    runner_identity: dict[str, Any],
    discovery: dict[str, Any],
) -> tuple[str | None, int | None, dict[str, Any] | None]:
    selection_value = getattr(args, "selection_artifact", None)
    discovery_value = getattr(args, "discovery_plan", None)
    if (selection_value is None) != (discovery_value is None):
        raise ValueError("selection artifact and discovery plan must be provided together")
    if selection_value is None:
        return None, None, None

    selection_path = Path(selection_value)
    discovery_path = Path(discovery_value)
    for path in (selection_path, discovery_path):
        if not _repo_path(path).is_file():
            raise FileNotFoundError(path)
    frozen_plan, plan_sha256 = _canonical_artifact(
        _repo_path(discovery_path), hash_field="plan_sha256", label="discovery plan"
    )
    if (
        frozen_plan.get("schema_version") != 1
        or frozen_plan.get("plan_id") != "strat-v1.1-fno-uno-reference-recipe-adequacy-v1"
        or frozen_plan.get("mode") != "validation_only"
        or frozen_plan.get("heldout_access") != "forbidden"
    ):
        raise ValueError("confirmation requires the canonical validation-only discovery plan")
    frozen_confirmation = frozen_plan.get("confirmation")
    if (
        not isinstance(frozen_confirmation, dict)
        or frozen_confirmation.get("runs") != []
        or frozen_confirmation.get("selected_architecture") is not None
        or frozen_confirmation.get("selected_epochs") is not None
    ):
        raise ValueError("discovery plan must not already contain confirmation runs")
    for key, expected in (
        ("bindings", bindings),
        ("config", config),
        ("runner_identity", runner_identity),
        ("discovery", discovery),
    ):
        if frozen_plan.get(key) != expected:
            raise ValueError(f"current confirmation inputs differ from discovery plan {key}")

    selection_artifact, selection_sha256 = _canonical_artifact(
        _repo_path(selection_path),
        hash_field="selection_sha256",
        label="selection artifact",
    )
    if selection_artifact.get("status") != "complete_validation_only":
        raise ValueError("selection artifact is not complete validation-only evidence")
    if (
        selection_artifact.get("schema_version") != 1
        or selection_artifact.get("selection_id")
        != "strat-v1.1-reference-recipe-adequacy-selection-v1"
    ):
        raise ValueError("selection artifact identity is unsupported")
    if selection_artifact.get("plan_sha256") != plan_sha256:
        raise ValueError("selection artifact is not bound to the supplied discovery plan")
    if selection_artifact.get("held_out_measurements") != 0:
        raise ValueError("selection artifact contains held-out measurements")
    if selection_artifact.get("no_eligible_architecture") is not False:
        raise ValueError("selection artifact has no eligible architecture")
    selected = selection_artifact.get("selection")
    if not isinstance(selected, dict):
        raise ValueError("selection artifact does not select an architecture")
    architecture = selected.get("architecture")
    epoch = selected.get("epoch")
    if architecture not in ARCHITECTURES or epoch not in EPOCH_RUNGS:
        raise ValueError("selection artifact chose an invalid architecture or epoch")
    architecture_rows = selection_artifact.get("architectures")
    if not isinstance(architecture_rows, dict):
        raise ValueError("selection artifact architecture evidence is missing")
    architecture_row = architecture_rows.get(architecture)
    if not isinstance(architecture_row, dict) or architecture_row.get("eligible") is not True:
        raise ValueError("selected architecture is not eligible in the selection artifact")
    if architecture_row.get("chosen_epoch") != epoch:
        raise ValueError("selected epoch disagrees with the eligible architecture evidence")
    if architecture_row.get("chosen_macro_primary_nrmse") != selected.get(
        "macro_primary_nrmse"
    ) or architecture_row.get("chosen_checkpoint") != selected.get("checkpoint"):
        raise ValueError("selection record disagrees with its architecture evidence")

    evidence = {
        "discovery_plan": {
            "path": str(discovery_path),
            "file_sha256": _file_sha256(_repo_path(discovery_path)),
            "plan_sha256": plan_sha256,
        },
        "selection_artifact": {
            "path": str(selection_path),
            "file_sha256": _file_sha256(_repo_path(selection_path)),
            "selection_sha256": selection_sha256,
        },
    }
    return architecture, int(epoch), evidence


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    addendum_path = Path(args.metric_addendum)
    training_lock_path = Path(args.training_lock)
    config_path = Path(args.config)
    for path in (addendum_path, training_lock_path, config_path):
        if not _repo_path(path).is_file():
            raise FileNotFoundError(path)
    # Keep command paths as supplied, but validate and hash against resolved repository paths.
    bindings = _load_and_validate_bindings(
        _repo_path(addendum_path), _repo_path(training_lock_path)
    )
    lock_sha256 = str(bindings["training_lock"]["lock_sha256"])
    discovery_runs = [
        _run_record(
            architecture=architecture,
            epochs=EPOCH_RUNGS[-1],
            seed=DISCOVERY_SEED,
            phase="discovery",
            args=args,
            lock_sha256=lock_sha256,
        )
        for architecture in ARCHITECTURES
    ]
    config = {"path": str(config_path), "file_sha256": _file_sha256(_repo_path(config_path))}
    runner_identity = {
        architecture: {
            "path": runner,
            "file_sha256": _file_sha256(REPO_ROOT / runner),
            "external_package": f"neuraloperator=={args.neuraloperator_version}",
        }
        for architecture, runner in RUNNERS.items()
    }
    discovery = {
        "architectures": list(ARCHITECTURES),
        "tasks": list(TASKS),
        "epoch_rungs": list(EPOCH_RUNGS),
        "maximum_epochs": EPOCH_RUNGS[-1],
        "seed": DISCOVERY_SEED,
        "trajectory_semantics": "one_continuous_training_trajectory_with_validation_checkpoint_at_each_rung",
        "independent_rung_restarts": False,
        "runs": discovery_runs,
    }
    selected_architecture, selected_epochs, confirmation_evidence = _confirmation_evidence(
        args,
        bindings=bindings,
        config=config,
        runner_identity=runner_identity,
        discovery=discovery,
    )
    confirmation_runs: list[dict[str, Any]] = []
    if selected_architecture is not None and selected_epochs is not None:
        confirmation_runs = [
            _run_record(
                architecture=selected_architecture,
                epochs=selected_epochs,
                seed=seed,
                phase="confirmation",
                args=args,
                lock_sha256=lock_sha256,
            )
            for seed in CONFIRMATION_SEEDS[1:]
        ]

    payload: dict[str, Any] = {
        "schema_version": 1,
        "plan_id": "strat-v1.1-fno-uno-reference-recipe-adequacy-v1",
        "mode": "validation_only",
        "execution_policy": "plan_only_no_runner_invocation",
        "heldout_access": "forbidden",
        "measurement_lock_access": "forbidden",
        "bindings": bindings,
        "config": config,
        "runner_identity": runner_identity,
        "discovery": discovery,
        "plateau_criterion": {
            "selection_metric": "macro_average_of_three_task_primary_validation_nrmse",
            "task_metrics": {
                "advection1d": "task_advection1d_decoded_rollout_nrmse",
                "burgers1d": "task_burgers1d_decoded_rollout_nrmse",
                "darcy2d": "task_darcy2d_decoded_solution_nrmse",
            },
            "best_so_far_relative_improvement_formula": "(previous_best-current_best)/previous_best",
            "plateau_when": "best_so_far_relative_improvement < 0.01 for two consecutive rung transitions",
            "consecutive_transitions_required": 2,
            "relative_improvement_threshold": 0.01,
            "operator": "strictly_less_than",
            "adequate_label": "adequate",
            "maximum_rung_without_plateau_label": "budget-capped",
            "non_finite_label": "invalid",
        },
        "selection": {
            "rung_policy": "best_validation_rung_from_the_continuous_trajectory_after_declared_plateau_else_no_adequate_recipe",
            "architecture_policy": "lowest_selected_rung_macro_nrmse_among_adequate_eligible_architectures",
            "secondary_eligibility_gate": {
                "source": "strat-v1.1 promotion_gate",
                "metric": "maximum_global_scale_regime_nrmse_to_task_primary_nrmse",
                "operator": "less_than_or_equal",
                "maximum": 1.5,
                "required_scope": "every_task",
            },
            "no_eligible_architecture_action": "stop_without_confirmation_or_heldout",
        },
        "confirmation": {
            "selected_architecture": selected_architecture,
            "selected_epochs": selected_epochs,
            "required_seeds": list(CONFIRMATION_SEEDS),
            "reuse_discovery_seed_17": selected_architecture is not None,
            "additional_seeds": list(CONFIRMATION_SEEDS[1:]),
            "evidence_binding": confirmation_evidence,
            "runs": confirmation_runs,
            "completion_rule": "one all-task artifact at seeds 17, 29, and 43; seed 17 is reused from discovery",
        },
        "forbidden_actions": [
            "test split access",
            "measurement lock access",
            "held-out metric computation",
            "selecting more than one architecture for confirmation",
            "calling a recipe strong when labeled budget-capped",
        ],
    }
    payload["plan_sha256"] = canonical_sha256(payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metric-addendum",
        default="docs/data/protocols/strat_v1_1_metric_addendum.yaml",
    )
    parser.add_argument("--training-lock", default=str(DEFAULT_RELEASE / "training.lock.json"))
    parser.add_argument("--config", default="configs/a4_strat_v1_baselines.yaml")
    parser.add_argument("--data-root", default="data/pdebench_strat_v1_training")
    parser.add_argument("--output-root", default="reports/research/reference_recipe_adequacy")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--neuraloperator-version", default="2.0.0")
    parser.add_argument("--discovery-plan")
    parser.add_argument("--selection-artifact")
    parser.add_argument("--output-plan", required=True)
    args = parser.parse_args(argv)
    plan = build_plan(args)
    output = Path(args.output_plan)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"WROTE_PLAN: {output}")
    print(f"DISCOVERY_RUNS: {len(plan['discovery']['runs'])}")
    print(f"CONFIRMATION_RUNS: {len(plan['confirmation']['runs'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
