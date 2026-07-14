#!/usr/bin/env python
"""Execute one frozen, validation-only reference-recipe adequacy run set.

The executor deliberately does not build or modify plans.  It revalidates the
complete plan, code, package, and local cache boundary before starting any
subprocess, then invokes only the selected plan commands from the repository
root.  Test and held-out access are categorically rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.plan_reference_recipe_adequacy import (  # noqa: E402
    ARCHITECTURES,
    CONFIRMATION_SEEDS,
    DISCOVERY_SEED,
    EPOCH_RUNGS,
    MODEL_ARGS,
    RUNNERS,
    TASKS,
    TRAINING_LOCK_SHA256,
)
from ups.data.manifests import canonical_sha256, load_data_lock  # noqa: E402

PLAN_ID = "strat-v1.1-fno-uno-reference-recipe-adequacy-v1"
SELECTION_ID = "strat-v1.1-reference-recipe-adequacy-selection-v1"
FORBIDDEN_FLAGS = frozenset(
    {
        "--allow-held-out-test-eval",
        "--allow-repeat-test",
        "--measurement-lock",
        "--test-split",
        "--residual",
        "--identity-scaling",
        "--dry-run",
    }
)
FIXED_RECIPE_OPTIONS = {
    "--max-train-samples": "288",
    "--max-eval-samples": "72",
    "--rollout-steps": "16",
    "--max-pairs-per-task": "1152",
    "--train-stride": "4",
    "--learning-rate": "0.001",
    "--weight-decay": "0.0001",
    "--batch-size": "8",
    "--metric": "decoded_rollout_nrmse",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _resolve_repo_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty path")
    path = Path(value)
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def _option(command: list[str], flag: str) -> str:
    if command.count(flag) != 1:
        raise ValueError(f"command must contain {flag} exactly once")
    index = command.index(flag)
    if index + 1 >= len(command) or command[index + 1].startswith("--"):
        raise ValueError(f"command lacks a value for {flag}")
    return command[index + 1]


def _option_values(command: list[str], flag: str) -> list[str]:
    if command.count(flag) != 1:
        raise ValueError(f"command must contain {flag} exactly once")
    index = command.index(flag) + 1
    values: list[str] = []
    while index < len(command) and not command[index].startswith("--"):
        values.append(command[index])
        index += 1
    if not values:
        raise ValueError(f"command lacks values for {flag}")
    return values


def _verify_plan_identity(plan: dict[str, Any]) -> None:
    recorded = plan.get("plan_sha256")
    payload = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if recorded != canonical_sha256(payload):
        raise ValueError("recipe-adequacy plan SHA-256 does not match its canonical payload")
    if plan.get("schema_version") != 1 or plan.get("plan_id") != PLAN_ID:
        raise ValueError("unsupported recipe-adequacy plan identity")
    if plan.get("mode") != "validation_only":
        raise ValueError("executor accepts only validation-only plans")
    if plan.get("execution_policy") != "plan_only_no_runner_invocation":
        raise ValueError("plan execution policy is not the frozen planner-only policy")
    if plan.get("heldout_access") != "forbidden":
        raise ValueError("plan does not forbid held-out access")
    if plan.get("measurement_lock_access") != "forbidden":
        raise ValueError("plan does not forbid measurement-lock access")


def _verify_bound_files(plan: dict[str, Any]) -> None:
    bindings = _mapping(plan.get("bindings"), "bindings")
    for name in ("metric_addendum", "training_lock"):
        record = _mapping(bindings.get(name), f"bindings.{name}")
        path = _resolve_repo_path(record.get("path"), f"bindings.{name}.path")
        if not path.is_file() or _file_sha256(path) != record.get("file_sha256"):
            raise ValueError(f"bound {name} file no longer matches the plan")

    config = _mapping(plan.get("config"), "config")
    config_path = _resolve_repo_path(config.get("path"), "config.path")
    if not config_path.is_file() or _file_sha256(config_path) != config.get("file_sha256"):
        raise ValueError("bound config file no longer matches the plan")

    lock_record = _mapping(bindings.get("training_lock"), "bindings.training_lock")
    lock_path = _resolve_repo_path(lock_record.get("path"), "bindings.training_lock.path")
    lock = load_data_lock(lock_path)
    if lock.lock_sha256 != TRAINING_LOCK_SHA256:
        raise ValueError("executor requires the frozen universal training lock")
    if lock_record.get("lock_sha256") != TRAINING_LOCK_SHA256:
        raise ValueError("plan is not bound to the frozen universal training lock")
    if lock.purpose != "training" or set(lock.requested_roles) != {"train", "valid"}:
        raise ValueError("training lock must contain exactly train and valid roles")
    if lock.measurement_contract_id is not None or any(
        item.role == "test" for item in lock.objects
    ):
        raise ValueError("training lock contains measurement or test authority")
    if lock_record.get("roles") != ["train", "valid"]:
        raise ValueError("plan training roles are not exactly train and valid")
    if lock_record.get("object_ids") != [item.object_id for item in lock.objects]:
        raise ValueError("plan training object inventory does not match its bound lock")


def _verify_runner_and_package_identity(plan: dict[str, Any]) -> None:
    identities = _mapping(plan.get("runner_identity"), "runner_identity")
    expected_version: str | None = None
    for architecture in ARCHITECTURES:
        identity = _mapping(identities.get(architecture), f"runner_identity.{architecture}")
        if identity.get("path") != RUNNERS[architecture]:
            raise ValueError(f"{architecture} runner path is not canonical")
        runner = REPO_ROOT / RUNNERS[architecture]
        if _file_sha256(runner) != identity.get("file_sha256"):
            raise ValueError(f"{architecture} runner file changed after planning")
        package = identity.get("external_package")
        if not isinstance(package, str) or not package.startswith("neuraloperator=="):
            raise ValueError(f"{architecture} package identity is malformed")
        version = package.removeprefix("neuraloperator==")
        if not version or (expected_version is not None and version != expected_version):
            raise ValueError("runner package identities disagree")
        expected_version = version

    assert expected_version is not None
    for package_name in ("neuraloperator", "neuralop"):
        try:
            observed = importlib.metadata.version(package_name)
            break
        except importlib.metadata.PackageNotFoundError:
            continue
    else:
        raise RuntimeError("executor requires the pinned neuraloperator package")
    if observed != expected_version:
        raise RuntimeError(
            f"neuraloperator version mismatch: expected {expected_version}, got {observed}"
        )


def _select_and_verify_run_set(plan: dict[str, Any], run_set: str) -> list[dict[str, Any]]:
    discovery = _mapping(plan.get("discovery"), "discovery")
    confirmation = _mapping(plan.get("confirmation"), "confirmation")
    if discovery.get("architectures") != list(ARCHITECTURES):
        raise ValueError("discovery architectures are not frozen")
    if discovery.get("tasks") != list(TASKS):
        raise ValueError("discovery tasks are not frozen")
    if discovery.get("epoch_rungs") != list(EPOCH_RUNGS):
        raise ValueError("discovery epoch rungs are not frozen")
    if discovery.get("maximum_epochs") != EPOCH_RUNGS[-1]:
        raise ValueError("discovery maximum epoch is not frozen")
    if discovery.get("seed") != DISCOVERY_SEED:
        raise ValueError("discovery seed is not frozen")
    if discovery.get("independent_rung_restarts") is not False:
        raise ValueError("discovery must use continuous trajectories")

    runs = _list(discovery.get("runs"), "discovery.runs")
    if run_set == "discovery":
        expected = [
            (architecture, DISCOVERY_SEED, EPOCH_RUNGS[-1]) for architecture in ARCHITECTURES
        ]
        observed = [(run.get("architecture"), run.get("seed"), run.get("epochs")) for run in runs]
        if observed != expected or len(runs) != 2:
            raise ValueError("plan does not contain exactly the frozen discovery run set")
        return [_mapping(run, "discovery run") for run in runs]

    architecture = confirmation.get("selected_architecture")
    epochs = confirmation.get("selected_epochs")
    if architecture not in ARCHITECTURES or epochs not in EPOCH_RUNGS:
        raise ValueError("confirmation requires one selected architecture and frozen epoch rung")
    if confirmation.get("required_seeds") != list(CONFIRMATION_SEEDS):
        raise ValueError("confirmation seed policy is not frozen")
    if confirmation.get("additional_seeds") != list(CONFIRMATION_SEEDS[1:]):
        raise ValueError("confirmation additional seeds are not frozen")
    if confirmation.get("reuse_discovery_seed_17") is not True:
        raise ValueError("confirmation must reuse discovery seed 17")
    runs = _list(confirmation.get("runs"), "confirmation.runs")
    expected = [(architecture, seed, epochs) for seed in CONFIRMATION_SEEDS[1:]]
    observed = [(run.get("architecture"), run.get("seed"), run.get("epochs")) for run in runs]
    if observed != expected or len(runs) != 2:
        raise ValueError("plan does not contain exactly the selected confirmation run set")
    return [_mapping(run, "confirmation run") for run in runs]


def _verify_confirmation_evidence(plan: dict[str, Any]) -> None:
    """Revalidate the immutable discovery-to-selection derivation for confirmation."""

    confirmation = _mapping(plan.get("confirmation"), "confirmation")
    binding = _mapping(confirmation.get("evidence_binding"), "confirmation.evidence_binding")
    discovery_binding = _mapping(
        binding.get("discovery_plan"), "confirmation.evidence_binding.discovery_plan"
    )
    selection_binding = _mapping(
        binding.get("selection_artifact"), "confirmation.evidence_binding.selection_artifact"
    )

    discovery_path = _resolve_repo_path(
        discovery_binding.get("path"), "confirmation.evidence_binding.discovery_plan.path"
    )
    if not discovery_path.is_file():
        raise FileNotFoundError(discovery_path)
    if _file_sha256(discovery_path) != discovery_binding.get("file_sha256"):
        raise ValueError("bound discovery plan file hash does not match")
    discovery_plan = json.loads(discovery_path.read_text(encoding="utf-8"))
    if not isinstance(discovery_plan, dict):
        raise ValueError("bound discovery plan must be a mapping")
    _verify_plan_identity(discovery_plan)
    if discovery_binding.get("plan_sha256") != discovery_plan.get("plan_sha256"):
        raise ValueError("bound discovery plan self hash does not match")
    _select_and_verify_run_set(discovery_plan, "discovery")
    discovery_confirmation = _mapping(
        discovery_plan.get("confirmation"), "discovery_plan.confirmation"
    )
    if (
        discovery_confirmation.get("selected_architecture") is not None
        or discovery_confirmation.get("selected_epochs") is not None
        or discovery_confirmation.get("evidence_binding") is not None
        or discovery_confirmation.get("runs") != []
    ):
        raise ValueError("bound discovery plan already contains confirmation authority")

    # Confirmation may add only evidence-derived confirmation fields and runs;
    # its experimental discovery authority must be byte-for-byte equivalent.
    for key in (
        "bindings",
        "config",
        "runner_identity",
        "discovery",
        "plateau_criterion",
        "selection",
    ):
        if plan.get(key) != discovery_plan.get(key):
            raise ValueError(f"confirmation plan changed frozen discovery field: {key}")

    selection_path = _resolve_repo_path(
        selection_binding.get("path"), "confirmation.evidence_binding.selection_artifact.path"
    )
    if not selection_path.is_file():
        raise FileNotFoundError(selection_path)
    if _file_sha256(selection_path) != selection_binding.get("file_sha256"):
        raise ValueError("bound selection artifact file hash does not match")
    artifact = json.loads(selection_path.read_text(encoding="utf-8"))
    if not isinstance(artifact, dict):
        raise ValueError("bound selection artifact must be a mapping")
    recorded_selection_sha = artifact.get("selection_sha256")
    selection_payload = {key: value for key, value in artifact.items() if key != "selection_sha256"}
    if recorded_selection_sha != canonical_sha256(selection_payload):
        raise ValueError("selection artifact SHA-256 does not match its canonical payload")
    if selection_binding.get("selection_sha256") != recorded_selection_sha:
        raise ValueError("bound selection artifact self hash does not match")
    if artifact.get("schema_version") != 1 or artifact.get("selection_id") != SELECTION_ID:
        raise ValueError("unsupported selection artifact identity")
    if artifact.get("status") != "complete_validation_only":
        raise ValueError("selection artifact is not complete validation-only evidence")
    if artifact.get("plan_sha256") != discovery_plan.get("plan_sha256"):
        raise ValueError("selection artifact does not derive from the bound discovery plan")
    if artifact.get("held_out_measurements") != 0:
        raise ValueError("selection artifact reports held-out measurements")
    if artifact.get("no_eligible_architecture") is not False:
        raise ValueError("confirmation cannot run without an eligible selected architecture")
    selected = _mapping(artifact.get("selection"), "selection_artifact.selection")
    architecture = selected.get("architecture")
    epoch = selected.get("epoch")
    if architecture not in ARCHITECTURES or epoch not in EPOCH_RUNGS:
        raise ValueError("selection artifact chose an invalid architecture or epoch")
    architecture_rows = _mapping(artifact.get("architectures"), "selection_artifact.architectures")
    architecture_row = _mapping(
        architecture_rows.get(architecture),
        f"selection_artifact.architectures.{architecture}",
    )
    if architecture_row.get("eligible") is not True:
        raise ValueError("materialized selection architecture is not eligible")
    if architecture_row.get("chosen_epoch") != epoch:
        raise ValueError("materialized selection epoch differs from architecture evidence")
    if architecture_row.get("chosen_macro_primary_nrmse") != selected.get(
        "macro_primary_nrmse"
    ) or architecture_row.get("chosen_checkpoint") != selected.get("checkpoint"):
        raise ValueError("materialized selection differs from architecture evidence")
    if architecture != confirmation.get("selected_architecture"):
        raise ValueError("confirmation architecture differs from materialized selection")
    if epoch != confirmation.get("selected_epochs"):
        raise ValueError("confirmation epoch differs from materialized selection")


def _verify_command(
    run: dict[str, Any], plan: dict[str, Any], run_set: str
) -> tuple[list[str], Path]:
    command = _list(run.get("command"), "run.command")
    if not command or not all(isinstance(value, str) and value for value in command):
        raise ValueError("run command must contain only non-empty strings")
    command = list(command)
    if run.get("command_sha256") != canonical_sha256(command):
        raise ValueError("run command hash does not match its canonical command")
    if len(command) < 2 or Path(command[0]).resolve() != Path(sys.executable).resolve():
        raise ValueError("run command does not use the current Python interpreter")
    if command[1] != RUNNERS.get(run.get("architecture")):
        raise ValueError("run command does not invoke its canonical architecture runner")
    if FORBIDDEN_FLAGS.intersection(command):
        raise ValueError("run command contains a forbidden execution flag")
    if any(value.startswith("--") and "=" in value for value in command):
        raise ValueError("run command must use canonical separate option values")
    if any(value.lower() in {"test", "heldout", "held-out"} for value in command):
        raise ValueError("run command contains a held-out role value")
    if command.count("--strict-contract") != 1 or command.count("--refuse-overwrite") != 1:
        raise ValueError("run command lacks strict-contract or overwrite protection")
    if _option(command, "--train-split") != "train" or _option(command, "--eval-split") != "val":
        raise ValueError("run command is not train plus validation only")
    if _option(command, "--expected-data-lock-sha256") != TRAINING_LOCK_SHA256:
        raise ValueError("run command does not require the frozen training lock")
    config_path = _resolve_repo_path(_option(command, "--config"), "--config")
    if config_path != _resolve_repo_path(plan["config"]["path"], "config.path"):
        raise ValueError("run command config differs from the plan binding")
    for flag, expected in FIXED_RECIPE_OPTIONS.items():
        if _option(command, flag) != expected:
            raise ValueError(f"run command changes frozen recipe option {flag}")
    architecture = run.get("architecture")
    for key, expected in MODEL_ARGS[architecture].items():
        flag = f"--{key.replace('_', '-')}"
        if _option(command, flag) != str(expected):
            raise ValueError(f"run command changes frozen {architecture} option {flag}")
    if _resolve_repo_path(_option(command, "--data-lock"), "--data-lock") != _resolve_repo_path(
        plan["bindings"]["training_lock"]["path"], "bindings.training_lock.path"
    ):
        raise ValueError("run command data lock differs from the plan binding")
    if run.get("tasks") != list(TASKS) or run.get("train_role") != "train":
        raise ValueError("run metadata does not bind the frozen all-task training scope")
    if _option_values(command, "--tasks") != list(TASKS):
        raise ValueError("run command does not bind the frozen all-task scope")
    if run.get("evaluation_role") != "valid":
        raise ValueError("run metadata is not validation-only")
    if run.get("phase") != run_set:
        raise ValueError("run phase differs from the requested run set")
    if int(_option(command, "--epochs")) != run.get("epochs"):
        raise ValueError("run epochs differ from run metadata")
    if int(_option(command, "--seed")) != run.get("seed"):
        raise ValueError("run seed differs from run metadata")
    expected_rungs = list(EPOCH_RUNGS) if run_set == "discovery" else [run["epochs"]]
    if [int(value) for value in _option_values(command, "--validation-rungs")] != expected_rungs:
        raise ValueError("validation checkpoint rungs differ from frozen run-set policy")
    if _option(command, "--name") != run.get("run_id"):
        raise ValueError("run name differs from run metadata")

    output_root = _resolve_repo_path(_option(command, "--output-root"), "--output-root")
    run_dir = output_root / str(run["run_id"])
    expected_summary = _resolve_repo_path(run.get("expected_summary"), "expected_summary")
    if expected_summary != (run_dir / "summary.json").resolve():
        raise ValueError("expected summary is outside the declared run output")
    return command, run_dir


def _verify_local_training_cache(plan: dict[str, Any], data_root: Path) -> None:
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)
    if any(data_root.rglob("*_test.h5")):
        raise ValueError("executor refuses a data root containing test HDF5 objects")
    lock_path = _resolve_repo_path(
        plan["bindings"]["training_lock"]["path"], "bindings.training_lock.path"
    )
    lock = load_data_lock(lock_path)
    root = data_root.resolve()
    for item in lock.objects:
        path = (root / item.path).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"training object path escapes data root: {item.path}") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != item.size_bytes or _file_sha256(path) != item.checksums["sha256"]:
            raise ValueError(f"training cache object does not match frozen lock: {path}")


def execute_plan(
    plan: dict[str, Any],
    *,
    run_set: str,
    subprocess_run: Callable[..., Any] = subprocess.run,
) -> None:
    """Validate the whole selected run set, then execute its exact commands."""

    _verify_plan_identity(plan)
    _verify_bound_files(plan)
    _verify_runner_and_package_identity(plan)
    runs = _select_and_verify_run_set(plan, run_set)
    if run_set == "confirmation":
        _verify_confirmation_evidence(plan)

    preflight: list[list[str]] = []
    data_roots: set[Path] = set()
    output_roots: set[Path] = set()
    run_dirs: set[Path] = set()
    for run in runs:
        command, run_dir = _verify_command(run, plan, run_set)
        data_roots.add(_resolve_repo_path(_option(command, "--data-root"), "--data-root"))
        output_roots.add(run_dir.parent)
        if run_dir in run_dirs:
            raise ValueError(f"duplicate run output in plan: {run_dir}")
        run_dirs.add(run_dir)
        if run_dir.exists():
            raise FileExistsError(f"refusing existing run output: {run_dir}")
        preflight.append(command)
    if len(data_roots) != 1:
        raise ValueError("selected run set must use exactly one training cache root")
    if len(output_roots) != 1:
        raise ValueError("selected run set must use exactly one output root")
    _verify_local_training_cache(plan, next(iter(data_roots)))

    for command in preflight:
        subprocess_run(command, cwd=REPO_ROOT, check=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--run-set", required=True, choices=("discovery", "confirmation"))
    parser.add_argument("--confirm-validation-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.confirm_validation_only:
        parser.error("execution requires explicit --confirm-validation-only")

    plan_path = Path(args.plan)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(plan, dict):
        parser.error("plan JSON must be a mapping")
    execute_plan(plan, run_set=args.run_set)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
