#!/usr/bin/env python
from __future__ import annotations

"""Pre-register the validation-only D6 modular shared-trunk experiment."""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ups.data.baseline_runtime import FROZEN_STRAT_V1_TRAINING_LOCK_SHA256  # noqa: E402
from ups.data.manifests import canonical_sha256, load_data_lock  # noqa: E402

TASKS = ("advection1d", "burgers1d", "darcy2d")
ARMS = (
    "joint-modular",
    "ablation-advection1d",
    "ablation-burgers1d",
    "ablation-darcy2d",
)
TASKS_BY_ARM = {
    "joint-modular": TASKS,
    **{f"ablation-{task}": (task,) for task in TASKS},
}
OBJECTS = {
    "advection1d-train": "aeaf3cc539f60e481b7028f4ec6293acd0c72e67612f20e4a17360239e47891d",
    "burgers1d-train": "9b7ae18e229641e2b75962673ca7699ff75fd2a51df4178ce2771d0c4ee4fd82",
    "darcy2d-train": "47945f27fa1f56f856733d3bc1aa1b0b5f498669a73cdb7352940292d71d09fe",
    "advection1d-valid": "0671198b32355842dbea16e41c3ab1ea59eb4065274b0a0ab477fb1dd383c726",
    "burgers1d-valid": "496a66bc4366d88d83fbbf9842ae14e2c93c2b726a27d9e6ac26ccd4ada68e73",
    "darcy2d-valid": "2b345a587f6f95a9ff4a12f6cce80ac4c8c83540a03c2a11f87ffdc91be1b595",
}
PERSISTENCE = {
    "advection1d": 0.673296470301147,
    "burgers1d": 0.6400475744644419,
    "darcy2d": 0.9721143218273548,
}
D5_SPECIALISTS = {
    "advection1d": 0.5323334224663255,
    "burgers1d": 0.5746504950528585,
    "darcy2d": 0.9614428185307234,
}
D5_SPECIALIST_MACRO = 0.6894755786833024
D5_SPECIALIST_CHECKPOINT_BYTES = 67_752_971
D5_PLAN_SHA256 = "5e44e12eb387eec037ac8b7200e7577f9f4d6f806a056b7516342702c9bd7bfd"
D5_RESULT_SHA256 = "737c08903ca4f45bdc992e5abb53ebe39fe948ea2feac375469bf901c9e9d762"
D3_DARCY = 0.11694165553982801
SOURCE_FILES = (
    "configs/d6_strat_v1_modular_shared_trunk.yaml",
    "configs/d5_strat_v1_shared_tier_b.yaml",
    "configs/a4_strat_v1_baselines.yaml",
    "configs/train_multitask_heterogeneous_light_best.yaml",
    "configs/defaults.yaml",
    "scripts/run_strat_v1_modular_shared_trunk.py",
    "scripts/plan_strat_v1_modular_shared_trunk.py",
    "scripts/materialize_strat_v1_modular_shared_trunk.py",
    "scripts/run_remote_strat_v1_modular_shared_trunk.sh",
    "scripts/launch_strat_v1_modular_shared_trunk_vast.sh",
    "scripts/d5_presigned_io.py",
    "scripts/generate_b2_presigned_bundle.py",
    "scripts/finalize_d5_presigned_transfer.py",
    "scripts/run_light_experiment.py",
    "scripts/train.py",
    "scripts/evaluate.py",
    "scripts/vast_remote_bootstrap.sh",
    "scripts/vast_launch.py",
    "scripts/vast_watchdog.py",
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_paths() -> tuple[str, ...]:
    shared = tuple(
        str(path.relative_to(REPO_ROOT)) for path in sorted((REPO_ROOT / "src/ups").rglob("*.py"))
    )
    return tuple(dict.fromkeys((*shared, *SOURCE_FILES)))


def source_manifest(commit: str) -> dict[str, str]:
    paths = source_paths()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *paths], cwd=REPO_ROOT, text=True
    )
    if status.strip():
        raise ValueError(f"D6 protocol sources must be clean and committed:\n{status.rstrip()}")
    result: dict[str, str] = {}
    for relative in paths:
        live = (REPO_ROOT / relative).read_bytes()
        committed = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=REPO_ROOT)
        if live != committed:
            raise ValueError(f"D6 source differs from implementation commit: {relative}")
        result[relative] = hashlib.sha256(live).hexdigest()
    return result


def _checked_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload.get("include") != "d5_strat_v1_shared_tier_b":
        raise ValueError("D6 config must inherit the frozen D5 contract")
    modular = payload.get("modular_shared_trunk", {})
    if modular.get("trunk") != "native_ups_tier_b":
        raise ValueError("D6 must retain the frozen native tier-b trunk")
    if modular.get("adapter_type") != "residual_bottleneck":
        raise ValueError("D6 adapter type must be residual_bottleneck")
    if modular.get("adapter_bottleneck_dim") != 16:
        raise ValueError("D6 adapter bottleneck must remain 16")
    if tuple(modular.get("adapter_inventory", ())) != TASKS:
        raise ValueError("D6 must declare the full three-task adapter inventory")
    if modular.get("task_specific_operator_blocks") is not False:
        raise ValueError("D6 forbids task-specific operator blocks")
    routed = payload.get("operator", {}).get("routed_adapters", {})
    expected_routed = {
        "enabled": True,
        "route_source": "task_id",
        "route_vocab": list(TASKS),
        "bottleneck_dim": 16,
        "input_enabled": True,
        "output_enabled": True,
        "zero_init": True,
    }
    if routed != expected_routed:
        raise ValueError("D6 routed adapter contract differs from the frozen task-id design")
    if payload.get("training", {}).get("patience", "missing") is not None:
        raise ValueError("D6 must disable loss-dependent early stopping for exposure parity")
    if payload.get("training", {}).get("fail_on_oom") is not True:
        raise ValueError("D6 must fail closed instead of skipping OOM samples or batches")
    arms = modular.get("arms", {})
    if tuple(arms) != ARMS:
        raise ValueError("D6 arm inventory or order differs from the frozen design")
    for arm in ARMS:
        if tuple(arms[arm].get("tasks", ())) != TASKS_BY_ARM[arm]:
            raise ValueError(f"D6 task assignment differs for {arm}")
        if tuple(arms[arm].get("adapter_inventory", TASKS)) != TASKS:
            raise ValueError(f"D6 arm {arm} lacks the full adapter inventory")
    parity = payload.get("update_parity", {})
    if parity != {
        "required": True,
        "comparison": "joint_task_to_matching_ablation",
        "dimensions": ["source_examples", "scheduled_compute_units"],
        "efficiency_reporting": "total_scheduled_optimizer_updates_by_arm",
        "fail_closed_on_missing_or_mismatch": True,
    }:
        raise ValueError("D6 update parity contract is incomplete")
    return payload


def build_plan(
    *,
    training_lock: Path,
    config: Path,
    implementation_commit: str,
    data_root: str,
    output_dir: str,
    output_plan: Path,
    device: str,
) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{40}", implementation_commit):
        raise ValueError("implementation commit must be a full lowercase commit")
    resolved = subprocess.check_output(
        ["git", "rev-parse", f"{implementation_commit}^{{commit}}"], cwd=REPO_ROOT, text=True
    ).strip()
    if resolved != implementation_commit:
        raise ValueError("implementation commit did not resolve exactly")
    lock = load_data_lock(training_lock)
    if lock.lock_sha256 != FROZEN_STRAT_V1_TRAINING_LOCK_SHA256 or lock.purpose != "training":
        raise ValueError("D6 requires the frozen universal training lock")
    if set(lock.requested_roles) != {"train", "valid"} or any(
        x.role == "test" for x in lock.objects
    ):
        raise PermissionError("D6 must remain train/validation-only")
    observed = {item.object_id: item.checksums.get("sha256") for item in lock.objects}
    if observed != OBJECTS:
        raise ValueError("D6 training object identities differ from the frozen contract")
    _checked_config(config)
    command = [
        "python",
        "scripts/run_strat_v1_modular_shared_trunk.py",
        "--training-lock",
        str(training_lock),
        "--data-root",
        data_root,
        "--config",
        str(config),
        "--output-dir",
        output_dir,
        "--plan-path",
        str(output_plan),
        "--plan-sha256",
        "__PLAN_SHA256__",
        "--stage-report",
        "reports/research/strat_v1_modular_shared_trunk_stage.json",
        "--device",
        device,
    ]
    plan: dict[str, Any] = {
        "schema_version": 1,
        "plan_id": "strat-v1-modular-shared-trunk-d6-v4",
        "mode": "validation_only",
        "heldout_access": "forbidden",
        "measurement_lock_access": "forbidden",
        "execution_authorization": "validation_only_runner_bound",
        "bindings": {
            "source": {"implementation_commit": resolved, "files": source_manifest(resolved)},
            "training_lock": {
                "path": str(training_lock),
                "lock_sha256": lock.lock_sha256,
                "file_sha256": file_sha256(training_lock),
                "objects": OBJECTS,
            },
            "config": {"path": str(config), "file_sha256": file_sha256(config)},
            "frozen_d5": {
                "plan_sha256": D5_PLAN_SHA256,
                "result_artifact_sha256": D5_RESULT_SHA256,
            },
        },
        "design": {
            "seed": 17,
            "tasks": list(TASKS),
            "arms": list(ARMS),
            "tasks_by_arm": {key: list(value) for key, value in TASKS_BY_ARM.items()},
            "architecture": "modular_adapters_shared_native_ups_tier_b_trunk",
            "adapter_type": "residual_bottleneck",
            "adapter_bottleneck_dim": 16,
            "adapter_placement": {
                "input": "after_time_and_conditioning_immediately_before_pde_transformer",
                "output": "after_shared_output_norm_before_outer_state_residual",
            },
            "adapter_inventory_by_arm": {arm: list(TASKS) for arm in ARMS},
            "stage_epochs": {"operator": 12, "decoder": 6, "operator_decoded": 6, "joint": 4},
            "stage_batch_sizes": {"operator": 16, "decoder": 2, "operator_decoded": 2, "joint": 2},
            "conditioning_schema": {"task_vocab": list(TASKS), "param_vocab": ["beta", "nu"]},
            "temporal_rollout_horizons": [1, 4, 16],
            "steady_semantics": "one_coefficient_to_solution_application",
            "objective_weighting": "equal_source_sample_weight",
            "update_parity": {
                "required": True,
                "comparison": "joint_task_to_matching_ablation",
                "dimensions": ["source_examples", "scheduled_compute_units"],
                "efficiency_reporting": "total_scheduled_optimizer_updates_by_arm",
            },
        },
        "frozen_references": {
            "d5_specialist_by_task": D5_SPECIALISTS,
            "d5_specialist_macro_primary_nrmse": D5_SPECIALIST_MACRO,
            "d5_specialist_ensemble_checkpoint_bytes": D5_SPECIALIST_CHECKPOINT_BYTES,
        },
        "gates": {
            "u1": {
                "joint_macro_ratio_to_frozen_d5_specialist_maximum": 1.10,
                "joint_per_task_ratio_to_frozen_d5_specialist_maximum": 1.20,
                "persistence_maximum_by_task": PERSISTENCE,
                "darcy_primary_maximum": D3_DARCY * 1.20,
                "maximum_corrected_regime_spread": 1.5,
                "shuffled_parameter_nrmse_degradation_minimum": 0.05,
                "joint_checkpoint_bytes_less_than_frozen_d5_ensemble": True,
                "joint_initialized_tensor_elements_less_than_matched_ablation_ensemble": True,
                "heldout_reads": 0,
            },
            "u2": {
                "joint_macro_ratio_to_matched_ablation_macro_maximum": 1.05,
                "joint_per_task_ratio_to_matched_ablation_maximum": 1.10,
                "update_parity_required": True,
            },
        },
        "command": command,
    }
    plan["command_sha256"] = canonical_sha256(command)
    plan["plan_sha256"] = canonical_sha256(plan)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-lock", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--output-plan", type=Path, required=True)
    args = parser.parse_args()
    plan = build_plan(
        training_lock=args.training_lock,
        config=args.config,
        implementation_commit=args.implementation_commit,
        data_root=args.data_root,
        output_dir=args.output_dir,
        output_plan=args.output_plan,
        device=args.device,
    )
    if args.output_plan.exists():
        raise FileExistsError(f"refusing to overwrite plan: {args.output_plan}")
    args.output_plan.parent.mkdir(parents=True, exist_ok=True)
    args.output_plan.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output_plan), "plan_sha256": plan["plan_sha256"]}))


if __name__ == "__main__":
    main()
