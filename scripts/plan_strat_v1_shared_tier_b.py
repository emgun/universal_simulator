#!/usr/bin/env python
from __future__ import annotations

"""Pre-register the validation-only D5 shared tier-b interference experiment."""

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
D3_DARCY = 0.11694165553982801
SOURCE_FILES = (
    "configs/d5_strat_v1_shared_tier_b.yaml",
    "scripts/run_strat_v1_shared_tier_b.py",
    "scripts/plan_strat_v1_shared_tier_b.py",
    "scripts/materialize_strat_v1_shared_tier_b.py",
    "scripts/run_remote_strat_v1_shared_tier_b.sh",
    "scripts/launch_strat_v1_shared_tier_b_vast.sh",
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
        raise ValueError(f"D5 source paths must be clean and committed:\n{status.rstrip()}")
    result: dict[str, str] = {}
    for relative in paths:
        live = (REPO_ROOT / relative).read_bytes()
        committed = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=REPO_ROOT)
        if live != committed:
            raise ValueError(f"D5 source differs from implementation commit: {relative}")
        result[relative] = hashlib.sha256(live).hexdigest()
    return result


def _checked_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload.get("include") != "a4_strat_v1_baselines":
        raise ValueError("D5 config must inherit the frozen A4 strat-v1 overlay")
    schema = payload.get("data", {}).get("conditioning_schema", {})
    if tuple(schema.get("task_vocab", ())) != TASKS:
        raise ValueError("D5 task vocabulary differs from the universal schema")
    if tuple(schema.get("param_vocab", ())) != ("beta", "nu"):
        raise ValueError("D5 parameter vocabulary differs from the universal schema")
    if payload.get("operator", {}).get("conditioning", {}).get("sources") is not None:
        raise ValueError("D5 must use automatic physical conditioning sources")
    training = payload.get("training", {})
    if training.get("sample_balanced_operator_loss") is not True:
        raise ValueError("D5 must use sample-balanced operator loss")
    if training.get("canonical_steady_operator_mapping") is not True:
        raise ValueError("D5 must use canonical steady-operator mappings")
    if float(training.get("lambda_semigroup", 0.0)) != 0.0:
        raise ValueError("D5 semigroup loss must remain disabled")
    return payload


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

    if not re.fullmatch(r"[0-9a-f]{40}", args.implementation_commit):
        raise ValueError("implementation commit must be a full lowercase commit")
    resolved = subprocess.check_output(
        ["git", "rev-parse", f"{args.implementation_commit}^{{commit}}"], cwd=REPO_ROOT, text=True
    ).strip()
    if resolved != args.implementation_commit:
        raise ValueError("implementation commit did not resolve exactly")
    lock = load_data_lock(args.training_lock)
    if lock.lock_sha256 != FROZEN_STRAT_V1_TRAINING_LOCK_SHA256 or lock.purpose != "training":
        raise ValueError("D5 requires the frozen universal training lock")
    if set(lock.requested_roles) != {"train", "valid"} or any(
        item.role == "test" for item in lock.objects
    ):
        raise PermissionError("D5 must remain train/validation-only")
    observed = {item.object_id: item.checksums.get("sha256") for item in lock.objects}
    if observed != OBJECTS:
        raise ValueError("D5 training object identities differ from the frozen contract")
    _checked_config(args.config)
    sources = source_manifest(resolved)
    command = [
        "python",
        "scripts/run_strat_v1_shared_tier_b.py",
        "--training-lock",
        str(args.training_lock),
        "--data-root",
        args.data_root,
        "--config",
        str(args.config),
        "--output-dir",
        args.output_dir,
        "--plan-path",
        str(args.output_plan),
        "--plan-sha256",
        "__PLAN_SHA256__",
        "--device",
        args.device,
    ]
    plan = {
        "schema_version": 1,
        "plan_id": "strat-v1-shared-tier-b-d5",
        "mode": "validation_only",
        "heldout_access": "forbidden",
        "measurement_lock_access": "forbidden",
        "question": "Can one conditioned tier-b model match three architecture-matched specialists?",
        "bindings": {
            "source": {"implementation_commit": resolved, "files": sources},
            "training_lock": {
                "path": str(args.training_lock),
                "lock_sha256": lock.lock_sha256,
                "file_sha256": file_sha256(args.training_lock),
                "objects": OBJECTS,
            },
            "config": {"path": str(args.config), "file_sha256": file_sha256(args.config)},
        },
        "design": {
            "seed": 17,
            "tasks": list(TASKS),
            "arms": [
                "specialist-advection1d",
                "specialist-burgers1d",
                "specialist-darcy2d",
                "shared",
            ],
            "architecture": "native_ups_tier_b",
            "stage_epochs": {"operator": 12, "decoder": 6, "operator_decoded": 6, "joint": 4},
            "conditioning_schema": {"task_vocab": list(TASKS), "param_vocab": ["beta", "nu"]},
            "temporal_rollout_horizons": [1, 4, 16],
            "steady_semantics": "one_coefficient_to_solution_application",
            "objective_weighting": "equal_source_sample_weight",
        },
        "gates": {
            "shared_macro_ratio_to_specialist_oracle_maximum": 1.05,
            "shared_per_task_ratio_to_specialist_maximum": 1.10,
            "persistence_maximum_by_task": PERSISTENCE,
            "darcy_primary_maximum": D3_DARCY * 1.20,
            "maximum_corrected_regime_spread": 1.5,
            "shuffled_parameter_nrmse_degradation_minimum": 0.05,
            "shared_checkpoint_bytes_less_than_specialist_ensemble": True,
            "heldout_reads": 0,
        },
        "command": command,
    }
    plan["command_sha256"] = canonical_sha256(command)
    plan["plan_sha256"] = canonical_sha256(plan)
    if args.output_plan.exists():
        raise FileExistsError(f"refusing to overwrite plan: {args.output_plan}")
    args.output_plan.parent.mkdir(parents=True, exist_ok=True)
    args.output_plan.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output_plan), "plan_sha256": plan["plan_sha256"]}))


if __name__ == "__main__":
    main()
