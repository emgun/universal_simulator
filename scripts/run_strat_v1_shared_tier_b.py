#!/usr/bin/env python
from __future__ import annotations

"""Run the validation-only D5 shared tier-b versus matched specialist controls."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from ups.data.baseline_runtime import (  # noqa: E402
    FROZEN_STRAT_V1_TRAINING_LOCK_SHA256,
    load_strat_v1_baseline_runtime,
)
from ups.data.manifests import canonical_sha256  # noqa: E402
from ups.utils.config_loader import load_config_with_includes  # noqa: E402

TASKS = ("advection1d", "burgers1d", "darcy2d")
ARMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("specialist-advection1d", ("advection1d",)),
    ("specialist-burgers1d", ("burgers1d",)),
    ("specialist-darcy2d", ("darcy2d",)),
    ("shared", TASKS),
)
STAGES = ("operator", "decoder", "operator_decoded", "joint_codec_operator")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checked_plan(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.get("plan_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "plan_sha256"}
    if recorded != canonical_sha256(unsigned):
        raise ValueError("D5 plan self hash does not match")
    if payload.get("mode") != "validation_only":
        raise PermissionError("D5 requires validation-only mode")
    if payload.get("heldout_access") != "forbidden":
        raise PermissionError("D5 held-out access must be forbidden")
    if payload.get("measurement_lock_access") != "forbidden":
        raise PermissionError("D5 measurement-lock access must be forbidden")
    return payload


def _verify_plan_bindings(plan: dict[str, Any], args: argparse.Namespace, runtime: Any) -> None:
    """Fail closed if the live run differs from the pre-registered evidence."""

    bindings = plan.get("bindings", {})
    lock_binding = bindings.get("training_lock", {})
    if lock_binding.get("lock_sha256") != runtime.lock.lock_sha256:
        raise ValueError("D5 live training lock differs from the plan binding")
    if lock_binding.get("file_sha256") != _sha256(Path(args.training_lock)):
        raise ValueError("D5 live training-lock file differs from the plan binding")
    config_binding = bindings.get("config", {})
    if config_binding.get("file_sha256") != _sha256(Path(args.config)):
        raise ValueError("D5 live config differs from the plan binding")
    source_binding = bindings.get("source", {})
    implementation_commit = source_binding.get("implementation_commit")
    source_files = source_binding.get("files")
    if not isinstance(implementation_commit, str) or len(implementation_commit) != 40:
        raise ValueError("D5 plan lacks a full implementation-commit binding")
    if not isinstance(source_files, dict) or not source_files:
        raise ValueError("D5 plan lacks source-file bindings")
    for relative, expected in source_files.items():
        path = REPO_ROOT / relative
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"D5 live source differs from the plan binding: {relative}")


def _arm_config(base: dict[str, Any], runtime: Any, tasks: tuple[str, ...], split: str) -> dict:
    cfg = runtime.apply_to_runner_config(base, condition_on_regime=True)
    cfg["data"]["task"] = list(tasks) if len(tasks) > 1 else tasks[0]
    cfg["data"]["split"] = split
    cfg["training"]["seed"] = 17
    cfg["seed"] = 17
    cfg.setdefault("logging", {}).setdefault("wandb", {})["enabled"] = False
    return cfg


def _checkpoint_evidence(directory: Path) -> dict[str, Any]:
    records: dict[str, Any] = {}
    total_bytes = 0
    total_elements = 0
    for path in sorted(directory.glob("*.pt")):
        state = torch.load(path, map_location="cpu", weights_only=False)
        tensors = state.values() if isinstance(state, dict) else ()
        elements = sum(int(value.numel()) for value in tensors if torch.is_tensor(value))
        total_elements += elements
        total_bytes += path.stat().st_size
        records[path.name] = {
            "path": str(path),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
            "tensor_elements": elements,
        }
    if not records:
        raise FileNotFoundError(f"D5 arm produced no checkpoints in {directory}")
    return {
        "files": records,
        "total_checkpoint_bytes": total_bytes,
        "total_checkpoint_tensor_elements": total_elements,
    }


def _validate_arm_summary(path: Path, tasks: tuple[str, ...]) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    extra = summary.get("extra", {})
    if extra.get("split") not in {"val", "valid", "validation"}:
        raise PermissionError(f"D5 arm is not validation-only: {path}")
    if extra.get("strict_stratified_metrics") is not True:
        raise ValueError(f"D5 arm omitted strict stratified metrics: {path}")
    metrics = summary.get("metrics", {})
    required = {"macro_primary_nrmse"}
    for task in tasks:
        suffix = "decoded_solution_nrmse" if task == "darcy2d" else "decoded_rollout_nrmse"
        required.add(f"task_{task}_{suffix}")
        required.add(f"task_{task}_maximum_corrected_regime_spread_ratio")
    missing = sorted(required - set(metrics))
    if missing:
        raise ValueError(f"D5 arm summary lacks strict metrics: {missing}")
    return summary


def run(args: argparse.Namespace) -> Path:
    output = Path(args.output_dir)
    if output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite D5 output: {output}")
    plan = _checked_plan(Path(args.plan_path))
    if plan["plan_sha256"] != args.plan_sha256:
        raise ValueError("D5 command plan hash differs from plan file")
    if plan["command_sha256"] != canonical_sha256(plan["command"]):
        raise ValueError("D5 plan command hash does not match")
    runtime = load_strat_v1_baseline_runtime(
        args.training_lock,
        args.data_root,
        expected_lock_sha256=FROZEN_STRAT_V1_TRAINING_LOCK_SHA256,
    )
    if set(runtime.lock.requested_roles) != {"train", "valid"} or any(
        item.role == "test" for item in runtime.lock.objects
    ):
        raise PermissionError("D5 refuses any training lock with held-out objects")
    _verify_plan_bindings(plan, args, runtime)

    base = load_config_with_includes(args.config)
    schema = base.get("data", {}).get("conditioning_schema", {})
    if tuple(schema.get("task_vocab", ())) != TASKS or tuple(schema.get("param_vocab", ())) != (
        "beta",
        "nu",
    ):
        raise ValueError("D5 config does not freeze the universal conditioning schema")
    if base.get("operator", {}).get("conditioning", {}).get("sources") is not None:
        raise ValueError("D5 requires auto-discovered conditioning sources")
    training = base.get("training", {})
    if training.get("sample_balanced_operator_loss") is not True:
        raise ValueError("D5 requires sample-balanced operator loss")
    if training.get("canonical_steady_operator_mapping") is not True:
        raise ValueError("D5 requires canonical steady-operator mappings")
    if float(training.get("lambda_semigroup", 0.0)) != 0.0:
        raise ValueError("D5 forbids the unconditioned semigroup loss")

    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema_version": 1,
        "plan_sha256": plan["plan_sha256"],
        "training_lock_sha256": runtime.lock.lock_sha256,
        "config_sha256": _sha256(Path(args.config)),
        "heldout_reads": 0,
    }
    identity["identity_sha256"] = canonical_sha256(identity)
    identity_path = output / "run_identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("D5 resume identity differs from existing output")
    else:
        identity_path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")

    arms_root = output / "arms"
    config_root = output / "resolved_configs"
    config_root.mkdir(parents=True, exist_ok=True)
    arm_records: dict[str, Any] = {}
    started = time.time()
    for arm, tasks in ARMS:
        train_cfg = _arm_config(base, runtime, tasks, "train")
        eval_cfg = _arm_config(base, runtime, tasks, "val")
        train_path = config_root / f"{arm}.train.yaml"
        eval_path = config_root / f"{arm}.eval.yaml"
        train_path.write_text(yaml.safe_dump(train_cfg, sort_keys=False), encoding="utf-8")
        eval_path.write_text(yaml.safe_dump(eval_cfg, sort_keys=False), encoding="utf-8")
        summary_path = arms_root / arm / "summary.json"
        arm_started = time.time()
        if not summary_path.exists():
            command = [
                sys.executable,
                "scripts/run_light_experiment.py",
                "--config",
                str(train_path),
                "--eval-config",
                str(eval_path),
                "--name",
                arm,
                "--output-root",
                str(arms_root),
                "--decoded",
                "--decoded-rollout-steps",
                "16",
                "--device",
                args.device,
            ]
            for stage in STAGES:
                command.extend(("--stage", stage))
            subprocess.run(command, cwd=REPO_ROOT, check=True)
        summary = _validate_arm_summary(summary_path, tasks)
        arm_records[arm] = {
            "tasks": list(tasks),
            "summary_path": str(summary_path),
            "summary_file_sha256": _sha256(summary_path),
            "metrics": summary["metrics"],
            "checkpoints": _checkpoint_evidence(arms_root / arm / "checkpoints"),
            "duration_sec_observed_by_orchestrator": time.time() - arm_started,
        }

    shuffled_name = "shared-parameter-shuffled"
    shuffled_eval_cfg = _arm_config(base, runtime, TASKS, "val")
    shuffled_eval_cfg.setdefault("evaluation", {})["conditioning_parameter_index_shift"] = 1
    shuffled_eval_path = config_root / f"{shuffled_name}.eval.yaml"
    shuffled_eval_path.write_text(
        yaml.safe_dump(shuffled_eval_cfg, sort_keys=False), encoding="utf-8"
    )
    shuffled_summary_path = arms_root / shuffled_name / "summary.json"
    if not shuffled_summary_path.exists():
        subprocess.run(
            [
                sys.executable,
                "scripts/run_light_experiment.py",
                "--config",
                str(config_root / "shared.train.yaml"),
                "--eval-config",
                str(shuffled_eval_path),
                "--name",
                shuffled_name,
                "--output-root",
                str(arms_root),
                "--skip-training",
                "--checkpoint-source",
                str(arms_root / "shared" / "checkpoints"),
                "--decoded",
                "--decoded-rollout-steps",
                "16",
                "--device",
                args.device,
            ],
            cwd=REPO_ROOT,
            check=True,
        )
    shuffled_summary = _validate_arm_summary(shuffled_summary_path, TASKS)
    reference_primary = float(arm_records["shared"]["metrics"]["macro_primary_nrmse"])
    shuffled_primary = float(shuffled_summary["metrics"]["macro_primary_nrmse"])
    conditioning_diagnostics = {
        "method": "deterministic_within_task_cyclic_parameter_shift_by_one_sample",
        "summary_path": str(shuffled_summary_path),
        "summary_file_sha256": _sha256(shuffled_summary_path),
        "reference_macro_primary_nrmse": reference_primary,
        "shuffled_macro_primary_nrmse": shuffled_primary,
        "relative_nrmse_degradation": (
            shuffled_primary / reference_primary - 1.0 if reference_primary > 0.0 else float("inf")
        ),
    }

    result = {
        "schema_version": 1,
        "artifact_id": "strat-v1-shared-tier-b-d5-summary",
        "status": "complete_validation_only",
        "plan_sha256": plan["plan_sha256"],
        "training_lock_sha256": runtime.lock.lock_sha256,
        "heldout_reads": 0,
        "arms": arm_records,
        "conditioning_diagnostics": conditioning_diagnostics,
        "duration_sec": time.time() - started,
    }
    result["artifact_sha256"] = canonical_sha256(result)
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-lock", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--config", default="configs/d5_strat_v1_shared_tier_b.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--plan-path", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    path = run(build_parser().parse_args())
    print(json.dumps({"output": str(path)}))


if __name__ == "__main__":
    main()
