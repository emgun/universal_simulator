#!/usr/bin/env python
from __future__ import annotations

"""Run the validation-only D6 modular shared-trunk and matched U2 ablations."""

import argparse
import copy
import hashlib
import json
import math
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.nn.parameter import UninitializedBuffer, UninitializedParameter

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
    ("joint-modular", TASKS),
    ("ablation-advection1d", ("advection1d",)),
    ("ablation-burgers1d", ("burgers1d",)),
    ("ablation-darcy2d", ("darcy2d",)),
)
ARM_NAMES = tuple(name for name, _ in ARMS)
ARM_TASKS = {name: tasks for name, tasks in ARMS}
STAGES = ("operator", "decoder", "operator_decoded", "joint_codec_operator")
VALIDATION_SPLITS = {"val", "valid", "validation"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unsigned_hash(payload: dict[str, Any], hash_key: str) -> str:
    return canonical_sha256({key: value for key, value in payload.items() if key != hash_key})


def _plan_arm_names(plan: dict[str, Any]) -> tuple[str, ...]:
    raw = plan.get("design", {}).get("arms")
    if not isinstance(raw, list):
        raise ValueError("D6 plan design.arms must be a list")
    names: list[str] = []
    for item in raw:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict) and isinstance(item.get("name"), str):
            names.append(item["name"])
        else:
            raise ValueError("D6 plan design.arms entries must be names or named objects")
    return tuple(names)


def _checked_plan(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("plan_sha256") != _unsigned_hash(payload, "plan_sha256"):
        raise ValueError("D6 plan self hash does not match")
    if payload.get("mode") != "validation_only":
        raise PermissionError("D6 requires validation-only mode")
    if payload.get("heldout_access") != "forbidden":
        raise PermissionError("D6 held-out access must be forbidden")
    if payload.get("measurement_lock_access") != "forbidden":
        raise PermissionError("D6 measurement-lock access must be forbidden")
    if _plan_arm_names(payload) != ARM_NAMES:
        raise ValueError(f"D6 plan arm mismatch; expected {list(ARM_NAMES)}")
    design = payload.get("design", {})
    if int(design.get("seed", -1)) != 17:
        raise ValueError("D6 plan seed must be 17")
    architecture = str(design.get("architecture", "")).lower()
    if "modular" not in architecture:
        raise ValueError("D6 plan architecture must identify the modular candidate")
    return payload


def _verify_plan_bindings(plan: dict[str, Any], args: argparse.Namespace, runtime: Any) -> None:
    """Fail closed when the live lock, config, or source differs from the plan."""

    bindings = plan.get("bindings", {})
    lock_binding = bindings.get("training_lock", {})
    if lock_binding.get("lock_sha256") != runtime.lock.lock_sha256:
        raise ValueError("D6 live training lock differs from the plan binding")
    if lock_binding.get("file_sha256") != _sha256(Path(args.training_lock)):
        raise ValueError("D6 live training-lock file differs from the plan binding")
    config_binding = bindings.get("config", {})
    if config_binding.get("file_sha256") != _sha256(Path(args.config)):
        raise ValueError("D6 live config differs from the plan binding")
    source_binding = bindings.get("source", {})
    implementation_commit = source_binding.get("implementation_commit")
    source_files = source_binding.get("files")
    if not isinstance(implementation_commit, str) or len(implementation_commit) != 40:
        raise ValueError("D6 plan lacks a full implementation-commit binding")
    if not isinstance(source_files, dict) or not source_files:
        raise ValueError("D6 plan lacks source-file bindings")
    for relative, expected in source_files.items():
        path = REPO_ROOT / str(relative)
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"D6 live source differs from the plan binding: {relative}")


def _contains_adapter_setting(value: Any, *, key: str = "") -> bool:
    if "adapter" in key.lower():
        return True
    if isinstance(value, dict):
        return any(_contains_adapter_setting(item, key=str(name)) for name, item in value.items())
    if isinstance(value, list):
        return any(_contains_adapter_setting(item) for item in value)
    return False


def _validate_base_config(base: dict[str, Any]) -> None:
    schema = base.get("data", {}).get("conditioning_schema", {})
    if tuple(schema.get("task_vocab", ())) != TASKS:
        raise ValueError("D6 config does not freeze the universal task vocabulary")
    if tuple(schema.get("param_vocab", ())) != ("beta", "nu"):
        raise ValueError("D6 config does not freeze the universal parameter vocabulary")
    if not _contains_adapter_setting(base.get("operator", {})):
        raise ValueError("D6 config does not enable a modular adapter setting")
    modular = base.get("modular_shared_trunk", {})
    if modular.get("adapter_type") != "residual_bottleneck":
        raise ValueError("D6 config adapter type must be residual_bottleneck")
    if int(modular.get("adapter_bottleneck_dim", -1)) != 16:
        raise ValueError("D6 config adapter bottleneck must remain 16")
    if tuple(modular.get("adapter_inventory", ())) != TASKS:
        raise ValueError("D6 config must retain the full three-task adapter inventory")
    configured_arms = modular.get("arms", {})
    if not isinstance(configured_arms, dict) or tuple(configured_arms) != ARM_NAMES:
        raise ValueError("D6 config arm inventory differs from the frozen design")
    for arm in ARM_NAMES:
        if tuple(configured_arms[arm].get("tasks", ())) != ARM_TASKS[arm]:
            raise ValueError(f"D6 config task assignment differs for {arm}")
        if tuple(configured_arms[arm].get("adapter_inventory", ())) != TASKS:
            raise ValueError(f"D6 config {arm} lacks the full adapter inventory")
    if base.get("operator", {}).get("conditioning", {}).get("sources") is not None:
        raise ValueError("D6 requires auto-discovered conditioning sources")
    training = base.get("training", {})
    if training.get("patience", "missing") is not None:
        raise ValueError("D6 must disable loss-dependent early stopping for exposure parity")
    if training.get("fail_on_oom") is not True:
        raise ValueError("D6 must fail closed instead of skipping OOM samples or batches")
    if training.get("sample_balanced_operator_loss") is not True:
        raise ValueError("D6 requires sample-balanced operator loss")
    if training.get("canonical_steady_operator_mapping") is not True:
        raise ValueError("D6 requires canonical steady-operator mappings")
    if float(training.get("lambda_semigroup", 0.0)) != 0.0:
        raise ValueError("D6 forbids the unconditioned semigroup loss")


def _arm_config(base: dict[str, Any], runtime: Any, tasks: tuple[str, ...], split: str) -> dict:
    if split not in {"train", *VALIDATION_SPLITS}:
        raise PermissionError(f"D6 refuses non-training/non-validation split {split!r}")
    cfg = runtime.apply_to_runner_config(copy.deepcopy(base), condition_on_regime=True)
    cfg["data"]["task"] = list(tasks) if len(tasks) > 1 else tasks[0]
    cfg["data"]["split"] = split
    cfg["training"]["seed"] = 17
    cfg["seed"] = 17
    cfg.setdefault("logging", {}).setdefault("wandb", {})["enabled"] = False
    return cfg


def _checkpoint_evidence(directory: Path) -> dict[str, Any]:
    records: dict[str, Any] = {}
    total_bytes = 0
    total_initialized_elements = 0
    total_adapter_elements = 0
    total_uninitialized_tensors = 0
    for path in sorted(directory.glob("*.pt")):
        state = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(state, dict):
            raise TypeError(f"D6 checkpoint is not a state dictionary: {path}")
        initialized_elements = 0
        adapter_elements = 0
        uninitialized_keys: list[str] = []
        saw_tensor = False
        for key, value in state.items():
            if not torch.is_tensor(value):
                continue
            saw_tensor = True
            if isinstance(value, (UninitializedParameter, UninitializedBuffer)):
                uninitialized_keys.append(str(key))
                continue
            elements = int(value.numel())
            initialized_elements += elements
            if "adapter" in str(key).lower():
                adapter_elements += elements
        if not saw_tensor:
            raise ValueError(f"D6 checkpoint contains no tensor state: {path}")
        total_initialized_elements += initialized_elements
        total_adapter_elements += adapter_elements
        total_uninitialized_tensors += len(uninitialized_keys)
        size_bytes = path.stat().st_size
        total_bytes += size_bytes
        records[path.name] = {
            "path": str(path),
            "sha256": _sha256(path),
            "size_bytes": size_bytes,
            "initialized_tensor_elements": initialized_elements,
            "adapter_tensor_elements": adapter_elements,
            "shared_tensor_elements": initialized_elements - adapter_elements,
            "uninitialized_tensor_count": len(uninitialized_keys),
            "uninitialized_tensor_keys": uninitialized_keys,
        }
    if not records:
        raise FileNotFoundError(f"D6 arm produced no checkpoints in {directory}")
    if total_adapter_elements <= 0:
        raise ValueError(f"D6 arm checkpoints contain no modular adapter state: {directory}")
    return {
        "files": records,
        "total_checkpoint_bytes": total_bytes,
        "total_initialized_tensor_elements": total_initialized_elements,
        "total_adapter_tensor_elements": total_adapter_elements,
        "total_shared_tensor_elements": total_initialized_elements - total_adapter_elements,
        "total_uninitialized_tensor_count": total_uninitialized_tensors,
    }


def _training_log_evidence(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "present": False, "records": 0, "stage_epochs": {}}
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"D6 invalid training log JSON at {path}:{line_number}") from exc
        if isinstance(row, dict):
            records.append(row)
    stage_epochs: dict[str, int] = {}
    epoch_time_sec = 0.0
    for row in records:
        for stage in STAGES:
            if f"{stage}/epoch" in row:
                stage_epochs[stage] = max(
                    stage_epochs.get(stage, 0), int(row[f"{stage}/epoch"]) + 1
                )
            if f"{stage}/epoch_time_sec" in row:
                epoch_time_sec += float(row[f"{stage}/epoch_time_sec"])
    return {
        "path": str(path),
        "present": True,
        "sha256": _sha256(path),
        "records": len(records),
        "stage_epochs": stage_epochs,
        "reported_epoch_time_sec": epoch_time_sec,
    }


def _resource_evidence(
    *, run_dir: Path, summary: dict[str, Any], wall_time_sec: float, child_max_rss_kib: int
) -> dict[str, Any]:
    extra = summary.get("extra", {})
    reported = summary.get("resource_accounting", extra.get("resource_accounting", {}))
    if not isinstance(reported, dict):
        reported = {}
    duration = summary.get("duration_sec")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)):
        raise ValueError("D6 arm summary lacks a numeric runner duration")
    duration = float(duration)
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("D6 arm runner duration must be finite and positive")
    rss_value = child_max_rss_kib if child_max_rss_kib > 0 else None
    return {
        "wall_time_sec_observed_by_orchestrator": wall_time_sec,
        "duration_sec_reported_by_runner": duration,
        "child_process_family_max_rss_kib_high_watermark": rss_value,
        "rss_scope": (
            "cumulative RUSAGE_CHILDREN process-family high-water mark; "
            "not attributable to this arm alone"
        ),
        "runner_reported": reported,
        "training_log": _training_log_evidence(run_dir / "logs" / "training.jsonl"),
    }


def _derive_scheduled_exposure(
    *, record: dict[str, Any], base: dict[str, Any], runtime: Any, task: str
) -> dict[str, int]:
    """Derive exact scheduled task exposure from sealed counts and completed epochs.

    The source count comes from the verified strat-v1 runtime, while completed
    epochs come from the arm's append-only training log. Every stage must reach
    its hash-bound epoch count; matching truncated arms are not valid parity.
    """

    task_runtime = getattr(runtime, "tasks", {}).get(task)
    train_runtime = getattr(task_runtime, "train", None)
    sample_count = getattr(train_runtime, "sample_count", None)
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count <= 0:
        raise ValueError(f"D6 sealed runtime lacks a positive training sample count for {task}")
    training_log = record.get("resources", {}).get("training_log", {})
    completed = training_log.get("stage_epochs")
    if not isinstance(completed, dict):
        raise ValueError(f"D6 arm lacks completed stage-epoch evidence for {task}")

    source_examples = 0
    scheduled_compute_units = 0
    for stage in STAGES:
        configured_epochs = int(base.get("stages", {}).get(stage, {}).get("epochs", 0) or 0)
        if configured_epochs <= 0:
            continue
        observed_epochs = completed.get(stage)
        if isinstance(observed_epochs, bool) or not isinstance(observed_epochs, int):
            raise ValueError(f"D6 arm training log lacks completed {stage} epochs for {task}")
        if observed_epochs != configured_epochs:
            raise ValueError(
                f"D6 arm must complete all {configured_epochs} scheduled {stage} epochs for {task}"
            )
        rollout_units = max(
            1, int(base.get("stages", {}).get(stage, {}).get("rollout_steps", 1) or 1)
        )
        source_examples += sample_count * observed_epochs
        scheduled_compute_units += sample_count * observed_epochs * rollout_units
    if source_examples <= 0 or scheduled_compute_units <= 0:
        raise ValueError(f"D6 derived empty update-parity exposure for {task}")
    return {
        "source_examples": source_examples,
        "scheduled_compute_units": scheduled_compute_units,
    }


def _derive_arm_optimizer_schedule(
    *, record: dict[str, Any], base: dict[str, Any], runtime: Any, tasks: tuple[str, ...]
) -> dict[str, Any]:
    sample_counts: dict[str, int] = {}
    for task in tasks:
        task_runtime = getattr(runtime, "tasks", {}).get(task)
        sample_count = getattr(getattr(task_runtime, "train", None), "sample_count", None)
        if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count <= 0:
            raise ValueError(f"D6 sealed runtime lacks a positive training sample count for {task}")
        sample_counts[task] = sample_count
    combined_samples = sum(sample_counts.values())
    completed = record.get("resources", {}).get("training_log", {}).get("stage_epochs")
    if not isinstance(completed, dict):
        raise ValueError("D6 arm lacks completed stage-epoch evidence for optimizer accounting")
    accum_steps = max(1, int(base.get("training", {}).get("accum_steps", 1)))
    stage_records: dict[str, Any] = {}
    total_updates = 0
    for stage in STAGES:
        configured_epochs = int(base.get("stages", {}).get(stage, {}).get("epochs", 0) or 0)
        if configured_epochs <= 0:
            continue
        observed_epochs = completed.get(stage)
        if isinstance(observed_epochs, bool) or not isinstance(observed_epochs, int):
            raise ValueError(f"D6 arm training log lacks completed {stage} epochs")
        if observed_epochs != configured_epochs:
            raise ValueError(
                f"D6 arm must complete all {configured_epochs} scheduled {stage} epochs"
            )
        batch_size = int(
            base.get("stages", {})
            .get(stage, {})
            .get("batch_size", base.get("training", {}).get("batch_size", 1))
        )
        if batch_size <= 0:
            raise ValueError(f"D6 stage {stage} has a nonpositive batch size")
        batches_per_epoch = math.ceil(combined_samples / batch_size)
        stage_accum = accum_steps if stage == "operator" else 1
        updates_per_epoch = math.ceil(batches_per_epoch / stage_accum)
        scheduled_updates = updates_per_epoch * observed_epochs
        total_updates += scheduled_updates
        stage_records[stage] = {
            "combined_source_samples": combined_samples,
            "batch_size": batch_size,
            "accumulation_steps": stage_accum,
            "completed_epochs": observed_epochs,
            "batches_per_epoch": batches_per_epoch,
            "optimizer_updates_per_epoch": updates_per_epoch,
            "scheduled_optimizer_updates": scheduled_updates,
        }
    if total_updates <= 0:
        raise ValueError("D6 derived an empty arm optimizer-update schedule")
    return {
        "scope": "whole_arm_combined_task_loader",
        "sample_counts_by_task": sample_counts,
        "combined_source_samples": combined_samples,
        "stages": stage_records,
        "total_scheduled_optimizer_updates": total_updates,
    }


def _ensure_update_exposure(
    *, record: dict[str, Any], base: dict[str, Any], runtime: Any, tasks: tuple[str, ...]
) -> None:
    resources = record.setdefault("resources", {})
    reported = resources.setdefault("runner_reported", {})
    by_task = reported.get("update_parity_by_task")
    if by_task is not None:
        if not isinstance(by_task, dict):
            raise ValueError("D6 runner-reported update_parity_by_task must be a mapping")
        # Normalize explicit evidence to the two legitimate per-task dimensions;
        # optimizer updates only have a truthful whole-arm meaning for joint batches.
        reported["update_parity_by_task"] = {
            task: {
                dimension: by_task.get(task, {}).get(dimension)
                for dimension in ("source_examples", "scheduled_compute_units")
            }
            for task in tasks
        }
    else:
        reported["update_parity_by_task"] = {
            task: _derive_scheduled_exposure(record=record, base=base, runtime=runtime, task=task)
            for task in tasks
        }
        resources["update_parity_derivation"] = {
            "method": "sealed_task_sample_count_x_observed_completed_stage_schedule",
            "sample_count_source": "verified_strat_v1_training_runtime",
            "epoch_count_source": "arm_training_log",
            "batch_and_rollout_source": "hash_bound_d6_config",
        }
    resources["optimizer_update_schedule"] = _derive_arm_optimizer_schedule(
        record=record, base=base, runtime=runtime, tasks=tasks
    )


def _reported_update_exposure(record: dict[str, Any], task: str) -> dict[str, float]:
    reported = record.get("resources", {}).get("runner_reported", {})
    by_task = reported.get("update_parity_by_task")
    if not isinstance(by_task, dict) or not isinstance(by_task.get(task), dict):
        raise ValueError(
            f"D6 arm {record.get('tasks')} lacks exact update_parity_by_task evidence for {task}"
        )
    exposure: dict[str, float] = {}
    for dimension in ("source_examples", "scheduled_compute_units"):
        raw = by_task[task].get(dimension)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"D6 update parity {task} {dimension} is missing or nonnumeric")
        value = float(raw)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"D6 update parity {task} {dimension} must be finite and nonnegative")
        exposure[dimension] = value
    return exposure


def _update_parity_evidence(arms: dict[str, Any]) -> dict[str, Any]:
    joint = {task: _reported_update_exposure(arms["joint-modular"], task) for task in TASKS}
    ablations = {task: _reported_update_exposure(arms[f"ablation-{task}"], task) for task in TASKS}
    for task in TASKS:
        for dimension in ("source_examples", "scheduled_compute_units"):
            if joint[task][dimension] != ablations[task][dimension]:
                raise ValueError(f"D6 update parity mismatch for {task} {dimension}")
    total_updates: dict[str, int] = {}
    for arm in ARM_NAMES:
        raw = (
            arms.get(arm, {})
            .get("resources", {})
            .get("optimizer_update_schedule", {})
            .get("total_scheduled_optimizer_updates")
        )
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
            raise ValueError(f"D6 arm {arm} lacks a positive optimizer-update schedule")
        total_updates[arm] = raw
    return {
        "comparison": "joint_task_to_matching_ablation",
        "joint_by_task": joint,
        "ablation_by_task": ablations,
        "total_scheduled_optimizer_updates_by_arm": total_updates,
    }


def _validate_arm_summary(path: Path, tasks: tuple[str, ...]) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    extra = summary.get("extra", {})
    if extra.get("split") not in VALIDATION_SPLITS:
        raise PermissionError(f"D6 arm is not validation-only: {path}")
    if extra.get("strict_stratified_metrics") is not True:
        raise ValueError(f"D6 arm omitted strict stratified metrics: {path}")
    reported_tasks = extra.get("task")
    if isinstance(reported_tasks, str):
        reported_tasks = [reported_tasks]
    if reported_tasks is not None and tuple(reported_tasks) != tasks:
        raise ValueError(f"D6 arm summary task mismatch at {path}")
    metrics = summary.get("metrics", {})
    required = {"macro_primary_nrmse"}
    for task in tasks:
        suffix = "decoded_solution_nrmse" if task == "darcy2d" else "decoded_rollout_nrmse"
        required.add(f"task_{task}_{suffix}")
        required.add(f"task_{task}_maximum_corrected_regime_spread_ratio")
    missing = sorted(required - set(metrics))
    if missing:
        raise ValueError(f"D6 arm summary lacks strict metrics: {missing}")
    return summary


def _run_arm_command(command: list[str]) -> tuple[float, int]:
    started = time.time()
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    # ru_maxrss is the cumulative child process-family high-water mark. It cannot
    # be differenced or truthfully attributed to one arm.
    child_max_rss = int(after.ru_maxrss)
    return time.time() - started, child_max_rss


def run(args: argparse.Namespace) -> Path:
    output = Path(args.output_dir)
    if output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite D6 output: {output}")
    plan = _checked_plan(Path(args.plan_path))
    if plan["plan_sha256"] != args.plan_sha256:
        raise ValueError("D6 command plan hash differs from plan file")
    if plan.get("command_sha256") != canonical_sha256(plan.get("command")):
        raise ValueError("D6 plan command hash does not match")

    runtime = load_strat_v1_baseline_runtime(
        args.training_lock,
        args.data_root,
        expected_lock_sha256=FROZEN_STRAT_V1_TRAINING_LOCK_SHA256,
    )
    if set(runtime.lock.requested_roles) != {"train", "valid"} or any(
        item.role == "test" for item in runtime.lock.objects
    ):
        raise PermissionError("D6 refuses any training lock with held-out objects")
    _verify_plan_bindings(plan, args, runtime)
    base = load_config_with_includes(args.config)
    _validate_base_config(base)

    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema_version": 1,
        "plan_sha256": plan["plan_sha256"],
        "training_lock_sha256": runtime.lock.lock_sha256,
        "config_sha256": _sha256(Path(args.config)),
        "arms": list(ARM_NAMES),
        "heldout_reads": 0,
    }
    identity["identity_sha256"] = canonical_sha256(identity)
    identity_path = output / "run_identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("D6 resume identity differs from existing output")
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
        resumed = summary_path.exists()
        wall_time_sec = 0.0
        child_max_rss_kib = 0
        if not resumed:
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
            wall_time_sec, child_max_rss_kib = _run_arm_command(command)
        summary = _validate_arm_summary(summary_path, tasks)
        run_dir = arms_root / arm
        arm_record = {
            "tasks": list(tasks),
            "adapter_inventory": list(TASKS),
            "adapter_bottleneck_dim": 16,
            "summary_path": str(summary_path),
            "summary_file_sha256": _sha256(summary_path),
            "resolved_train_config_sha256": _sha256(train_path),
            "resolved_eval_config_sha256": _sha256(eval_path),
            "metrics": summary["metrics"],
            "checkpoints": _checkpoint_evidence(run_dir / "checkpoints"),
            "resources": _resource_evidence(
                run_dir=run_dir,
                summary=summary,
                wall_time_sec=wall_time_sec,
                child_max_rss_kib=child_max_rss_kib,
            ),
            "resumed": resumed,
        }
        _ensure_update_exposure(record=arm_record, base=base, runtime=runtime, tasks=tasks)
        arm_records[arm] = arm_record

    if tuple(arm_records) != ARM_NAMES:
        raise ValueError("D6 executed arm set differs from the frozen four-arm design")
    update_parity = _update_parity_evidence(arm_records)

    shuffled_name = "joint-modular-parameter-shuffled"
    shuffled_cfg = _arm_config(base, runtime, TASKS, "val")
    shuffled_cfg.setdefault("evaluation", {})["conditioning_parameter_index_shift"] = 1
    shuffled_path = config_root / f"{shuffled_name}.eval.yaml"
    shuffled_path.write_text(yaml.safe_dump(shuffled_cfg, sort_keys=False), encoding="utf-8")
    shuffled_summary_path = arms_root / shuffled_name / "summary.json"
    if not shuffled_summary_path.exists():
        _run_arm_command(
            [
                sys.executable,
                "scripts/run_light_experiment.py",
                "--config",
                str(config_root / "joint-modular.train.yaml"),
                "--eval-config",
                str(shuffled_path),
                "--name",
                shuffled_name,
                "--output-root",
                str(arms_root),
                "--skip-training",
                "--checkpoint-source",
                str(arms_root / "joint-modular" / "checkpoints"),
                "--decoded",
                "--decoded-rollout-steps",
                "16",
                "--device",
                args.device,
            ]
        )
    shuffled_summary = _validate_arm_summary(shuffled_summary_path, TASKS)
    reference = float(arm_records["joint-modular"]["metrics"]["macro_primary_nrmse"])
    shuffled = float(shuffled_summary["metrics"]["macro_primary_nrmse"])
    conditioning_diagnostics = {
        "method": "deterministic_within_task_cyclic_parameter_shift_by_one_sample",
        "summary_path": str(shuffled_summary_path),
        "summary_file_sha256": _sha256(shuffled_summary_path),
        "reference_macro_primary_nrmse": reference,
        "shuffled_macro_primary_nrmse": shuffled,
        "relative_nrmse_degradation": (
            shuffled / reference - 1.0 if reference > 0.0 else float("inf")
        ),
    }

    result = {
        "schema_version": 1,
        "artifact_id": "strat-v1-modular-shared-trunk-d6-summary",
        "status": "complete_validation_only",
        "plan_sha256": plan["plan_sha256"],
        "training_lock_sha256": runtime.lock.lock_sha256,
        "config_sha256": _sha256(Path(args.config)),
        "heldout_reads": 0,
        "heldout_evidence": {
            "requested_roles": sorted(runtime.lock.requested_roles),
            "contains_test_object": False,
            "evaluation_splits": ["val"],
        },
        "arms": arm_records,
        "update_parity": update_parity,
        "conditioning_diagnostics": conditioning_diagnostics,
        "duration_sec": time.time() - started,
    }
    result["artifact_sha256"] = canonical_sha256(result)
    summary_path = output / "summary.json"
    # Preserve the preregistered arm order for the independent materializer.
    summary_path.write_text(json.dumps(result, indent=2) + "\n")
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-lock", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--config", default="configs/d6_strat_v1_modular_shared_trunk.yaml")
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
