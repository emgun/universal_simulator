#!/usr/bin/env python
from __future__ import annotations

"""Independently materialize the frozen D6 U1 and U2 validation gates."""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ups.data.manifests import canonical_sha256  # noqa: E402

TASKS = ("advection1d", "burgers1d", "darcy2d")
ARMS = (
    "joint-modular",
    "ablation-advection1d",
    "ablation-burgers1d",
    "ablation-darcy2d",
)


def _checked_self_hash(payload: dict[str, Any], key: str) -> None:
    expected = payload.get(key)
    unsigned = {name: value for name, value in payload.items() if name != key}
    if expected != canonical_sha256(unsigned):
        raise ValueError(f"{key} does not match payload")


def _finite(value: Any, *, name: str, signed: bool = False) -> float:
    number = float(value)
    if not math.isfinite(number) or (not signed and number < 0.0):
        raise ValueError(f"{name} must be finite" + ("" if signed else " and nonnegative"))
    return number


def _task_metric(metrics: dict[str, Any], task: str) -> float:
    suffix = "decoded_solution_nrmse" if task == "darcy2d" else "decoded_rollout_nrmse"
    return _finite(metrics[f"task_{task}_{suffix}"], name=f"{task} primary")


def _check_arm_contract(plan: dict[str, Any], arms: dict[str, Any]) -> None:
    if tuple(arms) != ARMS:
        raise ValueError("D6 summary arm inventory or order differs from the frozen design")
    expected_tasks = {
        "joint-modular": TASKS,
        **{f"ablation-{task}": (task,) for task in TASKS},
    }
    expected_inventory = tuple(TASKS)
    for arm in ARMS:
        evidence = arms[arm]
        if tuple(evidence.get("tasks", ())) != expected_tasks[arm]:
            raise ValueError(f"D6 task assignment differs for {arm}")
        if tuple(evidence.get("adapter_inventory", ())) != expected_inventory:
            raise ValueError(f"D6 arm {arm} lacks the full three-task adapter inventory")
        if evidence.get("adapter_bottleneck_dim") != 16:
            raise ValueError(f"D6 arm {arm} adapter bottleneck differs from 16")
    design = plan.get("design", {})
    if tuple(design.get("arms", ())) != ARMS:
        raise ValueError("D6 plan arm inventory differs from the materializer contract")
    if any(
        tuple(value) != expected_inventory for value in design["adapter_inventory_by_arm"].values()
    ):
        raise ValueError("D6 plan lacks the full adapter inventory in every arm")


def _checked_update_parity(summary: dict[str, Any]) -> dict[str, Any]:
    parity = summary.get("update_parity")
    if not isinstance(parity, dict):
        raise ValueError("D6 summary lacks update parity evidence")
    if parity.get("comparison") != "joint_task_to_matching_ablation":
        raise ValueError("D6 update parity comparison differs from the frozen contract")
    joint = parity.get("joint_by_task")
    ablations = parity.get("ablation_by_task")
    if not isinstance(joint, dict) or not isinstance(ablations, dict):
        raise ValueError("D6 update parity evidence must contain both sides")
    if tuple(joint) != TASKS or tuple(ablations) != TASKS:
        raise ValueError("D6 update parity task inventory is incomplete")
    dimensions = ("source_examples", "scheduled_compute_units")
    checked: dict[str, dict[str, float]] = {}
    for task in TASKS:
        checked[task] = {}
        if "optimizer_updates" in joint[task] or "optimizer_updates" in ablations[task]:
            raise ValueError("D6 optimizer updates must be reported separately from parity")
        for dimension in dimensions:
            left = _finite(joint[task][dimension], name=f"joint {task} {dimension}")
            right = _finite(ablations[task][dimension], name=f"ablation {task} {dimension}")
            if left != right:
                raise ValueError(f"D6 update parity mismatch for {task} {dimension}")
            checked[task][dimension] = left
    updates = parity.get("total_scheduled_optimizer_updates_by_arm")
    if not isinstance(updates, dict) or tuple(updates) != ARMS:
        raise ValueError("D6 lacks total scheduled optimizer-update efficiency evidence")
    checked_updates = {
        arm: _finite(updates[arm], name=f"{arm} total scheduled optimizer updates") for arm in ARMS
    }
    if any(value <= 0.0 for value in checked_updates.values()):
        raise ValueError("D6 total scheduled optimizer updates must be positive")
    ablation_total = sum(checked_updates[f"ablation-{task}"] for task in TASKS)
    return {
        "comparison": parity["comparison"],
        "matched_by_task": checked,
        "optimizer_update_efficiency": {
            "total_scheduled_by_arm": checked_updates,
            "joint_total": checked_updates["joint-modular"],
            "ablation_ensemble_total": ablation_total,
            "joint_to_ablation_ensemble_ratio": checked_updates["joint-modular"] / ablation_total,
        },
    }


def _checked_resource_evidence(summary: dict[str, Any], arms: dict[str, Any]) -> dict[str, Any]:
    duration = _finite(summary.get("duration_sec"), name="D6 run duration")
    if duration <= 0.0:
        raise ValueError("D6 run duration must be positive")
    checked_arms: dict[str, Any] = {}
    for arm in ARMS:
        checkpoints = arms[arm].get("checkpoints", {})
        checkpoint_bytes = int(checkpoints.get("total_checkpoint_bytes", 0))
        initialized = int(checkpoints.get("total_initialized_tensor_elements", 0))
        adapter = int(checkpoints.get("total_adapter_tensor_elements", 0))
        if checkpoint_bytes <= 0 or initialized <= 0 or adapter <= 0:
            raise ValueError(f"D6 arm {arm} lacks positive checkpoint tensor and adapter evidence")
        training_log = arms[arm].get("resources", {}).get("training_log", {})
        records = int(training_log.get("records", 0))
        reported_time = _finite(
            training_log.get("reported_epoch_time_sec"), name=f"{arm} training-log duration"
        )
        if training_log.get("present") is not True or records <= 0 or reported_time <= 0.0:
            raise ValueError(f"D6 arm {arm} lacks positive training-log evidence")
        attempt = arms[arm].get("attempt_evidence", {})
        _checked_self_hash(attempt, "artifact_sha256")
        if attempt.get("summary_file_sha256") != arms[arm].get("summary_file_sha256"):
            raise ValueError(f"D6 arm {arm} attempt evidence summary binding differs")
        resource_record = arms[arm].get("resources", {})
        if attempt.get("wall_time_sec_observed_by_orchestrator") != resource_record.get(
            "wall_time_sec_observed_by_orchestrator"
        ) or attempt.get("child_process_family_max_rss_kib_high_watermark") != resource_record.get(
            "child_process_family_max_rss_kib_high_watermark"
        ):
            raise ValueError(f"D6 arm {arm} attempt resource evidence differs from the summary")
        checked_arms[arm] = {
            "checkpoint_bytes": checkpoint_bytes,
            "initialized_tensor_elements": initialized,
            "adapter_tensor_elements": adapter,
            "training_log_records": records,
            "training_log_reported_epoch_time_sec": reported_time,
            "runner_wall_time_sec": _finite(
                arms[arm].get("resources", {}).get("wall_time_sec_observed_by_orchestrator"),
                name=f"{arm} runner wall time",
            ),
            "child_process_family_max_rss_kib_high_watermark": int(
                arms[arm]
                .get("resources", {})
                .get("child_process_family_max_rss_kib_high_watermark", 0)
            ),
        }
        if (
            checked_arms[arm]["runner_wall_time_sec"] <= 0.0
            or checked_arms[arm]["child_process_family_max_rss_kib_high_watermark"] <= 0
        ):
            raise ValueError(f"D6 arm {arm} lacks positive runner wall-time or RSS evidence")
    return {"run_duration_sec": duration, "arms": checked_arms}


def _checked_stage_report(
    plan: dict[str, Any], summary: dict[str, Any], stage_report: dict[str, Any]
) -> str:
    _checked_self_hash(stage_report, "artifact_sha256")
    if stage_report.get("status") != "complete":
        raise ValueError("D6 stage report is not complete")
    binding = plan.get("bindings", {}).get("training_lock", {})
    if stage_report.get("lock_sha256") != binding.get("lock_sha256"):
        raise ValueError("D6 stage report training lock differs from the plan")
    expected = binding.get("objects", {})
    observed = {
        str(item.get("id")): item.get("checksum", {}).get("value")
        for item in stage_report.get("objects", [])
    }
    if observed != expected or int(stage_report.get("object_count", -1)) != len(expected):
        raise ValueError("D6 stage report objects differ from the plan")
    if any(str(item.get("role")) == "test" for item in stage_report.get("objects", [])):
        raise PermissionError("D6 stage report contains held-out data")
    digest = str(stage_report["artifact_sha256"])
    if summary.get("stage_report_artifact_sha256") != digest:
        raise ValueError("D6 summary is not bound to the supplied stage report")
    return digest


def build_result(
    plan: dict[str, Any], summary: dict[str, Any], stage_report: dict[str, Any]
) -> dict[str, Any]:
    _checked_self_hash(plan, "plan_sha256")
    _checked_self_hash(summary, "artifact_sha256")
    if summary.get("plan_sha256") != plan["plan_sha256"]:
        raise ValueError("D6 summary and plan hashes differ")
    if summary.get("status") != "complete_validation_only" or summary.get("heldout_reads") != 0:
        raise PermissionError("D6 summary is not complete validation-only evidence")
    bindings = plan.get("bindings", {})
    if summary.get("training_lock_sha256") != bindings.get("training_lock", {}).get("lock_sha256"):
        raise ValueError("D6 summary training lock differs from the plan binding")
    if summary.get("config_sha256") != bindings.get("config", {}).get("file_sha256"):
        raise ValueError("D6 summary config differs from the plan binding")
    stage_report_sha256 = _checked_stage_report(plan, summary, stage_report)
    heldout = summary.get("heldout_evidence")
    if heldout != {
        "requested_roles": ["train", "valid"],
        "contains_test_object": False,
        "evaluation_splits": ["val"],
    }:
        raise PermissionError("D6 summary held-out evidence differs from validation-only contract")
    arms = summary.get("arms", {})
    _check_arm_contract(plan, arms)
    parity = _checked_update_parity(summary)
    resources = _checked_resource_evidence(summary, arms)

    joint_metrics = arms["joint-modular"]["metrics"]
    joint_by_task = {task: _task_metric(joint_metrics, task) for task in TASKS}
    joint_macro = _finite(joint_metrics["macro_primary_nrmse"], name="joint macro")
    if not math.isclose(
        joint_macro, sum(joint_by_task.values()) / 3.0, rel_tol=1e-9, abs_tol=1e-12
    ):
        raise ValueError("D6 joint macro does not match the equal-task primary mean")
    ablation_by_task = {
        task: _task_metric(arms[f"ablation-{task}"]["metrics"], task) for task in TASKS
    }
    ablation_macro = sum(ablation_by_task.values()) / 3.0
    frozen = plan["frozen_references"]
    frozen_by_task = {task: float(frozen["d5_specialist_by_task"][task]) for task in TASKS}
    frozen_macro = float(frozen["d5_specialist_macro_primary_nrmse"])
    joint_to_frozen = {task: joint_by_task[task] / frozen_by_task[task] for task in TASKS}
    joint_to_ablation = {task: joint_by_task[task] / ablation_by_task[task] for task in TASKS}
    spread_by_task = {
        task: _finite(
            joint_metrics[f"task_{task}_maximum_corrected_regime_spread_ratio"],
            name=f"{task} spread",
        )
        for task in TASKS
    }
    diagnostics = summary.get("conditioning_diagnostics", {})
    diagnostic_reference = _finite(
        diagnostics.get("reference_macro_primary_nrmse"), name="shuffle reference macro"
    )
    diagnostic_shuffled = _finite(
        diagnostics.get("shuffled_macro_primary_nrmse"), name="shuffled parameter macro"
    )
    if not math.isclose(diagnostic_reference, joint_macro, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("D6 shuffle reference differs from the joint metric")
    if diagnostic_reference <= 0.0:
        raise ValueError("D6 shuffle reference must be positive")
    degradation = diagnostic_shuffled / diagnostic_reference - 1.0
    supplied_degradation = _finite(
        diagnostics.get("relative_nrmse_degradation"),
        name="reported shuffled parameter degradation",
        signed=True,
    )
    if not math.isclose(degradation, supplied_degradation, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("D6 reported shuffled degradation is internally inconsistent")
    joint_bytes = int(arms["joint-modular"]["checkpoints"]["total_checkpoint_bytes"])
    joint_tensor_elements = resources["arms"]["joint-modular"]["initialized_tensor_elements"]
    ablation_tensor_elements = sum(
        resources["arms"][f"ablation-{task}"]["initialized_tensor_elements"] for task in TASKS
    )
    u1_gates = plan["gates"]["u1"]
    u2_gates = plan["gates"]["u2"]
    if u1_gates.get("joint_checkpoint_bytes_less_than_frozen_d5_ensemble") is not True:
        raise ValueError("D6 plan lacks the frozen checkpoint-consolidation gate")
    if (
        u1_gates.get("joint_initialized_tensor_elements_less_than_matched_ablation_ensemble")
        is not True
    ):
        raise ValueError("D6 plan lacks the initialized-tensor consolidation gate")
    u1_checks = {
        "macro_to_frozen_d5_specialists": joint_macro
        <= float(u1_gates["joint_macro_ratio_to_frozen_d5_specialist_maximum"]) * frozen_macro,
        "per_task_to_frozen_d5_specialists": max(joint_to_frozen.values())
        <= float(u1_gates["joint_per_task_ratio_to_frozen_d5_specialist_maximum"]),
        "beats_persistence_each_task": all(
            joint_by_task[task] < float(u1_gates["persistence_maximum_by_task"][task])
            for task in TASKS
        ),
        "darcy_absolute_wall": joint_by_task["darcy2d"] <= float(u1_gates["darcy_primary_maximum"]),
        "corrected_regime_spread": max(spread_by_task.values())
        <= float(u1_gates["maximum_corrected_regime_spread"]),
        "parameter_use": degradation
        >= float(u1_gates["shuffled_parameter_nrmse_degradation_minimum"]),
        "checkpoint_consolidation": joint_bytes
        < int(frozen["d5_specialist_ensemble_checkpoint_bytes"]),
        "initialized_tensor_consolidation": joint_tensor_elements < ablation_tensor_elements,
        "heldout_zero": summary["heldout_reads"] == int(u1_gates["heldout_reads"]),
    }
    u2_checks = {
        "macro_negative_transfer": joint_macro
        <= float(u2_gates["joint_macro_ratio_to_matched_ablation_macro_maximum"]) * ablation_macro,
        "per_task_negative_transfer": max(joint_to_ablation.values())
        <= float(u2_gates["joint_per_task_ratio_to_matched_ablation_maximum"]),
        "update_parity": True,
    }
    u1_passed = all(u1_checks.values())
    u2_passed = all(u2_checks.values())
    interpretation = (
        "u1_failed"
        if not u1_passed
        else "u2_negative_transfer" if not u2_passed else "modular_shared_trunk_validated"
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "artifact_id": "strat-v1-modular-shared-trunk-d6-result",
        "status": "complete_validation_only",
        "plan_sha256": plan["plan_sha256"],
        "source_summary_artifact_sha256": summary["artifact_sha256"],
        "source_stage_report_artifact_sha256": stage_report_sha256,
        "heldout_reads": 0,
        "metrics": {
            "joint_macro_primary_nrmse": joint_macro,
            "frozen_d5_specialist_macro_primary_nrmse": frozen_macro,
            "matched_ablation_macro_primary_nrmse": ablation_macro,
            "joint_by_task": joint_by_task,
            "frozen_d5_specialist_by_task": frozen_by_task,
            "matched_ablation_by_task": ablation_by_task,
            "joint_to_frozen_d5_ratio_by_task": joint_to_frozen,
            "joint_to_matched_ablation_ratio_by_task": joint_to_ablation,
            "joint_maximum_corrected_regime_spread_by_task": spread_by_task,
            "shuffled_parameter_relative_nrmse_degradation": degradation,
            "joint_checkpoint_bytes": joint_bytes,
            "joint_initialized_tensor_elements": joint_tensor_elements,
            "matched_ablation_ensemble_initialized_tensor_elements": ablation_tensor_elements,
        },
        "update_parity": parity,
        "resource_evidence": resources,
        "u1_checks": u1_checks,
        "u2_checks": u2_checks,
        "u1_passed": u1_passed,
        "u2_passed": u2_passed,
        "all_preregistered_gates_passed": u1_passed and u2_passed,
        "interpretation": interpretation,
    }
    result["artifact_sha256"] = canonical_sha256(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--stage-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_result(
        json.loads(args.plan.read_text(encoding="utf-8")),
        json.loads(args.summary.read_text(encoding="utf-8")),
        json.loads(args.stage_report.read_text(encoding="utf-8")),
    )
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite D6 result: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "u1_passed": result["u1_passed"],
                "u2_passed": result["u2_passed"],
            }
        )
    )


if __name__ == "__main__":
    main()
