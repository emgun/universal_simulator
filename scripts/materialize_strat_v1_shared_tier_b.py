#!/usr/bin/env python
from __future__ import annotations

"""Independently materialize the frozen D5 shared tier-b gates."""

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


def _checked_self_hash(payload: dict[str, Any], key: str) -> None:
    expected = payload.get(key)
    unsigned = {name: value for name, value in payload.items() if name != key}
    if expected != canonical_sha256(unsigned):
        raise ValueError(f"{key} does not match payload")


def _finite(value: Any, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return number


def _finite_signed(value: Any, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _task_metric(metrics: dict[str, Any], task: str) -> float:
    suffix = "decoded_solution_nrmse" if task == "darcy2d" else "decoded_rollout_nrmse"
    return _finite(metrics[f"task_{task}_{suffix}"], name=f"{task} primary")


def build_result(plan: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    _checked_self_hash(plan, "plan_sha256")
    _checked_self_hash(summary, "artifact_sha256")
    if summary.get("plan_sha256") != plan["plan_sha256"]:
        raise ValueError("D5 summary and plan hashes differ")
    if summary.get("status") != "complete_validation_only" or summary.get("heldout_reads") != 0:
        raise PermissionError("D5 summary is not complete validation-only evidence")
    arms = summary.get("arms", {})
    expected_arms = {
        "shared",
        "specialist-advection1d",
        "specialist-burgers1d",
        "specialist-darcy2d",
    }
    if set(arms) != expected_arms:
        raise ValueError("D5 summary arm set differs from the frozen design")
    expected_tasks_by_arm = {
        "shared": TASKS,
        **{f"specialist-{task}": (task,) for task in TASKS},
    }
    for arm, expected_tasks in expected_tasks_by_arm.items():
        if tuple(arms[arm].get("tasks", ())) != expected_tasks:
            raise ValueError(f"D5 summary task assignment differs for arm {arm}")

    shared_metrics = arms["shared"]["metrics"]
    shared_by_task = {task: _task_metric(shared_metrics, task) for task in TASKS}
    specialist_by_task = {
        task: _task_metric(arms[f"specialist-{task}"]["metrics"], task) for task in TASKS
    }
    shared_macro = _finite(shared_metrics["macro_primary_nrmse"], name="shared macro")
    reconstructed_shared_macro = sum(shared_by_task.values()) / len(TASKS)
    if not math.isclose(shared_macro, reconstructed_shared_macro, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("D5 shared macro does not match the equal-task primary mean")
    specialist_oracle_macro = sum(specialist_by_task.values()) / len(TASKS)
    per_task_ratios = {
        task: (
            shared_by_task[task] / specialist_by_task[task]
            if specialist_by_task[task] > 0.0
            else math.inf
        )
        for task in TASKS
    }
    spread_by_task = {
        task: _finite(
            shared_metrics[f"task_{task}_maximum_corrected_regime_spread_ratio"],
            name=f"{task} spread",
        )
        for task in TASKS
    }
    shared_bytes = int(arms["shared"]["checkpoints"]["total_checkpoint_bytes"])
    specialist_bytes = sum(
        int(arms[f"specialist-{task}"]["checkpoints"]["total_checkpoint_bytes"]) for task in TASKS
    )
    diagnostics = summary.get("conditioning_diagnostics", {})
    shuffled_degradation = _finite_signed(
        diagnostics.get("relative_nrmse_degradation"), name="shuffled parameter degradation"
    )
    gates = plan["gates"]
    checks = {
        "shared_macro_interference": shared_macro
        <= float(gates["shared_macro_ratio_to_specialist_oracle_maximum"])
        * specialist_oracle_macro,
        "shared_per_task_interference": max(per_task_ratios.values())
        <= float(gates["shared_per_task_ratio_to_specialist_maximum"]),
        "shared_beats_persistence_each_task": all(
            shared_by_task[task] < float(gates["persistence_maximum_by_task"][task])
            for task in TASKS
        ),
        "shared_darcy_specialist_wall": shared_by_task["darcy2d"]
        <= float(gates["darcy_primary_maximum"]),
        "shared_regime_spread": max(spread_by_task.values())
        <= float(gates["maximum_corrected_regime_spread"]),
        "shared_parameter_use": shuffled_degradation
        >= float(gates["shuffled_parameter_nrmse_degradation_minimum"]),
        "shared_consolidates_checkpoint_bytes": shared_bytes < specialist_bytes,
        "heldout_zero": summary["heldout_reads"] == int(gates["heldout_reads"]),
    }
    result = {
        "schema_version": 1,
        "artifact_id": "strat-v1-shared-tier-b-d5-result",
        "status": "complete_validation_only",
        "plan_sha256": plan["plan_sha256"],
        "source_summary_artifact_sha256": summary["artifact_sha256"],
        "heldout_reads": 0,
        "metrics": {
            "shared_macro_primary_nrmse": shared_macro,
            "specialist_oracle_macro_primary_nrmse": specialist_oracle_macro,
            "shared_macro_ratio_to_specialist_oracle": (
                shared_macro / specialist_oracle_macro
                if specialist_oracle_macro > 0.0
                else math.inf
            ),
            "shared_by_task": shared_by_task,
            "specialist_by_task": specialist_by_task,
            "shared_to_specialist_ratio_by_task": per_task_ratios,
            "shared_maximum_corrected_regime_spread_by_task": spread_by_task,
            "shuffled_parameter_relative_nrmse_degradation": shuffled_degradation,
            "shared_checkpoint_bytes": shared_bytes,
            "specialist_ensemble_checkpoint_bytes": specialist_bytes,
        },
        "gate_checks": checks,
        "all_gates_passed": all(checks.values()),
        "interpretation": (
            "shared_tier_b_validated" if all(checks.values()) else "shared_tier_b_not_validated"
        ),
    }
    result["artifact_sha256"] = canonical_sha256(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    result = build_result(plan, summary)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite D5 result: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "all_gates_passed": result["all_gates_passed"]}))


if __name__ == "__main__":
    main()
