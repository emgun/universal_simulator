#!/usr/bin/env python
from __future__ import annotations

"""Run a train/val-only transfer scorecard for the inferred transport gate."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.calibrate_residual_gate import _test_guard_result
from scripts.run_inferred_transport_shift_gate import (
    _candidate_shifts,
    _fit_linear_calibrator,
    _load_series,
    _sample_shift_pairs,
    _score_inferred_transport,
    run_gate,
)


def _split_path(data_root: str | Path, task: str, split: str) -> Path:
    return Path(data_root) / f"{task}_{split}.h5"


def _support_status(
    data_root: str | Path, task: str, train_split: str, val_split: str
) -> str | None:
    train_path = _split_path(data_root, task, train_split)
    val_path = _split_path(data_root, task, val_split)
    if not train_path.exists():
        return f"missing train split: {train_path}"
    if not val_path.exists():
        return f"missing validation split: {val_path}"
    if not task.endswith("1d"):
        return f"unsupported task for 1D transport gate: {task}"
    with h5py.File(train_path, "r") as handle:
        if "data" not in handle:
            return f"missing data dataset: {train_path}"
        shape = tuple(handle["data"].shape)
    if len(shape) != 4 or shape[-1] != 1:
        return f"unsupported data shape for 1D transport gate: {shape}"
    if len(shape) >= 3 and shape[1] <= shape[2]:
        return None
    return f"unsupported data shape for 1D transport gate: {shape}"


def _gate_args(args: argparse.Namespace, task: str, output_json: Path) -> argparse.Namespace:
    return argparse.Namespace(
        data_root=args.data_root,
        task=task,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split="",
        max_samples=args.max_samples,
        train_max_samples=None,
        val_max_samples=None,
        test_max_samples=None,
        context_transitions=args.context_transitions,
        rollout_steps=args.rollout_steps,
        shift=args.shift,
        metric=args.metric,
        fit_kind=args.fit_kind,
        fit_intercept=args.fit_intercept,
        refine_radius=args.refine_radius,
        fractional_refine_step=args.fractional_refine_step,
        reference_metric_value=args.reference_metric_value,
        val_min_relative_improvement=args.val_min_relative_improvement,
        test_ledger_json=None,
        allow_repeat_test=False,
        output_json=str(output_json),
    )


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _task_support(
    args: argparse.Namespace, output_dir: Path
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    tasks: dict[str, dict[str, Any]] = {}
    supported_tasks: list[str] = []
    for task in args.tasks:
        reason = _support_status(args.data_root, task, args.train_split, args.val_split)
        if reason:
            tasks[task] = {"status": "skipped", "reason": reason}
            continue
        supported_tasks.append(task)
        tasks[task] = {
            "status": "pending",
            "gate_json": str(output_dir / f"{task}_transfer_gate.json"),
        }
    return tasks, supported_tasks


def _run_shared_calibrator(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    tasks: dict[str, dict[str, Any]],
    supported_tasks: list[str],
) -> list[float]:
    shifts = _candidate_shifts(args.shift)
    inferred_groups: list[np.ndarray] = []
    target_groups: list[np.ndarray] = []
    train_pair_summaries: dict[str, dict[str, Any]] = {}
    for task in supported_tasks:
        train_fields = _load_series(
            root=args.data_root,
            task=task,
            split=args.train_split,
            max_samples=args.max_samples,
        )
        inferred, targets, records = _sample_shift_pairs(
            train_fields,
            shifts,
            context_transitions=args.context_transitions,
            rollout_steps=args.rollout_steps,
            metric=args.metric,
            refine_radius=args.refine_radius,
            fractional_refine_step=args.fractional_refine_step,
        )
        inferred_groups.append(inferred)
        target_groups.append(targets)
        train_pair_summaries[task] = {
            "count": len(records),
            "context_shift_mean": float(np.mean(inferred)),
            "context_shift_std": float(np.std(inferred)),
            "target_shift_mean": float(np.mean(targets)),
            "target_shift_std": float(np.std(targets)),
        }

    coefficients = _fit_linear_calibrator(
        np.concatenate(inferred_groups),
        np.concatenate(target_groups),
        fit_intercept=bool(args.fit_intercept),
    )
    shared_fit = {
        "model": "shared_linear_context_shift_calibrator",
        "calibration_scope": "shared_1d_transport",
        "fit_kind": args.fit_kind,
        "fit_intercept": bool(args.fit_intercept),
        "context_transitions": int(args.context_transitions),
        "refine_radius": int(args.refine_radius),
        "fractional_refine_step": float(args.fractional_refine_step),
        "candidate_shifts": shifts,
        "coefficients": coefficients,
        "train_task_count": len(supported_tasks),
        "train_tasks": supported_tasks,
        "train_shift_pair_summaries": train_pair_summaries,
    }
    shared_fit_path = output_dir / "shared_1d_transport_calibrator.json"
    shared_fit_path.write_text(json.dumps(shared_fit, indent=2, sort_keys=True), encoding="utf-8")

    validation_values: list[float] = []
    for task in supported_tasks:
        train_fields = _load_series(
            root=args.data_root,
            task=task,
            split=args.train_split,
            max_samples=args.max_samples,
        )
        val_fields = _load_series(
            root=args.data_root,
            task=task,
            split=args.val_split,
            max_samples=args.max_samples,
        )
        selected_train = _score_inferred_transport(
            train_fields,
            shifts,
            coefficients,
            context_transitions=args.context_transitions,
            rollout_steps=args.rollout_steps,
            metric=args.metric,
            refine_radius=args.refine_radius,
            fractional_refine_step=args.fractional_refine_step,
        )
        selected_validation = _score_inferred_transport(
            val_fields,
            shifts,
            coefficients,
            context_transitions=args.context_transitions,
            rollout_steps=args.rollout_steps,
            metric=args.metric,
            refine_radius=args.refine_radius,
            fractional_refine_step=args.fractional_refine_step,
        )
        validation_nrmse = float(selected_validation["nrmse"])
        validation_values.append(validation_nrmse)
        validation_guard = _test_guard_result(
            value=float(selected_validation[args.metric]),
            reference=args.reference_metric_value,
            min_relative_improvement=args.val_min_relative_improvement,
            mode="min",
        )
        task_record = {
            "task": task,
            "calibration_scope": "shared_1d_transport",
            "train_split": args.train_split,
            "val_split": args.val_split,
            "metric": args.metric,
            "fit": {
                "shared_calibrator_json": str(shared_fit_path),
                "coefficients": coefficients,
                "train": selected_train,
                "selected_validation": selected_validation,
                "validation_guard": validation_guard,
            },
            "validation_guard": validation_guard,
            "test": None,
            "test_eligible": False,
            "held_out_test_policy": (
                "No held-out test split is passed by the shared transfer scorecard."
            ),
        }
        output_json = output_dir / f"{task}_transfer_gate.json"
        output_json.write_text(json.dumps(task_record, indent=2, sort_keys=True), encoding="utf-8")
        tasks[task] = {
            "status": "validated",
            "calibration_scope": "shared_1d_transport",
            "gate_json": str(output_json),
            "train_nrmse": float(selected_train["nrmse"]),
            "validation_nrmse": validation_nrmse,
            "validation_guard_passed": bool(validation_guard["passed"]),
            "test_touched": False,
        }
    return validation_values


def run_scorecard(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks, supported_tasks = _task_support(args, output_dir)
    validation_values: list[float] = []
    shared_calibrator = bool(getattr(args, "shared_calibrator", False))
    if supported_tasks and shared_calibrator:
        validation_values = _run_shared_calibrator(
            args,
            output_dir=output_dir,
            tasks=tasks,
            supported_tasks=supported_tasks,
        )
    else:
        for task in supported_tasks:
            output_json = output_dir / f"{task}_transfer_gate.json"
            record = run_gate(_gate_args(args, task, output_json))
            output_json.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
            validation_nrmse = float(record["fit"]["selected_validation"]["nrmse"])
            validation_values.append(validation_nrmse)
            tasks[task] = {
                "status": "validated",
                "calibration_scope": "per_task_1d_transport",
                "gate_json": str(output_json),
                "train_nrmse": float(record["fit"]["train"]["nrmse"]),
                "validation_nrmse": validation_nrmse,
                "validation_guard_passed": bool(record["validation_guard"]["passed"]),
                "test_touched": record.get("test") is not None,
            }

    evaluated_task_count = sum(1 for payload in tasks.values() if payload["status"] == "validated")
    skipped_task_count = sum(1 for payload in tasks.values() if payload["status"] == "skipped")
    if evaluated_task_count == 0:
        status = "blocked_no_supported_transfer_tasks"
    elif skipped_task_count:
        status = "partial_transfer_validated"
    else:
        status = "transfer_validated"
    scorecard = {
        "status": status,
        "data_root": str(args.data_root),
        "train_split": args.train_split,
        "val_split": args.val_split,
        "metric": args.metric,
        "context_transitions": int(args.context_transitions),
        "rollout_steps": int(args.rollout_steps),
        "refine_radius": int(args.refine_radius),
        "fractional_refine_step": float(args.fractional_refine_step),
        "calibration_scope": (
            "shared_1d_transport" if shared_calibrator else "per_task_1d_transport"
        ),
        "evaluated_task_count": evaluated_task_count,
        "skipped_task_count": skipped_task_count,
        "mean_validation_nrmse": _mean(validation_values),
        "tasks": tasks,
        "held_out_policy": "train/val only; no held-out test split is passed to task gates",
    }
    shared_fit_path = output_dir / "shared_1d_transport_calibrator.json"
    if shared_calibrator and shared_fit_path.exists():
        scorecard["shared_fit"] = json.loads(shared_fit_path.read_text(encoding="utf-8"))
    (output_dir / "scorecard.json").write_text(
        json.dumps(scorecard, indent=2, sort_keys=True), encoding="utf-8"
    )
    return scorecard


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inferred transport transfer scorecard")
    parser.add_argument("--data-root", default="data/pdebench")
    parser.add_argument("--task", action="append", dest="tasks", default=None)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--context-transitions", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=16)
    parser.add_argument("--shift", action="append", type=int, default=None)
    parser.add_argument("--metric", choices=("mse", "nrmse"), default="nrmse")
    parser.add_argument("--fit-kind", choices=("linear",), default="linear")
    parser.add_argument("--fit-intercept", action="store_true")
    parser.add_argument("--refine-radius", type=int, default=4)
    parser.add_argument("--fractional-refine-step", type=float, default=0.025)
    parser.add_argument("--reference-metric-value", type=float, default=1.0)
    parser.add_argument("--val-min-relative-improvement", type=float, default=0.0)
    parser.add_argument(
        "--shared-calibrator",
        action="store_true",
        help="Fit one shared linear context-shift calibrator across all supported 1D tasks.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/research/sota_loop/inferred_transfer_scorecard",
    )
    args = parser.parse_args()
    if args.tasks is None:
        args.tasks = ["advection1d", "burgers1d", "darcy2d"]

    record = run_scorecard(args)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
