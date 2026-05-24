#!/usr/bin/env python
from __future__ import annotations

"""Gate a context-inferred transport-shift estimator.

This removes explicit beta/source identity conditioning from the official
Advection transport path. Train rows calibrate a shift function from early
observed context; validation measures the locked calibrator before any held-out
test can be recorded.
"""

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.calibrate_residual_gate import _relative_improvement, _test_guard_result
from scripts.fit_transport_shift_head import _candidate_shifts, _load_series, _select_best
from scripts.run_observed_transport_shift_gate import _load_test_ledger, _write_test_ledger
from scripts.run_parameter_conditioned_transport_shift_gate import _local_refined_shifts
from scripts.run_source_conditioned_transport_shift_gate import _periodic_shift
from scripts.run_transport_shift_gate import _split_source_record


def _candidate_scores_periodic_window(
    fields: np.ndarray,
    shifts: Sequence[float],
    *,
    start_step: int,
    transitions: int,
) -> list[dict[str, float]]:
    if transitions <= 0:
        raise ValueError("transitions must be positive")
    steps = min(int(transitions), fields.shape[1] - int(start_step) - 1)
    if steps <= 0:
        raise ValueError("Need enough trajectory frames to score a transport shift")
    previous = fields[:, start_step : start_step + steps]
    current = fields[:, start_step + 1 : start_step + steps + 1]
    rows = []
    for shift in shifts:
        shifted = _periodic_shift(previous, float(shift))
        squared_error = np.square(shifted - current)
        mse = float(np.mean(squared_error))
        nrmse = float(np.sqrt(np.mean(squared_error)) / max(float(np.std(current)), 1e-12))
        rows.append({"shift": float(shift), "mse": mse, "nrmse": nrmse})
    return rows


def _select_shift_for_sample(
    sample: np.ndarray,
    shifts: Sequence[int],
    *,
    start_step: int,
    transitions: int,
    metric: str,
    refine_radius: int,
    fractional_refine_step: float,
) -> dict[str, float]:
    coarse_rows = _candidate_scores_periodic_window(
        sample,
        shifts,
        start_step=start_step,
        transitions=transitions,
    )
    coarse_best = _select_best(coarse_rows, metric)
    local_shifts = _local_refined_shifts(
        float(coarse_best["shift"]),
        radius=refine_radius,
        fractional_step=fractional_refine_step,
        width=int(sample.shape[-1]),
    )
    refined_rows = _candidate_scores_periodic_window(
        sample,
        local_shifts,
        start_step=start_step,
        transitions=transitions,
    )
    refined_best = _select_best(refined_rows, metric)
    return {
        "coarse_shift": float(coarse_best["shift"]),
        "selected_shift": float(refined_best["shift"]),
        "mse": float(refined_best["mse"]),
        "nrmse": float(refined_best["nrmse"]),
    }


def _sample_shift_pairs(
    fields: np.ndarray,
    shifts: Sequence[int],
    *,
    context_transitions: int,
    rollout_steps: int,
    metric: str,
    refine_radius: int,
    fractional_refine_step: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    inferred: list[float] = []
    targets: list[float] = []
    records: list[dict[str, Any]] = []
    for sample_idx in range(fields.shape[0]):
        sample = fields[sample_idx : sample_idx + 1]
        context = _select_shift_for_sample(
            sample,
            shifts,
            start_step=0,
            transitions=context_transitions,
            metric=metric,
            refine_radius=refine_radius,
            fractional_refine_step=fractional_refine_step,
        )
        target = _select_shift_for_sample(
            sample,
            shifts,
            start_step=context_transitions,
            transitions=rollout_steps,
            metric=metric,
            refine_radius=refine_radius,
            fractional_refine_step=fractional_refine_step,
        )
        inferred_shift = float(context["selected_shift"])
        target_shift = float(target["selected_shift"])
        inferred.append(inferred_shift)
        targets.append(target_shift)
        records.append(
            {
                "sample_index": int(sample_idx),
                "context_shift": inferred_shift,
                "target_shift": target_shift,
                f"context_{metric}": float(context[metric]),
                f"target_{metric}": float(target[metric]),
            }
        )
    return (
        np.asarray(inferred, dtype=np.float64),
        np.asarray(targets, dtype=np.float64),
        records,
    )


def _fit_linear_calibrator(
    inferred_shifts: np.ndarray,
    target_shifts: np.ndarray,
    *,
    fit_intercept: bool,
) -> dict[str, float]:
    if inferred_shifts.size == 0:
        raise ValueError("Need at least one train sample to fit inferred transport calibrator")
    if fit_intercept:
        design = np.column_stack([inferred_shifts, np.ones_like(inferred_shifts)])
        slope, intercept = np.linalg.lstsq(design, target_shifts, rcond=None)[0]
    else:
        denom = float(np.dot(inferred_shifts, inferred_shifts))
        if denom <= 0:
            raise ValueError(
                "Cannot fit through-origin inferred calibrator with zero shift support"
            )
        slope = float(np.dot(inferred_shifts, target_shifts) / denom)
        intercept = 0.0
    return {"slope": float(slope), "intercept": float(intercept)}


def _calibrate_shifts(raw_shifts: np.ndarray, coefficients: Mapping[str, float]) -> np.ndarray:
    return raw_shifts * float(coefficients["slope"]) + float(coefficients["intercept"])


def _score_inferred_transport(
    fields: np.ndarray,
    shifts: Sequence[int],
    coefficients: Mapping[str, float],
    *,
    context_transitions: int,
    rollout_steps: int,
    metric: str,
    refine_radius: int,
    fractional_refine_step: float,
) -> dict[str, Any]:
    raw_shifts, _target_shifts, records = _sample_shift_pairs(
        fields,
        shifts,
        context_transitions=context_transitions,
        rollout_steps=rollout_steps,
        metric=metric,
        refine_radius=refine_radius,
        fractional_refine_step=fractional_refine_step,
    )
    predicted_shifts = _calibrate_shifts(raw_shifts, coefficients)
    steps = min(int(rollout_steps), fields.shape[1] - int(context_transitions) - 1)
    if steps <= 0:
        raise ValueError("Need enough post-context trajectory frames to score inferred transport")
    previous = fields[:, context_transitions : context_transitions + steps]
    current = fields[:, context_transitions + 1 : context_transitions + steps + 1]
    shifted = np.empty_like(previous)
    for sample_idx, shift in enumerate(predicted_shifts.tolist()):
        shifted[sample_idx : sample_idx + 1] = _periodic_shift(
            previous[sample_idx : sample_idx + 1], float(shift)
        )
    squared_error = np.square(shifted - current)
    mse = float(np.mean(squared_error))
    nrmse = float(np.sqrt(np.mean(squared_error)) / max(float(np.std(current)), 1e-12))
    return {
        "mse": mse,
        "nrmse": nrmse,
        "metric_value": mse if metric == "mse" else nrmse,
        "raw_shift_mean": float(np.mean(raw_shifts)),
        "raw_shift_std": float(np.std(raw_shifts)),
        "predicted_shift_mean": float(np.mean(predicted_shifts)),
        "predicted_shift_std": float(np.std(predicted_shifts)),
        "predicted_shift_min": float(np.min(predicted_shifts)),
        "predicted_shift_max": float(np.max(predicted_shifts)),
        "scored_steps": int(steps),
        "sample_records": records,
    }


def _summarize_pairs(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    context = np.asarray([float(record["context_shift"]) for record in records], dtype=np.float64)
    target = np.asarray([float(record["target_shift"]) for record in records], dtype=np.float64)
    return {
        "count": int(len(records)),
        "context_shift_mean": float(np.mean(context)),
        "context_shift_std": float(np.std(context)),
        "target_shift_mean": float(np.mean(target)),
        "target_shift_std": float(np.std(target)),
    }


def _test_measurement_key(
    *,
    args: argparse.Namespace,
    shifts: Sequence[int],
    data_sources: dict[str, dict[str, Any]],
    coefficients: Mapping[str, float],
) -> str:
    payload = {
        "candidate_shifts": [int(shift) for shift in shifts],
        "coefficients": {key: float(value) for key, value in coefficients.items()},
        "context_transitions": args.context_transitions,
        "data_sources": data_sources,
        "estimator": "inferred_context_periodic_shift",
        "fit_intercept": bool(args.fit_intercept),
        "fit_kind": args.fit_kind,
        "max_samples": args.max_samples,
        "metric": args.metric,
        "reference_metric_value": args.reference_metric_value,
        "refine_radius": args.refine_radius,
        "rollout_steps": args.rollout_steps,
        "task": args.task,
        "test_max_samples": args.test_max_samples,
        "test_split": args.test_split,
        "train_max_samples": args.train_max_samples,
        "train_split": args.train_split,
        "val_max_samples": args.val_max_samples,
        "val_min_relative_improvement": args.val_min_relative_improvement,
        "val_split": args.val_split,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    if args.train_split == args.val_split:
        raise ValueError("--train-split and --val-split must differ")
    if args.fit_kind != "linear":
        raise ValueError(f"Unsupported fit_kind: {args.fit_kind}")

    shifts = _candidate_shifts(args.shift)
    train_fields = _load_series(
        root=args.data_root,
        task=args.task,
        split=args.train_split,
        max_samples=(
            args.train_max_samples if args.train_max_samples is not None else args.max_samples
        ),
    )
    val_fields = _load_series(
        root=args.data_root,
        task=args.task,
        split=args.val_split,
        max_samples=args.val_max_samples if args.val_max_samples is not None else args.max_samples,
    )
    inferred_shifts, target_shifts, train_records = _sample_shift_pairs(
        train_fields,
        shifts,
        context_transitions=args.context_transitions,
        rollout_steps=args.rollout_steps,
        metric=args.metric,
        refine_radius=args.refine_radius,
        fractional_refine_step=args.fractional_refine_step,
    )
    coefficients = _fit_linear_calibrator(
        inferred_shifts,
        target_shifts,
        fit_intercept=bool(args.fit_intercept),
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
    validation_guard = _test_guard_result(
        value=float(selected_validation[args.metric]),
        reference=args.reference_metric_value,
        min_relative_improvement=args.val_min_relative_improvement,
        mode="min",
    )
    blockers: list[str] = []
    if not bool(validation_guard["passed"]):
        blockers.append(
            "context-inferred train-calibrated validation metric did not pass the configured SOTA guard"
        )
    test_eligible = bool(validation_guard["passed"] and not blockers)

    source_splits = [args.train_split, args.val_split]
    if args.test_split:
        source_splits.append(args.test_split)
    data_sources = {
        split: _split_source_record(args.data_root, args.task, split)
        for split in dict.fromkeys(source_splits)
    }

    ledger_path = getattr(args, "test_ledger_json", None)
    allow_repeat_test = bool(getattr(args, "allow_repeat_test", False))
    test_measurement_key = None
    test_ledger_recorded = False
    if test_eligible and args.test_split:
        test_measurement_key = _test_measurement_key(
            args=args,
            shifts=shifts,
            data_sources=data_sources,
            coefficients=coefficients,
        )
        ledger = _load_test_ledger(ledger_path)
        existing_keys = {
            str(entry.get("measurement_key"))
            for entry in ledger.get("measurements", [])
            if isinstance(entry, dict)
        }
        if test_measurement_key in existing_keys and not allow_repeat_test:
            raise RuntimeError(
                "held-out test measurement already recorded for this inferred transport gate; "
                "set --allow-repeat-test only for explicit debugging"
            )

    test_record = None
    if test_eligible and args.test_split:
        test_fields = _load_series(
            root=args.data_root,
            task=args.task,
            split=args.test_split,
            max_samples=(
                args.test_max_samples if args.test_max_samples is not None else args.max_samples
            ),
        )
        test_record = {
            "split": args.test_split,
            "selected_test": _score_inferred_transport(
                test_fields,
                shifts,
                coefficients,
                context_transitions=args.context_transitions,
                rollout_steps=args.rollout_steps,
                metric=args.metric,
                refine_radius=args.refine_radius,
                fractional_refine_step=args.fractional_refine_step,
            ),
        }
        if ledger_path and test_measurement_key and not allow_repeat_test:
            ledger = _load_test_ledger(ledger_path)
            ledger.setdefault("measurements", []).append(
                {
                    "measurement_key": test_measurement_key,
                    "metric": args.metric,
                    "test_metric_value": test_record["selected_test"]["metric_value"],
                    "test_split": args.test_split,
                    "validation_metric_value": selected_validation["metric_value"],
                }
            )
            _write_test_ledger(ledger_path, ledger)
            test_ledger_recorded = True

    return {
        "task": args.task,
        "data_root": str(args.data_root),
        "data_sources": data_sources,
        "train_split": args.train_split,
        "val_split": args.val_split,
        "metric": args.metric,
        "candidate_shifts": shifts,
        "rollout_steps": args.rollout_steps,
        "estimator": {
            "name": "inferred_context_periodic_shift",
            "fit_uses": "train_split context-inferred shifts only",
            "causality_note": (
                "Infers shift from early observed frames and calibrates that inferred shift "
                "on train only; does not use beta metadata or source_file_index as the learned key."
            ),
        },
        "fit": {
            "model": "linear_context_shift_calibrator",
            "fit_kind": args.fit_kind,
            "fit_intercept": bool(args.fit_intercept),
            "context_transitions": int(args.context_transitions),
            "refine_radius": int(args.refine_radius),
            "fractional_refine_step": float(args.fractional_refine_step),
            "candidate_shifts": shifts,
            "coefficients": coefficients,
            "train": selected_train,
            "selected_validation": selected_validation,
            "train_shift_pair_summary": _summarize_pairs(train_records),
            "validation_guard": validation_guard,
            "validation_gap_to_reference": (
                _relative_improvement(
                    float(selected_validation[args.metric]),
                    float(args.reference_metric_value),
                    mode="min",
                )
                if args.reference_metric_value is not None
                else None
            ),
        },
        "validation_guard": validation_guard,
        "held_out_test_policy": {
            "allow_repeat_test": allow_repeat_test,
            "ledger_path": ledger_path,
            "measurement_key": test_measurement_key,
            "recorded": test_ledger_recorded,
        },
        "test_eligible": test_eligible,
        "test": test_record,
        "blockers": blockers,
        "next_action": (
            "held-out test measured"
            if test_record
            else (
                "run exactly one held-out test with locked inferred transport estimator"
                if test_eligible
                else "do not run held-out test; improve train-calibrated inferred transport head first"
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run context-inferred transport-shift gate")
    parser.add_argument("--data-root", default="data/pdebench")
    parser.add_argument("--task", default="advection1d")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--test-split", default="")
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--train-max-samples", type=int)
    parser.add_argument("--val-max-samples", type=int)
    parser.add_argument("--test-max-samples", type=int)
    parser.add_argument("--context-transitions", type=int, default=1)
    parser.add_argument("--rollout-steps", type=int, default=16)
    parser.add_argument("--shift", action="append", type=int, default=None)
    parser.add_argument("--metric", choices=("mse", "nrmse"), default="nrmse")
    parser.add_argument("--fit-kind", choices=("linear",), default="linear")
    parser.add_argument("--fit-intercept", action="store_true")
    parser.add_argument("--refine-radius", type=int, default=0)
    parser.add_argument("--fractional-refine-step", type=float, default=0.0)
    parser.add_argument("--reference-metric-value", type=float, required=True)
    parser.add_argument("--val-min-relative-improvement", type=float, default=0.0)
    parser.add_argument("--test-ledger-json")
    parser.add_argument("--allow-repeat-test", action="store_true")
    parser.add_argument(
        "--output-json",
        default="reports/research/sota_loop/inferred_transport_shift_gate.json",
    )
    args = parser.parse_args()

    record = run_gate(args)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
