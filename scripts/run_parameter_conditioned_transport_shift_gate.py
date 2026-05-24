#!/usr/bin/env python
from __future__ import annotations

"""Gate a beta-parameter-conditioned transport-shift estimator.

This is the benchmark-clean successor to the source-file-conditioned shift map:
fit a shift function from Advection beta parameters on train rows only, validate
the locked function on val, and measure held-out test only after the guard passes.
"""

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.calibrate_residual_gate import _relative_improvement, _test_guard_result
from scripts.fit_transport_shift_head import _candidate_shifts, _select_best
from scripts.run_observed_transport_shift_gate import _load_test_ledger, _write_test_ledger
from scripts.run_source_conditioned_transport_shift_gate import (
    _candidate_scores_periodic,
    _fractional_refined_shifts,
    _periodic_shift,
    _refined_shifts,
    _split_source_record,
)

_BETA_PATTERN = re.compile(r"beta(?P<beta>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")


def _max_samples(value: int | None) -> int | None:
    return None if value is not None and value < 0 else value


def _beta_from_source_path(source_path: str) -> float:
    match = _BETA_PATTERN.search(source_path)
    if not match:
        raise ValueError(f"Could not parse beta from source path: {source_path}")
    return float(match.group("beta"))


def _json_safe_attr(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _load_series_source_beta(
    *,
    root: str | Path,
    task: str,
    split: str,
    max_samples: int | None,
) -> tuple[np.ndarray, np.ndarray, dict[int, float], list[str]]:
    path = Path(root) / f"{task}_{split}.h5"
    if not path.exists():
        raise FileNotFoundError(path)
    sample_limit = _max_samples(max_samples)
    sample_slice = slice(0, sample_limit) if sample_limit is not None else slice(None)
    with h5py.File(path, "r") as handle:
        if "source_file_index" not in handle:
            raise KeyError(f"{path} does not contain source_file_index provenance")
        data = np.asarray(handle["data"][sample_slice], dtype=np.float32)
        source_file_index = np.asarray(handle["source_file_index"][sample_slice], dtype=np.int64)
        source_paths = [
            str(value) for value in _json_safe_attr(handle.attrs.get("source_paths", []))
        ]
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(
            f"Expected 1D task data shaped (samples, steps, width[, 1]), got {tuple(data.shape)}"
        )
    if data.shape[0] != source_file_index.shape[0]:
        raise ValueError(f"{path} data/source_file_index sample counts differ")
    beta_by_source = {
        index: _beta_from_source_path(value) for index, value in enumerate(source_paths)
    }
    missing_sources = sorted(
        set(int(value) for value in source_file_index.tolist()) - set(beta_by_source)
    )
    if missing_sources:
        raise ValueError(
            f"{path} has source_file_index values without beta metadata: {missing_sources}"
        )
    return data, source_file_index, beta_by_source, source_paths


def _local_refined_shifts(
    base_shift: float,
    *,
    radius: int,
    fractional_step: float,
    width: int,
) -> list[float]:
    if radius <= 0:
        return [float(base_shift)]
    if fractional_step > 0:
        return _fractional_refined_shifts(
            float(base_shift), int(radius), float(fractional_step), width=int(width)
        )
    return [
        float(value) for value in _refined_shifts(int(round(base_shift)), int(radius), width=width)
    ]


def _fit_sample_shift_labels(
    fields: np.ndarray,
    source_file_index: np.ndarray,
    beta_by_source: Mapping[int, float],
    shifts: Sequence[int],
    *,
    rollout_steps: int,
    metric: str,
    refine_radius: int,
    fractional_refine_step: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    beta_values: list[float] = []
    shift_labels: list[float] = []
    records: list[dict[str, Any]] = []
    for sample_idx in range(fields.shape[0]):
        sample = fields[sample_idx : sample_idx + 1]
        coarse_rows = _candidate_scores_periodic(sample, shifts, rollout_steps=rollout_steps)
        coarse_best = _select_best(coarse_rows, metric)
        local_shifts = _local_refined_shifts(
            float(coarse_best["shift"]),
            radius=refine_radius,
            fractional_step=fractional_refine_step,
            width=int(fields.shape[-1]),
        )
        refined_rows = _candidate_scores_periodic(sample, local_shifts, rollout_steps=rollout_steps)
        refined_best = _select_best(refined_rows, metric)
        source_index = int(source_file_index[sample_idx])
        beta = float(beta_by_source[source_index])
        selected_shift = float(refined_best["shift"])
        beta_values.append(beta)
        shift_labels.append(selected_shift)
        records.append(
            {
                "sample_index": int(sample_idx),
                "source_file_index": source_index,
                "beta": beta,
                "coarse_shift": float(coarse_best["shift"]),
                "selected_shift": selected_shift,
                f"selected_{metric}": float(refined_best[metric]),
            }
        )
    return (
        np.asarray(beta_values, dtype=np.float64),
        np.asarray(shift_labels, dtype=np.float64),
        records,
    )


def _fit_linear_beta_shift(
    beta_values: np.ndarray,
    shift_labels: np.ndarray,
    *,
    fit_intercept: bool,
) -> dict[str, float]:
    if beta_values.size == 0:
        raise ValueError("Need at least one train sample to fit beta-conditioned transport")
    if fit_intercept:
        design = np.column_stack([beta_values, np.ones_like(beta_values)])
        slope, intercept = np.linalg.lstsq(design, shift_labels, rcond=None)[0]
    else:
        denom = float(np.dot(beta_values, beta_values))
        if denom <= 0:
            raise ValueError("Cannot fit through-origin beta shift model with zero beta support")
        slope = float(np.dot(beta_values, shift_labels) / denom)
        intercept = 0.0
    return {"slope": float(slope), "intercept": float(intercept)}


def _predict_shifts(beta_values: np.ndarray, coefficients: Mapping[str, float]) -> np.ndarray:
    return beta_values * float(coefficients["slope"]) + float(coefficients["intercept"])


def _score_parameter_shifts(
    fields: np.ndarray,
    source_file_index: np.ndarray,
    beta_by_source: Mapping[int, float],
    coefficients: Mapping[str, float],
    *,
    rollout_steps: int,
    metric: str,
) -> dict[str, Any]:
    steps = min(int(rollout_steps), fields.shape[1] - 1)
    if steps <= 0:
        raise ValueError("Need at least two trajectory frames to score a transport shift")
    previous = fields[:, :steps]
    current = fields[:, 1 : steps + 1]
    beta_values = np.asarray(
        [float(beta_by_source[int(source)]) for source in source_file_index.tolist()],
        dtype=np.float64,
    )
    predicted_shifts = _predict_shifts(beta_values, coefficients)
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
        "predicted_shift_mean": float(np.mean(predicted_shifts)),
        "predicted_shift_std": float(np.std(predicted_shifts)),
        "predicted_shift_min": float(np.min(predicted_shifts)),
        "predicted_shift_max": float(np.max(predicted_shifts)),
        "scored_steps": int(steps),
    }


def _summarize_labels(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[float]] = {}
    for record in records:
        key = f"{float(record['beta']):g}"
        grouped.setdefault(key, []).append(float(record["selected_shift"]))
    return {
        key: {
            "count": int(len(values)),
            "mean_shift": float(np.mean(values)),
            "std_shift": float(np.std(values)),
        }
        for key, values in sorted(grouped.items(), key=lambda item: float(item[0]))
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
        "data_sources": data_sources,
        "estimator": "parameter_conditioned_periodic_shift",
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
    train_fields, train_source, train_beta_by_source, train_source_paths = _load_series_source_beta(
        root=args.data_root,
        task=args.task,
        split=args.train_split,
        max_samples=(
            args.train_max_samples if args.train_max_samples is not None else args.max_samples
        ),
    )
    val_fields, val_source, val_beta_by_source, val_source_paths = _load_series_source_beta(
        root=args.data_root,
        task=args.task,
        split=args.val_split,
        max_samples=args.val_max_samples if args.val_max_samples is not None else args.max_samples,
    )

    beta_values, shift_labels, sample_records = _fit_sample_shift_labels(
        train_fields,
        train_source,
        train_beta_by_source,
        shifts,
        rollout_steps=args.rollout_steps,
        metric=args.metric,
        refine_radius=args.refine_radius,
        fractional_refine_step=args.fractional_refine_step,
    )
    coefficients = _fit_linear_beta_shift(
        beta_values, shift_labels, fit_intercept=bool(args.fit_intercept)
    )
    selected_train = _score_parameter_shifts(
        train_fields,
        train_source,
        train_beta_by_source,
        coefficients,
        rollout_steps=args.rollout_steps,
        metric=args.metric,
    )
    selected_validation = _score_parameter_shifts(
        val_fields,
        val_source,
        val_beta_by_source,
        coefficients,
        rollout_steps=args.rollout_steps,
        metric=args.metric,
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
            "parameter-conditioned train-fitted validation metric did not pass the configured SOTA guard"
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
                "held-out test measurement already recorded for this parameter transport gate; "
                "set --allow-repeat-test only for explicit debugging"
            )

    test_record = None
    if test_eligible and args.test_split:
        test_fields, test_source, test_beta_by_source, _test_source_paths = (
            _load_series_source_beta(
                root=args.data_root,
                task=args.task,
                split=args.test_split,
                max_samples=(
                    args.test_max_samples if args.test_max_samples is not None else args.max_samples
                ),
            )
        )
        test_record = {
            "split": args.test_split,
            "selected_test": _score_parameter_shifts(
                test_fields,
                test_source,
                test_beta_by_source,
                coefficients,
                rollout_steps=args.rollout_steps,
                metric=args.metric,
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
            "name": "parameter_conditioned_periodic_shift",
            "fit_uses": "train_split beta parameters and trajectories only",
            "causality_note": (
                "Uses known PDE beta metadata as the conditioning parameter and does not use "
                "held-out transitions or source_file_index as the learned shift key."
            ),
        },
        "fit": {
            "model": "beta_linear_periodic_shift",
            "fit_kind": args.fit_kind,
            "fit_intercept": bool(args.fit_intercept),
            "refine_radius": int(args.refine_radius),
            "fractional_refine_step": float(args.fractional_refine_step),
            "candidate_shifts": shifts,
            "coefficients": coefficients,
            "train": selected_train,
            "selected_validation": selected_validation,
            "train_label_summary_by_beta": _summarize_labels(sample_records),
            "train_source_paths": train_source_paths,
            "val_source_paths": val_source_paths,
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
                "run exactly one held-out test with locked beta-conditioned transport estimator"
                if test_eligible
                else "do not run held-out test; improve train-fitted beta transport head first"
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run beta-parameter-conditioned train/val transport-shift gate"
    )
    parser.add_argument("--data-root", default="data/pdebench")
    parser.add_argument("--task", default="advection1d")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--test-split", default="")
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--train-max-samples", type=int)
    parser.add_argument("--val-max-samples", type=int)
    parser.add_argument("--test-max-samples", type=int)
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
        default="reports/research/sota_loop/parameter_conditioned_transport_shift_gate.json",
    )
    args = parser.parse_args()

    record = run_gate(args)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
