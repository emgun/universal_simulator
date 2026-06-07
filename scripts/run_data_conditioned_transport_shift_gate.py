#!/usr/bin/env python
from __future__ import annotations

"""Fit a train-only data-conditioned transport shift estimator and validate it."""

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.calibrate_residual_gate import _test_guard_result
from scripts.calibrate_roll_shift import _candidate_shifts
from scripts.run_inferred_transport_shift_gate import _select_shift_for_sample
from scripts.run_source_conditioned_transport_shift_gate import _periodic_shift

DEFAULT_FEATURE_NAMES = (
    "bias",
    "horizon_norm",
    "mean",
    "std",
    "rms",
    "abs_mean",
    "max",
    "min",
)
CONTEXT_FEATURE_NAMES = {"context_shift", "context_shift_abs"}


def _load_series(
    *,
    root: str | Path,
    task: str,
    split: str,
    max_samples: int | None,
    normalize: bool = True,
) -> np.ndarray:
    path = Path(root) / f"{task}_{split}.h5"
    if not path.exists():
        raise FileNotFoundError(path)
    sample_limit = None if max_samples is not None and max_samples < 0 else max_samples
    sample_slice = slice(0, sample_limit) if sample_limit is not None else slice(None)
    with h5py.File(path, "r") as handle:
        data = np.asarray(handle["data"][sample_slice], dtype=np.float32)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(
            f"Expected 1D task data shaped (samples, steps, width[, 1]), got {tuple(data.shape)}"
        )
    if normalize:
        std = float(np.std(data))
        if std < 1e-6:
            std = 1.0
        data = (data - float(np.mean(data))) / std
    return data.astype(np.float32, copy=False)


def _field_features(
    previous: np.ndarray,
    *,
    horizon: int,
    rollout_steps: int,
    context_shift: float | None,
) -> dict[str, float]:
    features = {
        "bias": 1.0,
        "horizon_norm": float(horizon) / max(float(rollout_steps), 1.0),
        "mean": float(np.mean(previous)),
        "std": float(np.std(previous)),
        "rms": float(np.sqrt(np.mean(np.square(previous)))),
        "abs_mean": float(np.mean(np.abs(previous))),
        "max": float(np.max(previous)),
        "min": float(np.min(previous)),
    }
    if context_shift is not None:
        features["context_shift"] = float(context_shift)
        features["context_shift_abs"] = abs(float(context_shift))
    return features


def _context_shifts(
    fields: np.ndarray,
    shifts: Sequence[int],
    *,
    context_transitions: int,
    metric: str,
    refine_radius: int,
    fractional_refine_step: float,
) -> np.ndarray:
    if context_transitions <= 0:
        raise ValueError("context_transitions must be positive when context features are used")
    rows = []
    for sample_idx in range(fields.shape[0]):
        selected = _select_shift_for_sample(
            fields[sample_idx : sample_idx + 1],
            shifts,
            start_step=0,
            transitions=context_transitions,
            metric=metric,
            refine_radius=refine_radius,
            fractional_refine_step=fractional_refine_step,
        )
        rows.append(float(selected["selected_shift"]))
    return np.asarray(rows, dtype=np.float64)


def _feature_matrix(
    fields: np.ndarray,
    *,
    rollout_steps: int,
    feature_names: Sequence[str],
    min_horizon: int,
    context_shifts: np.ndarray | None,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    steps = min(int(rollout_steps), fields.shape[1] - 1)
    if steps <= 0:
        raise ValueError("Need at least two trajectory frames to fit a transport shift")
    rows: list[list[float]] = []
    row_index: list[tuple[int, int]] = []
    for sample_idx in range(fields.shape[0]):
        for step in range(steps):
            horizon = step + 1
            if horizon < int(min_horizon):
                continue
            context_shift = None if context_shifts is None else float(context_shifts[sample_idx])
            features = _field_features(
                fields[sample_idx, step],
                horizon=horizon,
                rollout_steps=steps,
                context_shift=context_shift,
            )
            rows.append([float(features.get(name, 0.0)) for name in feature_names])
            row_index.append((sample_idx, step))
    if not rows:
        raise ValueError("No transition rows remained after applying min_horizon")
    return np.asarray(rows, dtype=np.float64), row_index


def _best_shift_labels(
    fields: np.ndarray,
    row_index: Sequence[tuple[int, int]],
    shifts: Sequence[int],
    *,
    metric: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    labels: list[float] = []
    records: list[dict[str, Any]] = []
    for sample_idx, step in row_index:
        previous = fields[sample_idx : sample_idx + 1, step : step + 1]
        current = fields[sample_idx : sample_idx + 1, step + 1 : step + 2]
        rows = []
        for shift in shifts:
            shifted = _periodic_shift(previous, float(shift))
            squared_error = np.square(shifted - current)
            mse = float(np.mean(squared_error))
            nrmse = float(np.sqrt(np.mean(squared_error)) / max(float(np.std(current)), 1e-12))
            rows.append({"shift": float(shift), "mse": mse, "nrmse": nrmse})
        best = min(rows, key=lambda row: float(row[metric]))
        labels.append(float(best["shift"]))
        records.append(
            {
                "sample_index": int(sample_idx),
                "horizon": int(step + 1),
                "selected_shift": float(best["shift"]),
                f"selected_{metric}": float(best[metric]),
            }
        )
    return np.asarray(labels, dtype=np.float64), records


def _fit_ridge(
    X: np.ndarray,
    y: np.ndarray,
    *,
    ridge: float,
    feature_names: Sequence[str],
) -> np.ndarray:
    penalty = np.eye(X.shape[1], dtype=np.float64) * float(ridge)
    if "bias" in feature_names:
        bias_index = int(list(feature_names).index("bias"))
        penalty[bias_index, bias_index] = 0.0
    return np.linalg.solve(X.T @ X + penalty, X.T @ y)


def _predict_shift(
    X: np.ndarray,
    coefficients: np.ndarray,
    *,
    min_shift: float | None,
    max_shift: float | None,
) -> np.ndarray:
    predicted = X @ coefficients
    if min_shift is not None:
        predicted = np.maximum(predicted, float(min_shift))
    if max_shift is not None:
        predicted = np.minimum(predicted, float(max_shift))
    return predicted


def _score_predicted_shifts(
    fields: np.ndarray,
    row_index: Sequence[tuple[int, int]],
    predicted_shifts: np.ndarray,
    *,
    metric: str,
) -> dict[str, Any]:
    squared_errors = []
    current_values = []
    for (sample_idx, step), shift in zip(row_index, predicted_shifts.tolist(), strict=True):
        previous = fields[sample_idx : sample_idx + 1, step : step + 1]
        current = fields[sample_idx : sample_idx + 1, step + 1 : step + 2]
        shifted = _periodic_shift(previous, float(shift))
        squared_errors.append(np.square(shifted - current).reshape(-1))
        current_values.append(current.reshape(-1))
    squared_error = np.concatenate(squared_errors)
    current_concat = np.concatenate(current_values)
    mse = float(np.mean(squared_error))
    nrmse = float(np.sqrt(mse) / max(float(np.std(current_concat)), 1e-12))
    return {
        "mse": mse,
        "nrmse": nrmse,
        "metric_value": mse if metric == "mse" else nrmse,
        "predicted_shift_mean": float(np.mean(predicted_shifts)),
        "predicted_shift_std": float(np.std(predicted_shifts)),
        "predicted_shift_min": float(np.min(predicted_shifts)),
        "predicted_shift_max": float(np.max(predicted_shifts)),
        "transition_count": int(len(predicted_shifts)),
    }


def _label_summary(labels: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(labels)),
        "std": float(np.std(labels)),
        "min": float(np.min(labels)),
        "max": float(np.max(labels)),
    }


def _selected_override(
    *,
    coefficients: Mapping[str, float],
    feature_names: Sequence[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    estimator: dict[str, Any] = {
        "enabled": True,
        "mode": "roll_persistence",
        "tasks": [str(args.task)],
        "min_horizon": int(args.min_horizon),
        "feature_names": list(feature_names),
        "coefficients": {key: float(value) for key, value in coefficients.items()},
        "calibration_scope": "train_fit_field_features",
    }
    if any(name in CONTEXT_FEATURE_NAMES for name in feature_names):
        estimator["candidate_shifts"] = [float(shift) for shift in _candidate_shifts(args.shift)]
        estimator["context_transitions"] = int(args.context_transitions)
        estimator["calibration_scope"] = "train_fit_context_field_features"
    if args.min_shift is not None:
        estimator["min_shift"] = float(args.min_shift)
    if args.max_shift is not None:
        estimator["max_shift"] = float(args.max_shift)
    return {"evaluation.decoded_data_conditioned_roll_shift_estimator": estimator}


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    if args.train_split == args.val_split:
        raise ValueError("--train-split and --val-split must differ")
    shifts = _candidate_shifts(args.shift)
    feature_names = [str(name) for name in (args.feature or DEFAULT_FEATURE_NAMES)]
    uses_context = any(name in CONTEXT_FEATURE_NAMES for name in feature_names)
    if uses_context and int(args.min_horizon) <= int(args.context_transitions):
        raise ValueError(
            "--min-horizon must be greater than --context-transitions when context features are used"
        )
    train_fields = _load_series(
        root=args.data_root,
        task=args.task,
        split=args.train_split,
        max_samples=(
            args.train_max_samples if args.train_max_samples is not None else args.max_samples
        ),
        normalize=not args.no_normalize,
    )
    val_fields = _load_series(
        root=args.data_root,
        task=args.task,
        split=args.val_split,
        max_samples=args.val_max_samples if args.val_max_samples is not None else args.max_samples,
        normalize=not args.no_normalize,
    )
    train_context_shifts = (
        _context_shifts(
            train_fields,
            shifts,
            context_transitions=int(args.context_transitions),
            metric=args.metric,
            refine_radius=int(args.refine_radius),
            fractional_refine_step=float(args.fractional_refine_step),
        )
        if uses_context
        else None
    )
    val_context_shifts = (
        _context_shifts(
            val_fields,
            shifts,
            context_transitions=int(args.context_transitions),
            metric=args.metric,
            refine_radius=int(args.refine_radius),
            fractional_refine_step=float(args.fractional_refine_step),
        )
        if uses_context
        else None
    )
    train_X, train_index = _feature_matrix(
        train_fields,
        rollout_steps=args.rollout_steps,
        feature_names=feature_names,
        min_horizon=int(args.min_horizon),
        context_shifts=train_context_shifts,
    )
    train_labels, train_label_records = _best_shift_labels(
        train_fields,
        train_index,
        shifts,
        metric=args.metric,
    )
    coefficient_vector = _fit_ridge(
        train_X,
        train_labels,
        ridge=args.ridge,
        feature_names=feature_names,
    )
    coefficients = {
        name: float(value) for name, value in zip(feature_names, coefficient_vector, strict=True)
    }
    train_pred = _predict_shift(
        train_X,
        coefficient_vector,
        min_shift=args.min_shift,
        max_shift=args.max_shift,
    )
    train_score = _score_predicted_shifts(
        train_fields,
        train_index,
        train_pred,
        metric=args.metric,
    )
    val_X, val_index = _feature_matrix(
        val_fields,
        rollout_steps=args.rollout_steps,
        feature_names=feature_names,
        min_horizon=int(args.min_horizon),
        context_shifts=val_context_shifts,
    )
    val_pred = _predict_shift(
        val_X,
        coefficient_vector,
        min_shift=args.min_shift,
        max_shift=args.max_shift,
    )
    validation_score = _score_predicted_shifts(
        val_fields,
        val_index,
        val_pred,
        metric=args.metric,
    )
    validation_guard = _test_guard_result(
        value=float(validation_score[args.metric]),
        reference=args.reference_metric_value,
        min_relative_improvement=args.val_min_relative_improvement,
        mode="min",
    )
    blockers = []
    if not bool(validation_guard["passed"]):
        blockers.append(
            "data-conditioned train-fitted validation metric did not pass the configured guard"
        )
    return {
        "task": args.task,
        "data_root": str(args.data_root),
        "train_split": args.train_split,
        "val_split": args.val_split,
        "held_out_test_used": False,
        "held_out_test_data_read": False,
        "max_samples": args.max_samples,
        "train_max_samples": args.train_max_samples,
        "val_max_samples": args.val_max_samples,
        "rollout_steps": args.rollout_steps,
        "metric": args.metric,
        "candidate_shifts": [int(shift) for shift in shifts],
        "feature_names": feature_names,
        "ridge": float(args.ridge),
        "normalize": not args.no_normalize,
        "min_horizon": int(args.min_horizon),
        "context_transitions": int(args.context_transitions) if uses_context else None,
        "context_refine_radius": int(args.refine_radius) if uses_context else None,
        "context_fractional_refine_step": (
            float(args.fractional_refine_step) if uses_context else None
        ),
        "train_context_shift_summary": (
            _label_summary(train_context_shifts) if train_context_shifts is not None else None
        ),
        "val_context_shift_summary": (
            _label_summary(val_context_shifts) if val_context_shifts is not None else None
        ),
        "min_shift": args.min_shift,
        "max_shift": args.max_shift,
        "fit": {
            "model": "ridge_linear_field_feature_shift",
            "coefficients": coefficients,
            "train_label_summary": _label_summary(train_labels),
            "train_label_records_preview": train_label_records[: min(len(train_label_records), 16)],
            "train": train_score,
            "selected_validation": validation_score,
            "validation_guard": validation_guard,
        },
        "selected_override": _selected_override(
            coefficients=coefficients,
            feature_names=feature_names,
            args=args,
        ),
        "test_eligible": bool(validation_guard["passed"] and not blockers),
        "blockers": blockers,
        "next_action": (
            "validation guard cleared; write a separate held-out pre-test contract before test"
            if bool(validation_guard["passed"] and not blockers)
            else "do not run held-out test; improve train-fitted data-conditioned estimator first"
        ),
        "notes": [
            "Fit uses train split trajectories only.",
            "Validation scores locked train-fitted coefficients; it does not refit on validation.",
            "No held-out test split is loaded or measured by this script.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/pdebench")
    parser.add_argument("--task", default="advection1d")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--train-max-samples", type=int)
    parser.add_argument("--val-max-samples", type=int)
    parser.add_argument("--rollout-steps", type=int, default=16)
    parser.add_argument("--shift", action="append", type=int, default=None)
    parser.add_argument("--feature", action="append", default=None)
    parser.add_argument("--metric", choices=("mse", "nrmse"), default="nrmse")
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--min-horizon", type=int, default=1)
    parser.add_argument("--context-transitions", type=int, default=0)
    parser.add_argument("--refine-radius", type=int, default=0)
    parser.add_argument("--fractional-refine-step", type=float, default=0.0)
    parser.add_argument("--min-shift", type=float)
    parser.add_argument("--max-shift", type=float)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--reference-metric-value", type=float, required=True)
    parser.add_argument("--val-min-relative-improvement", type=float, default=0.0)
    parser.add_argument(
        "--output-json",
        default="reports/research/sota_loop/data_conditioned_transport_phase/gate.json",
    )
    args = parser.parse_args(argv)
    record = run_gate(args)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
