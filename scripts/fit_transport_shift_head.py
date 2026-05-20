#!/usr/bin/env python
from __future__ import annotations

"""Fit a causal periodic transport-shift rule on train data, then validate it."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.calibrate_residual_gate import _relative_improvement, _test_guard_result
from scripts.calibrate_roll_shift import _candidate_shifts, _shift_override


def _load_series(
    *,
    root: str | Path,
    task: str,
    split: str,
    max_samples: int | None,
) -> np.ndarray:
    path = Path(root) / f"{task}_{split}.h5"
    if not path.exists():
        raise FileNotFoundError(path)
    sample_slice = slice(0, max_samples) if max_samples is not None else slice(None)
    with h5py.File(path, "r") as handle:
        data = np.asarray(handle["data"][sample_slice], dtype=np.float32)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Expected 1D task data shaped (samples, steps, width[, 1]), got {tuple(data.shape)}")
    return data


def _candidate_scores(fields: np.ndarray, shifts: Sequence[int], *, rollout_steps: int) -> list[dict[str, float | int]]:
    if rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive")
    steps = min(int(rollout_steps), fields.shape[1] - 1)
    if steps <= 0:
        raise ValueError("Need at least two trajectory frames to fit a transport shift")
    previous = fields[:, :steps]
    current = fields[:, 1 : steps + 1]
    rows: list[dict[str, float | int]] = []
    for shift in shifts:
        shifted = np.roll(previous, shift=int(shift), axis=-1)
        squared_error = np.square(shifted - current)
        mse = float(np.mean(squared_error))
        nrmse = float(np.sqrt(np.mean(squared_error)) / max(float(np.std(current)), 1e-12))
        rows.append({"shift": int(shift), "mse": mse, "nrmse": nrmse})
    return rows


def _select_best(rows: Sequence[Mapping[str, Any]], metric: str) -> dict[str, Any]:
    if not rows:
        raise ValueError("No candidate rows to select from")
    return dict(sorted(rows, key=lambda row: float(row[metric]))[0])


def fit_and_validate(args: argparse.Namespace) -> dict[str, Any]:
    if args.train_split == args.val_split and not args.allow_same_split_smoke:
        raise ValueError("--train-split and --val-split must differ unless --allow-same-split-smoke is set")
    shifts = _candidate_shifts(args.shift)
    train_fields = _load_series(root=args.data_root, task=args.task, split=args.train_split, max_samples=args.max_samples)
    val_fields = _load_series(root=args.data_root, task=args.task, split=args.val_split, max_samples=args.max_samples)
    train_rows = _candidate_scores(train_fields, shifts, rollout_steps=args.rollout_steps)
    val_rows = _candidate_scores(val_fields, shifts, rollout_steps=args.rollout_steps)
    selected_train = _select_best(train_rows, args.metric)
    selected_shift = int(selected_train["shift"])
    selected_val = next(row for row in val_rows if int(row["shift"]) == selected_shift)
    oracle_val = _select_best(val_rows, args.metric)
    validation_guard = _test_guard_result(
        value=float(selected_val[args.metric]),
        reference=args.reference_metric_value,
        min_relative_improvement=args.val_min_relative_improvement,
        mode="min",
    )
    return {
        "task": args.task,
        "kind": args.kind,
        "key": args.key,
        "data_root": str(args.data_root),
        "train_split": args.train_split,
        "val_split": args.val_split,
        "max_samples": args.max_samples,
        "rollout_steps": args.rollout_steps,
        "metric": args.metric,
        "candidate_shifts": shifts,
        "train_candidates": train_rows,
        "validation_candidates": val_rows,
        "selected_train_shift": selected_shift,
        "selected_train": selected_train,
        "selected_validation": selected_val,
        "oracle_validation": oracle_val,
        "validation_gap_to_oracle": _relative_improvement(
            float(selected_val[args.metric]),
            float(oracle_val[args.metric]),
            mode="min",
        ),
        "validation_guard": validation_guard,
        "selected_override": _shift_override(args.kind, args.key, selected_shift),
        "notes": [
            "Fit uses only train_split trajectories.",
            "Validation selects no new shift; it only measures the train-fitted shift.",
            "Run held-out test separately only if validation_guard.passed is true.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a train-split periodic transport shift and validate it")
    parser.add_argument("--data-root", default="data/pdebench")
    parser.add_argument("--task", default="advection1d")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--rollout-steps", type=int, default=16)
    parser.add_argument("--shift", action="append", type=int, default=None)
    parser.add_argument("--metric", choices=("mse", "nrmse"), default="nrmse")
    parser.add_argument("--kind", choices=("task", "family"), default="task")
    parser.add_argument("--key", default="advection1d")
    parser.add_argument("--reference-metric-value", type=float)
    parser.add_argument("--val-min-relative-improvement", type=float)
    parser.add_argument("--allow-same-split-smoke", action="store_true")
    parser.add_argument("--output-json", default="reports/research/sota_loop/transport_head_fit/fit_record.json")
    parser.add_argument("--export-selected-config")
    args = parser.parse_args()

    record = fit_and_validate(args)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    if args.export_selected_config:
        selected_path = Path(args.export_selected_config)
        selected_path.parent.mkdir(parents=True, exist_ok=True)
        selected_path.write_text(
            json.dumps(
                {
                    "selected_override": record["selected_override"],
                    "selected_train_shift": record["selected_train_shift"],
                    "selected_train": record["selected_train"],
                    "selected_validation": record["selected_validation"],
                    "validation_guard": record["validation_guard"],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
