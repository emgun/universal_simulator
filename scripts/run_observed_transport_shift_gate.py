#!/usr/bin/env python
from __future__ import annotations

"""Gate a lagged observed-transition transport estimator on train/val/test.

The constant train-fitted shift fails on the current light-v1 Advection shards
because train/val/test have different transport regimes. This script evaluates
the next stricter candidate: estimate each step's periodic shift from the
previous observed transition, then predict the next frame with that shift.

Train is used to lock and report the estimator contract. Validation measures the
same contract without selecting a validation shift. Test is measured only when
the validation guard passes.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.calibrate_residual_gate import _test_guard_result
from scripts.fit_transport_shift_head import _candidate_shifts, _load_series
from scripts.run_transport_shift_gate import _split_source_record


def _load_test_ledger(path: str | None) -> dict[str, Any]:
    if not path:
        return {"measurements": []}
    ledger_path = Path(path)
    if not ledger_path.exists():
        return {"measurements": []}
    return json.loads(ledger_path.read_text(encoding="utf-8"))


def _write_test_ledger(path: str | None, ledger: dict[str, Any]) -> None:
    if not path:
        return
    ledger_path = Path(path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True), encoding="utf-8")


def _test_measurement_key(
    *,
    args: argparse.Namespace,
    shifts: Sequence[int],
    data_sources: dict[str, dict[str, Any]],
) -> str:
    payload = {
        "candidate_shifts": [int(shift) for shift in shifts],
        "data_sources": data_sources,
        "estimator": "lagged_observed_transition_shift",
        "max_samples": args.max_samples,
        "metric": args.metric,
        "reference_metric_value": args.reference_metric_value,
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


def _estimate_shift_indices(
    *,
    previous: torch.Tensor,
    current: torch.Tensor,
    shifts: Sequence[int],
) -> torch.Tensor:
    if not shifts:
        raise ValueError("At least one candidate shift is required")
    errors = [
        torch.roll(previous, shifts=int(shift), dims=-1).sub(current).pow(2).mean(dim=-1)
        for shift in shifts
    ]
    return torch.stack(errors, dim=0).argmin(dim=0)


def _score_observed_transport(
    fields: torch.Tensor,
    shifts: Sequence[int],
    *,
    rollout_steps: int,
    metric: str,
) -> dict[str, Any]:
    if rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive")
    steps = min(int(rollout_steps), fields.shape[1] - 2)
    if steps <= 0:
        raise ValueError("Need at least three trajectory frames for lagged observed transport scoring")

    shift_tensor = torch.tensor([int(shift) for shift in shifts], dtype=torch.long)
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    selected_shifts: list[torch.Tensor] = []

    for step in range(1, steps + 1):
        previous = fields[:, step - 1]
        current = fields[:, step]
        target = fields[:, step + 1]
        selected_idx = _estimate_shift_indices(previous=previous, current=current, shifts=shifts)
        pred = torch.empty_like(current)
        for idx, shift in enumerate(shifts):
            mask = selected_idx == idx
            if bool(mask.any()):
                pred[mask] = torch.roll(current[mask], shifts=int(shift), dims=-1)
        predictions.append(pred)
        targets.append(target)
        selected_shifts.append(shift_tensor[selected_idx])

    pred_stack = torch.stack(predictions, dim=1)
    target_stack = torch.stack(targets, dim=1)
    shift_stack = torch.stack(selected_shifts, dim=1).float()
    mse = float((pred_stack - target_stack).pow(2).mean().item())
    nrmse = float(torch.sqrt((pred_stack - target_stack).pow(2).mean()) / target_stack.std().clamp_min(1e-12))
    return {
        "mse": mse,
        "nrmse": nrmse,
        "metric_value": mse if metric == "mse" else nrmse,
        "shift_mean": float(shift_stack.mean().item()),
        "shift_std": float(shift_stack.std(unbiased=False).item()),
        "shift_min": int(shift_stack.min().item()),
        "shift_max": int(shift_stack.max().item()),
        "scored_steps": steps,
        "scored_transitions": int(shift_stack.numel()),
    }


def _split_record(
    *,
    data_root: str | Path,
    task: str,
    split: str,
    max_samples: int | None,
    shifts: Sequence[int],
    rollout_steps: int,
    metric: str,
) -> dict[str, Any]:
    fields = _load_series(root=data_root, task=task, split=split, max_samples=max_samples)
    score = _score_observed_transport(fields, shifts, rollout_steps=rollout_steps, metric=metric)
    return {
        "split": split,
        "max_samples": max_samples,
        "shape": list(fields.shape),
        **score,
    }


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    if args.train_split == args.val_split:
        raise ValueError("--train-split and --val-split must differ")

    shifts = _candidate_shifts(args.shift)
    train_record = _split_record(
        data_root=args.data_root,
        task=args.task,
        split=args.train_split,
        max_samples=args.train_max_samples or args.max_samples,
        shifts=shifts,
        rollout_steps=args.rollout_steps,
        metric=args.metric,
    )
    validation_record = _split_record(
        data_root=args.data_root,
        task=args.task,
        split=args.val_split,
        max_samples=args.val_max_samples or args.max_samples,
        shifts=shifts,
        rollout_steps=args.rollout_steps,
        metric=args.metric,
    )
    validation_guard = _test_guard_result(
        value=float(validation_record[args.metric]),
        reference=args.reference_metric_value,
        min_relative_improvement=args.val_min_relative_improvement,
        mode="min",
    )
    test_eligible = bool(validation_guard["passed"])
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
        test_measurement_key = _test_measurement_key(args=args, shifts=shifts, data_sources=data_sources)
        ledger = _load_test_ledger(ledger_path)
        existing_keys = {
            str(entry.get("measurement_key"))
            for entry in ledger.get("measurements", [])
            if isinstance(entry, dict)
        }
        if test_measurement_key in existing_keys and not allow_repeat_test:
            raise RuntimeError(
                "held-out test measurement already recorded for this observed transport gate; "
                "set --allow-repeat-test only for explicit debugging"
            )

    test_record = None
    if test_eligible and args.test_split:
        test_record = _split_record(
            data_root=args.data_root,
            task=args.task,
            split=args.test_split,
            max_samples=args.test_max_samples or args.max_samples,
            shifts=shifts,
            rollout_steps=args.rollout_steps,
            metric=args.metric,
        )
        if ledger_path and test_measurement_key and not allow_repeat_test:
            ledger = _load_test_ledger(ledger_path)
            ledger.setdefault("measurements", []).append(
                {
                    "measurement_key": test_measurement_key,
                    "metric": args.metric,
                    "test_metric_value": test_record["metric_value"],
                    "test_split": args.test_split,
                    "validation_metric_value": validation_record["metric_value"],
                }
            )
            _write_test_ledger(ledger_path, ledger)
            test_ledger_recorded = True

    return {
        "task": args.task,
        "data_root": str(args.data_root),
        "data_sources": data_sources,
        "metric": args.metric,
        "candidate_shifts": shifts,
        "rollout_steps": args.rollout_steps,
        "estimator": {
            "name": "lagged_observed_transition_shift",
            "fit_scope": "train locks the estimator contract and candidate shift support; validation does not select shifts",
            "causality_note": (
                "Uses the previous observed transition to predict the next frame. "
                "This is state-conditioned and benchmark-clean for the gate, but not a fully autonomous rollout head."
            ),
        },
        "train": train_record,
        "validation": validation_record,
        "validation_guard": validation_guard,
        "test_eligible": test_eligible,
        "held_out_test_policy": {
            "allow_repeat_test": allow_repeat_test,
            "ledger_path": ledger_path,
            "measurement_key": test_measurement_key,
            "recorded": test_ledger_recorded,
        },
        "test": test_record,
        "next_action": (
            "held-out test measured"
            if test_record
            else "run exactly one held-out test with the locked observed-transport estimator"
            if test_eligible
            else "do not run held-out test; train a causal transport head or rebuild compatible shards"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate a lagged observed-transition transport estimator")
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
    parser.add_argument("--reference-metric-value", type=float)
    parser.add_argument("--val-min-relative-improvement", type=float)
    parser.add_argument(
        "--test-ledger-json",
        help="Optional ledger that prevents measuring the same guarded held-out test more than once",
    )
    parser.add_argument("--allow-repeat-test", action="store_true", help="Bypass the held-out test ledger guard")
    parser.add_argument("--output-json", default="reports/research/sota_loop/observed_transport_shift_gate.json")
    args = parser.parse_args()

    record = run_gate(args)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
