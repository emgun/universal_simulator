#!/usr/bin/env python
from __future__ import annotations

"""Run the benchmark-clean train/val gate for transport-shift candidates."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.diagnose_transport_shift_splits import diagnose as diagnose_splits
from scripts.fit_transport_shift_head import (
    _candidate_scores,
    _load_series,
    _select_best,
    fit_and_validate,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_source_record(data_root: str, task: str, split: str) -> dict[str, Any]:
    path = Path(data_root) / f"{task}_{split}.h5"
    if not path.exists():
        return {"split": split, "path": str(path), "exists": False}
    return {
        "split": split,
        "path": str(path),
        "exists": True,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    diagnostic_args = argparse.Namespace(
        data_root=args.data_root,
        task=args.task,
        splits=f"{args.train_split},{args.val_split}",
        max_samples=args.max_samples,
        rollout_steps=args.rollout_steps,
        shift=args.shift,
        top_k=args.top_k,
    )
    diagnostic = diagnose_splits(diagnostic_args)

    fit_args = argparse.Namespace(
        data_root=args.data_root,
        task=args.task,
        train_split=args.train_split,
        val_split=args.val_split,
        max_samples=args.max_samples,
        rollout_steps=args.rollout_steps,
        shift=args.shift,
        metric=args.metric,
        kind=args.kind,
        key=args.key,
        reference_metric_value=args.reference_metric_value,
        val_min_relative_improvement=args.val_min_relative_improvement,
        allow_same_split_smoke=False,
    )
    fit_record = fit_and_validate(fit_args)
    validation_guard = fit_record["validation_guard"]
    split_consistent = bool(diagnostic["consistent_best_shift"])
    guard_passed = bool(validation_guard["passed"])
    test_eligible = split_consistent and guard_passed
    blockers: list[str] = []
    if not split_consistent:
        blockers.append(
            "train and validation best transport shifts differ; a constant train-fitted shift is not benchmark-clean"
        )
    if not guard_passed:
        blockers.append("train-fitted validation metric did not pass the configured SOTA guard")
    test_record = None
    if test_eligible and args.test_split:
        test_fields = _load_series(
            root=args.data_root,
            task=args.task,
            split=args.test_split,
            max_samples=args.test_max_samples or args.max_samples,
        )
        test_rows = _candidate_scores(
            test_fields,
            fit_record["candidate_shifts"],
            rollout_steps=args.rollout_steps,
        )
        selected_shift = int(fit_record["selected_train_shift"])
        selected_test = next(row for row in test_rows if int(row["shift"]) == selected_shift)
        oracle_test = _select_best(test_rows, args.metric)
        test_record = {
            "split": args.test_split,
            "selected_shift": selected_shift,
            "selected_test": selected_test,
            "oracle_test": oracle_test,
            "candidate_scores": test_rows,
        }

    source_splits = [args.train_split, args.val_split]
    if args.test_split:
        source_splits.append(args.test_split)
    data_sources = {
        split: _split_source_record(args.data_root, args.task, split)
        for split in dict.fromkeys(source_splits)
    }

    return {
        "task": args.task,
        "data_root": args.data_root,
        "data_sources": data_sources,
        "train_split": args.train_split,
        "val_split": args.val_split,
        "metric": args.metric,
        "reference_metric_value": args.reference_metric_value,
        "val_min_relative_improvement": args.val_min_relative_improvement,
        "diagnostic": diagnostic,
        "fit": fit_record,
        "test_eligible": test_eligible,
        "test": test_record,
        "blockers": blockers,
        "next_action": (
            "held-out test measured"
            if test_record
            else (
                "run exactly one held-out test with fit.selected_override"
                if test_eligible and not test_record
                else "do not run held-out test; fix split construction or train a per-trajectory head first"
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the train/val transport-shift promotion gate")
    parser.add_argument("--data-root", default="data/pdebench")
    parser.add_argument("--task", default="advection1d")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument(
        "--test-split",
        default="",
        help="Optional held-out test split to evaluate only if train/val gate passes",
    )
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument(
        "--test-max-samples", type=int, help="Optional max samples for held-out test split"
    )
    parser.add_argument("--rollout-steps", type=int, default=16)
    parser.add_argument("--shift", action="append", type=int, default=None)
    parser.add_argument("--metric", choices=("mse", "nrmse"), default="nrmse")
    parser.add_argument("--kind", choices=("task", "family"), default="task")
    parser.add_argument("--key", default="advection1d")
    parser.add_argument("--reference-metric-value", type=float, required=True)
    parser.add_argument("--val-min-relative-improvement", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--output-json", default="reports/research/sota_loop/transport_shift_gate.json"
    )
    args = parser.parse_args()

    record = run_gate(args)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
