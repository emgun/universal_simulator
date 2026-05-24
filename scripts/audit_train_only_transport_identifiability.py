#!/usr/bin/env python
from __future__ import annotations

"""Audit train-only identifiability of transport-shift labels before testing."""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.fit_transport_shift_head import (
    _candidate_scores,
    _candidate_shifts,
    _load_series,
    _select_best,
)


def _max_samples(value: int | None) -> int | None:
    return None if value is not None and value < 0 else value


def _per_sample_best_shifts(
    fields: torch.Tensor,
    shifts: Sequence[int],
    *,
    rollout_steps: int,
    metric: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    fields = torch.as_tensor(fields, dtype=torch.float32)
    labels: list[int] = []
    margins: list[float] = []
    for sample in fields:
        rows = _candidate_scores(sample.unsqueeze(0), shifts, rollout_steps=rollout_steps)
        ranked = sorted(rows, key=lambda row: float(row[metric]))
        labels.append(int(ranked[0]["shift"]))
        margins.append(
            float(ranked[1][metric]) - float(ranked[0][metric]) if len(ranked) > 1 else 0.0
        )
    return torch.tensor(labels, dtype=torch.long), torch.tensor(margins, dtype=torch.float32)


def _histogram(values: torch.Tensor) -> dict[str, int]:
    hist: dict[str, int] = {}
    for value in values.tolist():
        key = str(int(value))
        hist[key] = hist.get(key, 0) + 1
    return hist


def _summary(values: torch.Tensor) -> dict[str, float]:
    return {
        "min": float(values.min().item()),
        "mean": float(values.mean().item()),
        "max": float(values.max().item()),
    }


def _split_candidate_summary(
    fields: torch.Tensor,
    shifts: Sequence[int],
    *,
    rollout_steps: int,
    metric: str,
    top_k: int,
) -> dict[str, Any]:
    fields = torch.as_tensor(fields, dtype=torch.float32)
    rows = _candidate_scores(fields, shifts, rollout_steps=rollout_steps)
    return {
        "best": _select_best(rows, metric),
        "top": sorted(rows, key=lambda row: float(row[metric]))[:top_k],
    }


def audit_identifiability(args: argparse.Namespace) -> dict[str, Any]:
    shifts = _candidate_shifts(args.shift)
    train_fields = _load_series(
        root=args.data_root,
        task=args.task,
        split=args.train_split,
        max_samples=_max_samples(args.max_samples),
    )
    val_max_samples = args.max_samples if args.val_max_samples is None else args.val_max_samples
    val_fields = _load_series(
        root=args.data_root,
        task=args.task,
        split=args.val_split,
        max_samples=_max_samples(val_max_samples),
    )

    train_labels, train_margins = _per_sample_best_shifts(
        train_fields, shifts, rollout_steps=args.rollout_steps, metric=args.metric
    )
    val_labels, val_margins = _per_sample_best_shifts(
        val_fields, shifts, rollout_steps=args.rollout_steps, metric=args.metric
    )
    train_label_set = sorted(set(int(value) for value in train_labels.tolist()))
    val_label_set = sorted(set(int(value) for value in val_labels.tolist()))
    unsupported_val_shifts = sorted(set(val_label_set) - set(train_label_set))
    train_single_regime = len(train_label_set) == 1
    val_requires_unseen_regime = bool(unsupported_val_shifts)
    split_train = _split_candidate_summary(
        train_fields, shifts, rollout_steps=args.rollout_steps, metric=args.metric, top_k=args.top_k
    )
    split_val = _split_candidate_summary(
        val_fields, shifts, rollout_steps=args.rollout_steps, metric=args.metric, top_k=args.top_k
    )

    if train_single_regime and val_requires_unseen_regime:
        status = "blocked_underidentified_train_only_shift"
        blockers = [
            "train split contains a single observed shift label",
            "validation requires at least one shift label absent from train",
            "a train-only supervised shift rule has no evidence selecting the unseen validation regime",
        ]
    elif val_requires_unseen_regime:
        status = "blocked_unsupported_validation_shift"
        blockers = ["validation requires at least one shift label absent from train"]
    else:
        status = "train_shift_support_covers_validation"
        blockers = []

    return {
        "status": status,
        "blockers": blockers,
        "task": args.task,
        "data_root": str(args.data_root),
        "train_split": args.train_split,
        "val_split": args.val_split,
        "max_samples": _max_samples(args.max_samples),
        "val_max_samples": _max_samples(val_max_samples),
        "rollout_steps": args.rollout_steps,
        "metric": args.metric,
        "candidate_shifts": shifts,
        "train_shape": list(train_fields.shape),
        "val_shape": list(val_fields.shape),
        "train_shift_histogram": _histogram(train_labels),
        "val_shift_histogram": _histogram(val_labels),
        "train_shift_support": train_label_set,
        "val_shift_support": val_label_set,
        "unsupported_val_shifts": unsupported_val_shifts,
        "train_single_regime": train_single_regime,
        "val_requires_unseen_regime": val_requires_unseen_regime,
        "train_best_margin_summary": _summary(train_margins),
        "val_best_margin_summary": _summary(val_margins),
        "split_level_candidates": {
            "train": split_train,
            "validation": split_val,
        },
        "interpretation": (
            "No train-only shift-label learner can identify an unseen validation shift from supervised train evidence alone."
            if status == "blocked_underidentified_train_only_shift"
            else "Train support covers validation shift labels; a train-only learned head remains plausible."
        ),
        "notes": [
            "Uses train and validation splits only.",
            "Validation labels are used only to audit support before any held-out test.",
            "Does not read or evaluate the held-out test split.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit train-only transport-shift identifiability")
    parser.add_argument("--data-root", default="data/pdebench")
    parser.add_argument("--task", default="advection1d")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument(
        "--max-samples", type=int, default=128, help="Train sample cap; use -1 for full split"
    )
    parser.add_argument(
        "--val-max-samples", type=int, help="Validation sample cap; use -1 for full split"
    )
    parser.add_argument("--rollout-steps", type=int, default=16)
    parser.add_argument("--shift", action="append", type=int, default=None)
    parser.add_argument("--metric", choices=("mse", "nrmse"), default="nrmse")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--output-json",
        default="reports/research/sota_loop/train_only_transport_identifiability_audit.json",
    )
    args = parser.parse_args()

    record = audit_identifiability(args)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
