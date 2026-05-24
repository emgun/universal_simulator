#!/usr/bin/env python
from __future__ import annotations

"""Diagnose whether train-only first-frame features can predict transport shift."""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.fit_transport_shift_head import _candidate_scores, _candidate_shifts, _load_series


def _per_sample_shift_stats(
    fields: torch.Tensor, shifts: Sequence[int], *, rollout_steps: int, metric: str
) -> tuple[torch.Tensor, torch.Tensor]:
    fields = torch.as_tensor(fields, dtype=torch.float32)
    labels: list[int] = []
    margins: list[float] = []
    for sample in fields:
        rows = _candidate_scores(sample.unsqueeze(0), shifts, rollout_steps=rollout_steps)
        ranked = sorted(rows, key=lambda row: float(row[metric]))
        best = ranked[0]
        labels.append(int(best["shift"]))
        margins.append(float(ranked[1][metric]) - float(best[metric]) if len(ranked) > 1 else 0.0)
    return torch.tensor(labels, dtype=torch.long), torch.tensor(margins, dtype=torch.float32)


def _first_frame_features(fields: torch.Tensor) -> torch.Tensor:
    fields = torch.as_tensor(fields, dtype=torch.float32)
    first = fields[:, 0]
    centered = first - first.mean(dim=-1, keepdim=True)
    spectrum = torch.fft.rfft(centered, dim=-1).abs()
    low_freq = spectrum[:, 1:9]
    stats = torch.stack(
        [
            first.mean(dim=-1),
            first.std(dim=-1),
            first.min(dim=-1).values,
            first.max(dim=-1).values,
        ],
        dim=-1,
    )
    return torch.cat([stats, low_freq], dim=-1).float()


def _standardize(train: torch.Tensor, other: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = train.mean(dim=0, keepdim=True)
    std = train.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (train - mean) / std, (other - mean) / std


def _nearest_centroid_predict(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, list[float]]]:
    centroids: dict[int, torch.Tensor] = {}
    for label in sorted(set(int(value) for value in train_labels.tolist())):
        centroids[label] = train_features[train_labels == label].mean(dim=0)
    ordered_labels = sorted(centroids)
    centroid_tensor = torch.stack([centroids[label] for label in ordered_labels], dim=0)
    distances = torch.cdist(val_features, centroid_tensor)
    predicted_indices = distances.argmin(dim=1)
    predictions = torch.tensor(
        [ordered_labels[int(index)] for index in predicted_indices], dtype=torch.long
    )
    return predictions, {str(label): centroids[label].tolist() for label in ordered_labels}


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


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    shifts = _candidate_shifts(args.shift)
    train_max_samples = (
        None if args.max_samples is not None and args.max_samples < 0 else args.max_samples
    )
    val_max_samples = args.max_samples if args.val_max_samples is None else args.val_max_samples
    val_max_samples = (
        None if val_max_samples is not None and val_max_samples < 0 else val_max_samples
    )
    train_fields = _load_series(
        root=args.data_root, task=args.task, split=args.train_split, max_samples=train_max_samples
    )
    val_fields = _load_series(
        root=args.data_root, task=args.task, split=args.val_split, max_samples=val_max_samples
    )
    train_labels, train_margins = _per_sample_shift_stats(
        train_fields, shifts, rollout_steps=args.rollout_steps, metric=args.metric
    )
    val_labels, val_margins = _per_sample_shift_stats(
        val_fields, shifts, rollout_steps=args.rollout_steps, metric=args.metric
    )
    train_features = _first_frame_features(train_fields)
    val_features = _first_frame_features(val_fields)
    train_features_std, val_features_std = _standardize(train_features, val_features)
    val_predictions, centroids = _nearest_centroid_predict(
        train_features_std, train_labels, val_features_std
    )
    accuracy = float((val_predictions == val_labels).float().mean().item())
    train_label_set = set(int(value) for value in train_labels.tolist())
    val_label_set = set(int(value) for value in val_labels.tolist())
    unsupported_val_shifts = sorted(val_label_set - train_label_set)
    return {
        "task": args.task,
        "data_root": str(args.data_root),
        "train_split": args.train_split,
        "val_split": args.val_split,
        "max_samples": train_max_samples,
        "val_max_samples": val_max_samples,
        "rollout_steps": args.rollout_steps,
        "metric": args.metric,
        "candidate_shifts": shifts,
        "feature_set": "first_frame_stats_plus_low_frequency_spectrum",
        "model": "nearest_train_shift_centroid",
        "train_shift_histogram": _histogram(train_labels),
        "val_shift_histogram": _histogram(val_labels),
        "val_prediction_histogram": _histogram(val_predictions),
        "train_best_margin_summary": _summary(train_margins),
        "val_best_margin_summary": _summary(val_margins),
        "unsupported_val_shifts": unsupported_val_shifts,
        "val_accuracy": accuracy,
        "centroids": centroids,
        "conclusion": (
            "blocked_no_train_support_for_validation_shift"
            if unsupported_val_shifts
            else (
                "train_feature_probe_supports_validation_shift"
                if accuracy > 0.0
                else "train_feature_probe_failed"
            )
        ),
        "notes": [
            "Uses train split only to fit shift centroids.",
            "Validation labels are used only for measurement.",
            "Does not read or evaluate the held-out test split.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose train-only transport-shift feature support"
    )
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
    parser.add_argument(
        "--output-json",
        default="reports/research/sota_loop/train_only_transport_feature_diagnostic.json",
    )
    args = parser.parse_args()

    record = diagnose(args)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
