#!/usr/bin/env python
from __future__ import annotations

"""Scan temporal start windows for train/val transport-shift support."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import h5py
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.fit_transport_shift_head import _candidate_scores, _candidate_shifts, _select_best


def _load_time_window(
    *,
    root: str | Path,
    task: str,
    split: str,
    max_samples: int | None,
    start_step: int,
    rollout_steps: int,
) -> torch.Tensor:
    path = Path(root) / f"{task}_{split}.h5"
    if not path.exists():
        raise FileNotFoundError(path)
    sample_slice = slice(0, max_samples) if max_samples is not None else slice(None)
    stop_step = start_step + rollout_steps + 1
    with h5py.File(path, "r") as handle:
        total_steps = int(handle["data"].shape[1])
        if start_step < 0 or stop_step > total_steps:
            raise ValueError(f"Temporal window [{start_step}, {stop_step}) is outside {path} with {total_steps} steps")
        data = torch.from_numpy(handle["data"][sample_slice, start_step:stop_step]).float()
    if data.dim() == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.dim() != 3:
        raise ValueError(f"Expected 1D task data shaped (samples, steps, width[, 1]), got {tuple(data.shape)}")
    return data


def _scan_split(
    *,
    root: str | Path,
    task: str,
    split: str,
    max_samples: int | None,
    start_steps: Sequence[int],
    rollout_steps: int,
    shifts: Sequence[int],
    metric: str,
    top_k: int,
) -> list[dict[str, Any]]:
    rows = []
    for start_step in start_steps:
        fields = _load_time_window(
            root=root,
            task=task,
            split=split,
            max_samples=max_samples,
            start_step=int(start_step),
            rollout_steps=rollout_steps,
        )
        scores = _candidate_scores(fields, shifts, rollout_steps=rollout_steps)
        best = _select_best(scores, metric)
        rows.append(
            {
                "start_step": int(start_step),
                "sample_count": int(fields.shape[0]),
                "best": best,
                "top": sorted(scores, key=lambda row: float(row[metric]))[:top_k],
            }
        )
    return rows


def _histogram(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    hist: dict[str, int] = {}
    for row in rows:
        key = str(int(row["best"]["shift"]))
        hist[key] = hist.get(key, 0) + 1
    return hist


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    shifts = _candidate_shifts(args.shift)
    train_max_samples = None if args.max_samples is not None and args.max_samples < 0 else args.max_samples
    val_max_samples = args.max_samples if args.val_max_samples is None else args.val_max_samples
    val_max_samples = None if val_max_samples is not None and val_max_samples < 0 else val_max_samples
    with h5py.File(Path(args.data_root) / f"{args.task}_{args.train_split}.h5", "r") as handle:
        total_steps = int(handle["data"].shape[1])
    latest_start = total_steps - args.rollout_steps - 1
    if latest_start < 0:
        raise ValueError("rollout_steps leaves no valid temporal windows")
    start_steps = list(range(args.start_step, latest_start + 1, args.stride))
    if args.max_windows is not None:
        start_steps = start_steps[: args.max_windows]

    train_rows = _scan_split(
        root=args.data_root,
        task=args.task,
        split=args.train_split,
        max_samples=train_max_samples,
        start_steps=start_steps,
        rollout_steps=args.rollout_steps,
        shifts=shifts,
        metric=args.metric,
        top_k=args.top_k,
    )
    val_rows = _scan_split(
        root=args.data_root,
        task=args.task,
        split=args.val_split,
        max_samples=val_max_samples,
        start_steps=start_steps,
        rollout_steps=args.rollout_steps,
        shifts=shifts,
        metric=args.metric,
        top_k=args.top_k,
    )
    train_shifts = {int(row["best"]["shift"]) for row in train_rows}
    val_shifts = {int(row["best"]["shift"]) for row in val_rows}
    common_shifts = sorted(train_shifts & val_shifts)
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
        "start_step": args.start_step,
        "stride": args.stride,
        "max_windows": args.max_windows,
        "windows_scanned": len(start_steps),
        "train_shift_histogram": _histogram(train_rows),
        "val_shift_histogram": _histogram(val_rows),
        "common_temporal_best_shifts": common_shifts,
        "train_windows": train_rows,
        "val_windows": val_rows,
        "conclusion": "temporal_train_val_shift_support_found" if common_shifts else "blocked_no_temporal_common_shift",
        "notes": [
            "Uses train and validation splits only.",
            "Scans temporal start windows; does not read or evaluate held-out test.",
            "This is a support diagnostic, not a held-out promotion gate.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan train/val temporal transport-shift support")
    parser.add_argument("--data-root", default="data/pdebench")
    parser.add_argument("--task", default="advection1d")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--max-samples", type=int, default=128, help="Train sample cap; use -1 for full split")
    parser.add_argument("--val-max-samples", type=int, help="Validation sample cap; use -1 for full split")
    parser.add_argument("--rollout-steps", type=int, default=16)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--shift", action="append", type=int, default=None)
    parser.add_argument("--metric", choices=("mse", "nrmse"), default="nrmse")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output-json", default="reports/research/sota_loop/transport_temporal_window_diagnostic.json")
    args = parser.parse_args()

    record = diagnose(args)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
