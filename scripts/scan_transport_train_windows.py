#!/usr/bin/env python
from __future__ import annotations

"""Scan train-source windows for periodic transport-shift regimes.

This is a data-construction diagnostic for the train-fitted transport gate. It
uses only a source split, normally `train`, and reports which source windows
would fit which constant shift. The intended remote use is against large/full
PDEBench train shards before constructing a smaller light shard.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.fit_transport_shift_head import _candidate_scores, _candidate_shifts, _select_best


def _load_window(path: Path, *, start: int, count: int, rollout_steps: int) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        total = int(handle["data"].shape[0])
        if start >= total:
            raise ValueError(f"Window start {start} is outside {path} with {total} samples")
        stop = min(start + count, total)
        data = torch.from_numpy(handle["data"][start:stop, : rollout_steps + 1]).float()
    if data.dim() == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.dim() != 3:
        raise ValueError(f"Expected 1D data shaped (samples, steps, width[, 1]), got {tuple(data.shape)}")
    return data


def scan_windows(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.data_root) / f"{args.task}_{args.split}.h5"
    shifts = _candidate_shifts(args.shift)
    with h5py.File(path, "r") as handle:
        total_samples = int(handle["data"].shape[0])
        source_shape = [int(dim) for dim in handle["data"].shape]
    if args.window_size <= 0:
        raise ValueError("--window-size must be positive")
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    starts = list(range(args.start_index, total_samples, args.stride))
    if args.max_windows is not None:
        starts = starts[: args.max_windows]

    rows = []
    for start in starts:
        if start + args.min_samples > total_samples:
            break
        fields = _load_window(path, start=start, count=args.window_size, rollout_steps=args.rollout_steps)
        scores = _candidate_scores(fields, shifts, rollout_steps=args.rollout_steps)
        best = _select_best(scores, args.metric)
        rows.append(
            {
                "start_index": int(start),
                "sample_count": int(fields.shape[0]),
                "best": best,
                "top": sorted(scores, key=lambda row: float(row[args.metric]))[: args.top_k],
            }
        )

    histogram: dict[str, int] = {}
    for row in rows:
        shift = str(int(row["best"]["shift"]))
        histogram[shift] = histogram.get(shift, 0) + 1
    return {
        "task": args.task,
        "data_root": str(args.data_root),
        "split": args.split,
        "path": str(path),
        "source_shape": source_shape,
        "total_samples": total_samples,
        "window_size": args.window_size,
        "stride": args.stride,
        "start_index": args.start_index,
        "max_windows": args.max_windows,
        "rollout_steps": args.rollout_steps,
        "metric": args.metric,
        "candidate_shifts": shifts,
        "windows_scanned": len(rows),
        "best_shift_histogram": histogram,
        "windows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan transport-shift regimes across source-train windows")
    parser.add_argument("--data-root", default="data/pdebench")
    parser.add_argument("--task", default="advection1d")
    parser.add_argument("--split", default="train")
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--min-samples", type=int, default=1)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--rollout-steps", type=int, default=16)
    parser.add_argument("--shift", action="append", type=int, default=None)
    parser.add_argument("--metric", choices=("mse", "nrmse"), default="nrmse")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output-json", default="reports/research/sota_loop/transport_train_window_scan.json")
    args = parser.parse_args()

    record = scan_windows(args)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
