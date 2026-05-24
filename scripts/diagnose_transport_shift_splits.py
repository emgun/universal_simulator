#!/usr/bin/env python
from __future__ import annotations

"""Diagnose periodic transport-shift regimes across train/val/test shards."""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.calibrate_roll_shift import _candidate_shifts


def _load_series(path: Path, *, max_samples: int | None, rollout_steps: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    sample_slice = slice(0, max_samples) if max_samples is not None else slice(None)
    with h5py.File(path, "r") as handle:
        data = np.asarray(handle["data"][sample_slice, : rollout_steps + 1], dtype=np.float32)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(
            f"Expected 1D shard shaped (samples, steps, width[, 1]), got {tuple(data.shape)}"
        )
    if data.shape[1] <= 1:
        raise ValueError("Need at least two time steps to diagnose transport shift")
    return data


def _score_shift(fields: np.ndarray, shift: int) -> dict[str, float | int]:
    previous = fields[:, :-1]
    current = fields[:, 1:]
    shifted = np.roll(previous, shift=int(shift), axis=-1)
    squared_error = np.square(shifted - current)
    mse = float(np.mean(squared_error))
    nrmse = float(np.sqrt(np.mean(squared_error)) / max(float(np.std(current)), 1e-12))
    return {"shift": int(shift), "mse": mse, "nrmse": nrmse}


def _diagnose_split(
    *,
    root: Path,
    task: str,
    split: str,
    shifts: Sequence[int],
    max_samples: int | None,
    rollout_steps: int,
    top_k: int,
) -> dict[str, Any]:
    path = root / f"{task}_{split}.h5"
    fields = _load_series(path, max_samples=max_samples, rollout_steps=rollout_steps)
    scores = [_score_shift(fields, shift) for shift in shifts]
    ranked = sorted(scores, key=lambda row: float(row["nrmse"]))
    return {
        "split": split,
        "path": str(path),
        "shape": list(fields.shape),
        "mean": float(np.mean(fields)),
        "std": float(np.std(fields)),
        "best": ranked[0],
        "top": ranked[:top_k],
    }


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    shifts = _candidate_shifts(args.shift)
    splits = [split.strip() for split in args.splits.replace(";", ",").split(",") if split.strip()]
    rows = [
        _diagnose_split(
            root=Path(args.data_root),
            task=args.task,
            split=split,
            shifts=shifts,
            max_samples=args.max_samples,
            rollout_steps=args.rollout_steps,
            top_k=args.top_k,
        )
        for split in splits
    ]
    best_shifts = {row["split"]: int(row["best"]["shift"]) for row in rows}
    unique_best_shifts = sorted(set(best_shifts.values()))
    return {
        "task": args.task,
        "data_root": args.data_root,
        "splits": splits,
        "max_samples": args.max_samples,
        "rollout_steps": args.rollout_steps,
        "candidate_shifts": shifts,
        "best_shifts": best_shifts,
        "unique_best_shifts": unique_best_shifts,
        "consistent_best_shift": len(unique_best_shifts) == 1,
        "diagnostics": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose transport-shift consistency across PDEBench splits"
    )
    parser.add_argument("--data-root", default="data/pdebench")
    parser.add_argument("--task", default="advection1d")
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--rollout-steps", type=int, default=16)
    parser.add_argument("--shift", action="append", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--output-json", default="reports/research/sota_loop/transport_shift_split_diagnostic.json"
    )
    args = parser.parse_args()

    record = diagnose(args)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
