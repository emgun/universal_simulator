#!/usr/bin/env python
from __future__ import annotations

"""Protocol-level diagnostic: advection transport-shift distributions per split.

Two independent held-out failures (the no-context model-side candidate and the
Poseidon channel_lift Option A candidate) share one signature: advection
validation quality that collapses on the held-out test split while Burgers and
Darcy transfer fine. This diagnostic characterizes whether the light-v1
advection splits differ in transport-speed composition, using a deterministic,
model-free, per-sample shift estimate (circular cross-correlation of
consecutive frames).

This is NOT a candidate measurement: nothing is trained, tuned, selected, or
scored, and no ledger is written. Reading the held-out test split requires the
explicit --include-test flag and is recorded as held_out_test_data_read=true
with purpose=protocol_distribution_diagnostic in the output JSON.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Official PDEBench Advection source files are named by beta; index order
# matches source_file_index in the hydrated shards.
OFFICIAL_BETAS = (0.1, 0.2, 0.4, 0.7, 1.0, 2.0, 4.0, 7.0)


def estimate_shift_per_step(fields: np.ndarray, *, transitions: int) -> float:
    """Median subpixel circular shift per raw timestep for one sample.

    fields: (T, W) array. Positive shift means rightward transport.
    """

    steps = fields.shape[0]
    use = min(int(transitions), steps - 1)
    indices = np.linspace(0, steps - 2, num=use, dtype=int)
    shifts: list[float] = []
    width = fields.shape[1]
    for t in indices:
        a = fields[t]
        b = fields[t + 1]
        fa = np.fft.rfft(a - a.mean())
        fb = np.fft.rfft(b - b.mean())
        corr = np.fft.irfft(fb * np.conj(fa), n=width)
        k = int(np.argmax(corr))
        # Parabolic subpixel refinement around the integer peak.
        left = corr[(k - 1) % width]
        mid = corr[k]
        right = corr[(k + 1) % width]
        denom = left - 2.0 * mid + right
        offset = 0.0 if denom == 0.0 else 0.5 * (left - right) / denom
        shift = k + float(np.clip(offset, -0.5, 0.5))
        if shift > width / 2:
            shift -= width
        shifts.append(shift)
    return float(np.median(shifts))


def summarize(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(arr.max()),
    }


def nearest_beta(shift_per_step: float, pixels_per_beta: float) -> float:
    implied = abs(shift_per_step) / pixels_per_beta
    return float(min(OFFICIAL_BETAS, key=lambda beta: abs(beta - implied)))


def analyze_split(
    path: Path,
    *,
    transitions: int,
    max_samples: int | None,
    pixels_per_beta: float,
) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        data = handle["data"]
        count = data.shape[0] if max_samples is None else min(int(max_samples), data.shape[0])
        shifts = []
        for index in range(count):
            fields = np.asarray(data[index], dtype=np.float64)
            if fields.ndim == 3:
                fields = fields[..., 0]
            shifts.append(estimate_shift_per_step(fields, transitions=transitions))
        provenance = None
        if "source_file_index" in handle:
            provenance = [int(v) for v in np.asarray(handle["source_file_index"][:count])]

    betas = [nearest_beta(s, pixels_per_beta) for s in shifts]
    beta_counts: dict[str, int] = {}
    for beta in betas:
        key = f"{beta:g}"
        beta_counts[key] = beta_counts.get(key, 0) + 1

    record: dict[str, Any] = {
        "file": str(path),
        "samples_analyzed": len(shifts),
        "shift_per_step_pixels": summarize(shifts),
        "implied_nearest_official_beta_counts": dict(
            sorted(beta_counts.items(), key=lambda kv: float(kv[0]))
        ),
        "per_sample_shift_per_step": [round(s, 4) for s in shifts],
    }
    if provenance is not None:
        official_counts: dict[str, int] = {}
        for idx in provenance:
            key = f"{OFFICIAL_BETAS[idx]:g}"
            official_counts[key] = official_counts.get(key, 0) + 1
        record["official_beta_counts_from_provenance"] = dict(
            sorted(official_counts.items(), key=lambda kv: float(kv[0]))
        )
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="PDEBench-shaped root with advection1d_*.h5")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Explicitly allow reading advection1d_test.h5 for this protocol diagnostic",
    )
    parser.add_argument("--transitions", type=int, default=24)
    parser.add_argument("--max-samples", type=int, default=0, help="0 = all samples")
    parser.add_argument(
        "--pixels-per-beta",
        type=float,
        default=10.24,
        help="Pixels of shift per raw step per unit beta (dt*W/domain: 0.01*1024)",
    )
    parser.add_argument("--output-json", default="")
    args = parser.parse_args(argv)

    splits = list(args.splits)
    if "test" in splits and not args.include_test:
        raise SystemExit(
            "Reading the held-out test split requires --include-test. This diagnostic "
            "characterizes the data distribution only; it never scores a candidate."
        )

    root = Path(args.root)
    results: dict[str, Any] = {
        "measurement_type": "advection_split_shift_distribution_diagnostic",
        "purpose": "protocol_distribution_diagnostic",
        "candidate_scored": False,
        "test_ledger_writes": [],
        "held_out_test_data_read": "test" in splits,
        "root": str(root),
        "transitions_per_sample": int(args.transitions),
        "pixels_per_beta": float(args.pixels_per_beta),
        "splits": {},
    }
    for split in splits:
        path = root / f"advection1d_{split}.h5"
        if not path.exists():
            results["splits"][split] = {"status": "missing", "file": str(path)}
            continue
        results["splits"][split] = analyze_split(
            path,
            transitions=args.transitions,
            max_samples=None if args.max_samples <= 0 else args.max_samples,
            pixels_per_beta=args.pixels_per_beta,
        )

    payload = json.dumps(results, indent=2, sort_keys=True)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")
    compact = {
        split: {
            "mean_shift_per_step": record.get("shift_per_step_pixels", {}).get("mean"),
            "beta_counts": record.get("implied_nearest_official_beta_counts"),
        }
        for split, record in results["splits"].items()
        if isinstance(record, dict) and record.get("status") != "missing"
    }
    print(json.dumps(compact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
