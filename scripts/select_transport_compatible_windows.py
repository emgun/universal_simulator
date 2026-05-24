#!/usr/bin/env python
from __future__ import annotations

"""Select compatible train/val/test transport windows from scan artifacts."""

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _load_scan(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "windows" not in payload:
        raise ValueError(f"{path} is missing windows")
    return payload


def _windows_by_shift(scan: Mapping[str, Any]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in scan.get("windows", []):
        shift = int(row["best"]["shift"])
        grouped.setdefault(shift, []).append(dict(row))
    return grouped


def select_compatible(args: argparse.Namespace) -> dict[str, Any]:
    scans = {
        "train": _load_scan(Path(args.train_scan)),
        "val": _load_scan(Path(args.val_scan)),
        "test": _load_scan(Path(args.test_scan)) if args.test_scan else None,
    }
    grouped = {split: _windows_by_shift(scan) for split, scan in scans.items() if scan is not None}
    required_splits = ["train", "val"] + (
        ["test"] if args.require_test and grouped.get("test") is not None else []
    )
    common_shifts = set(grouped[required_splits[0]])
    for split in required_splits[1:]:
        common_shifts &= set(grouped[split])

    candidates: list[dict[str, Any]] = []
    for shift in sorted(common_shifts):
        split_rows = {split: grouped[split][shift][0] for split in required_splits}
        score = sum(float(row["best"][args.metric]) for row in split_rows.values())
        candidates.append(
            {
                "shift": int(shift),
                "score": score,
                "windows": split_rows,
            }
        )
    candidates.sort(key=lambda row: (float(row["score"]), int(row["shift"])))
    selected = candidates[0] if candidates else None
    return {
        "metric": args.metric,
        "required_splits": required_splits,
        "histograms": {
            split: scan.get("best_shift_histogram", {})
            for split, scan in scans.items()
            if scan is not None
        },
        "common_shifts": sorted(int(shift) for shift in common_shifts),
        "selected": selected,
        "compatible": selected is not None,
        "candidate_count": len(candidates),
        "candidates": candidates[: args.top_k],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select compatible transport windows from scan JSON files"
    )
    parser.add_argument("--train-scan", required=True)
    parser.add_argument("--val-scan", required=True)
    parser.add_argument("--test-scan")
    parser.add_argument("--require-test", action="store_true")
    parser.add_argument("--metric", choices=("mse", "nrmse"), default="nrmse")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    record = select_compatible(args)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
