#!/usr/bin/env python
from __future__ import annotations

"""Preflight local disk and raw-file state before official Advection hydration."""

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _raw_file_path(raw_out: str | Path, logical_path: str) -> Path:
    return Path(raw_out) / logical_path


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    plan = _load_json(args.plan_json)
    raw_out = Path(str(plan.get("raw_out") or args.raw_out))
    selected = [str(path) for path in plan.get("selected_official_advection_train_files", [])]
    size_by_path = {
        str(entry.get("path")): int(entry.get("size_bytes") or 0)
        for entry in plan.get("remote_entries", [])
        if entry.get("path")
    }
    if not size_by_path:
        # Plans before remote_entries existed still carry the aggregate size. Split only for reporting.
        aggregate = int(plan.get("estimated_download_bytes") or 0)
        per_file = aggregate // max(1, len(selected))
        size_by_path = {path: per_file for path in selected}

    file_rows = []
    remaining_bytes = 0
    largest_missing_file_bytes = 0
    for logical_path in selected:
        path = _raw_file_path(raw_out, logical_path)
        expected_size = int(size_by_path.get(logical_path) or 0)
        exists = path.exists()
        actual_size = path.stat().st_size if exists else 0
        complete = exists and expected_size > 0 and actual_size == expected_size
        if not complete:
            missing_bytes = max(expected_size - actual_size, 0)
            remaining_bytes += missing_bytes
            largest_missing_file_bytes = max(largest_missing_file_bytes, missing_bytes)
        file_rows.append(
            {
                "logical_path": logical_path,
                "local_path": str(path),
                "exists": exists,
                "expected_size_bytes": expected_size,
                "actual_size_bytes": actual_size,
                "complete": complete,
            }
        )

    disk_root = Path(args.disk_root or raw_out)
    disk_root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(disk_root)
    mode = str(getattr(args, "mode", "all"))
    if mode == "sequential":
        required_download_bytes = largest_missing_file_bytes
    elif mode == "all":
        required_download_bytes = remaining_bytes
    else:
        raise ValueError(f"Unsupported preflight mode: {mode}")
    required_free_bytes = int(required_download_bytes * float(args.safety_factor))
    blockers = []
    if remaining_bytes <= 0:
        status = "ready_raw_files_present"
    elif usage.free < required_free_bytes:
        status = "blocked_insufficient_disk"
        blockers.append(
            f"available bytes {usage.free} below required {required_free_bytes} "
            f"for {mode} download bytes {required_download_bytes} with safety factor {args.safety_factor}"
        )
    else:
        status = "ready_for_sequential_download" if mode == "sequential" else "ready_for_download"

    return {
        "status": status,
        "blockers": blockers,
        "plan_json": str(args.plan_json),
        "raw_out": str(raw_out),
        "disk_root": str(disk_root),
        "disk": {
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
        },
        "selected_file_count": len(selected),
        "complete_file_count": sum(1 for row in file_rows if row["complete"]),
        "remaining_download_bytes": remaining_bytes,
        "largest_missing_file_bytes": largest_missing_file_bytes,
        "required_download_bytes": required_download_bytes,
        "required_free_bytes": required_free_bytes,
        "mode": mode,
        "safety_factor": float(args.safety_factor),
        "files": file_rows,
        "held_out_test_policy": plan.get("held_out_test_policy"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preflight official Advection hydration disk state"
    )
    parser.add_argument(
        "--plan-json", default="reports/research/sota_loop/official_advection_hydration_plan.json"
    )
    parser.add_argument("--raw-out", default="data/pdebench/raw")
    parser.add_argument("--disk-root")
    parser.add_argument("--safety-factor", type=float, default=1.15)
    parser.add_argument(
        "--mode",
        choices=("all", "sequential"),
        default="sequential",
        help="Require space for all remaining files or only the largest missing file for sequential hydration.",
    )
    parser.add_argument(
        "--output-json",
        default="reports/research/sota_loop/official_advection_hydration_preflight.json",
    )
    args = parser.parse_args()

    record = preflight(args)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))
    raise SystemExit(0 if not record["blockers"] else 2)


if __name__ == "__main__":
    main()
