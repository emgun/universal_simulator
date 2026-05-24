#!/usr/bin/env python
from __future__ import annotations

"""Recommend storage roots for official Advection hydration."""

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _candidate_usage(path: str, required_free_bytes: int) -> dict[str, Any]:
    root = Path(path)
    exists = root.exists()
    probe = root if exists else root.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        usage = shutil.disk_usage(probe)
        accessible = True
        error = None
    except OSError as exc:
        usage = None
        accessible = False
        error = str(exc)
    free = int(usage.free) if usage else 0
    return {
        "path": path,
        "probe_path": str(probe),
        "exists": exists,
        "accessible": accessible,
        "error": error,
        "total_bytes": int(usage.total) if usage else None,
        "used_bytes": int(usage.used) if usage else None,
        "free_bytes": free,
        "satisfies_required_free": bool(accessible and free >= required_free_bytes),
    }


def recommend_storage(args: argparse.Namespace) -> dict[str, Any]:
    plan = _load_json(args.plan_json)
    total_download_bytes = int(plan.get("estimated_download_bytes") or 0)
    largest_file_bytes = max(
        (int(entry.get("size_bytes") or 0) for entry in plan.get("remote_entries", [])),
        default=total_download_bytes,
    )
    mode = str(getattr(args, "mode", "all"))
    if mode == "sequential":
        required_download_bytes = largest_file_bytes
    elif mode == "all":
        required_download_bytes = total_download_bytes
    else:
        raise ValueError(f"Unsupported storage recommendation mode: {mode}")
    required_free_bytes = int(required_download_bytes * float(args.safety_factor))
    candidate_paths = args.candidate_root or [
        str(plan.get("raw_out") or "data/pdebench/raw"),
        "/private/tmp",
        "/Volumes",
    ]
    candidates = [_candidate_usage(path, required_free_bytes) for path in candidate_paths]
    viable = [row for row in candidates if row["satisfies_required_free"]]
    status = "storage_root_available" if viable else "external_or_freed_space_required"
    return {
        "status": status,
        "blockers": (
            []
            if viable
            else [
                f"no candidate root has required free bytes {required_free_bytes}",
                "free local disk space or provide a larger mounted volume and regenerate the hydration plan with that root",
            ]
        ),
        "plan_json": str(args.plan_json),
        "remaining_download_bytes": total_download_bytes,
        "largest_file_bytes": largest_file_bytes,
        "required_download_bytes": required_download_bytes,
        "required_free_bytes": required_free_bytes,
        "mode": mode,
        "safety_factor": float(args.safety_factor),
        "recommended_root": viable[0]["path"] if viable else None,
        "candidates": candidates,
        "example_replan_command": (
            "python scripts/plan_transport_official_hydration.py "
            "--raw-out /Volumes/<large-volume>/pdebench/raw "
            "--hydrated-source-root /Volumes/<large-volume>/pdebench/official_advection_hydrated "
            "--hydrated-light-root /Volumes/<large-volume>/pdebench/official_advection_light"
        ),
        "notes": [
            "This audit does not create directories or download data.",
            "The held-out test policy remains governed by the hydration plan and gated transport runner.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recommend storage for official Advection hydration"
    )
    parser.add_argument(
        "--plan-json", default="reports/research/sota_loop/official_advection_hydration_plan.json"
    )
    parser.add_argument("--candidate-root", action="append", default=None)
    parser.add_argument("--safety-factor", type=float, default=1.15)
    parser.add_argument(
        "--mode",
        choices=("all", "sequential"),
        default="sequential",
        help="Recommend storage for all raw files or the largest file needed by sequential hydration.",
    )
    parser.add_argument(
        "--output-json",
        default="reports/research/sota_loop/official_advection_hydration_storage_recommendation.json",
    )
    args = parser.parse_args()

    record = recommend_storage(args)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))
    raise SystemExit(0 if record["status"] == "storage_root_available" else 2)


if __name__ == "__main__":
    main()
