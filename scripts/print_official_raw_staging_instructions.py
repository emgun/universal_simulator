#!/usr/bin/env python
from __future__ import annotations

"""Print exact staged-raw requirements for official Advection hydration."""

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _readiness_files(readiness: dict[str, Any]) -> dict[str, dict[str, Any]]:
    staged_raw = readiness.get("staged_raw") or {}
    return {str(row.get("logical_path") or ""): dict(row) for row in staged_raw.get("files") or []}


def _file_record(raw_root: Path, entry: dict[str, Any], readiness_by_path: dict[str, dict[str, Any]]) -> dict[str, Any]:
    logical_path = str(entry.get("path") or "")
    ready_row = readiness_by_path.get(logical_path, {})
    return {
        "logical_path": logical_path,
        "local_path": str(ready_row.get("local_path") or raw_root / logical_path),
        "file_id": entry.get("file_id"),
        "expected_size_bytes": int(ready_row.get("expected_size_bytes") or entry.get("size_bytes") or 0),
        "actual_size_bytes": int(ready_row.get("actual_size_bytes") or 0),
        "checksum_type": str(ready_row.get("checksum_type") or entry.get("checksum_type") or "md5").lower(),
        "expected_checksum": ready_row.get("expected_checksum") or entry.get("checksum"),
        "actual_checksum": ready_row.get("actual_checksum"),
        "exists": bool(ready_row.get("exists", False)),
        "checksum_matches": bool(ready_row.get("checksum_matches", False)),
        "complete": bool(ready_row.get("complete", False)),
    }


def _next_command(plan_json: str, run_json: str) -> str:
    return (
        "SEQUENTIAL_HYDRATION=1 SEQUENTIAL_USE_EXISTING_RAW=1 EXECUTE=1 EXECUTE_DOWNLOADS=0 "
        f"PLAN_JSON={plan_json} SEQUENTIAL_HYDRATION_JSON={run_json} "
        "bash scripts/run_remote_official_hydration.sh"
    )


def build_staging_instructions(args: argparse.Namespace) -> dict[str, Any]:
    plan = _load_json(args.plan_json)
    readiness = _load_json(args.readiness_json)
    raw_root = Path(args.raw_out or plan.get("raw_out") or "data/pdebench/raw")
    entries = list(plan.get("remote_entries") or [])
    readiness_by_path = _readiness_files(readiness)
    files = [_file_record(raw_root, entry, readiness_by_path) for entry in entries]
    complete_count = sum(1 for row in files if row["complete"])
    selected_count = len(files)
    all_complete = bool(selected_count and complete_count == selected_count)

    return {
        "status": "ready_for_existing_raw_hydration" if all_complete else "needs_staging",
        "plan_json": str(args.plan_json),
        "readiness_json": str(args.readiness_json),
        "raw_root": str(raw_root),
        "selected_file_count": selected_count,
        "complete_file_count": complete_count,
        "missing_or_incomplete_file_count": selected_count - complete_count,
        "local_sequential_required_bytes": (readiness.get("disk") or {}).get("local_sequential_required_bytes"),
        "local_route_blockers": (readiness.get("route_blockers") or {}).get("local_sequential_hydration", []),
        "files": files,
        "next_command": _next_command(str(args.plan_json), str(args.run_json)),
        "held_out_test_policy": plan.get("held_out_test_policy"),
    }


def _print_human(record: dict[str, Any]) -> None:
    print(f"status: {record['status']}")
    print(f"raw_root: {record['raw_root']}")
    print(
        "files: "
        f"{record['complete_file_count']}/{record['selected_file_count']} complete "
        f"({record['missing_or_incomplete_file_count']} missing or incomplete)"
    )
    if record.get("local_sequential_required_bytes") is not None:
        print(f"sequential_required_bytes: {record['local_sequential_required_bytes']}")
    blockers = record.get("local_route_blockers") or []
    if blockers:
        print("local_route_blockers:")
        for blocker in blockers:
            print(f"  - {blocker}")
    print("required_raw_files:")
    for row in record["files"]:
        marker = "complete" if row["complete"] else "needed"
        checksum = row["expected_checksum"] or "not provided"
        print(f"  - {marker}: {row['local_path']}")
        print(f"    size_bytes: {row['expected_size_bytes']}")
        print(f"    {row['checksum_type']}: {checksum}")
    print("next_command:")
    print(f"  {record['next_command']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Print official raw staging instructions")
    parser.add_argument("--plan-json", default="reports/research/sota_loop/official_advection_hydration_plan.json")
    parser.add_argument(
        "--readiness-json",
        default="reports/research/sota_loop/official_execution_readiness.json",
    )
    parser.add_argument(
        "--run-json",
        default="reports/research/sota_loop/official_advection_sequential_hydration_run.json",
    )
    parser.add_argument("--raw-out")
    parser.add_argument("--output-json")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of a checklist.")
    args = parser.parse_args()

    record = build_staging_instructions(args)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    if args.json:
        print(json.dumps(record, indent=2, sort_keys=True))
    else:
        _print_human(record)
    raise SystemExit(0 if record["status"] == "ready_for_existing_raw_hydration" else 2)


if __name__ == "__main__":
    main()
