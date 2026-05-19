#!/usr/bin/env python
from __future__ import annotations

"""Validate the official Advection hydration plan before any large download."""

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _download_paths(plan: Mapping[str, Any]) -> list[str]:
    commands = (plan.get("commands") or {}).get("download_official_train_files") or []
    paths: list[str] = []
    for command in commands:
        text = str(command)
        marker = "scripts/download_pdebench_file.py "
        if marker not in text:
            continue
        tail = text.split(marker, 1)[1].strip()
        if tail.startswith("'"):
            paths.append(tail.split("'", 2)[1])
        else:
            paths.append(tail.split()[0])
    return paths


def validate_plan(args: argparse.Namespace) -> dict[str, Any]:
    plan = _load_json(args.plan_json)
    blockers: list[str] = []
    warnings: list[str] = []

    selected_paths = [str(path) for path in plan.get("selected_official_advection_train_files", [])]
    download_paths = _download_paths(plan)
    if plan.get("status") != "ready_for_explicit_hydration":
        blockers.append(f"plan status is {plan.get('status')}, expected ready_for_explicit_hydration")
    if not selected_paths:
        blockers.append("plan selects no official Advection train files")
    if selected_paths != download_paths:
        blockers.append("download command paths do not match selected official Advection train files")
    non_train_paths = [path for path in selected_paths if "1D/Advection/Train/" not in path]
    if non_train_paths:
        blockers.append(f"plan includes non-train or non-Advection paths: {non_train_paths}")

    estimated_bytes = int(plan.get("estimated_download_bytes") or 0)
    if estimated_bytes < int(args.min_download_bytes):
        blockers.append(f"estimated download bytes {estimated_bytes} below expected minimum {args.min_download_bytes}")

    policy = plan.get("held_out_test_policy") or {}
    if policy.get("test_split_downloaded") is not False:
        blockers.append("held-out test policy must not download test split during hydration")
    if policy.get("test_split_sharded") is not False:
        blockers.append("held-out test policy must not shard test split during hydration")
    if policy.get("test_may_run_only_after_validation_guard") is not True:
        blockers.append("held-out test must remain gated behind validation")

    commands = plan.get("commands") or {}
    required_command_keys = [
        "download_official_train_files",
        "build_train_val_source",
        "build_light_train_val_shards",
        "validate_without_test",
        "objective_audit_after_validation",
    ]
    missing_commands = [key for key in required_command_keys if not commands.get(key)]
    if missing_commands:
        blockers.append(f"missing required commands: {missing_commands}")
    if "--test-count 0" not in str(commands.get("build_light_train_val_shards")):
        blockers.append("build_light_train_val_shards command must set --test-count 0")
    if "--test-split" in str(commands.get("validate_without_test")):
        blockers.append("validate_without_test command must not pass --test-split")
    if "REQUIRE_STATUS=literal-test-ready" not in str(commands.get("objective_audit_after_validation")):
        blockers.append("objective_audit_after_validation must require literal-test-ready, not final achievement")
    if "reports/light_experiments" in json.dumps(plan):
        blockers.append("plan must not use synthetic report artifacts")
    if "The current workspace has not performed these downloads." not in plan.get("notes", []):
        warnings.append("plan notes do not explicitly state downloads have not been performed")

    return {
        "status": "valid" if not blockers else "invalid",
        "blockers": blockers,
        "warnings": warnings,
        "plan_json": str(args.plan_json),
        "selected_file_count": len(selected_paths),
        "estimated_download_bytes": estimated_bytes,
        "estimated_download_gib": plan.get("estimated_download_gib"),
        "download_paths": download_paths,
        "held_out_test_policy": policy,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate official Advection hydration plan")
    parser.add_argument(
        "--plan-json",
        default="reports/research/sota_loop/official_advection_hydration_plan.json",
    )
    parser.add_argument("--min-download-bytes", type=int, default=1)
    parser.add_argument(
        "--output-json",
        default="reports/research/sota_loop/official_advection_hydration_plan_validation.json",
    )
    args = parser.parse_args()

    record = validate_plan(args)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))
    raise SystemExit(0 if record["status"] == "valid" else 2)


if __name__ == "__main__":
    main()
