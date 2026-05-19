#!/usr/bin/env python
from __future__ import annotations

"""Run or preview the official Advection hydration plan in explicit stages."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.validate_transport_hydration_plan import validate_plan


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _normalize_commands(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _stage_commands(plan: dict[str, Any], stages: Sequence[str]) -> list[dict[str, Any]]:
    commands = plan.get("commands") or {}
    stage_map = {
        "download": _normalize_commands(commands.get("download_official_train_files")),
        "convert": _normalize_commands(commands.get("build_train_val_source")),
        "shard": _normalize_commands(commands.get("build_light_train_val_shards")),
        "validate": _normalize_commands(commands.get("validate_without_test")),
        "audit": _normalize_commands(commands.get("objective_audit_after_validation")),
    }
    return [{"stage": stage, "commands": stage_map[stage]} for stage in stages]


def run_plan(args: argparse.Namespace) -> dict[str, Any]:
    validation = validate_plan(
        argparse.Namespace(
            plan_json=args.plan_json,
            min_download_bytes=args.min_download_bytes,
            output_json=args.validation_json,
        )
    )
    if validation["status"] != "valid":
        return {
            "status": "invalid_plan",
            "blockers": validation["blockers"],
            "validation": validation,
            "executed": [],
        }

    plan = _load_json(args.plan_json)
    stages = args.stage or ["download", "convert", "shard", "validate", "audit"]
    stage_records = _stage_commands(plan, stages)
    blockers: list[str] = []
    if "download" in stages and not args.execute_downloads:
        blockers.append("download stage requires --execute-downloads")
    if "validate" in stages and "--test-split" in " ".join(
        command for record in stage_records for command in record["commands"]
    ):
        blockers.append("validation stage must not pass --test-split")

    should_execute = bool(args.execute and not blockers)
    executed: list[dict[str, Any]] = []
    for record in stage_records:
        for command in record["commands"]:
            command_record = {
                "stage": record["stage"],
                "command": command,
                "executed": False,
                "returncode": None,
            }
            if should_execute:
                completed = subprocess.run(command, shell=True, check=False)
                command_record["executed"] = True
                command_record["returncode"] = completed.returncode
                if completed.returncode != 0:
                    blockers.append(f"{record['stage']} command failed with exit code {completed.returncode}")
                    should_execute = False
            executed.append(command_record)

    status = "executed" if args.execute and not blockers else "dry_run" if not args.execute else "blocked"
    return {
        "status": status,
        "blockers": blockers,
        "plan_json": args.plan_json,
        "validation": validation,
        "execute_requested": bool(args.execute),
        "execute_downloads": bool(args.execute_downloads),
        "stages": stages,
        "executed": executed,
        "held_out_test_policy": plan.get("held_out_test_policy"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or preview official Advection hydration stages")
    parser.add_argument("--plan-json", default="reports/research/sota_loop/official_advection_hydration_plan.json")
    parser.add_argument(
        "--validation-json",
        default="reports/research/sota_loop/official_advection_hydration_plan_validation.json",
    )
    parser.add_argument("--min-download-bytes", type=int, default=1)
    parser.add_argument("--stage", action="append", choices=("download", "convert", "shard", "validate", "audit"))
    parser.add_argument("--execute", action="store_true", help="Execute non-download stages")
    parser.add_argument("--execute-downloads", action="store_true", help="Allow the large official download stage")
    parser.add_argument(
        "--output-json",
        default="reports/research/sota_loop/official_advection_hydration_plan_run.json",
    )
    args = parser.parse_args()

    record = run_plan(args)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))
    raise SystemExit(0 if record["status"] in {"dry_run", "executed"} else 2)


if __name__ == "__main__":
    main()
