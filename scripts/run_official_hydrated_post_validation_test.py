#!/usr/bin/env python
from __future__ import annotations

"""Run or preview the official hydrated held-out test after literal test readiness."""

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _shift_args(shifts: list[int]) -> str:
    return " ".join(f"--shift {shift}" for shift in shifts)


def _commands(args: argparse.Namespace) -> list[dict[str, str]]:
    test_start_index = int(args.train_count) + int(args.val_count)
    build_test = (
        "python scripts/make_light_hdf5_shards.py "
        f"--root {args.hydrated_source_root} "
        f"--out-root {args.hydrated_light_root} "
        "--tasks advection1d --source-split train "
        "--split-source test=train "
        f"--split-start-index test={test_start_index} "
        "--train-count 0 --val-count 0 "
        f"--test-count {args.test_count} "
        f"--manifest {args.output_root}/official_hydrated_test_manifest.yaml "
        "--overwrite"
    )
    run_gate = (
        "python scripts/run_transport_shift_gate.py "
        f"--data-root {args.hydrated_light_root} "
        "--task advection1d --train-split train --val-split val --test-split test "
        f"--max-samples {args.train_count} "
        f"--test-max-samples {args.test_count} "
        f"--rollout-steps {args.rollout_steps} "
        f"{_shift_args(args.shift)} "
        "--metric nrmse "
        f"--reference-metric-value {args.reference_metric_value} "
        f"--val-min-relative-improvement {args.val_min_relative_improvement} "
        f"--output-json {args.official_hydrated_gate_json}"
    )
    return [
        {"stage": "build_test_shard", "command": build_test},
        {"stage": "run_gated_test", "command": run_gate},
    ]


def run_post_validation_test(args: argparse.Namespace) -> dict[str, Any]:
    objective_status = _load_json(args.objective_status_json)
    commands = _commands(args)
    blockers: list[str] = []
    if objective_status.get("status") != "literal_test_ready":
        blockers.append(f"objective status is {objective_status.get('status')}, expected literal_test_ready")
    if not bool(args.execute_test):
        blockers.append("held-out test execution requires --execute-test")

    should_execute = bool(args.execute and args.execute_test and not blockers)
    executed: list[dict[str, Any]] = []
    for row in commands:
        command_record = {
            "stage": row["stage"],
            "command": row["command"],
            "executed": False,
            "returncode": None,
        }
        if should_execute:
            completed = subprocess.run(row["command"], shell=True, check=False)
            command_record["executed"] = True
            command_record["returncode"] = completed.returncode
            if completed.returncode != 0:
                blockers.append(f"{row['stage']} command failed with exit code {completed.returncode}")
                should_execute = False
        executed.append(command_record)

    status = "executed" if args.execute and not blockers else "dry_run" if not args.execute else "blocked"
    return {
        "status": status,
        "blockers": blockers,
        "objective_status_json": args.objective_status_json,
        "objective_status": objective_status.get("status"),
        "execute_requested": bool(args.execute),
        "execute_test": bool(args.execute_test),
        "held_out_test_policy": {
            "requires_literal_test_ready": True,
            "builds_test_shard": bool(args.execute and args.execute_test and not blockers),
            "test_start_index": int(args.train_count) + int(args.val_count),
            "test_count": int(args.test_count),
            "gate_measures_test_only_if_validation_passes": True,
        },
        "executed": executed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run official hydrated held-out test after literal test readiness")
    parser.add_argument("--objective-status-json", default="reports/research/sota_loop/transport_objective_status.json")
    parser.add_argument("--hydrated-source-root", default="data/pdebench_official_advection_hydrated")
    parser.add_argument("--hydrated-light-root", default="data/pdebench_official_advection_light")
    parser.add_argument("--output-root", default="reports/research/sota_loop")
    parser.add_argument(
        "--official-hydrated-gate-json",
        default="reports/research/sota_loop/official_hydrated_transport_shift_gate.json",
    )
    parser.add_argument("--train-count", type=int, default=256)
    parser.add_argument("--val-count", type=int, default=64)
    parser.add_argument("--test-count", type=int, default=32)
    parser.add_argument("--rollout-steps", type=int, default=16)
    parser.add_argument("--shift", action="append", type=int, default=None)
    parser.add_argument("--reference-metric-value", type=float, default=0.30780652221851373)
    parser.add_argument("--val-min-relative-improvement", type=float, default=0.0)
    parser.add_argument("--execute", action="store_true", help="Execute the post-validation stages")
    parser.add_argument("--execute-test", action="store_true", help="Allow creating/reading the held-out test shard")
    parser.add_argument(
        "--output-json",
        default="reports/research/sota_loop/official_hydrated_post_validation_test_run.json",
    )
    args = parser.parse_args()
    if args.shift is None:
        args.shift = list(range(-96, 97, 8))

    record = run_post_validation_test(args)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))
    raise SystemExit(0 if record["status"] in {"dry_run", "executed"} else 2)


if __name__ == "__main__":
    main()
