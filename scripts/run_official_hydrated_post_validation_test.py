#!/usr/bin/env python
from __future__ import annotations

"""Run or preview the official hydrated held-out test after literal test readiness."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_test_ledger(path: str | None) -> dict[str, Any]:
    if not path:
        return {"measurements": []}
    ledger_path = Path(path)
    if not ledger_path.exists():
        return {"measurements": []}
    return json.loads(ledger_path.read_text(encoding="utf-8"))


def _write_test_ledger(path: str | None, ledger: dict[str, Any]) -> None:
    if not path:
        return
    ledger_path = Path(path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True), encoding="utf-8")


def _shift_args(shifts: list[int]) -> str:
    return " ".join(f"--shift {shift}" for shift in shifts)


def _test_measurement_key(args: argparse.Namespace, objective_status: dict[str, Any]) -> str:
    payload = {
        "estimator": "official_hydrated_source_conditioned_transport_shift",
        "fit_strategy": args.fit_strategy,
        "hydrated_light_root": args.hydrated_light_root,
        "hydrated_source_root": args.hydrated_source_root,
        "objective_status": objective_status.get("status"),
        "official_hydrated_gate_json": args.official_hydrated_gate_json,
        "reference_metric_value": args.reference_metric_value,
        "fractional_refine_step": float(args.fractional_refine_step),
        "refine_radius": int(args.refine_radius),
        "rollout_steps": args.rollout_steps,
        "shift": [int(shift) for shift in args.shift],
        "split_block_size": int(getattr(args, "split_block_size", 0)),
        "test_block_offset": int(getattr(args, "test_block_offset", 0)),
        "test_count": int(args.test_count),
        "train_count": int(args.train_count),
        "val_count": int(args.val_count),
        "val_min_relative_improvement": args.val_min_relative_improvement,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _gate_test_result_count(path: str | Path) -> int:
    gate_path = Path(path)
    if not gate_path.exists():
        return 0
    gate = _load_json(gate_path)
    test_payload = gate.get("test")
    if not test_payload:
        return 0
    if isinstance(test_payload, list):
        return len(test_payload)
    return 1


def _commands(args: argparse.Namespace) -> list[dict[str, str]]:
    test_start_index = int(args.train_count) + int(args.val_count)
    split_block_size = int(
        getattr(args, "split_block_size", 0) or test_start_index + int(args.test_count)
    )
    test_block_offset = int(getattr(args, "test_block_offset", test_start_index))
    build_test = (
        "python scripts/make_light_hdf5_shards.py "
        f"--root {args.hydrated_source_root} "
        f"--out-root {args.hydrated_light_root} "
        "--tasks advection1d --source-split train "
        "--split-source test=train "
        f"--split-block-size {split_block_size} "
        f"--split-block-offset test={test_block_offset} "
        "--train-count 0 --val-count 0 "
        f"--test-count {args.test_count} "
        f"--manifest {args.output_root}/official_hydrated_test_manifest.yaml "
        "--overwrite"
    )
    run_gate = (
        "python scripts/run_source_conditioned_transport_shift_gate.py "
        f"--data-root {args.hydrated_light_root} "
        "--task advection1d --train-split train --val-split val --test-split test "
        f"--max-samples {args.train_count} "
        f"--test-max-samples {args.test_count} "
        f"--rollout-steps {args.rollout_steps} "
        f"{_shift_args(args.shift)} "
        "--metric nrmse "
        f"--fit-strategy {args.fit_strategy} "
        f"--refine-radius {args.refine_radius} "
        f"--fractional-refine-step {args.fractional_refine_step} "
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
    measurement_key = _test_measurement_key(args, objective_status)
    test_start_index = int(args.train_count) + int(args.val_count)
    split_block_size = int(
        getattr(args, "split_block_size", 0) or test_start_index + int(args.test_count)
    )
    test_block_offset = int(getattr(args, "test_block_offset", test_start_index))
    blockers: list[str] = []
    if objective_status.get("status") != "literal_test_ready":
        blockers.append(
            f"objective status is {objective_status.get('status')}, expected literal_test_ready"
        )
    if not bool(args.execute_test):
        blockers.append("held-out test execution requires --execute-test")
    ledger_path = getattr(args, "test_ledger_json", None)
    allow_repeat_test = bool(getattr(args, "allow_repeat_test", False))
    test_ledger_recorded = False
    if args.execute_test and ledger_path:
        ledger = _load_test_ledger(ledger_path)
        existing_keys = {
            str(entry.get("measurement_key"))
            for entry in ledger.get("measurements", [])
            if isinstance(entry, dict)
        }
        if measurement_key in existing_keys and not allow_repeat_test:
            blockers.append(
                "held-out test measurement already recorded for this official hydrated test configuration"
            )

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
                blockers.append(
                    f"{row['stage']} command failed with exit code {completed.returncode}"
                )
                should_execute = False
        executed.append(command_record)

    test_result_count = 0
    if args.execute and args.execute_test and not blockers:
        test_result_count = _gate_test_result_count(args.official_hydrated_gate_json)
        if test_result_count != 1:
            blockers.append(
                f"gated test command recorded {test_result_count} held-out test results; expected exactly 1"
            )
        elif ledger_path and not allow_repeat_test:
            ledger = _load_test_ledger(ledger_path)
            ledger.setdefault("measurements", []).append(
                {
                    "measurement_key": measurement_key,
                    "official_hydrated_gate_json": args.official_hydrated_gate_json,
                    "test_result_count": test_result_count,
                }
            )
            _write_test_ledger(ledger_path, ledger)
            test_ledger_recorded = True

    status = (
        "executed"
        if args.execute and not blockers
        else "dry_run" if not args.execute else "blocked"
    )
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
            "ledger_path": ledger_path,
            "measurement_key": measurement_key,
            "allow_repeat_test": allow_repeat_test,
            "ledger_recorded": test_ledger_recorded,
            "test_result_count": test_result_count,
            "test_start_index": test_start_index,
            "split_block_size": split_block_size,
            "test_block_offset": test_block_offset,
            "test_count": int(args.test_count),
            "gate_measures_test_only_if_validation_passes": True,
        },
        "executed": executed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run official hydrated held-out test after literal test readiness"
    )
    parser.add_argument(
        "--objective-status-json",
        default="reports/research/sota_loop/transport_objective_status.json",
    )
    parser.add_argument(
        "--hydrated-source-root", default="data/pdebench_official_advection_hydrated"
    )
    parser.add_argument("--hydrated-light-root", default="data/pdebench_official_advection_light")
    parser.add_argument("--output-root", default="reports/research/sota_loop")
    parser.add_argument(
        "--official-hydrated-gate-json",
        default="reports/research/sota_loop/official_hydrated_transport_shift_gate.json",
    )
    parser.add_argument("--train-count", type=int, default=256)
    parser.add_argument("--val-count", type=int, default=64)
    parser.add_argument("--test-count", type=int, default=64)
    parser.add_argument("--split-block-size", type=int, default=48)
    parser.add_argument("--test-block-offset", type=int, default=40)
    parser.add_argument("--rollout-steps", type=int, default=16)
    parser.add_argument("--shift", action="append", type=int, default=None)
    parser.add_argument("--reference-metric-value", type=float, default=0.30780652221851373)
    parser.add_argument("--val-min-relative-improvement", type=float, default=0.0)
    parser.add_argument(
        "--fit-strategy", choices=("aggregate", "sample_mode"), default="sample_mode"
    )
    parser.add_argument("--refine-radius", type=int, default=4)
    parser.add_argument("--fractional-refine-step", type=float, default=0.5)
    parser.add_argument(
        "--test-ledger-json",
        default="reports/research/sota_loop/official_hydrated_transport_shift_test_ledger.json",
        help="Ledger that prevents measuring the same official hydrated held-out test more than once",
    )
    parser.add_argument(
        "--allow-repeat-test", action="store_true", help="Bypass the held-out test ledger guard"
    )
    parser.add_argument("--execute", action="store_true", help="Execute the post-validation stages")
    parser.add_argument(
        "--execute-test", action="store_true", help="Allow creating/reading the held-out test shard"
    )
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
