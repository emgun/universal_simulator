#!/usr/bin/env python
from __future__ import annotations

"""Calibrate decoded periodic roll-shift transport corrections on validation data."""

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.calibrate_residual_gate import _relative_improvement, _select_best, _test_guard_result


def _safe_shift(shift: int) -> str:
    return str(int(shift)).replace("-", "m")


def _shift_override(kind: str, key: str, shift: int) -> str:
    if kind == "family":
        return f"evaluation.decoded_roll_shift_by_family={{{json.dumps(key)}:{int(shift)}}}"
    if kind == "task":
        return f"evaluation.decoded_roll_shift_by_task={{{json.dumps(key)}:{int(shift)}}}"
    raise ValueError(f"Unsupported roll-shift kind: {kind}")


def _schedule_override(kind: str, key: str, schedule: Mapping[int, int]) -> str:
    payload = {key: {str(int(horizon)): int(shift) for horizon, shift in sorted(schedule.items())}}
    encoded = json.dumps(payload, separators=(",", ":"))
    if kind == "family":
        return f"evaluation.decoded_roll_shift_by_family_horizon={encoded}"
    if kind == "task":
        return f"evaluation.decoded_roll_shift_by_task_horizon={encoded}"
    raise ValueError(f"Unsupported roll-shift kind: {kind}")


def _candidate_shifts(values: Sequence[int] | None) -> list[int]:
    if values:
        return [int(value) for value in values]
    return [-4, -2, -1, 0, 1, 2, 4, 8, 16, 24, 32, 40, 48, 64]


def _read_metrics(summary_path: Path) -> dict[str, float]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = summary.get("metrics", {})
    if not isinstance(metrics, Mapping):
        return {}
    return {
        str(key): float(value) for key, value in metrics.items() if isinstance(value, (int, float))
    }


def _run_light_eval(
    *,
    args: argparse.Namespace,
    name: str,
    output_root: str,
    split: str,
    shift_override: str,
    report_all_horizon_metrics: bool = False,
) -> Path:
    summary_path = Path(output_root) / name / "summary.json"
    if args.reuse_existing and summary_path.exists():
        return summary_path
    cmd = [
        sys.executable,
        "scripts/run_light_experiment.py",
        "--config",
        args.config,
        "--name",
        name,
        "--output-root",
        output_root,
        "--checkpoint-source",
        args.checkpoint_source,
        "--skip-training",
        "--device",
        args.device,
        "--decoded",
        "--override",
        f"data.root={args.data_root}",
        "--eval-override",
        f"data.root={args.data_root}",
        "--eval-override",
        f"data.split={split}",
        "--decoded-rollout-steps",
        str(args.decoded_rollout_steps),
        "--override",
        f"evaluation.decoded_persistence_residual_alpha={args.default_alpha:g}",
        "--override",
        shift_override,
    ]
    if report_all_horizon_metrics:
        cmd.extend(["--override", "evaluation.report_all_horizon_metrics=true"])
    if args.eval_max_samples is not None:
        cmd.extend(["--eval-override", f"data.max_samples={args.eval_max_samples}"])
    for override in args.override:
        cmd.extend(["--override", override])
    for override in args.eval_override:
        cmd.extend(["--eval-override", override])
    for rule in args.promotion_rule:
        cmd.extend(["--promotion-rule", rule])
    subprocess.run(cmd, check=True)
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate decoded periodic roll-shift correction on validation data"
    )
    parser.add_argument("--config", default="configs/train_multitask_heterogeneous_light_best.yaml")
    parser.add_argument("--checkpoint-source", required=True)
    parser.add_argument("--data-root", default="data/pdebench")
    parser.add_argument("--output-root", default="reports/research/sota_loop/transport_shift_sweep")
    parser.add_argument("--test-output-root", default=None)
    parser.add_argument("--run-prefix", default="ups_light_advection_roll")
    parser.add_argument("--kind", choices=("family", "task"), default="task")
    parser.add_argument("--key", default="advection1d")
    parser.add_argument("--default-alpha", type=float, default=0.0)
    parser.add_argument(
        "--shift", action="append", type=int, default=None, help="Candidate shift; repeatable"
    )
    parser.add_argument(
        "--schedule-by-horizon",
        action="store_true",
        help="Select a validation-best shift per rollout horizon",
    )
    parser.add_argument(
        "--schedule-min-relative-improvement",
        type=float,
        default=0.01,
        help="Minimum aggregate validation improvement required to select a horizon schedule over the best constant shift",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse existing run summary files instead of rerunning them",
    )
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--metric", default="decoded_rollout_nrmse")
    parser.add_argument("--mode", choices=("min", "max"), default="min")
    parser.add_argument("--reference-metric-value", type=float)
    parser.add_argument("--test-min-relative-improvement", type=float)
    parser.add_argument("--eval-max-samples", type=int, default=32)
    parser.add_argument("--decoded-rollout-steps", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--eval-override", action="append", default=[])
    parser.add_argument("--promotion-rule", action="append", default=["decoded_rollout_nrmse<=1.0"])
    parser.add_argument("--output-json", help="Calibration record path; defaults under output root")
    parser.add_argument(
        "--export-selected-shift-config",
        help="Optional JSON path for frozen selected override payload",
    )
    args = parser.parse_args()

    shifts = _candidate_shifts(args.shift)
    rows: list[dict[str, Any]] = []
    for shift in shifts:
        name = f"{args.run_prefix}_shift_{_safe_shift(shift)}_{args.val_split}"
        summary_path = _run_light_eval(
            args=args,
            name=name,
            output_root=args.output_root,
            split=args.val_split,
            shift_override=_shift_override(args.kind, args.key, shift),
            report_all_horizon_metrics=args.schedule_by_horizon,
        )
        metrics = _read_metrics(summary_path)
        rows.append(
            {
                "shift": int(shift),
                "run_name": name,
                "summary": str(summary_path),
                args.metric: metrics[args.metric],
                "metrics": metrics,
            }
        )

    best = _select_best(rows, metric=args.metric, mode=args.mode)
    schedule: dict[int, int] | None = None
    schedule_selections: list[dict[str, Any]] | None = None
    if args.schedule_by_horizon:
        schedule, schedule_selections = _select_horizon_schedule(
            rows, kind=args.kind, key=args.key, mode=args.mode
        )

    record: dict[str, Any] = {
        "config": args.config,
        "checkpoint_source": args.checkpoint_source,
        "data_root": args.data_root,
        "kind": args.kind,
        "key": args.key,
        "default_alpha": args.default_alpha,
        "metric": args.metric,
        "mode": args.mode,
        "val_split": args.val_split,
        "test_split": args.test_split,
        "validation": rows,
        "best_validation": best,
    }
    if schedule is not None and schedule_selections is not None:
        record["best_validation_schedule"] = {
            "schedule": {str(horizon): shift for horizon, shift in sorted(schedule.items())},
            "selections": schedule_selections,
        }
        schedule_val_name = f"{args.run_prefix}_horizon_schedule_{args.val_split}"
        schedule_val_summary = _run_light_eval(
            args=args,
            name=schedule_val_name,
            output_root=args.output_root,
            split=args.val_split,
            shift_override=_schedule_override(args.kind, args.key, schedule),
            report_all_horizon_metrics=True,
        )
        schedule_val_metrics = _read_metrics(schedule_val_summary)
        record["validation_schedule"] = {
            "run_name": schedule_val_name,
            "summary": str(schedule_val_summary),
            args.metric: schedule_val_metrics[args.metric],
            "metrics": schedule_val_metrics,
        }
        record["schedule_min_relative_improvement"] = args.schedule_min_relative_improvement
        record["schedule_relative_improvement"] = _relative_improvement(
            float(schedule_val_metrics[args.metric]),
            float(best[args.metric]),
            mode=args.mode,
        )

    validation_schedule = record.get("validation_schedule")
    use_schedule = (
        schedule is not None
        and validation_schedule is not None
        and float(record.get("schedule_relative_improvement", 0.0))
        >= args.schedule_min_relative_improvement
    )
    if use_schedule:
        selected_override = _schedule_override(args.kind, args.key, schedule)
        record["selected_validation_shift"] = {
            "selected_shift": "horizon_schedule",
            "schedule": {str(horizon): shift for horizon, shift in sorted(schedule.items())},
            "overrides": [selected_override],
            "validation": validation_schedule,
        }
    else:
        selected_override = _shift_override(args.kind, args.key, int(best["shift"]))
        record["selected_validation_shift"] = {
            "selected_shift": int(best["shift"]),
            "overrides": [selected_override],
            "validation": best,
        }
    selected_metric_value = float(record["selected_validation_shift"]["validation"][args.metric])
    record["selected_validation_shift"]["test_guard"] = _test_guard_result(
        value=selected_metric_value,
        reference=args.reference_metric_value,
        min_relative_improvement=args.test_min_relative_improvement,
        mode=args.mode,
    )

    test_output_root = args.test_output_root or args.output_root
    if args.skip_test:
        record["test_skipped"] = {"reason": "--skip-test"}
    elif not record["selected_validation_shift"]["test_guard"]["passed"]:
        record["test_skipped"] = {
            "reason": "selected validation shift did not pass held-out test guard"
        }
    else:
        if use_schedule:
            test_name = f"{args.run_prefix}_horizon_schedule_{args.test_split}"
            test_shift: int | dict[str, int] = record["selected_validation_shift"]["schedule"]
        else:
            test_name = (
                f"{args.run_prefix}_shift_{_safe_shift(int(best['shift']))}_{args.test_split}"
            )
            test_shift = int(best["shift"])
        test_summary = _run_light_eval(
            args=args,
            name=test_name,
            output_root=test_output_root,
            split=args.test_split,
            shift_override=selected_override,
            report_all_horizon_metrics=use_schedule,
        )
        test_metrics = _read_metrics(test_summary)
        record["test"] = {
            "shift": test_shift,
            "run_name": test_name,
            "summary": str(test_summary),
            args.metric: test_metrics[args.metric],
            "metrics": test_metrics,
        }

    output_path = (
        Path(args.output_json)
        if args.output_json
        else Path(args.output_root) / f"{args.run_prefix}_calibration.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    if args.export_selected_shift_config:
        export_path = Path(args.export_selected_shift_config)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text(
            json.dumps(record["selected_validation_shift"], indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(json.dumps(record, indent=2, sort_keys=True))


def _horizon_metric_pattern(kind: str, key: str) -> str:
    if kind == "family":
        return f"family_{key}_decoded_h"
    if kind == "task":
        return f"task_{key}_decoded_h"
    raise ValueError(f"Unsupported roll-shift kind: {kind}")


def _select_horizon_schedule(
    rows: Sequence[dict[str, Any]],
    *,
    kind: str,
    key: str,
    mode: str,
) -> tuple[dict[int, int], list[dict[str, Any]]]:
    prefix = _horizon_metric_pattern(kind, key)
    candidates: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        for metric_name, value in row.get("metrics", {}).items():
            text = str(metric_name)
            if not text.startswith(prefix) or not text.endswith("_nrmse"):
                continue
            horizon_text = text[len(prefix) : -len("_nrmse")]
            if not horizon_text.isdigit():
                continue
            candidates.setdefault(int(horizon_text), []).append(
                {
                    "horizon": int(horizon_text),
                    "shift": int(row["shift"]),
                    "run_name": row["run_name"],
                    "summary": row["summary"],
                    "metric": metric_name,
                    "value": float(value),
                }
            )
    if not candidates:
        raise ValueError(f"No horizon metrics found for {kind} '{key}'")
    reverse = mode == "max"
    selections = [
        sorted(rows, key=lambda row: row["value"], reverse=reverse)[0]
        for _, rows in sorted(candidates.items())
    ]
    return {int(row["horizon"]): int(row["shift"]) for row in selections}, selections


if __name__ == "__main__":
    main()
