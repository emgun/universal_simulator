#!/usr/bin/env python
from __future__ import annotations

"""Calibrate decoded persistence-residual gates on validation data, then test once."""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _safe_alpha(alpha: float) -> str:
    return f"{alpha:g}".replace("-", "m").replace(".", "p")


def _alpha_override(kind: str, key: str, alpha: float) -> str:
    if kind == "global":
        return f"evaluation.decoded_persistence_residual_alpha={alpha:g}"
    if kind == "family":
        return f"evaluation.decoded_persistence_residual_alpha_by_family={{{json.dumps(key)}:{alpha:g}}}"
    if kind == "task":
        return (
            f"evaluation.decoded_persistence_residual_alpha_by_task={{{json.dumps(key)}:{alpha:g}}}"
        )
    raise ValueError(f"Unsupported gate kind: {kind}")


def _schedule_override(kind: str, key: str, schedule: dict[int, float]) -> str:
    if kind == "global":
        encoded = json.dumps(
            {str(int(horizon)): float(alpha) for horizon, alpha in sorted(schedule.items())},
            separators=(",", ":"),
        )
        return f"evaluation.decoded_persistence_residual_alpha_by_horizon={encoded}"
    payload = {
        key: {str(int(horizon)): float(alpha) for horizon, alpha in sorted(schedule.items())}
    }
    encoded = json.dumps(payload, separators=(",", ":"))
    if kind == "family":
        return f"evaluation.decoded_persistence_residual_alpha_by_family_horizon={encoded}"
    if kind == "task":
        return f"evaluation.decoded_persistence_residual_alpha_by_task_horizon={encoded}"
    raise ValueError(f"Unsupported gate kind: {kind}")


def _gate_config_override(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, separators=(",", ":"), sort_keys=True)
    return f"evaluation.decoded_persistence_residual_gate={encoded}"


def _parse_json_mapping(text: str, *, setting: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{setting} must be valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{setting} must decode to a JSON object")
    return payload


def _parse_float_map(values: Sequence[str], *, setting: str) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{setting} entries must be name=value")
        key, raw = value.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"{setting} entries require a non-empty key")
        parsed[key] = float(raw)
    return parsed


def _merge_gate_config(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            nested = dict(merged[key])
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = value
    return merged


def _gate_candidates(args: argparse.Namespace) -> list[dict[str, Any] | None]:
    if not args.use_decoded_residual_gate and not args.gate_config_candidate:
        return [None]
    base: dict[str, Any] = {
        "min_alpha": args.gate_min_alpha,
        "max_alpha": args.gate_max_alpha,
    }
    if args.gate_bias != 0.0:
        base["bias"] = args.gate_bias
    feature_weights = _parse_float_map(args.gate_feature_weight, setting="--gate-feature-weight")
    if feature_weights:
        base["feature_weights"] = feature_weights
    raw_candidates = args.gate_config_candidate or ["{}"]
    return [
        _merge_gate_config(base, _parse_json_mapping(raw, setting="--gate-config-candidate"))
        for raw in raw_candidates
    ]


def _gate_suffix(index: int, total: int, gate_config: dict[str, Any] | None) -> str:
    if gate_config is None:
        return ""
    return "_gate" if total == 1 else f"_gate{index}"


def _gate_overrides(primary_override: str, gate_config: dict[str, Any] | None) -> list[str]:
    overrides = [primary_override]
    if gate_config is not None:
        overrides.append(_gate_config_override(gate_config))
    return overrides


def _read_metrics(summary_path: Path) -> dict[str, float]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = summary.get("metrics", {})
    return {
        str(key): float(value) for key, value in metrics.items() if isinstance(value, (int, float))
    }


def _read_metric(summary_path: Path, metric: str) -> float:
    metrics = _read_metrics(summary_path)
    if metric not in metrics:
        raise KeyError(f"Metric '{metric}' not found in {summary_path}")
    return float(metrics[metric])


def _select_best(rows: Sequence[dict[str, Any]], *, metric: str, mode: str) -> dict[str, Any]:
    if not rows:
        raise ValueError("No calibration rows to select from")
    reverse = mode == "max"
    return sorted(rows, key=lambda row: float(row[metric]), reverse=reverse)[0]


def _is_better(value: float, reference: float, *, mode: str) -> bool:
    if mode == "min":
        return value < reference
    if mode == "max":
        return value > reference
    raise ValueError(f"Unsupported selection mode: {mode}")


def _relative_improvement(value: float, reference: float, *, mode: str) -> float:
    scale = max(abs(reference), 1e-12)
    if mode == "min":
        return (reference - value) / scale
    if mode == "max":
        return (value - reference) / scale
    raise ValueError(f"Unsupported selection mode: {mode}")


def _test_guard_result(
    *,
    value: float,
    reference: float | None,
    min_relative_improvement: float | None,
    mode: str,
) -> dict[str, Any]:
    if reference is None or min_relative_improvement is None:
        return {"enabled": False, "passed": True}
    relative_improvement = _relative_improvement(value, reference, mode=mode)
    return {
        "enabled": True,
        "reference_metric_value": reference,
        "min_relative_improvement": min_relative_improvement,
        "relative_improvement": relative_improvement,
        "passed": relative_improvement >= min_relative_improvement,
    }


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


def _test_measurement_key(
    *,
    args: argparse.Namespace,
    selected_overrides: Sequence[str],
    selected_gate: Mapping[str, Any],
) -> str:
    payload = {
        "calibrator": "decoded_persistence_residual_gate",
        "checkpoint_source": args.checkpoint_source,
        "config": args.config,
        "data_root": args.data_root,
        "decoded_rollout_steps": args.decoded_rollout_steps,
        "default_alpha": args.default_alpha,
        "eval_max_samples": args.eval_max_samples,
        "eval_override": list(args.eval_override),
        "kind": args.kind,
        "key": args.key,
        "metric": args.metric,
        "mode": args.mode,
        "override": list(args.override),
        "promotion_rule": list(args.promotion_rule),
        "reference_metric_value": args.reference_metric_value,
        "selection_split": getattr(args, "selection_split", args.val_split),
        "selected_gate": selected_gate.get("selected_gate"),
        "selected_gate_alpha": selected_gate.get("alpha"),
        "selected_gate_schedule": selected_gate.get("schedule"),
        "selected_overrides": list(selected_overrides),
        "test_min_relative_improvement": args.test_min_relative_improvement,
        "test_split": args.test_split,
        "val_split": args.val_split,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _guard_test_measurement(
    *,
    ledger_path: str | None,
    measurement_key: str,
    allow_repeat_test: bool,
) -> dict[str, Any]:
    ledger = _load_test_ledger(ledger_path)
    existing_keys = {
        str(entry.get("measurement_key"))
        for entry in ledger.get("measurements", [])
        if isinstance(entry, dict)
    }
    already_recorded = measurement_key in existing_keys
    if already_recorded and not allow_repeat_test:
        raise RuntimeError(
            "held-out test measurement already recorded for this residual gate; "
            "set --allow-repeat-test only for explicit debugging"
        )
    return {
        "allow_repeat_test": allow_repeat_test,
        "ledger_path": ledger_path,
        "measurement_key": measurement_key,
        "recorded": False,
        "already_recorded": already_recorded,
    }


def _record_test_measurement(
    *,
    ledger_path: str | None,
    measurement_key: str,
    allow_repeat_test: bool,
    metric: str,
    test_metric_value: float,
    validation_metric_value: float,
    test_split: str,
    selected_gate: str,
    selected_overrides: Sequence[str],
    run_name: str,
    summary: str,
) -> bool:
    if not ledger_path or allow_repeat_test:
        return False
    ledger = _load_test_ledger(ledger_path)
    ledger.setdefault("measurements", []).append(
        {
            "measurement_key": measurement_key,
            "metric": metric,
            "run_name": run_name,
            "selected_gate": selected_gate,
            "selected_overrides": list(selected_overrides),
            "summary": summary,
            "test_metric_value": test_metric_value,
            "test_split": test_split,
            "validation_metric_value": validation_metric_value,
        }
    )
    _write_test_ledger(ledger_path, ledger)
    return True


def _horizon_metric_pattern(kind: str, key: str) -> re.Pattern[str]:
    if kind == "global":
        prefix = "decoded_h"
    elif kind == "family":
        prefix = f"family_{re.escape(key)}_decoded_h"
    elif kind == "task":
        prefix = f"task_{re.escape(key)}_decoded_h"
    else:
        raise ValueError(f"Unsupported gate kind: {kind}")
    return re.compile(rf"^{prefix}(\d+)_nrmse$")


def _select_horizon_schedule(
    rows: Sequence[dict[str, Any]],
    *,
    kind: str,
    key: str,
    mode: str,
) -> tuple[dict[int, float], list[dict[str, Any]]]:
    if not rows:
        raise ValueError("No calibration rows to select from")
    pattern = _horizon_metric_pattern(kind, key)
    candidates: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        metrics = row.get("metrics", {})
        for metric_name, value in metrics.items():
            match = pattern.match(str(metric_name))
            if not match:
                continue
            candidates.setdefault(int(match.group(1)), []).append(
                {
                    "horizon": int(match.group(1)),
                    "alpha": float(row["alpha"]),
                    "run_name": row["run_name"],
                    "summary": row["summary"],
                    "metric": metric_name,
                    "value": float(value),
                }
            )
    if not candidates:
        raise ValueError(f"No horizon metrics found for {kind} '{key}'")

    selections: list[dict[str, Any]] = []
    reverse = mode == "max"
    for horizon, horizon_rows in sorted(candidates.items()):
        selected = sorted(horizon_rows, key=lambda row: row["value"], reverse=reverse)[0]
        selections.append(selected)
    schedule = {int(row["horizon"]): float(row["alpha"]) for row in selections}
    return schedule, selections


def _run_light_eval(
    *,
    args: argparse.Namespace,
    name: str,
    split: str,
    gate_overrides: Sequence[str],
    report_all_horizon_metrics: bool = False,
    reuse_existing: bool | None = None,
) -> Path:
    summary_path = Path(args.output_root) / name / "summary.json"
    if (
        args.reuse_existing if reuse_existing is None else reuse_existing
    ) and summary_path.exists():
        return summary_path
    cmd = [
        sys.executable,
        "scripts/run_light_experiment.py",
        "--config",
        args.config,
        "--name",
        name,
        "--output-root",
        args.output_root,
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
    ]
    for gate_override in gate_overrides:
        cmd.extend(["--override", gate_override])
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
        description="Calibrate decoded residual gate alpha on a selection split, then validate and test once"
    )
    parser.add_argument("--config", default="configs/train_multitask_heterogeneous_light_best.yaml")
    parser.add_argument("--checkpoint-source", required=True)
    parser.add_argument("--data-root", default="data/pdebench")
    parser.add_argument("--output-root", default="reports/light_experiments_remote")
    parser.add_argument("--run-prefix", default="ups_light_transport_gate_calibrated")
    parser.add_argument("--kind", choices=("global", "family", "task"), default="family")
    parser.add_argument("--key", default="transport")
    parser.add_argument("--default-alpha", type=float, default=0.0)
    parser.add_argument(
        "--alpha", action="append", type=float, default=None, help="Candidate alpha; repeatable"
    )
    parser.add_argument(
        "--schedule-by-horizon",
        action="store_true",
        help="Select a validation-best alpha per rollout horizon",
    )
    parser.add_argument(
        "--schedule-min-relative-improvement",
        type=float,
        default=0.01,
        help="Minimum aggregate validation improvement required to select a horizon schedule over the best constant gate",
    )
    parser.add_argument(
        "--use-decoded-residual-gate",
        action="store_true",
        help="Route candidates through evaluation.decoded_persistence_residual_gate so alpha metrics and gate config are exported",
    )
    parser.add_argument(
        "--gate-config-candidate",
        action="append",
        default=[],
        help="JSON object merged into the decoded residual gate config; repeat to sweep configs",
    )
    parser.add_argument("--gate-min-alpha", type=float, default=0.0)
    parser.add_argument("--gate-max-alpha", type=float, default=1.0)
    parser.add_argument("--gate-bias", type=float, default=0.0)
    parser.add_argument(
        "--gate-feature-weight",
        action="append",
        default=[],
        help="Feature weight as name=value; repeated entries populate decoded gate feature_weights",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse existing run summary files instead of rerunning them",
    )
    parser.add_argument("--val-split", default="val")
    parser.add_argument(
        "--selection-split",
        help="Split used to select candidate gates; defaults to --val-split for backward compatibility",
    )
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--metric", default="decoded_rollout_nrmse")
    parser.add_argument("--mode", choices=("min", "max"), default="min")
    parser.add_argument(
        "--reference-metric-value",
        type=float,
        help="Validation reference for guarding held-out test, for example the clean constant-gate validation metric",
    )
    parser.add_argument(
        "--test-min-relative-improvement",
        type=float,
        help="Skip held-out test unless selected validation metric improves over reference by this fraction",
    )
    parser.add_argument(
        "--test-ledger-json",
        help="Optional ledger that prevents measuring the same guarded held-out test more than once",
    )
    parser.add_argument(
        "--allow-repeat-test",
        action="store_true",
        help="Bypass the held-out test ledger guard for explicit debugging repeats",
    )
    parser.add_argument("--eval-max-samples", type=int, default=32)
    parser.add_argument("--decoded-rollout-steps", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--eval-override", action="append", default=[])
    parser.add_argument("--promotion-rule", action="append", default=["decoded_rollout_nrmse<=1.0"])
    parser.add_argument("--output-json", help="Calibration record path; defaults under output root")
    parser.add_argument(
        "--export-selected-gate-config",
        help="Optional JSON path for frozen selected override payload",
    )
    args = parser.parse_args()
    args.selection_split = args.selection_split or args.val_split

    alphas = args.alpha or [0.0, 0.1, 0.2, 0.3, 0.4, 0.42, 0.44, 0.5, 0.75, 1.0]
    gate_candidates = _gate_candidates(args)
    if args.schedule_by_horizon and len(gate_candidates) > 1:
        raise ValueError("--schedule-by-horizon supports at most one decoded gate config candidate")
    rows: list[dict[str, Any]] = []
    for alpha in alphas:
        for gate_index, gate_config in enumerate(gate_candidates):
            name = (
                f"{args.run_prefix}_{args.selection_split}_{args.kind}_{args.key}_alpha{_safe_alpha(alpha)}"
                f"{_gate_suffix(gate_index, len(gate_candidates), gate_config)}"
            )
            summary_path = _run_light_eval(
                args=args,
                name=name,
                split=args.selection_split,
                gate_overrides=_gate_overrides(
                    _alpha_override(args.kind, args.key, alpha), gate_config
                ),
                report_all_horizon_metrics=args.schedule_by_horizon,
            )
            metrics = _read_metrics(summary_path)
            row: dict[str, Any] = {
                "alpha": alpha,
                "run_name": name,
                "summary": str(summary_path),
                args.metric: metrics[args.metric],
                "metrics": metrics,
            }
            if gate_config is not None:
                row["gate_config"] = gate_config
                row["gate_candidate_index"] = gate_index
            rows.append(row)

    best = _select_best(rows, metric=args.metric, mode=args.mode)
    schedule: dict[int, float] | None = None
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
        "selection_split": args.selection_split,
        "val_split": args.val_split,
        "test_split": args.test_split,
        "selection": rows,
        "best_selection": best,
    }
    if args.selection_split == args.val_split:
        record["validation"] = rows
        record["best_validation"] = best
    if schedule is not None and schedule_selections is not None:
        record["best_selection_schedule"] = {
            "schedule": {str(horizon): alpha for horizon, alpha in sorted(schedule.items())},
            "selections": schedule_selections,
        }
        schedule_val_name = (
            f"{args.run_prefix}_{args.selection_split}_{args.kind}_{args.key}_horizon_schedule"
        )
        schedule_val_summary = _run_light_eval(
            args=args,
            name=schedule_val_name,
            split=args.selection_split,
            gate_overrides=_gate_overrides(
                _schedule_override(args.kind, args.key, schedule), gate_candidates[0]
            ),
            report_all_horizon_metrics=True,
        )
        schedule_val_metrics = _read_metrics(schedule_val_summary)
        record["selection_schedule"] = {
            "run_name": schedule_val_name,
            "summary": str(schedule_val_summary),
            args.metric: schedule_val_metrics[args.metric],
            "metrics": schedule_val_metrics,
        }
        if gate_candidates[0] is not None:
            record["selection_schedule"]["gate_config"] = gate_candidates[0]
        if args.selection_split == args.val_split:
            record["best_validation_schedule"] = record["best_selection_schedule"]
            record["validation_schedule"] = record["selection_schedule"]
        record["schedule_min_relative_improvement"] = args.schedule_min_relative_improvement
        record["schedule_relative_improvement"] = _relative_improvement(
            float(schedule_val_metrics[args.metric]),
            float(best[args.metric]),
            mode=args.mode,
        )

    selection_schedule = record.get("selection_schedule")
    use_schedule = (
        schedule is not None
        and selection_schedule is not None
        and _is_better(
            float(selection_schedule[args.metric]),
            float(best[args.metric]),
            mode=args.mode,
        )
        and float(record.get("schedule_relative_improvement", 0.0))
        >= args.schedule_min_relative_improvement
    )
    if use_schedule:
        selected_overrides = _gate_overrides(
            _schedule_override(args.kind, args.key, schedule), gate_candidates[0]
        )
        record["selected_validation_gate"] = {
            "selected_gate": "horizon_schedule",
            "schedule": {str(horizon): alpha for horizon, alpha in sorted(schedule.items())},
            "overrides": selected_overrides,
            "selection": selection_schedule,
        }
    else:
        selected_gate_config = best.get("gate_config")
        selected_overrides = _gate_overrides(
            _alpha_override(args.kind, args.key, float(best["alpha"])),
            selected_gate_config if isinstance(selected_gate_config, dict) else None,
        )
        record["selected_validation_gate"] = {
            "selected_gate": "constant_alpha",
            "alpha": float(best["alpha"]),
            "overrides": selected_overrides,
            "selection": best,
        }
    if args.selection_split == args.val_split:
        validation_confirmation = record["selected_validation_gate"]["selection"]
    elif use_schedule:
        validation_name = f"{args.run_prefix}_{args.val_split}_{args.kind}_{args.key}_horizon_schedule_validation_confirmation"
        validation_summary = _run_light_eval(
            args=args,
            name=validation_name,
            split=args.val_split,
            gate_overrides=selected_overrides,
            report_all_horizon_metrics=True,
        )
        validation_metrics = _read_metrics(validation_summary)
        validation_confirmation = {
            "run_name": validation_name,
            "summary": str(validation_summary),
            args.metric: validation_metrics[args.metric],
            "metrics": validation_metrics,
        }
    else:
        validation_name = (
            f"{args.run_prefix}_{args.val_split}_{args.kind}_{args.key}_alpha"
            f"{_safe_alpha(float(best['alpha']))}_validation_confirmation"
        )
        validation_summary = _run_light_eval(
            args=args,
            name=validation_name,
            split=args.val_split,
            gate_overrides=selected_overrides,
        )
        validation_metrics = _read_metrics(validation_summary)
        validation_confirmation = {
            "run_name": validation_name,
            "summary": str(validation_summary),
            args.metric: validation_metrics[args.metric],
            "metrics": validation_metrics,
        }
    record["validation_confirmation"] = validation_confirmation
    record["selected_validation_gate"]["validation"] = validation_confirmation
    selected_metric_value = float(validation_confirmation[args.metric])
    record["selected_validation_gate"]["test_guard"] = _test_guard_result(
        value=selected_metric_value,
        reference=args.reference_metric_value,
        min_relative_improvement=args.test_min_relative_improvement,
        mode=args.mode,
    )
    record["held_out_test_policy"] = {
        "allow_repeat_test": args.allow_repeat_test,
        "ledger_path": args.test_ledger_json,
        "measurement_key": None,
        "recorded": False,
        "already_recorded": False,
    }

    if args.skip_test:
        record["test_skipped"] = {"reason": "--skip-test"}
    elif not record["selected_validation_gate"]["test_guard"]["passed"]:
        record["test_skipped"] = {
            "reason": "selected validation gate did not pass held-out test guard"
        }
    else:
        test_measurement_key = _test_measurement_key(
            args=args,
            selected_overrides=selected_overrides,
            selected_gate=record["selected_validation_gate"],
        )
        record["held_out_test_policy"] = _guard_test_measurement(
            ledger_path=args.test_ledger_json,
            measurement_key=test_measurement_key,
            allow_repeat_test=args.allow_repeat_test,
        )
        if use_schedule:
            test_name = (
                f"{args.run_prefix}_{args.test_split}_{args.kind}_{args.key}_horizon_schedule"
            )
            test_summary = _run_light_eval(
                args=args,
                name=test_name,
                split=args.test_split,
                gate_overrides=selected_overrides,
                report_all_horizon_metrics=True,
                reuse_existing=False,
            )
            record["test_schedule"] = {
                str(horizon): alpha for horizon, alpha in sorted(schedule.items())
            }
            test_alpha: float | dict[str, float] = record["test_schedule"]
            selected_gate = "horizon_schedule"
        else:
            test_name = f"{args.run_prefix}_{args.test_split}_{args.kind}_{args.key}_alpha{_safe_alpha(float(best['alpha']))}"
            test_summary = _run_light_eval(
                args=args,
                name=test_name,
                split=args.test_split,
                gate_overrides=selected_overrides,
                reuse_existing=False,
            )
            test_alpha = float(best["alpha"])
            selected_gate = "constant_alpha"
        test_metrics = _read_metrics(test_summary)
        record["test"] = {
            "alpha": test_alpha,
            "selected_gate": selected_gate,
            "run_name": test_name,
            "summary": str(test_summary),
            args.metric: test_metrics[args.metric],
            "metrics": test_metrics,
        }
        record["held_out_test_policy"]["recorded"] = _record_test_measurement(
            ledger_path=args.test_ledger_json,
            measurement_key=test_measurement_key,
            allow_repeat_test=args.allow_repeat_test,
            metric=args.metric,
            test_metric_value=float(test_metrics[args.metric]),
            validation_metric_value=selected_metric_value,
            test_split=args.test_split,
            selected_gate=selected_gate,
            selected_overrides=selected_overrides,
            run_name=test_name,
            summary=str(test_summary),
        )

    output_path = (
        Path(args.output_json)
        if args.output_json
        else Path(args.output_root) / f"{args.run_prefix}_calibration.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    if args.export_selected_gate_config:
        export_path = Path(args.export_selected_gate_config)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text(
            json.dumps(record["selected_validation_gate"], indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
