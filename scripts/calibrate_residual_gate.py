#!/usr/bin/env python
from __future__ import annotations

"""Calibrate decoded persistence-residual gates on validation data, then test once."""

import argparse
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
    if kind == "family":
        return f"evaluation.decoded_persistence_residual_alpha_by_family={{{json.dumps(key)}:{alpha:g}}}"
    if kind == "task":
        return (
            f"evaluation.decoded_persistence_residual_alpha_by_task={{{json.dumps(key)}:{alpha:g}}}"
        )
    raise ValueError(f"Unsupported gate kind: {kind}")


def _schedule_override(kind: str, key: str, schedule: dict[int, float]) -> str:
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


def _horizon_metric_pattern(kind: str, key: str) -> re.Pattern[str]:
    if kind == "family":
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
) -> Path:
    summary_path = Path(args.output_root) / name / "summary.json"
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
        description="Calibrate decoded residual gate alpha on validation data"
    )
    parser.add_argument("--config", default="configs/train_multitask_heterogeneous_light_best.yaml")
    parser.add_argument("--checkpoint-source", required=True)
    parser.add_argument("--data-root", default="data/pdebench")
    parser.add_argument("--output-root", default="reports/light_experiments_remote")
    parser.add_argument("--run-prefix", default="ups_light_transport_gate_calibrated")
    parser.add_argument("--kind", choices=("family", "task"), default="family")
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

    alphas = args.alpha or [0.0, 0.1, 0.2, 0.3, 0.4, 0.42, 0.44, 0.5, 0.75, 1.0]
    gate_candidates = _gate_candidates(args)
    if args.schedule_by_horizon and len(gate_candidates) > 1:
        raise ValueError("--schedule-by-horizon supports at most one decoded gate config candidate")
    rows: list[dict[str, Any]] = []
    for alpha in alphas:
        for gate_index, gate_config in enumerate(gate_candidates):
            name = (
                f"{args.run_prefix}_{args.val_split}_{args.kind}_{args.key}_alpha{_safe_alpha(alpha)}"
                f"{_gate_suffix(gate_index, len(gate_candidates), gate_config)}"
            )
            summary_path = _run_light_eval(
                args=args,
                name=name,
                split=args.val_split,
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
        "val_split": args.val_split,
        "test_split": args.test_split,
        "validation": rows,
        "best_validation": best,
    }
    if schedule is not None and schedule_selections is not None:
        record["best_validation_schedule"] = {
            "schedule": {str(horizon): alpha for horizon, alpha in sorted(schedule.items())},
            "selections": schedule_selections,
        }
        schedule_val_name = (
            f"{args.run_prefix}_{args.val_split}_{args.kind}_{args.key}_horizon_schedule"
        )
        schedule_val_summary = _run_light_eval(
            args=args,
            name=schedule_val_name,
            split=args.val_split,
            gate_overrides=_gate_overrides(
                _schedule_override(args.kind, args.key, schedule), gate_candidates[0]
            ),
            report_all_horizon_metrics=True,
        )
        schedule_val_metrics = _read_metrics(schedule_val_summary)
        record["validation_schedule"] = {
            "run_name": schedule_val_name,
            "summary": str(schedule_val_summary),
            args.metric: schedule_val_metrics[args.metric],
            "metrics": schedule_val_metrics,
        }
        if gate_candidates[0] is not None:
            record["validation_schedule"]["gate_config"] = gate_candidates[0]
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
        and _is_better(
            float(validation_schedule[args.metric]),
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
            "validation": validation_schedule,
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
            "validation": best,
        }
    selected_metric_value = float(record["selected_validation_gate"]["validation"][args.metric])
    record["selected_validation_gate"]["test_guard"] = _test_guard_result(
        value=selected_metric_value,
        reference=args.reference_metric_value,
        min_relative_improvement=args.test_min_relative_improvement,
        mode=args.mode,
    )

    if args.skip_test:
        record["test_skipped"] = {"reason": "--skip-test"}
    elif not record["selected_validation_gate"]["test_guard"]["passed"]:
        record["test_skipped"] = {
            "reason": "selected validation gate did not pass held-out test guard"
        }
    else:
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
