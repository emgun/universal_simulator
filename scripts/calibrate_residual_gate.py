#!/usr/bin/env python
from __future__ import annotations

"""Calibrate decoded persistence-residual gates on validation data, then test once."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


def _safe_alpha(alpha: float) -> str:
    return f"{alpha:g}".replace("-", "m").replace(".", "p")


def _alpha_override(kind: str, key: str, alpha: float) -> str:
    if kind == "family":
        return f'evaluation.decoded_persistence_residual_alpha_by_family={{{json.dumps(key)}:{alpha:g}}}'
    if kind == "task":
        return f'evaluation.decoded_persistence_residual_alpha_by_task={{{json.dumps(key)}:{alpha:g}}}'
    raise ValueError(f"Unsupported gate kind: {kind}")


def _read_metric(summary_path: Path, metric: str) -> float:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = summary.get("metrics", {})
    if metric not in metrics:
        raise KeyError(f"Metric '{metric}' not found in {summary_path}")
    return float(metrics[metric])


def _select_best(rows: Sequence[dict[str, Any]], *, metric: str, mode: str) -> dict[str, Any]:
    if not rows:
        raise ValueError("No calibration rows to select from")
    reverse = mode == "max"
    return sorted(rows, key=lambda row: float(row[metric]), reverse=reverse)[0]


def _run_light_eval(
    *,
    args: argparse.Namespace,
    name: str,
    split: str,
    alpha: float,
) -> Path:
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
        "--override",
        _alpha_override(args.kind, args.key, alpha),
    ]
    if args.eval_max_samples is not None:
        cmd.extend(["--eval-override", f"data.max_samples={args.eval_max_samples}"])
    for override in args.override:
        cmd.extend(["--override", override])
    for override in args.eval_override:
        cmd.extend(["--eval-override", override])
    for rule in args.promotion_rule:
        cmd.extend(["--promotion-rule", rule])
    subprocess.run(cmd, check=True)
    return Path(args.output_root) / name / "summary.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate decoded residual gate alpha on validation data")
    parser.add_argument("--config", default="configs/train_multitask_heterogeneous_light_best.yaml")
    parser.add_argument("--checkpoint-source", required=True)
    parser.add_argument("--data-root", default="data/pdebench")
    parser.add_argument("--output-root", default="reports/light_experiments_remote")
    parser.add_argument("--run-prefix", default="ups_light_transport_gate_calibrated")
    parser.add_argument("--kind", choices=("family", "task"), default="family")
    parser.add_argument("--key", default="transport")
    parser.add_argument("--default-alpha", type=float, default=0.0)
    parser.add_argument("--alpha", action="append", type=float, default=None, help="Candidate alpha; repeatable")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--metric", default="decoded_rollout_nrmse")
    parser.add_argument("--mode", choices=("min", "max"), default="min")
    parser.add_argument("--eval-max-samples", type=int, default=32)
    parser.add_argument("--decoded-rollout-steps", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--eval-override", action="append", default=[])
    parser.add_argument("--promotion-rule", action="append", default=["decoded_rollout_nrmse<=1.0"])
    parser.add_argument("--output-json", help="Calibration record path; defaults under output root")
    args = parser.parse_args()

    alphas = args.alpha or [0.0, 0.1, 0.2, 0.3, 0.4, 0.42, 0.44, 0.5, 0.75, 1.0]
    rows: list[dict[str, Any]] = []
    for alpha in alphas:
        name = f"{args.run_prefix}_{args.val_split}_{args.kind}_{args.key}_alpha{_safe_alpha(alpha)}"
        summary_path = _run_light_eval(args=args, name=name, split=args.val_split, alpha=alpha)
        rows.append(
            {
                "alpha": alpha,
                "run_name": name,
                "summary": str(summary_path),
                args.metric: _read_metric(summary_path, args.metric),
            }
        )

    best = _select_best(rows, metric=args.metric, mode=args.mode)
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

    if not args.skip_test:
        test_name = f"{args.run_prefix}_{args.test_split}_{args.kind}_{args.key}_alpha{_safe_alpha(float(best['alpha']))}"
        test_summary = _run_light_eval(args=args, name=test_name, split=args.test_split, alpha=float(best["alpha"]))
        record["test"] = {
            "alpha": best["alpha"],
            "run_name": test_name,
            "summary": str(test_summary),
            args.metric: _read_metric(test_summary, args.metric),
        }

    output_path = Path(args.output_json) if args.output_json else Path(args.output_root) / f"{args.run_prefix}_calibration.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
