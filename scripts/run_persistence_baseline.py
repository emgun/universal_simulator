#!/usr/bin/env python
from __future__ import annotations

"""Run a physical-space persistence baseline and write light-experiment summaries."""

import argparse
import copy
import csv
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ups.eval.persistence_baselines import evaluate_persistence_decoded
from ups.eval.promotion import evaluate_promotion_rules, parse_promotion_rule
from ups.utils.config_loader import load_config_with_includes


def _parse_override(text: str) -> tuple[str, Any]:
    if "=" not in text:
        raise ValueError(f"Invalid override '{text}'. Expected key=value")
    key, raw = text.split("=", 1)
    return key.strip(), yaml.safe_load(raw)


def _set_dotpath(cfg: dict[str, Any], path: str, value: Any) -> None:
    cursor = cfg
    parts = [part for part in path.split(".") if part]
    if not parts:
        raise ValueError(f"Invalid override path '{path}'")
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def _apply_overrides(cfg: dict[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    updated = copy.deepcopy(cfg)
    for text in overrides:
        key, value = _parse_override(text)
        _set_dotpath(updated, key, value)
    return updated


def _main_metric(metrics: dict[str, float]) -> tuple[str, float]:
    for key in ("decoded_rollout_nrmse", "decoded_step1_nrmse", "mse"):
        if key in metrics:
            return key, float(metrics[key])
    first_key = next(iter(metrics))
    return first_key, float(metrics[first_key])


def _append_results_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_name",
        "timestamp",
        "stages",
        "decoded",
        "train_split",
        "eval_split",
        "transfer_tasks",
        "promotion_passed",
        "main_metric_name",
        "main_metric_value",
        "summary_json",
    ]
    row_map: dict[str, dict[str, Any]] = {}
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for existing in reader:
                run_name = existing.get("run_name")
                if run_name:
                    row_map[run_name] = dict(existing)
    row_map[str(row["run_name"])] = row
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(row_map[name] for name in sorted(row_map))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run decoded persistence baseline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-root", default="reports/light_experiments")
    parser.add_argument(
        "--override", action="append", default=[], help="Config override like data.root=..."
    )
    parser.add_argument("--data-root", help="Override data.root")
    parser.add_argument("--split", help="Override data.split")
    parser.add_argument(
        "--task", action="append", default=[], help="Override data.task; repeat for multitask"
    )
    parser.add_argument("--max-samples", type=int, help="Override data.max_samples")
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--promotion-rule", action="append", default=[])
    args = parser.parse_args()

    cfg = _apply_overrides(load_config_with_includes(args.config), args.override)
    data_cfg = cfg.setdefault("data", {})
    if args.data_root:
        data_cfg["root"] = args.data_root
    if args.split:
        data_cfg["split"] = args.split
    if args.task:
        data_cfg["task"] = args.task if len(args.task) > 1 else args.task[0]
    if args.max_samples is not None:
        data_cfg["max_samples"] = int(args.max_samples)

    output_root = Path(args.output_root)
    run_dir = output_root / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_cfg_path = run_dir / "resolved_eval.yaml"
    resolved_cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    started = time.time()
    report = evaluate_persistence_decoded(cfg, rollout_steps=args.rollout_steps)
    finished = time.time()
    if args.promotion_rule:
        result = evaluate_promotion_rules(
            report.metrics,
            [parse_promotion_rule(rule) for rule in args.promotion_rule],
        )
        report.extra["promotion_passed"] = result.passed
        report.extra["promotion_failed_rules"] = result.failed_rules
        report.extra["promotion_missing_metrics"] = result.missing_metrics

    summary = {
        "metrics": report.metrics,
        "extra": report.extra,
        "details": {},
        "checkpoints": {"operator": None, "encoder": None, "decoder": None},
        "run_name": args.name,
        "stages": ["persistence"],
        "config": str(resolved_cfg_path),
        "eval_config": str(resolved_cfg_path),
        "duration_sec": finished - started,
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    main_metric_name, main_metric_value = _main_metric(report.metrics)
    _append_results_row(
        output_root / "results.tsv",
        {
            "run_name": args.name,
            "timestamp": int(finished),
            "stages": "persistence",
            "decoded": True,
            "train_split": "",
            "eval_split": data_cfg.get("split", ""),
            "transfer_tasks": "",
            "promotion_passed": report.extra.get("promotion_passed"),
            "main_metric_name": main_metric_name,
            "main_metric_value": main_metric_value,
            "summary_json": str(summary_path),
        },
    )
    print(
        json.dumps(
            {"summary": str(summary_path), "main_metric": {main_metric_name: main_metric_value}},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
