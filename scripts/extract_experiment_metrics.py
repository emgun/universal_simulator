#!/usr/bin/env python
from __future__ import annotations

"""Extract compact experiment metrics from UPS light experiment summaries."""

import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


DEFAULT_FIELDS = (
    "timestamp",
    "branch",
    "commit",
    "run_name",
    "summary_json",
    "status",
    "primary_metric",
    "primary_metric_value",
    "baseline_metric_value",
    "baseline_ratio",
    "baseline_improvement_fraction",
    "advection_nrmse",
    "burgers_nrmse",
    "darcy_nrmse",
    "transport_nrmse",
    "conservation_nrmse",
    "elliptic_nrmse",
    "h1_nrmse",
    "h4_nrmse",
    "h8_nrmse",
    "h16_nrmse",
    "spectral_error",
    "duration_sec",
    "wandb_urls",
    "description",
)


def _git_value(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ""


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON in {path}")
    return payload


def _metric(metrics: Mapping[str, Any], key: str) -> float | str:
    value = metrics.get(key)
    if value is None:
        return ""
    return float(value)


def _main_metric(metrics: Mapping[str, Any], preferred: str) -> tuple[str, float | str]:
    if preferred in metrics:
        return preferred, float(metrics[preferred])
    for key in ("decoded_rollout_nrmse", "decoded_step1_nrmse", "mse", "mae", "rmse"):
        if key in metrics:
            return key, float(metrics[key])
    if not metrics:
        return "", ""
    key = sorted(str(item) for item in metrics)[0]
    return key, float(metrics[key])


def _wandb_urls(summary: Mapping[str, Any]) -> str:
    tracking = summary.get("tracking", {})
    if not isinstance(tracking, Mapping):
        return ""
    wandb = tracking.get("wandb", {})
    if not isinstance(wandb, Mapping):
        return ""
    runs = wandb.get("runs", [])
    if not isinstance(runs, list):
        return ""
    urls = [str(run.get("url")) for run in runs if isinstance(run, Mapping) and run.get("url")]
    return ",".join(urls)


def extract_row(
    summary_path: str | Path,
    *,
    baseline_summary_path: str | Path | None = None,
    primary_metric: str = "decoded_rollout_nrmse",
    status: str = "",
    description: str = "",
) -> dict[str, Any]:
    summary_path = Path(summary_path)
    summary = _load_json(summary_path)
    metrics = summary.get("metrics", {})
    if not isinstance(metrics, Mapping):
        metrics = {}

    metric_name, metric_value = _main_metric(metrics, primary_metric)
    baseline_value: float | str = ""
    baseline_ratio: float | str = ""
    improvement: float | str = ""
    if baseline_summary_path:
        baseline = _load_json(baseline_summary_path)
        baseline_metrics = baseline.get("metrics", {})
        if isinstance(baseline_metrics, Mapping) and metric_name in baseline_metrics and metric_value != "":
            baseline_value = float(baseline_metrics[metric_name])
            if baseline_value != 0.0:
                baseline_ratio = float(metric_value) / baseline_value
                improvement = (baseline_value - float(metric_value)) / abs(baseline_value)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "branch": _git_value("branch", "--show-current"),
        "commit": _git_value("rev-parse", "--short", "HEAD"),
        "run_name": summary.get("run_name", summary_path.parent.name),
        "summary_json": str(summary_path),
        "status": status,
        "primary_metric": metric_name,
        "primary_metric_value": metric_value,
        "baseline_metric_value": baseline_value,
        "baseline_ratio": baseline_ratio,
        "baseline_improvement_fraction": improvement,
        "advection_nrmse": _metric(metrics, "task_advection1d_decoded_rollout_nrmse"),
        "burgers_nrmse": _metric(metrics, "task_burgers1d_decoded_rollout_nrmse"),
        "darcy_nrmse": _metric(metrics, "task_darcy2d_decoded_rollout_nrmse"),
        "transport_nrmse": _metric(metrics, "family_transport_decoded_rollout_nrmse"),
        "conservation_nrmse": _metric(metrics, "family_conservation_decoded_rollout_nrmse"),
        "elliptic_nrmse": _metric(metrics, "family_elliptic_decoded_rollout_nrmse"),
        "h1_nrmse": _metric(metrics, "decoded_h1_nrmse") or _metric(metrics, "decoded_step1_nrmse"),
        "h4_nrmse": _metric(metrics, "decoded_h4_nrmse"),
        "h8_nrmse": _metric(metrics, "decoded_h8_nrmse"),
        "h16_nrmse": _metric(metrics, "decoded_h16_nrmse"),
        "spectral_error": _metric(metrics, "decoded_rollout_spectral_energy_error"),
        "duration_sec": summary.get("duration_sec", ""),
        "wandb_urls": _wandb_urls(summary),
        "description": description,
    }


def append_row(path: str | Path, row: Mapping[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    with output_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DEFAULT_FIELDS, delimiter="\t", extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract compact metrics from a UPS summary.json")
    parser.add_argument("summary", help="Path to experiment summary.json")
    parser.add_argument("--baseline-summary", help="Optional baseline summary.json for ratio/improvement")
    parser.add_argument("--primary-metric", default="decoded_rollout_nrmse")
    parser.add_argument("--status", default="")
    parser.add_argument("--description", default="")
    parser.add_argument("--output-tsv", help="Append row to a TSV ledger")
    parser.add_argument("--print-header", action="store_true", help="Print TSV header before the row")
    args = parser.parse_args()

    row = extract_row(
        args.summary,
        baseline_summary_path=args.baseline_summary,
        primary_metric=args.primary_metric,
        status=args.status,
        description=args.description,
    )
    if args.output_tsv:
        append_row(args.output_tsv, row)
    writer = csv.DictWriter(
        __import__("sys").stdout,
        fieldnames=DEFAULT_FIELDS,
        delimiter="\t",
        lineterminator="\n",
        extrasaction="ignore",
    )
    if args.print_header:
        writer.writeheader()
    writer.writerow(row)


if __name__ == "__main__":
    main()
