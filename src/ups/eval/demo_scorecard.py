from __future__ import annotations

"""Utilities for turning light experiment summaries into demo scorecards."""

import csv
import html
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ups.eval.promotion import evaluate_promotion_rules, parse_promotion_rule


MAIN_METRIC_ORDER = (
    "decoded_rollout_nrmse",
    "decoded_step1_nrmse",
    "mse",
    "mae",
    "rmse",
)

BASE_FIELDS = (
    "run_name",
    "summary_json",
    "config",
    "eval_config",
    "stages",
    "duration_sec",
    "main_metric_name",
    "main_metric_value",
    "promotion_passed",
    "data_manifest",
    "commit",
    "cost_source",
    "cost_provider",
    "cost_instance_type",
    "cost_gpu_type",
    "cost_gpu_count",
    "cost_wall_clock_hours",
    "cost_gpu_hours",
    "cost_estimated_usd",
)


@dataclass(frozen=True)
class Scorecard:
    rows: list[dict[str, Any]]
    metric_keys: list[str]


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _main_metric(metrics: Mapping[str, Any]) -> tuple[str, float | None]:
    for key in MAIN_METRIC_ORDER:
        if key in metrics:
            return key, float(metrics[key])
    if not metrics:
        return "", None
    key = sorted(metrics)[0]
    return key, float(metrics[key])


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, str)):
        return str(value)
    return json.dumps(value, sort_keys=True)


def load_summary(path: str | Path) -> dict[str, Any]:
    summary_path = Path(path)
    with summary_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["_summary_json"] = str(summary_path)
    return payload


def _index_cost_record(
    cost_index: dict[str, dict[str, Any]],
    record: Mapping[str, Any],
    *,
    source: str,
    fallback_run_name: str,
) -> None:
    payload = dict(record)
    payload.setdefault("_cost_source", source)
    keys = {
        fallback_run_name,
        str(payload.get("run_name", "")),
        str(payload.get("summary_json", "")),
    }
    for key in keys:
        if key:
            cost_index[key] = payload


def load_cost_index(paths: Iterable[str | Path]) -> dict[str, dict[str, Any]]:
    """Load optional run cost records keyed by run name or summary path."""
    cost_index: dict[str, dict[str, Any]] = {}
    for path in paths:
        cost_path = Path(path)
        with cost_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        records = payload.get("runs") if isinstance(payload, Mapping) else None
        if isinstance(records, list):
            for record in records:
                if isinstance(record, Mapping):
                    _index_cost_record(
                        cost_index,
                        record,
                        source=str(cost_path),
                        fallback_run_name=cost_path.parent.name,
                    )
            continue
        if isinstance(payload, Mapping):
            _index_cost_record(
                cost_index,
                payload,
                source=str(cost_path),
                fallback_run_name=cost_path.parent.name,
            )
    return cost_index


def _cost_record_for_summary(
    summary: Mapping[str, Any],
    cost_index: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    summary_path = str(summary.get("_summary_json", ""))
    run_name = str(summary.get("run_name", Path(summary_path).parent.name if summary_path else ""))
    candidates = [run_name, summary_path]
    if summary_path:
        candidates.append(str(Path(summary_path).resolve()))
    for key in candidates:
        if key in cost_index:
            return cost_index[key]

    extra = summary.get("extra", {})
    if isinstance(extra, Mapping) and isinstance(extra.get("cost"), Mapping):
        return extra["cost"]
    if isinstance(summary.get("cost"), Mapping):
        return summary["cost"]
    return {}


def _cost_fields(cost_record: Mapping[str, Any]) -> dict[str, Any]:
    hourly_usd = _as_float(cost_record.get("hourly_usd", cost_record.get("cost_per_hour_usd")))
    wall_clock_hours = _as_float(
        cost_record.get(
            "wall_clock_hours",
            cost_record.get("duration_hours", cost_record.get("runtime_hours")),
        )
    )
    gpu_count = _as_float(cost_record.get("gpu_count", cost_record.get("gpus")))
    gpu_hours = _as_float(cost_record.get("gpu_hours"))
    estimated_usd = _as_float(
        cost_record.get(
            "estimated_usd",
            cost_record.get("estimated_cost_usd", cost_record.get("cost_usd")),
        )
    )

    if gpu_hours is None and wall_clock_hours is not None and gpu_count is not None:
        gpu_hours = wall_clock_hours * gpu_count
    if estimated_usd is None and wall_clock_hours is not None and hourly_usd is not None:
        estimated_usd = wall_clock_hours * hourly_usd

    return {
        "cost_source": cost_record.get("_cost_source", ""),
        "cost_provider": cost_record.get("provider", ""),
        "cost_instance_type": cost_record.get("instance_type", cost_record.get("machine_type", "")),
        "cost_gpu_type": cost_record.get("gpu_type", cost_record.get("gpu_model", "")),
        "cost_gpu_count": gpu_count,
        "cost_wall_clock_hours": wall_clock_hours,
        "cost_gpu_hours": gpu_hours,
        "cost_estimated_usd": estimated_usd,
    }


def scorecard_row_from_summary(
    summary: Mapping[str, Any],
    *,
    data_manifest: str | None = None,
    commit: str | None = None,
    promotion_rules: Sequence[str] = (),
    cost_index: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    metrics = {str(key): float(value) for key, value in dict(summary.get("metrics", {})).items()}
    extra = dict(summary.get("extra", {}))
    main_metric_name, main_metric_value = _main_metric(metrics)
    cost_fields = _cost_fields(_cost_record_for_summary(summary, cost_index or {}))

    promotion_passed: bool | None = extra.get("promotion_passed")
    failed_rules: list[str] = list(extra.get("promotion_failed_rules", []))
    missing_metrics: list[str] = list(extra.get("promotion_missing_metrics", []))
    if promotion_rules:
        parsed = [parse_promotion_rule(rule) for rule in promotion_rules]
        result = evaluate_promotion_rules(metrics, parsed)
        promotion_passed = result.passed
        failed_rules = result.failed_rules
        missing_metrics = result.missing_metrics

    row: dict[str, Any] = {
        "run_name": summary.get("run_name", Path(str(summary.get("_summary_json", ""))).parent.name),
        "summary_json": summary.get("_summary_json", ""),
        "config": summary.get("config", ""),
        "eval_config": summary.get("eval_config", ""),
        "stages": ",".join(str(stage) for stage in summary.get("stages", [])),
        "duration_sec": summary.get("duration_sec"),
        "main_metric_name": main_metric_name,
        "main_metric_value": main_metric_value,
        "promotion_passed": promotion_passed,
        "promotion_failed_rules": failed_rules,
        "promotion_missing_metrics": missing_metrics,
        "data_manifest": data_manifest,
        "commit": commit,
        **cost_fields,
    }
    for key, value in metrics.items():
        row[f"metric:{key}"] = value
    return row


def collect_scorecard(
    summary_paths: Iterable[str | Path],
    *,
    data_manifest: str | None = None,
    commit: str | None = None,
    promotion_rules: Sequence[str] = (),
    cost_paths: Iterable[str | Path] = (),
) -> Scorecard:
    cost_index = load_cost_index(cost_paths)
    rows = [
        scorecard_row_from_summary(
            load_summary(path),
            data_manifest=data_manifest,
            commit=commit,
            promotion_rules=promotion_rules,
            cost_index=cost_index,
        )
        for path in summary_paths
    ]
    metric_keys = sorted({key for row in rows for key in row if key.startswith("metric:")})
    rows.sort(key=lambda row: str(row.get("run_name", "")))
    return Scorecard(rows=rows, metric_keys=metric_keys)


def write_scorecard_tsv(scorecard: Scorecard, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(BASE_FIELDS) + scorecard.metric_keys + [
        "promotion_failed_rules",
        "promotion_missing_metrics",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in scorecard.rows:
            writer.writerow({key: _stringify(row.get(key)) for key in fieldnames})


def write_scorecard_json(scorecard: Scorecard, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metric_keys": scorecard.metric_keys,
        "rows": scorecard.rows,
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def render_scorecard_html(
    scorecard: Scorecard,
    *,
    title: str = "UPS Demo Scorecard",
    plots: Mapping[str, str] | None = None,
) -> str:
    fields = list(BASE_FIELDS) + scorecard.metric_keys
    header_cells = "".join(f"<th>{html.escape(field)}</th>" for field in fields)
    body_rows = []
    for row in scorecard.rows:
        cells = "".join(f"<td>{html.escape(_stringify(row.get(field)))}</td>" for field in fields)
        body_rows.append(f"<tr>{cells}</tr>")
    body = "\n".join(body_rows) or f"<tr><td colspan=\"{len(fields)}\">No runs found.</td></tr>"
    escaped_title = html.escape(title)
    plot_html = ""
    if plots:
        cards = []
        for label, src in plots.items():
            cards.append(
                f"<figure><img src=\"{html.escape(src)}\" alt=\"{html.escape(label)}\"><figcaption>{html.escape(label)}</figcaption></figure>"
            )
        plot_html = "<section><h2>Metric Plots</h2>" + "\n".join(cards) + "</section>"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{escaped_title}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 2rem; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 0.85rem; }}
    th, td {{ border: 1px solid #d0d7de; padding: 0.4rem; text-align: left; vertical-align: top; }}
    th {{ background: #f6f8fa; position: sticky; top: 0; }}
    .note {{ color: #57606a; max-width: 68rem; }}
    figure {{ display: inline-block; margin: 0 1rem 1rem 0; max-width: 48%; vertical-align: top; }}
    img {{ max-width: 100%; border: 1px solid #d0d7de; }}
    figcaption {{ color: #57606a; font-size: 0.9rem; margin-top: 0.25rem; }}
  </style>
</head>
<body>
  <h1>{escaped_title}</h1>
  <p class="note">Generated from UPS light experiment <code>summary.json</code> artifacts. Use held-out split rows for benchmark claims; smoke rows are plumbing checks only.</p>
  {plot_html}
  <table>
    <thead><tr>{header_cells}</tr></thead>
    <tbody>{body}</tbody>
  </table>
</body>
</html>
"""
