from __future__ import annotations

"""Utilities for turning light experiment summaries into demo scorecards."""

import csv
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

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
)


@dataclass(frozen=True)
class Scorecard:
    rows: list[dict[str, Any]]
    metric_keys: list[str]


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


def scorecard_row_from_summary(
    summary: Mapping[str, Any],
    *,
    data_manifest: str | None = None,
    commit: str | None = None,
    promotion_rules: Sequence[str] = (),
) -> dict[str, Any]:
    metrics = {str(key): float(value) for key, value in dict(summary.get("metrics", {})).items()}
    extra = dict(summary.get("extra", {}))
    main_metric_name, main_metric_value = _main_metric(metrics)

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
) -> Scorecard:
    rows = [
        scorecard_row_from_summary(
            load_summary(path),
            data_manifest=data_manifest,
            commit=commit,
            promotion_rules=promotion_rules,
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


def render_scorecard_html(scorecard: Scorecard, *, title: str = "UPS Demo Scorecard") -> str:
    fields = list(BASE_FIELDS) + scorecard.metric_keys
    header_cells = "".join(f"<th>{html.escape(field)}</th>" for field in fields)
    body_rows = []
    for row in scorecard.rows:
        cells = "".join(f"<td>{html.escape(_stringify(row.get(field)))}</td>" for field in fields)
        body_rows.append(f"<tr>{cells}</tr>")
    body = "\n".join(body_rows) or f"<tr><td colspan=\"{len(fields)}\">No runs found.</td></tr>"
    escaped_title = html.escape(title)
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
  </style>
</head>
<body>
  <h1>{escaped_title}</h1>
  <p class="note">Generated from UPS light experiment <code>summary.json</code> artifacts. Use held-out split rows for benchmark claims; smoke rows are plumbing checks only.</p>
  <table>
    <thead><tr>{header_cells}</tr></thead>
    <tbody>{body}</tbody>
  </table>
</body>
</html>
"""

