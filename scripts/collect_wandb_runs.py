#!/usr/bin/env python
from __future__ import annotations

"""Collect W&B run summaries into local JSON/TSV experiment registry files."""

import argparse
import csv
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    json_dict = getattr(value, "_json_dict", None)
    if isinstance(json_dict, Mapping):
        return json_dict
    try:
        return dict(value)
    except Exception:
        return {}


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, str)):
        return str(value)
    return json.dumps(value, sort_keys=True)


def _scalar(value: Any) -> bool:
    return isinstance(value, (int, float, str, bool)) or value is None


def _matches_prefix(key: str, prefixes: Iterable[str]) -> bool:
    prefix_list = [prefix for prefix in prefixes if prefix]
    return not prefix_list or any(key.startswith(prefix) for prefix in prefix_list)


def run_to_record(run: Any, *, metric_prefixes: Iterable[str] = ()) -> dict[str, Any]:
    summary = _as_mapping(getattr(run, "summary", {}))
    config = _as_mapping(getattr(run, "config", {}))
    tags = list(getattr(run, "tags", []) or [])
    record: dict[str, Any] = {
        "wandb_id": getattr(run, "id", ""),
        "wandb_name": getattr(run, "name", ""),
        "wandb_state": getattr(run, "state", ""),
        "wandb_url": getattr(run, "url", ""),
        "wandb_project": getattr(run, "project", ""),
        "wandb_entity": getattr(run, "entity", ""),
        "wandb_group": getattr(run, "group", ""),
        "wandb_job_type": getattr(run, "job_type", ""),
        "wandb_tags": ",".join(str(tag) for tag in tags),
        "created_at": str(getattr(run, "created_at", "") or ""),
        "updated_at": str(getattr(run, "updated_at", "") or ""),
        "config_run_name": config.get("logging", {}).get("wandb", {}).get("run_name", "")
        if isinstance(config.get("logging"), Mapping)
        else "",
    }
    for key, value in summary.items():
        text_key = str(key)
        if text_key.startswith("_") or not _scalar(value) or not _matches_prefix(text_key, metric_prefixes):
            continue
        record[f"metric:{text_key}"] = value
    return record


def write_json(rows: list[dict[str, Any]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"runs": rows}, indent=2, sort_keys=True), encoding="utf-8")


def write_tsv(rows: list[dict[str, Any]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _stringify(row.get(key)) for key in fieldnames})


def _include_run(run: Any, *, tags: list[str], group: str, name_contains: str) -> bool:
    run_tags = {str(tag) for tag in (getattr(run, "tags", []) or [])}
    if tags and not set(tags).issubset(run_tags):
        return False
    if group and str(getattr(run, "group", "") or "") != group:
        return False
    if name_contains and name_contains not in str(getattr(run, "name", "") or ""):
        return False
    return True


def collect_records(
    *,
    entity: str,
    project: str,
    limit: int,
    tags: list[str],
    group: str,
    name_contains: str,
    metric_prefixes: list[str],
) -> list[dict[str, Any]]:
    try:
        import wandb  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit("wandb is required. Install with `pip install wandb`.") from exc

    api = wandb.Api()
    rows: list[dict[str, Any]] = []
    for run in api.runs(f"{entity}/{project}", per_page=min(max(limit, 1), 200)):
        if not _include_run(run, tags=tags, group=group, name_contains=name_contains):
            continue
        rows.append(run_to_record(run, metric_prefixes=metric_prefixes))
        if len(rows) >= limit:
            break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect W&B run summaries into local registry artifacts")
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "universal-simulator"))
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", ""))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--tag", action="append", default=[], help="Require this W&B tag; can be repeated")
    parser.add_argument("--group", default=os.environ.get("WANDB_GROUP", ""))
    parser.add_argument("--name-contains", default="")
    parser.add_argument("--metric-prefix", action="append", default=[], help="Metric prefix to retain; can be repeated")
    parser.add_argument("--out-json", default="reports/wandb/runs.json")
    parser.add_argument("--out-tsv", default="reports/wandb/runs.tsv")
    args = parser.parse_args()

    if not args.entity:
        raise SystemExit("Set --entity or WANDB_ENTITY.")

    rows = collect_records(
        entity=args.entity,
        project=args.project,
        limit=args.limit,
        tags=args.tag,
        group=args.group,
        name_contains=args.name_contains,
        metric_prefixes=args.metric_prefix,
    )
    write_json(rows, args.out_json)
    write_tsv(rows, args.out_tsv)
    print(args.out_json)
    print(args.out_tsv)


if __name__ == "__main__":
    main()
