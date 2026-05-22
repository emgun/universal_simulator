#!/usr/bin/env python
from __future__ import annotations

"""Resolve official Dataverse file URLs into a derived hydration plan."""

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.download_pdebench_file import DATAFILE_URL, _resolve_redirect_url_with_curl


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _entry_url(entry: dict[str, Any]) -> str:
    explicit_url = entry.get("url") or entry.get("download_url") or entry.get("source_url")
    if explicit_url:
        return str(explicit_url)
    return DATAFILE_URL.format(file_id=entry.get("file_id", ""), path=entry.get("path", ""))


def _download_commands(entries: list[dict[str, Any]], out_root: str) -> list[str]:
    commands = []
    for entry in entries:
        path = str(entry.get("path") or "")
        source_url = str(entry.get("source_url") or entry.get("url") or entry.get("download_url") or "")
        prefix = (
            f"PDEBENCH_DATAFILE_URL_TEMPLATE={shlex.quote(source_url)} "
            if source_url
            else ""
        )
        commands.append(f"{prefix}python scripts/download_pdebench_file.py {shlex.quote(path)} --out {out_root}")
    return commands


def resolve_plan_urls(args: argparse.Namespace) -> dict[str, Any]:
    plan = _load_json(args.plan_json)
    entries = [dict(entry) for entry in plan.get("remote_entries") or []]
    records: list[dict[str, Any]] = []
    blockers: list[str] = []
    resolved_entries: list[dict[str, Any]] = []

    for entry in entries:
        original_url = _entry_url(entry)
        record = {
            "path": entry.get("path"),
            "file_id": entry.get("file_id"),
            "source_url": None,
            "resolved": False,
            "error": None,
        }
        try:
            resolved_url = _resolve_redirect_url_with_curl(
                original_url,
                timeout=args.timeout,
                retries=args.retries,
                retry_backoff=args.retry_backoff,
            )
        except Exception as exc:
            record["error"] = str(exc)
            blockers.append(f"{entry.get('path')}: {exc}")
            if not args.continue_on_error:
                records.append(record)
                break
            resolved_entries.append(entry)
        else:
            entry["source_url"] = resolved_url
            entry["resolved_from_url"] = original_url
            record["source_url"] = resolved_url
            record["resolved"] = True
            resolved_entries.append(entry)
        records.append(record)

    if blockers and not args.continue_on_error:
        status = "blocked"
    elif blockers:
        status = "partial"
    else:
        status = "resolved"

    output_plan = None
    if status in {"resolved", "partial"}:
        output_plan = dict(plan)
        output_plan["remote_entries"] = resolved_entries
        output_plan["url_resolution"] = {
            "source_plan_json": str(args.plan_json),
            "status": status,
            "resolved_count": sum(1 for row in records if row["resolved"]),
            "selected_file_count": len(entries),
        }
        raw_out = str(output_plan.get("raw_out") or "data/pdebench/raw")
        commands = dict(output_plan.get("commands") or {})
        commands["download_official_train_files"] = _download_commands(resolved_entries, raw_out)
        output_plan["commands"] = commands
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output_plan, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "status": status,
        "blockers": blockers,
        "plan_json": str(args.plan_json),
        "output_json": str(args.output_json),
        "selected_file_count": len(entries),
        "resolved_count": sum(1 for row in records if row["resolved"]),
        "entries": records,
        "wrote_output_plan": output_plan is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve official hydration plan URLs to direct source URLs")
    parser.add_argument("--plan-json", default="reports/research/sota_loop/official_advection_hydration_plan.json")
    parser.add_argument(
        "--output-json",
        default="reports/research/sota_loop/official_advection_hydration_plan_resolved_urls.json",
    )
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--retry-backoff", type=float, default=2.0)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    record = resolve_plan_urls(args)
    print(json.dumps(record, indent=2, sort_keys=True))
    raise SystemExit(0 if record["status"] in {"resolved", "partial"} else 2)


if __name__ == "__main__":
    main()
