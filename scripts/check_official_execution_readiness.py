#!/usr/bin/env python
from __future__ import annotations

"""Check whether the official Advection transport objective can execute here."""

import argparse
import json
import os
import shutil
import socket
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


DEFAULT_DATAFILE_URL = "https://darus.uni-stuttgart.de/api/access/datafile/{file_id}?format=original"


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _dns_record(host: str) -> dict[str, Any]:
    try:
        addresses = sorted({row[4][0] for row in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)})
    except OSError as exc:
        return {"host": host, "resolves": False, "error": str(exc), "addresses": []}
    return {"host": host, "resolves": True, "error": None, "addresses": addresses}


def _disk_record(root: str | Path) -> dict[str, Any]:
    path = Path(root)
    probe = path if path.exists() else path.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    return {
        "root": str(path),
        "probe_root": str(probe),
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "free_bytes": int(usage.free),
    }


def _entry_url(entry: dict[str, Any], default_host: str) -> str:
    explicit_url = entry.get("url") or entry.get("download_url") or entry.get("source_url")
    if explicit_url:
        return str(explicit_url)
    template = os.environ.get(
        "PDEBENCH_DATAFILE_URL_TEMPLATE",
        f"https://{default_host}/api/access/datafile/{{file_id}}?format=original",
    )
    return template.format(file_id=entry.get("file_id", ""), path=entry.get("path", ""))


def _official_data_dns_records(entries: list[dict[str, Any]], default_host: str) -> dict[str, Any]:
    hosts = []
    for entry in entries:
        parsed = urlparse(_entry_url(entry, default_host))
        host = parsed.hostname or default_host
        if host not in hosts:
            hosts.append(host)
    if not hosts:
        hosts = [default_host]
    records = [_dns_record(host) for host in hosts]
    return {
        "hosts": hosts,
        "records": records,
        "resolves": all(bool(record["resolves"]) for record in records),
        "unresolved_hosts": [str(record["host"]) for record in records if not record["resolves"]],
    }


def _staged_raw_record(raw_root: str | Path, entries: list[dict[str, Any]]) -> dict[str, Any]:
    root = Path(raw_root)
    files = []
    all_present = bool(entries)
    for entry in entries:
        logical_path = str(entry.get("path") or "")
        expected_size = int(entry.get("size_bytes") or 0)
        path = root / logical_path
        exists = path.exists()
        actual_size = path.stat().st_size if exists else 0
        complete = bool(exists and expected_size > 0 and actual_size == expected_size)
        all_present = all_present and complete
        files.append(
            {
                "logical_path": logical_path,
                "local_path": str(path),
                "exists": exists,
                "expected_size_bytes": expected_size,
                "actual_size_bytes": actual_size,
                "complete": complete,
            }
        )
    return {
        "raw_root": str(root),
        "all_present": all_present,
        "complete_file_count": sum(1 for row in files if row["complete"]),
        "selected_file_count": len(entries),
        "files": files,
    }


def check_readiness(args: argparse.Namespace) -> dict[str, Any]:
    plan = _load_json(args.plan_json)
    remote_plan = _load_json(args.remote_plan_json)
    local_disk = _disk_record(args.local_disk_root)
    remote_dns = _dns_record(args.remote_api_host)
    remote_entries = list(plan.get("remote_entries", []))
    official_data_dns = _official_data_dns_records(remote_entries, args.official_data_host)
    raw_root = str(plan.get("raw_out") or args.local_disk_root)
    staged_raw = _staged_raw_record(raw_root, remote_entries)

    largest_file_bytes = max(
        (int(entry.get("size_bytes") or 0) for entry in remote_entries),
        default=0,
    )
    total_download_bytes = int(plan.get("estimated_download_bytes") or 0)
    local_required_bytes = int(max(largest_file_bytes, total_download_bytes) * float(args.local_safety_factor))
    sequential_required_bytes = int(largest_file_bytes * float(args.local_safety_factor))
    remote_required_gb = int(remote_plan.get("required_disk_gb") or 0)

    remote_blockers: list[str] = []
    local_blockers: list[str] = []
    if not remote_dns["resolves"]:
        remote_blockers.append(f"remote API host {args.remote_api_host} does not resolve")
    if not staged_raw["all_present"] and not official_data_dns["resolves"]:
        local_blockers.append(
            "official data host(s) do not resolve: "
            + ", ".join(official_data_dns["unresolved_hosts"])
        )
    if not staged_raw["all_present"] and int(local_disk["free_bytes"]) < sequential_required_bytes:
        local_blockers.append(
            f"local free bytes {local_disk['free_bytes']} below sequential requirement {sequential_required_bytes}"
        )

    remote_ready = not remote_blockers
    local_ready = not local_blockers
    blockers = [] if remote_ready or local_ready else sorted(set(remote_blockers + local_blockers))
    status = "ready" if remote_ready or local_ready else "blocked"
    return {
        "status": status,
        "blockers": blockers,
        "plan_json": str(args.plan_json),
        "remote_plan_json": str(args.remote_plan_json),
        "remote_launch_ready": remote_ready,
        "local_sequential_hydration_ready": local_ready,
        "route_blockers": {
            "remote_launch": remote_blockers,
            "local_sequential_hydration": local_blockers,
        },
        "dns": {
            "remote_api": remote_dns,
            "official_data": official_data_dns,
        },
        "staged_raw": staged_raw,
        "disk": {
            "local": local_disk,
            "largest_official_file_bytes": largest_file_bytes,
            "estimated_total_download_bytes": total_download_bytes,
            "local_all_files_required_bytes": local_required_bytes,
            "local_sequential_required_bytes": sequential_required_bytes,
            "remote_required_disk_gb": remote_required_gb,
        },
        "next_action": (
            "run pinned remote actual_launcher"
            if remote_ready
            else "run local sequential hydration"
            if local_ready
            else "restore DNS and provide enough local disk or a reachable remote executor"
        ),
        "held_out_test_policy": plan.get("held_out_test_policy"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check official transport execution readiness")
    parser.add_argument("--plan-json", default="reports/research/sota_loop/official_advection_hydration_plan.json")
    parser.add_argument(
        "--remote-plan-json",
        default="reports/research/sota_loop/remote_official_advection_hydration_plan.json",
    )
    parser.add_argument("--local-disk-root", default="data/pdebench/raw")
    parser.add_argument("--local-safety-factor", type=float, default=1.15)
    parser.add_argument("--remote-api-host", default="console.vast.ai")
    parser.add_argument("--official-data-host", default="darus.uni-stuttgart.de")
    parser.add_argument(
        "--output-json",
        default="reports/research/sota_loop/official_execution_readiness.json",
    )
    args = parser.parse_args()

    record = check_readiness(args)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))
    raise SystemExit(0 if record["status"] == "ready" else 2)


if __name__ == "__main__":
    main()
