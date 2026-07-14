#!/usr/bin/env python
"""Local, contract-aware billing watchdog for one-shot Vast jobs."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def update_receipt(path: Path, **updates: Any) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    payload.update(updates)
    payload["updated_at"] = utc_now()
    atomic_write_json(path, payload)
    return payload


def vast(args: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["vastai", *args], capture_output=True, text=True, check=False, timeout=30
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            ["vastai", *args], 124, exc.stdout or "", exc.stderr or "command timed out"
        )


def destroy(instance_id: int, *, attempts: int | None = None) -> bool:
    """Reconcile until destruction/absence; a finite attempt count is test-only."""
    attempt = 0
    while attempts is None or attempt < attempts:
        result = vast(["destroy", "instance", str(instance_id)])
        combined = f"{result.stdout}\n{result.stderr}".lower()
        if result.returncode == 0 or any(
            marker in combined
            for marker in ("not found", "does not exist", "already destroyed", "no such instance")
        ):
            return True
        attempt += 1
        if attempts is None or attempt < attempts:
            time.sleep(min(60.0, 2.0 ** min(attempt, 6)))
    return False


def instance_exists(instance_id: int) -> bool:
    result = vast(["show", "instance", str(instance_id), "--raw"])
    combined = f"{result.stdout}\n{result.stderr}".lower()
    if result.returncode == 0:
        return True
    return not any(
        marker in combined for marker in ("not found", "no such instance", "does not exist")
    )


def remote_logs(instance_id: int) -> str:
    result = vast(["logs", str(instance_id), "--tail", "1000"])
    return f"{result.stdout}\n{result.stderr}"


def terminal_reason(logs: str, success_marker: str) -> tuple[str, str] | None:
    exit_statuses = []
    for line in logs.splitlines():
        if "REMOTE_BOOTSTRAP_EXIT_STATUS=" in line:
            exit_statuses.append(line.rsplit("=", 1)[-1].strip())
    if exit_statuses:
        status = exit_statuses[-1]
        if status != "0":
            return "remote_failed", f"remote bootstrap exited {status}"
        if success_marker in logs:
            return "succeeded", "success marker and zero bootstrap exit observed"
        return "remote_failed", "bootstrap exited zero without publication success marker"
    return None


def monitor(
    receipt: Path,
    *,
    poll_seconds: float = 30.0,
    clock=time.time,
    sleeper=time.sleep,
) -> int:
    payload = json.loads(receipt.read_text())
    instance_id = int(payload["instance_id"])
    deadline = float(payload["deadline_unix"])
    success_marker = str(payload["success_marker"])
    stop_reason: list[str] = []

    def request_stop(signum: int, _frame: object) -> None:
        stop_reason.append(f"signal_{signum}")

    previous_handlers: dict[int, Any] = {}
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        previous_handlers[signum] = signal.signal(signum, request_stop)
    update_receipt(receipt, status="watching", watchdog_pid=os.getpid())
    try:
        while True:
            if stop_reason:
                status, reason = "watchdog_stopped", stop_reason[-1]
                break
            if clock() >= deadline:
                status, reason = "timed_out", "maximum paid runtime reached"
                break
            logs = remote_logs(instance_id)
            terminal = terminal_reason(logs, success_marker)
            if terminal is not None:
                status, reason = terminal
                break
            if not instance_exists(instance_id):
                status, reason = "instance_absent", "instance no longer exists"
                update_receipt(receipt, status=status, terminal_reason=reason, destroyed=True)
                return 0
            sleeper(max(1.0, poll_seconds))

        update_receipt(receipt, status="destroying", terminal_reason=reason)
        destroyed = destroy(instance_id)
        update_receipt(
            receipt,
            status=status if destroyed else "destroy_failed",
            terminal_reason=reason,
            destroyed=destroyed,
            destroyed_at=utc_now() if destroyed else None,
        )
        return 0 if destroyed else 1
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.receipt.is_file():
        raise SystemExit(f"receipt not found: {args.receipt}")
    raise SystemExit(monitor(args.receipt, poll_seconds=args.poll_seconds))


if __name__ == "__main__":
    main()
