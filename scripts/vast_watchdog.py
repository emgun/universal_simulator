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


def instance_state(instance_id: int) -> dict[str, Any] | None:
    # The singular CLI endpoint currently raises an internal TypeError for a
    # destroyed instance. Query the collection instead and fail safe on API or
    # JSON errors so a transient control-plane failure never stops monitoring.
    result = vast(["show", "instances", "--raw"])
    if result.returncode != 0:
        return {}
    try:
        rows = json.loads(result.stdout or "[]")
        return next((row for row in rows if int(row.get("id", -1)) == instance_id), None)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}


def instance_exists(instance_id: int) -> bool:
    return instance_state(instance_id) is not None


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
    startup_deadline = float(payload.get("startup_deadline_unix", deadline))
    success_marker = str(payload["success_marker"])
    bootstrap_started = bool(payload.get("bootstrap_started", False))
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
            if not bootstrap_started and "REMOTE_BOOTSTRAP_STARTED=1" in logs:
                bootstrap_started = True
                update_receipt(
                    receipt,
                    bootstrap_started=True,
                    bootstrap_started_at=utc_now(),
                )
            terminal = terminal_reason(logs, success_marker)
            if terminal is not None:
                status, reason = terminal
                break
            state = instance_state(instance_id)
            if state is None:
                status = "instance_absent" if bootstrap_started else "startup_failed"
                reason = (
                    "instance no longer exists"
                    if bootstrap_started
                    else "instance disappeared before tracked bootstrap"
                )
                update_receipt(receipt, status=status, terminal_reason=reason, destroyed=True)
                return 0
            if not bootstrap_started and clock() >= startup_deadline:
                status, reason = (
                    "startup_failed",
                    "tracked bootstrap did not start before startup deadline",
                )
                if state:
                    update_receipt(
                        receipt,
                        last_instance_status=state.get("actual_status"),
                        last_instance_status_message=state.get("status_msg"),
                    )
                break
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
