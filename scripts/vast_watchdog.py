#!/usr/bin/env python3
"""
Lightweight watchdog for a running Vast.ai instance.

It polls the instance state and tail logs; if it detects a failure marker it
will issue a `vastai stop instance <id>` so the instance does not remain idle
and billable after an error.

Usage:
  python scripts/vast_watchdog.py --instance-id 12345 [--interval 60] [--runtime-mins 180]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from typing import Optional


def _run(cmd: list[str], timeout: int = 60) -> tuple[int, str]:
    """Run a command and return (returncode, stdout+stderr)."""
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=timeout)
        return 0, out
    except subprocess.CalledProcessError as exc:  # pragma: no cover - best effort helper
        return exc.returncode, exc.output
    except Exception as exc:  # pragma: no cover
        return 1, str(exc)


def fetch_instance_state(instance_id: int) -> Optional[str]:
    """Return cur_state of instance or None on failure."""
    rc, out = _run(["vastai", "show", "instance", str(instance_id), "--raw"])
    if rc != 0:
        return None
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return None
    return data.get("cur_state") or data.get("actual_status")


def tail_logs(instance_id: int, lines: int = 200) -> str:
    """Return the last N lines of logs (best effort)."""
    rc, out = _run(["vastai", "logs", str(instance_id), "--tail", str(lines)], timeout=120)
    return out if rc == 0 else ""


def stop_instance(instance_id: int) -> None:
    """Stop the instance (idempotent)."""
    _run(["vastai", "stop", "instance", str(instance_id)])


def contains_failure(log_text: str) -> bool:
    """Heuristic markers for failure in the Vast logs."""
    failure_markers = (
        "Training pipeline failed",
        "Traceback (most recent call last)",
        "MisconfigurationException",
        "RuntimeError",
    )
    return any(marker in log_text for marker in failure_markers)


def contains_success(log_text: str) -> bool:
    return "Training pipeline completed successfully" in log_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Watchdog for Vast.ai instance; stops it on failure.")
    parser.add_argument("--instance-id", type=int, required=True, help="Instance ID to monitor")
    parser.add_argument("--interval", type=int, default=60, help="Polling interval in seconds")
    parser.add_argument(
        "--runtime-mins",
        type=int,
        default=180,
        help="Maximum runtime for this watchdog before exiting (minutes)",
    )
    args = parser.parse_args()

    deadline = time.time() + args.runtime_mins * 60
    seen_failure = False

    print(f"[watchdog] Monitoring instance {args.instance_id} every {args.interval}s (max {args.runtime_mins} mins)")
    while time.time() < deadline:
        state = fetch_instance_state(args.instance_id)
        if state is None:
            print("[watchdog] Unable to fetch instance state; retrying...")
        else:
            print(f"[watchdog] Instance state: {state}")
            if state.lower() != "running":
                print("[watchdog] Instance not running; exiting watchdog")
                return

        logs = tail_logs(args.instance_id, lines=200)
        if not logs:
            print("[watchdog] No logs retrieved this cycle")
        elif contains_success(logs):
            print("[watchdog] Detected successful completion; exiting watchdog")
            return
        elif contains_failure(logs):
            if not seen_failure:
                print("[watchdog] Failure detected, stopping instance...")
                stop_instance(args.instance_id)
                seen_failure = True
            else:
                print("[watchdog] Failure already handled; still running in case stop is delayed")

        time.sleep(args.interval)

    print("[watchdog] Reached runtime limit; exiting without stopping instance.")


if __name__ == "__main__":
    sys.exit(main())
