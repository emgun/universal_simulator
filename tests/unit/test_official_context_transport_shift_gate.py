from __future__ import annotations

import os
import subprocess


def test_official_context_transport_gate_dry_run_describes_guarded_command():
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    proc = subprocess.run(
        ["bash", "scripts/run_official_context_transport_shift_gate.sh"],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert "official context advection1d" in proc.stdout
    assert "measures held-out test only if validation passes" in proc.stdout
    assert "context_transport_shift_gate.json" in proc.stdout
    assert "context_transport_shift_test_ledger.json" in proc.stdout
    assert "context_transport_shift_goal_audit.json" in proc.stdout
    assert "require_status=achieved" in proc.stdout


def test_official_context_transport_gate_dry_run_respects_repeat_flag():
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["ALLOW_REPEAT_TEST"] = "1"
    proc = subprocess.run(
        ["bash", "scripts/run_official_context_transport_shift_gate.sh"],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert "allow_repeat_test=1" in proc.stdout
