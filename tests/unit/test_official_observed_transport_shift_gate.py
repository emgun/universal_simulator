from __future__ import annotations

import os
import subprocess


def test_official_observed_transport_shift_gate_dry_run_is_local_safe():
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    proc = subprocess.run(
        ["bash", "scripts/run_official_observed_transport_shift_gate.sh"],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert "run official observed advection1d train=train val=val gate" in proc.stdout
    assert "gate measures held-out test only if validation passes" in proc.stdout
    assert "observed_transport_shift_gate_real_light_v1.json" in proc.stdout
    assert "enforce exactly-once held-out test ledger" in proc.stdout
    assert "observed_transport_shift_goal_audit.json" in proc.stdout
    assert "require_status=achieved" in proc.stdout
    assert "allow_repeat_test=0" in proc.stdout


def test_official_observed_transport_shift_gate_dry_run_reports_repeat_override():
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["ALLOW_REPEAT_TEST"] = "1"
    proc = subprocess.run(
        ["bash", "scripts/run_official_observed_transport_shift_gate.sh"],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert "allow_repeat_test=1" in proc.stdout
