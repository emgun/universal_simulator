from __future__ import annotations

import os
import subprocess


def test_official_transport_shift_audit_dry_run_is_local_safe():
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    proc = subprocess.run(
        ["bash", "scripts/run_official_transport_shift_audit.sh"],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert "run official advection1d train=train val=val gate" in proc.stdout
    assert "gate measures held-out test only if validation passes" in proc.stdout
    assert "require_status=achieved" in proc.stdout
    assert "require data SHA-256" in proc.stdout


def test_official_transport_shift_audit_dry_run_can_refresh_report_only():
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["AUDIT_REQUIRE_STATUS"] = "report"
    proc = subprocess.run(
        ["bash", "scripts/run_official_transport_shift_audit.sh"],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert "require_status=report" in proc.stdout
