from __future__ import annotations

import os
import subprocess


def test_official_transport_objective_status_dry_run_defaults_to_literal_release_check():
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    proc = subprocess.run(
        ["bash", "scripts/run_official_transport_objective_status.sh"],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert "aggregate transport objective status" in proc.stdout
    assert "transport_shift_goal_audit.json" in proc.stdout
    assert "observed_transport_shift_goal_audit.json" in proc.stdout
    assert "train_only_transport_feature_diagnostic_full.json" in proc.stdout
    assert "require_status=literal-achieved" in proc.stdout
    assert "accept_observed_context=0" in proc.stdout


def test_official_transport_objective_status_dry_run_reports_observed_policy_acceptance():
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["ACCEPT_OBSERVED_CONTEXT"] = "1"
    env["REQUIRE_STATUS"] = "observed-accepted"
    proc = subprocess.run(
        ["bash", "scripts/run_official_transport_objective_status.sh"],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert "require_status=observed-accepted" in proc.stdout
    assert "accept_observed_context=1" in proc.stdout
