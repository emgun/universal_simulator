from __future__ import annotations

import os
from pathlib import Path
import subprocess


def _write_json(path: Path, payload: str) -> None:
    path.write_text(payload, encoding="utf-8")


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
    assert "context_transport_shift_goal_audit.json" in proc.stdout
    assert "observed_transport_shift_goal_audit.json" in proc.stdout
    assert "train_only_transport_feature_diagnostic_full.json" in proc.stdout
    assert "require_status=literal-achieved" in proc.stdout
    assert "accept_context_transport=0" in proc.stdout
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


def test_official_transport_objective_status_dry_run_reports_context_policy_acceptance():
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["ACCEPT_CONTEXT_TRANSPORT"] = "1"
    env["REQUIRE_STATUS"] = "context-accepted"
    proc = subprocess.run(
        ["bash", "scripts/run_official_transport_objective_status.sh"],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert "require_status=context-accepted" in proc.stdout
    assert "accept_context_transport=1" in proc.stdout


def test_official_transport_objective_status_executes_default_literal_check(tmp_path):
    constant = tmp_path / "constant.json"
    context = tmp_path / "context.json"
    observed = tmp_path / "observed.json"
    features = tmp_path / "features.json"
    output = tmp_path / "objective.json"
    _write_json(constant, '{"status":"blocked_incompatible_splits"}')
    _write_json(context, '{"status":"achieved","result_record_policy":{"passed":true}}')
    _write_json(observed, '{"status":"achieved","result_record_policy":{"passed":true}}')
    _write_json(features, '{"conclusion":"blocked_no_train_support_for_validation_shift"}')
    env = os.environ.copy()
    env["CONSTANT_AUDIT_JSON"] = str(constant)
    env["CONTEXT_AUDIT_JSON"] = str(context)
    env["OBSERVED_AUDIT_JSON"] = str(observed)
    env["TRAIN_FEATURE_DIAGNOSTIC_JSON"] = str(features)
    env["OBJECTIVE_STATUS_JSON"] = str(output)
    env["REQUIRE_STATUS"] = "report"

    proc = subprocess.run(
        ["bash", "scripts/run_official_transport_objective_status.sh"],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert '"status": "literal_blocked"' in proc.stdout
    assert output.exists()
