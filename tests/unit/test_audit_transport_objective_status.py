from __future__ import annotations

from argparse import Namespace
import json

from scripts.audit_transport_objective_status import audit_objective, exit_code_for_status


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _args(tmp_path, *, accept_observed_context: bool = False, accept_context_transport: bool = False):
    return Namespace(
        constant_audit_json=str(tmp_path / "constant.json"),
        observed_audit_json=str(tmp_path / "observed.json"),
        context_audit_json=str(tmp_path / "context.json"),
        train_feature_diagnostic_json=str(tmp_path / "features.json"),
        train_identifiability_audit_json=str(tmp_path / "identifiability.json"),
        hydration_options_json=str(tmp_path / "hydration.json"),
        hydration_plan_json=str(tmp_path / "hydration_plan.json"),
        hydration_plan_validation_json=str(tmp_path / "hydration_plan_validation.json"),
        hydration_plan_run_json=str(tmp_path / "hydration_plan_run.json"),
        hydration_preflight_json=str(tmp_path / "hydration_preflight.json"),
        accept_observed_context=accept_observed_context,
        accept_context_transport=accept_context_transport,
        require_status="report",
        output_json=str(tmp_path / "objective.json"),
    )


def test_objective_audit_marks_literal_achieved_when_constant_audit_achieved(tmp_path):
    _write_json(tmp_path / "constant.json", {"status": "achieved", "result_record_policy": {"passed": True}})
    _write_json(tmp_path / "observed.json", {"status": "achieved"})
    _write_json(tmp_path / "context.json", {"status": "achieved"})
    _write_json(tmp_path / "features.json", {"conclusion": "blocked_no_train_support_for_validation_shift"})
    _write_json(tmp_path / "identifiability.json", {"status": "blocked_underidentified_train_only_shift"})
    _write_json(tmp_path / "hydration.json", {"status": "remote_official_hydration_required"})
    _write_json(tmp_path / "hydration_plan.json", {"status": "ready_for_explicit_hydration"})
    _write_json(tmp_path / "hydration_plan_validation.json", {"status": "valid"})
    _write_json(tmp_path / "hydration_plan_run.json", {"status": "dry_run"})
    _write_json(tmp_path / "hydration_preflight.json", {"status": "blocked_insufficient_disk"})

    record = audit_objective(_args(tmp_path))

    assert record["status"] == "literal_achieved"
    assert not record["blockers"]


def test_objective_audit_keeps_observed_context_policy_explicit(tmp_path):
    _write_json(tmp_path / "constant.json", {"status": "blocked_incompatible_splits"})
    _write_json(tmp_path / "context.json", {"status": "missing_evidence"})
    _write_json(
        tmp_path / "observed.json",
        {"status": "achieved", "result_record_policy": {"passed": True}, "held_out_test_policy": {"test_result_count": 1}},
    )
    _write_json(tmp_path / "features.json", {"conclusion": "blocked_no_train_support_for_validation_shift"})
    _write_json(tmp_path / "identifiability.json", {"status": "blocked_underidentified_train_only_shift"})
    _write_json(tmp_path / "hydration.json", {"status": "remote_official_hydration_required"})
    _write_json(tmp_path / "hydration_plan.json", {"status": "ready_for_explicit_hydration"})
    _write_json(tmp_path / "hydration_plan_validation.json", {"status": "valid"})
    _write_json(tmp_path / "hydration_plan_run.json", {"status": "dry_run"})
    _write_json(tmp_path / "hydration_preflight.json", {"status": "blocked_insufficient_disk"})

    blocked = audit_objective(_args(tmp_path))
    accepted = audit_objective(_args(tmp_path, accept_observed_context=True))

    assert blocked["status"] == "literal_blocked"
    assert "observed-context result is achieved but not accepted" in blocked["blockers"][-1]
    assert accepted["status"] == "observed_context_achieved"
    assert accepted["accept_observed_context"] is True


def test_objective_audit_keeps_context_transport_policy_explicit(tmp_path):
    _write_json(tmp_path / "constant.json", {"status": "blocked_incompatible_splits"})
    _write_json(tmp_path / "observed.json", {"status": "missing_evidence"})
    _write_json(
        tmp_path / "context.json",
        {"status": "achieved", "result_record_policy": {"passed": True}, "held_out_test_policy": {"test_result_count": 1}},
    )
    _write_json(tmp_path / "features.json", {"conclusion": "blocked_no_train_support_for_validation_shift"})
    _write_json(tmp_path / "identifiability.json", {"status": "blocked_underidentified_train_only_shift"})
    _write_json(tmp_path / "hydration.json", {"status": "remote_official_hydration_required"})
    _write_json(tmp_path / "hydration_plan.json", {"status": "ready_for_explicit_hydration"})
    _write_json(tmp_path / "hydration_plan_validation.json", {"status": "valid"})
    _write_json(tmp_path / "hydration_plan_run.json", {"status": "dry_run"})
    _write_json(tmp_path / "hydration_preflight.json", {"status": "blocked_insufficient_disk"})

    blocked = audit_objective(_args(tmp_path))
    accepted = audit_objective(_args(tmp_path, accept_context_transport=True))

    assert blocked["status"] == "literal_blocked"
    assert "two-frame context transport result is achieved but not accepted" in blocked["blockers"][-1]
    assert accepted["status"] == "context_transport_achieved"
    assert accepted["accept_context_transport"] is True


def test_objective_audit_reports_train_feature_blocker(tmp_path):
    _write_json(tmp_path / "constant.json", {"status": "blocked_incompatible_splits"})
    _write_json(tmp_path / "observed.json", {"status": "missing_evidence"})
    _write_json(tmp_path / "context.json", {"status": "missing_evidence"})
    _write_json(tmp_path / "features.json", {"conclusion": "blocked_no_train_support_for_validation_shift"})
    _write_json(tmp_path / "identifiability.json", {"status": "blocked_underidentified_train_only_shift"})
    _write_json(tmp_path / "hydration.json", {"status": "remote_official_hydration_required"})
    _write_json(tmp_path / "hydration_plan.json", {"status": "ready_for_explicit_hydration"})
    _write_json(tmp_path / "hydration_plan_validation.json", {"status": "valid"})
    _write_json(tmp_path / "hydration_plan_run.json", {"status": "dry_run"})
    _write_json(tmp_path / "hydration_preflight.json", {"status": "blocked_insufficient_disk"})

    record = audit_objective(_args(tmp_path))

    assert record["status"] == "literal_blocked"
    assert any("blocked_no_train_support_for_validation_shift" in blocker for blocker in record["blockers"])
    assert any("blocked_underidentified_train_only_shift" in blocker for blocker in record["blockers"])
    assert any("remote_official_hydration_required" in blocker for blocker in record["blockers"])
    assert any("ready_for_explicit_hydration" in blocker for blocker in record["blockers"])
    assert any("valid" in blocker for blocker in record["blockers"])
    assert any("dry_run" in blocker for blocker in record["blockers"])
    assert any("blocked_insufficient_disk" in blocker for blocker in record["blockers"])


def test_objective_exit_policy():
    assert exit_code_for_status("literal_achieved", "literal-achieved") == 0
    assert exit_code_for_status("observed_context_achieved", "literal-achieved") == 2
    assert exit_code_for_status("observed_context_achieved", "observed-accepted") == 0
    assert exit_code_for_status("context_transport_achieved", "context-accepted") == 0
    assert exit_code_for_status("observed_context_achieved", "context-accepted") == 2
    assert exit_code_for_status("literal_blocked", "report") == 0
