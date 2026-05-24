from __future__ import annotations

import json
from argparse import Namespace

from scripts.audit_transport_objective_status import audit_objective, exit_code_for_status


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _args(
    tmp_path, *, accept_observed_context: bool = False, accept_context_transport: bool = False
):
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
        hydration_storage_json=str(tmp_path / "hydration_storage.json"),
        remote_hydration_plan_json=str(tmp_path / "remote_hydration.json"),
        execution_readiness_json=str(tmp_path / "execution_readiness.json"),
        official_hydrated_gate_json=str(tmp_path / "official_hydrated_gate.json"),
        accept_observed_context=accept_observed_context,
        accept_context_transport=accept_context_transport,
        require_status="report",
        output_json=str(tmp_path / "objective.json"),
    )


def test_objective_audit_marks_literal_achieved_when_constant_audit_achieved(tmp_path):
    _write_json(
        tmp_path / "constant.json", {"status": "achieved", "result_record_policy": {"passed": True}}
    )
    _write_json(tmp_path / "observed.json", {"status": "achieved"})
    _write_json(tmp_path / "context.json", {"status": "achieved"})
    _write_json(
        tmp_path / "features.json", {"conclusion": "blocked_no_train_support_for_validation_shift"}
    )
    _write_json(
        tmp_path / "identifiability.json", {"status": "blocked_underidentified_train_only_shift"}
    )
    _write_json(tmp_path / "hydration.json", {"status": "remote_official_hydration_required"})
    _write_json(tmp_path / "hydration_plan.json", {"status": "ready_for_explicit_hydration"})
    _write_json(tmp_path / "hydration_plan_validation.json", {"status": "valid"})
    _write_json(tmp_path / "hydration_plan_run.json", {"status": "dry_run"})
    _write_json(tmp_path / "hydration_preflight.json", {"status": "blocked_insufficient_disk"})
    _write_json(tmp_path / "hydration_storage.json", {"status": "external_or_freed_space_required"})
    _write_json(tmp_path / "remote_hydration.json", {"status": "ready_for_remote_hydration"})
    _write_json(tmp_path / "execution_readiness.json", {"status": "blocked"})
    _write_json(tmp_path / "official_hydrated_gate.json", {})

    record = audit_objective(_args(tmp_path))

    assert record["status"] == "literal_achieved"
    assert not record["blockers"]


def test_objective_audit_keeps_observed_context_policy_explicit(tmp_path):
    _write_json(tmp_path / "constant.json", {"status": "blocked_incompatible_splits"})
    _write_json(tmp_path / "context.json", {"status": "missing_evidence"})
    _write_json(
        tmp_path / "observed.json",
        {
            "status": "achieved",
            "result_record_policy": {"passed": True},
            "held_out_test_policy": {"test_result_count": 1},
        },
    )
    _write_json(
        tmp_path / "features.json", {"conclusion": "blocked_no_train_support_for_validation_shift"}
    )
    _write_json(
        tmp_path / "identifiability.json", {"status": "blocked_underidentified_train_only_shift"}
    )
    _write_json(tmp_path / "hydration.json", {"status": "remote_official_hydration_required"})
    _write_json(tmp_path / "hydration_plan.json", {"status": "ready_for_explicit_hydration"})
    _write_json(tmp_path / "hydration_plan_validation.json", {"status": "valid"})
    _write_json(tmp_path / "hydration_plan_run.json", {"status": "dry_run"})
    _write_json(tmp_path / "hydration_preflight.json", {"status": "blocked_insufficient_disk"})
    _write_json(tmp_path / "hydration_storage.json", {"status": "external_or_freed_space_required"})
    _write_json(tmp_path / "remote_hydration.json", {"status": "ready_for_remote_hydration"})
    _write_json(tmp_path / "execution_readiness.json", {"status": "blocked"})
    _write_json(tmp_path / "official_hydrated_gate.json", {})

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
        {
            "status": "achieved",
            "result_record_policy": {"passed": True},
            "held_out_test_policy": {"test_result_count": 1},
        },
    )
    _write_json(
        tmp_path / "features.json", {"conclusion": "blocked_no_train_support_for_validation_shift"}
    )
    _write_json(
        tmp_path / "identifiability.json", {"status": "blocked_underidentified_train_only_shift"}
    )
    _write_json(tmp_path / "hydration.json", {"status": "remote_official_hydration_required"})
    _write_json(tmp_path / "hydration_plan.json", {"status": "ready_for_explicit_hydration"})
    _write_json(tmp_path / "hydration_plan_validation.json", {"status": "valid"})
    _write_json(tmp_path / "hydration_plan_run.json", {"status": "dry_run"})
    _write_json(tmp_path / "hydration_preflight.json", {"status": "blocked_insufficient_disk"})
    _write_json(tmp_path / "hydration_storage.json", {"status": "external_or_freed_space_required"})
    _write_json(tmp_path / "remote_hydration.json", {"status": "ready_for_remote_hydration"})
    _write_json(tmp_path / "execution_readiness.json", {"status": "blocked"})
    _write_json(tmp_path / "official_hydrated_gate.json", {})

    blocked = audit_objective(_args(tmp_path))
    accepted = audit_objective(_args(tmp_path, accept_context_transport=True))

    assert blocked["status"] == "literal_blocked"
    assert (
        "two-frame context transport result is achieved but not accepted" in blocked["blockers"][-1]
    )
    assert accepted["status"] == "context_transport_achieved"
    assert accepted["accept_context_transport"] is True
    assert any(
        requirement["name"] == "fit_transport_shift_only_on_train"
        and requirement["status"] == "satisfied"
        and "context_transport_achieved=True" in requirement["evidence"]
        for requirement in accepted["requirements"]
    )


def test_objective_audit_reports_train_feature_blocker(tmp_path):
    _write_json(tmp_path / "constant.json", {"status": "blocked_incompatible_splits"})
    _write_json(tmp_path / "observed.json", {"status": "missing_evidence"})
    _write_json(tmp_path / "context.json", {"status": "missing_evidence"})
    _write_json(
        tmp_path / "features.json", {"conclusion": "blocked_no_train_support_for_validation_shift"}
    )
    _write_json(
        tmp_path / "identifiability.json", {"status": "blocked_underidentified_train_only_shift"}
    )
    _write_json(tmp_path / "hydration.json", {"status": "remote_official_hydration_required"})
    _write_json(tmp_path / "hydration_plan.json", {"status": "ready_for_explicit_hydration"})
    _write_json(tmp_path / "hydration_plan_validation.json", {"status": "valid"})
    _write_json(tmp_path / "hydration_plan_run.json", {"status": "dry_run"})
    _write_json(tmp_path / "hydration_preflight.json", {"status": "blocked_insufficient_disk"})
    _write_json(tmp_path / "hydration_storage.json", {"status": "external_or_freed_space_required"})
    _write_json(tmp_path / "remote_hydration.json", {"status": "ready_for_remote_hydration"})
    _write_json(
        tmp_path / "execution_readiness.json",
        {
            "status": "blocked",
            "route_blockers": {
                "remote_launch": ["remote API host console.vast.ai does not resolve"],
                "local_sequential_hydration": ["local free bytes 1 below sequential requirement 2"],
            },
        },
    )
    _write_json(tmp_path / "official_hydrated_gate.json", {})

    record = audit_objective(_args(tmp_path))

    assert record["status"] == "literal_blocked"
    assert any(
        "blocked_no_train_support_for_validation_shift" in blocker for blocker in record["blockers"]
    )
    assert any(
        "blocked_underidentified_train_only_shift" in blocker for blocker in record["blockers"]
    )
    assert any("remote_official_hydration_required" in blocker for blocker in record["blockers"])
    assert any("dry_run" in blocker for blocker in record["blockers"])
    assert any("blocked_insufficient_disk" in blocker for blocker in record["blockers"])
    assert any("external_or_freed_space_required" in blocker for blocker in record["blockers"])
    assert not any("ready_for_explicit_hydration" in blocker for blocker in record["blockers"])
    assert not any("ready_for_remote_hydration" in blocker for blocker in record["blockers"])
    assert any(
        "official execution readiness status is blocked" in blocker
        for blocker in record["blockers"]
    )
    assert any("remote_launch blocker" in blocker for blocker in record["blockers"])
    assert record["evidence"]["execution_readiness_status"] == "blocked"


def test_objective_audit_reports_literal_test_ready_after_official_hydrated_validation(tmp_path):
    _write_json(tmp_path / "constant.json", {"status": "blocked_incompatible_splits"})
    _write_json(tmp_path / "observed.json", {"status": "missing_evidence"})
    _write_json(tmp_path / "context.json", {"status": "missing_evidence"})
    _write_json(
        tmp_path / "features.json", {"conclusion": "blocked_no_train_support_for_validation_shift"}
    )
    _write_json(
        tmp_path / "identifiability.json", {"status": "blocked_underidentified_train_only_shift"}
    )
    _write_json(tmp_path / "hydration.json", {"status": "remote_official_hydration_required"})
    _write_json(tmp_path / "hydration_plan.json", {"status": "ready_for_explicit_hydration"})
    _write_json(tmp_path / "hydration_plan_validation.json", {"status": "valid"})
    _write_json(tmp_path / "hydration_plan_run.json", {"status": "executed"})
    _write_json(tmp_path / "hydration_preflight.json", {"status": "ready_for_download"})
    _write_json(tmp_path / "hydration_storage.json", {"status": "ready"})
    _write_json(tmp_path / "remote_hydration.json", {"status": "ready_for_remote_hydration"})
    _write_json(tmp_path / "execution_readiness.json", {"status": "ready"})
    _write_json(
        tmp_path / "official_hydrated_gate.json",
        {
            "test_eligible": True,
            "test": None,
            "fit": {"validation_guard": {"passed": True}},
        },
    )

    record = audit_objective(_args(tmp_path))

    assert record["status"] == "literal_test_ready"
    assert record["evidence"]["official_hydrated_validation_passed"] is True
    assert record["evidence"]["official_hydrated_test_result_count"] == 0
    assert any(
        requirement["name"] == "validate_on_val_against_sota_guard"
        and requirement["status"] == "satisfied"
        for requirement in record["requirements"]
    )
    assert any(
        requirement["name"] == "exactly_one_held_out_test_only_after_validation"
        and requirement["status"] == "blocked"
        for requirement in record["requirements"]
    )


def test_objective_audit_marks_literal_achieved_after_one_official_hydrated_test(tmp_path):
    _write_json(tmp_path / "constant.json", {"status": "blocked_incompatible_splits"})
    _write_json(tmp_path / "observed.json", {"status": "missing_evidence"})
    _write_json(tmp_path / "context.json", {"status": "missing_evidence"})
    _write_json(
        tmp_path / "features.json", {"conclusion": "blocked_no_train_support_for_validation_shift"}
    )
    _write_json(
        tmp_path / "identifiability.json", {"status": "blocked_underidentified_train_only_shift"}
    )
    _write_json(tmp_path / "hydration.json", {"status": "remote_official_hydration_required"})
    _write_json(tmp_path / "hydration_plan.json", {"status": "ready_for_explicit_hydration"})
    _write_json(tmp_path / "hydration_plan_validation.json", {"status": "valid"})
    _write_json(tmp_path / "hydration_plan_run.json", {"status": "executed"})
    _write_json(tmp_path / "hydration_preflight.json", {"status": "ready_for_download"})
    _write_json(tmp_path / "hydration_storage.json", {"status": "ready"})
    _write_json(tmp_path / "remote_hydration.json", {"status": "ready_for_remote_hydration"})
    _write_json(tmp_path / "execution_readiness.json", {"status": "ready"})
    _write_json(
        tmp_path / "official_hydrated_gate.json",
        {
            "test_eligible": True,
            "test": {"selected_test": {"nrmse": 0.01}},
            "fit": {"validation_guard": {"passed": True}},
        },
    )

    record = audit_objective(_args(tmp_path))

    assert record["status"] == "literal_achieved"
    assert record["blockers"] == []
    assert record["evidence"]["official_hydrated_test_result_count"] == 1
    assert any(
        requirement["name"] == "exactly_one_held_out_test_only_after_validation"
        and requirement["status"] == "satisfied"
        for requirement in record["requirements"]
    )


def test_objective_exit_policy():
    assert exit_code_for_status("literal_achieved", "literal-achieved") == 0
    assert exit_code_for_status("literal_test_ready", "literal-test-ready") == 0
    assert exit_code_for_status("literal_test_ready", "literal-achieved") == 2
    assert exit_code_for_status("observed_context_achieved", "literal-achieved") == 2
    assert exit_code_for_status("observed_context_achieved", "observed-accepted") == 0
    assert exit_code_for_status("context_transport_achieved", "context-accepted") == 0
    assert exit_code_for_status("observed_context_achieved", "context-accepted") == 2
    assert exit_code_for_status("literal_blocked", "report") == 0
