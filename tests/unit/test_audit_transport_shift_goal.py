from __future__ import annotations

from argparse import Namespace
import json

from scripts.audit_transport_shift_goal import audit_goal


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _base_gate(*, guard_passed: bool, test_eligible: bool = False, with_test: bool = False):
    payload = {
        "diagnostic": {"best_shifts": {"train": 0, "val": 40}},
        "fit": {
            "selected_train_shift": 0,
            "selected_validation": {"shift": 0, "nrmse": 0.5},
            "validation_guard": {"passed": guard_passed, "relative_improvement": -0.1},
        },
        "test_eligible": test_eligible,
    }
    if with_test:
        payload["test"] = {"selected_shift": 0, "selected_test": {"nrmse": 0.1}}
    return payload


def _args(tmp_path, gate, selection):
    return Namespace(
        official_gate_json=str(gate),
        compatible_window_selection_json=str(selection),
        output_json=str(tmp_path / "audit.json"),
    )


def test_audit_reports_incompatible_full_source_splits_as_blocker(tmp_path):
    gate = tmp_path / "gate.json"
    selection = tmp_path / "selection.json"
    _write_json(gate, _base_gate(guard_passed=False))
    _write_json(
        selection,
        {
            "compatible": False,
            "common_shifts": [],
            "histograms": {"train": {"0": 1}, "val": {"40": 1}, "test": {"72": 1}},
        },
    )

    record = audit_goal(_args(tmp_path, gate, selection))

    assert record["status"] == "blocked_incompatible_splits"
    assert record["test_allowed"] is False
    assert record["constant_shift"]["best_shifts"] == {"train": 0, "val": 40}
    assert record["constant_shift"]["common_full_source_shifts"] == []
    assert any(req["name"] == "validation_sota_guard_passed" and req["status"] == "failed" for req in record["requirements"])


def test_audit_allows_exactly_one_test_only_after_validation_pass(tmp_path):
    gate = tmp_path / "gate.json"
    selection = tmp_path / "selection.json"
    _write_json(gate, _base_gate(guard_passed=True, test_eligible=True))
    _write_json(selection, {"compatible": True, "common_shifts": [0], "histograms": {}})

    record = audit_goal(_args(tmp_path, gate, selection))

    assert record["status"] == "test_ready"
    assert record["test_allowed"] is True


def test_audit_marks_achieved_only_when_gate_passed_and_test_exists(tmp_path):
    gate = tmp_path / "gate.json"
    selection = tmp_path / "selection.json"
    _write_json(gate, _base_gate(guard_passed=True, test_eligible=True, with_test=True))
    _write_json(selection, {"compatible": True, "common_shifts": [0], "histograms": {}})

    record = audit_goal(_args(tmp_path, gate, selection))

    assert record["status"] == "achieved"
    assert record["test_allowed"] is False


def test_audit_reports_missing_evidence(tmp_path):
    record = audit_goal(_args(tmp_path, tmp_path / "missing_gate.json", tmp_path / "missing_selection.json"))

    assert record["status"] == "missing_evidence"
    assert record["test_allowed"] is False
    assert len(record["blockers"]) == 2
