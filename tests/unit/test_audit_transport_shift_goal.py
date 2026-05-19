from __future__ import annotations

from argparse import Namespace
import json

import h5py
import torch

from scripts.audit_transport_shift_goal import audit_goal, exit_code_for_status


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_data_split(root, split: str, *, with_metadata: bool = False) -> None:
    with h5py.File(root / f"advection1d_{split}.h5", "w") as handle:
        data = handle.create_dataset("data", data=torch.zeros(2, 3, 4, 1).numpy())
        if with_metadata:
            handle.attrs["source"] = "synthetic"
            data.attrs["dt"] = 0.1
            handle.create_dataset("beta", data=torch.ones(2, 1).numpy())


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


def _add_gate_sources(payload, tmp_path):
    for split in ("train", "val", "test"):
        path = tmp_path / f"advection1d_{split}.h5"
        if not path.exists():
            _write_data_split(tmp_path, split)
        payload.setdefault("data_sources", {})[split] = {
            "split": split,
            "path": str(path),
            "exists": True,
            "bytes": path.stat().st_size,
            "sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
        }
    return payload


def _args(tmp_path, gate, selection):
    for split in ("train", "val", "test"):
        if not (tmp_path / f"advection1d_{split}.h5").exists():
            _write_data_split(tmp_path, split)
    return Namespace(
        official_gate_json=str(gate),
        compatible_window_selection_json=str(selection),
        data_root=str(tmp_path),
        task="advection1d",
        schema_splits=["train", "val", "test"],
        expected_data_sha256=None,
        require_data_identity=False,
        result_record=None,
        require_result_records=False,
        output_json=str(tmp_path / "audit.json"),
    )


def test_audit_reports_incompatible_full_source_splits_as_blocker(tmp_path):
    gate = tmp_path / "gate.json"
    selection = tmp_path / "selection.json"
    _write_json(gate, _add_gate_sources(_base_gate(guard_passed=False), tmp_path))
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
    assert record["held_out_test_policy"]["test_result_count"] == 0
    assert record["held_out_test_policy"]["leaked_test_result"] is False
    assert record["constant_shift"]["best_shifts"] == {"train": 0, "val": 40}
    assert record["constant_shift"]["common_full_source_shifts"] == []
    assert record["data_schema"]["parameter_metadata_available"] is False
    assert record["data_schema"]["splits"]["train"]["datasets"]["data"]["shape"] == [2, 3, 4, 1]
    assert record["data_schema"]["splits"]["train"]["bytes"] > 0
    assert len(record["data_schema"]["splits"]["train"]["sha256"]) == 64
    assert any(req["name"] == "validation_sota_guard_passed" and req["status"] == "failed" for req in record["requirements"])


def test_audit_allows_exactly_one_test_only_after_validation_pass(tmp_path):
    gate = tmp_path / "gate.json"
    selection = tmp_path / "selection.json"
    _write_json(gate, _add_gate_sources(_base_gate(guard_passed=True, test_eligible=True), tmp_path))
    _write_json(selection, {"compatible": True, "common_shifts": [0], "histograms": {}})

    record = audit_goal(_args(tmp_path, gate, selection))

    assert record["status"] == "test_ready"
    assert record["test_allowed"] is True


def test_audit_marks_achieved_only_when_gate_passed_and_test_exists(tmp_path):
    gate = tmp_path / "gate.json"
    selection = tmp_path / "selection.json"
    _write_json(gate, _add_gate_sources(_base_gate(guard_passed=True, test_eligible=True, with_test=True), tmp_path))
    _write_json(selection, {"compatible": True, "common_shifts": [0], "histograms": {}})

    record = audit_goal(_args(tmp_path, gate, selection))

    assert record["status"] == "achieved"
    assert record["test_allowed"] is False
    assert record["held_out_test_policy"]["exactly_one_test_after_validation"] is True


def test_audit_flags_test_result_when_gate_not_eligible(tmp_path):
    gate = tmp_path / "gate.json"
    selection = tmp_path / "selection.json"
    _write_json(gate, _add_gate_sources(_base_gate(guard_passed=False, test_eligible=False, with_test=True), tmp_path))
    _write_json(selection, {"compatible": False, "common_shifts": [], "histograms": {}})

    record = audit_goal(_args(tmp_path, gate, selection))

    assert record["status"] == "invalid_test_leakage"
    assert record["held_out_test_policy"]["leaked_test_result"] is True
    assert any(
        req["name"] == "exactly_one_held_out_test_after_validation" and req["status"] == "violated"
        for req in record["requirements"]
    )


def test_audit_flags_multiple_test_results(tmp_path):
    gate = tmp_path / "gate.json"
    selection = tmp_path / "selection.json"
    payload = _base_gate(guard_passed=True, test_eligible=True)
    payload["test"] = [{"selected_test": {"nrmse": 0.1}}, {"selected_test": {"nrmse": 0.2}}]
    _write_json(gate, _add_gate_sources(payload, tmp_path))
    _write_json(selection, {"compatible": True, "common_shifts": [0], "histograms": {}})

    record = audit_goal(_args(tmp_path, gate, selection))

    assert record["status"] == "invalid_multiple_tests"
    assert record["held_out_test_policy"]["test_result_count"] == 2


def test_audit_reports_missing_evidence(tmp_path):
    record = audit_goal(_args(tmp_path, tmp_path / "missing_gate.json", tmp_path / "missing_selection.json"))

    assert record["status"] == "missing_evidence"
    assert record["test_allowed"] is False
    assert len(record["blockers"]) == 2


def test_audit_reports_available_parameter_metadata(tmp_path):
    gate = tmp_path / "gate.json"
    selection = tmp_path / "selection.json"
    _write_json(gate, _add_gate_sources(_base_gate(guard_passed=False), tmp_path))
    _write_json(selection, {"compatible": False, "common_shifts": [], "histograms": {}})
    for split in ("train", "val", "test"):
        _write_data_split(tmp_path, split, with_metadata=True)

    args = Namespace(
        official_gate_json=str(gate),
        compatible_window_selection_json=str(selection),
        data_root=str(tmp_path),
        task="advection1d",
        schema_splits=["train", "val", "test"],
        expected_data_sha256=None,
        require_data_identity=False,
        result_record=None,
        require_result_records=False,
        output_json=str(tmp_path / "audit.json"),
    )

    record = audit_goal(args)

    assert record["data_schema"]["parameter_metadata_available"] is True
    assert record["data_schema"]["splits"]["train"]["sample_aligned_auxiliary_datasets"] == ["beta"]


def test_audit_enforces_expected_data_identity(tmp_path):
    gate = tmp_path / "gate.json"
    selection = tmp_path / "selection.json"
    args = _args(tmp_path, gate, selection)
    _write_json(gate, _add_gate_sources(_base_gate(guard_passed=False), tmp_path))
    _write_json(selection, {"compatible": False, "common_shifts": [], "histograms": {}})
    args.expected_data_sha256 = ["train=" + ("0" * 64)]

    record = audit_goal(args)

    assert record["status"] == "invalid_data_identity"
    assert record["data_identity_policy"]["passed"] is False
    assert "train sha256 mismatch" in record["blockers"][0]


def test_audit_can_require_identity_for_all_existing_splits(tmp_path):
    gate = tmp_path / "gate.json"
    selection = tmp_path / "selection.json"
    args = _args(tmp_path, gate, selection)
    _write_json(gate, _add_gate_sources(_base_gate(guard_passed=False), tmp_path))
    _write_json(selection, {"compatible": False, "common_shifts": [], "histograms": {}})
    train_sha = audit_goal(args)["data_schema"]["splits"]["train"]["sha256"]
    args.expected_data_sha256 = [f"train={train_sha}"]
    args.require_data_identity = True

    record = audit_goal(args)

    assert record["status"] == "invalid_data_identity"
    assert record["data_identity_policy"]["missing"]
    assert any("val missing required expected sha256" in blocker for blocker in record["blockers"])


def test_audit_can_require_result_records(tmp_path):
    gate = tmp_path / "gate.json"
    selection = tmp_path / "selection.json"
    record_path = tmp_path / "worklog.md"
    record_path.write_text("status: blocked_incompatible_splits\nvalidation_nrmse: 0.5\n", encoding="utf-8")
    _write_json(gate, _add_gate_sources(_base_gate(guard_passed=False), tmp_path))
    _write_json(selection, {"compatible": False, "common_shifts": [], "histograms": {}})
    args = _args(tmp_path, gate, selection)
    args.result_record = [str(record_path)]
    args.require_result_records = True

    record = audit_goal(args)

    assert record["status"] == "blocked_incompatible_splits"
    assert record["result_record_policy"]["passed"] is True
    assert record["result_record_policy"]["required_tokens"] == ["blocked_incompatible_splits", "0.5"]


def test_audit_flags_missing_result_record_status(tmp_path):
    gate = tmp_path / "gate.json"
    selection = tmp_path / "selection.json"
    record_path = tmp_path / "worklog.md"
    record_path.write_text("status: stale\n", encoding="utf-8")
    _write_json(gate, _add_gate_sources(_base_gate(guard_passed=False), tmp_path))
    _write_json(selection, {"compatible": False, "common_shifts": [], "histograms": {}})
    args = _args(tmp_path, gate, selection)
    args.result_record = [str(record_path)]
    args.require_result_records = True

    record = audit_goal(args)

    assert record["status"] == "invalid_result_record"
    assert record["result_record_policy"]["passed"] is False
    assert "blocked_incompatible_splits" in record["blockers"][0]


def test_audit_flags_missing_result_record_metric(tmp_path):
    gate = tmp_path / "gate.json"
    selection = tmp_path / "selection.json"
    record_path = tmp_path / "worklog.md"
    record_path.write_text("status: blocked_incompatible_splits\n", encoding="utf-8")
    _write_json(gate, _add_gate_sources(_base_gate(guard_passed=False), tmp_path))
    _write_json(selection, {"compatible": False, "common_shifts": [], "histograms": {}})
    args = _args(tmp_path, gate, selection)
    args.result_record = [str(record_path)]
    args.require_result_records = True

    record = audit_goal(args)

    assert record["status"] == "invalid_result_record"
    assert record["result_record_policy"]["passed"] is False
    assert "0.5" in record["blockers"][0]


def test_audit_flags_gate_source_mismatch(tmp_path):
    gate = tmp_path / "gate.json"
    selection = tmp_path / "selection.json"
    payload = _add_gate_sources(_base_gate(guard_passed=False), tmp_path)
    payload["data_sources"]["train"]["sha256"] = "0" * 64
    _write_json(gate, payload)
    _write_json(selection, {"compatible": False, "common_shifts": [], "histograms": {}})

    record = audit_goal(_args(tmp_path, gate, selection))

    assert record["status"] == "invalid_data_identity"
    assert any("gate data_sources train.sha256 mismatch" in blocker for blocker in record["blockers"])


def test_audit_exit_policy_fails_closed_for_blocked_status():
    assert exit_code_for_status("blocked_incompatible_splits", "report") == 0
    assert exit_code_for_status("blocked_incompatible_splits", "test-ready") == 2
    assert exit_code_for_status("blocked_incompatible_splits", "achieved") == 2


def test_audit_exit_policy_distinguishes_test_ready_from_achieved():
    assert exit_code_for_status("test_ready", "test-ready") == 0
    assert exit_code_for_status("test_ready", "achieved") == 2
    assert exit_code_for_status("achieved", "test-ready") == 0
    assert exit_code_for_status("achieved", "achieved") == 0
