from __future__ import annotations

from argparse import Namespace
import hashlib
import json
import subprocess

import h5py
import torch

from scripts.audit_context_transport_shift_result import audit_context_result, exit_code_for_status


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_data_split(root, split: str) -> None:
    with h5py.File(root / f"advection1d_{split}.h5", "w") as handle:
        handle.create_dataset("data", data=torch.zeros(2, 4, 8, 1).numpy())


def _add_sources(payload, tmp_path):
    for split in ("train", "val", "test"):
        path = tmp_path / f"advection1d_{split}.h5"
        if not path.exists():
            _write_data_split(tmp_path, split)
        payload.setdefault("data_sources", {})[split] = {
            "split": split,
            "path": str(path),
            "exists": True,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return payload


def _base_gate(*, guard_passed: bool = True, test_eligible: bool = True, with_test: bool = True):
    payload = {
        "estimator": {"name": "two_frame_context_shift"},
        "train": {"split": "train", "nrmse": 0.11},
        "validation": {"split": "val", "nrmse": 0.12},
        "validation_guard": {"passed": guard_passed, "relative_improvement": 0.6},
        "test_eligible": test_eligible,
        "held_out_test_policy": {"measurement_key": "abc", "recorded": True},
    }
    if with_test:
        payload["test"] = {"split": "test", "nrmse": 0.04}
    return payload


def _args(tmp_path, gate):
    for split in ("train", "val", "test"):
        if not (tmp_path / f"advection1d_{split}.h5").exists():
            _write_data_split(tmp_path, split)
    return Namespace(
        context_gate_json=str(gate),
        data_root=str(tmp_path),
        task="advection1d",
        schema_splits=["train", "val", "test"],
        expected_data_sha256=None,
        require_data_identity=False,
        result_record=None,
        require_result_records=False,
        output_json=str(tmp_path / "audit.json"),
    )


def test_context_audit_marks_achieved_for_passed_gate_and_one_test(tmp_path):
    gate = tmp_path / "context.json"
    _write_json(gate, _add_sources(_base_gate(), tmp_path))

    record = audit_context_result(_args(tmp_path, gate))

    assert record["status"] == "achieved"
    assert record["held_out_test_policy"]["exactly_one_test_after_validation"] is True
    assert record["context_transport"]["validation"]["nrmse"] == 0.12


def test_context_audit_marks_test_ready_when_validation_passes_without_test(tmp_path):
    gate = tmp_path / "context.json"
    _write_json(gate, _add_sources(_base_gate(with_test=False), tmp_path))

    record = audit_context_result(_args(tmp_path, gate))

    assert record["status"] == "test_ready"
    assert record["held_out_test_policy"]["test_result_count"] == 0


def test_context_audit_flags_test_leakage(tmp_path):
    gate = tmp_path / "context.json"
    _write_json(gate, _add_sources(_base_gate(guard_passed=False, test_eligible=False, with_test=True), tmp_path))

    record = audit_context_result(_args(tmp_path, gate))

    assert record["status"] == "invalid_test_leakage"
    assert record["held_out_test_policy"]["leaked_test_result"] is True


def test_context_audit_enforces_result_record_tokens(tmp_path):
    gate = tmp_path / "context.json"
    record_path = tmp_path / "worklog.md"
    record_path.write_text("status: achieved\nvalidation: 0.12\ntest: 0.04\n", encoding="utf-8")
    _write_json(gate, _add_sources(_base_gate(), tmp_path))
    args = _args(tmp_path, gate)
    args.result_record = [str(record_path)]
    args.require_result_records = True

    record = audit_context_result(args)

    assert record["status"] == "achieved"
    assert record["result_record_policy"]["required_tokens"] == ["achieved", "0.12", "0.04"]
    assert record["result_record_policy"]["passed"] is True


def test_context_audit_flags_missing_result_record_metric(tmp_path):
    gate = tmp_path / "context.json"
    record_path = tmp_path / "worklog.md"
    record_path.write_text("status: achieved\nvalidation: 0.12\n", encoding="utf-8")
    _write_json(gate, _add_sources(_base_gate(), tmp_path))
    args = _args(tmp_path, gate)
    args.result_record = [str(record_path)]
    args.require_result_records = True

    record = audit_context_result(args)

    assert record["status"] == "invalid_result_record"
    assert "0.04" in record["blockers"][0]


def test_context_audit_exit_policy_matches_transport_audit_modes():
    assert exit_code_for_status("achieved", "achieved") == 0
    assert exit_code_for_status("test_ready", "achieved") == 2
    assert exit_code_for_status("validation_failed", "report") == 0


def test_context_audit_cli_runs_from_repo_root(tmp_path):
    output = tmp_path / "audit.json"

    proc = subprocess.run(
        [
            "/opt/anaconda3/bin/python",
            "scripts/audit_context_transport_shift_result.py",
            "--context-gate-json",
            str(tmp_path / "missing.json"),
            "--data-root",
            str(tmp_path),
            "--output-json",
            str(output),
            "--require-status",
            "report",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert '"status": "missing_evidence"' in proc.stdout
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "missing_evidence"
