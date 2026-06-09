from __future__ import annotations

import hashlib
import json

from scripts.validate_durable_audit_inputs_evidence import validate_evidence

BASELINE_RUN = "persistence_light_v1_test"
BASELINE_VALUE = 0.5701633411507036


def _write_artifact(root, rel_path, payload):
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True)
    path.write_text(text, encoding="utf-8")
    raw = path.read_bytes()
    return {
        "artifact_path": rel_path,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _scorecard_payload():
    return {
        "metric_keys": ["metric:decoded_rollout_nrmse"],
        "rows": [
            {
                "run_name": BASELINE_RUN,
                "metric:decoded_rollout_nrmse": BASELINE_VALUE,
            },
            {
                "run_name": "ups_light_candidate",
                "metric:decoded_rollout_nrmse": 0.42,
            },
        ],
    }


def _transport_payload():
    return {"status": "literal_achieved", "blockers": []}


def _transfer_payload():
    return {
        "status": "partial_transfer_validated",
        "evaluated_task_count": 2,
        "skipped_task_count": 1,
        "mean_validation_nrmse": 0.003,
        "held_out_policy": "train/val only; no held-out test split is passed to task gates",
    }


def _evidence(root, *, scorecard=None, transport=None, transfer=None):
    scorecard_record = _write_artifact(
        root,
        "docs/claim_evidence/artifacts/light_v1_demo_scorecard.json",
        scorecard if scorecard is not None else _scorecard_payload(),
    )
    scorecard_record.update(
        {
            "original_report_path": "reports/demo/light_latest/scorecard.json",
            "key_recorded_fields": {
                "baseline_run_name": BASELINE_RUN,
                "baseline_metric_name": "decoded_rollout_nrmse",
                "baseline_metric_value": BASELINE_VALUE,
                "row_count": 2,
            },
        }
    )
    transport_record = _write_artifact(
        root,
        "docs/claim_evidence/artifacts/transport_objective_status.json",
        transport if transport is not None else _transport_payload(),
    )
    transport_record["original_report_path"] = (
        "reports/research/sota_loop/transport_objective_status.json"
    )
    transfer_record = _write_artifact(
        root,
        "docs/claim_evidence/artifacts/inferred_transport_transfer_scorecard.json",
        transfer if transfer is not None else _transfer_payload(),
    )
    transfer_record["original_report_path"] = (
        "reports/research/sota_loop/inferred_transfer_scorecard/scorecard.json"
    )
    return {
        "measurement_type": "durable_audit_inputs_evidence",
        "status": "complete",
        "held_out_test_data_read": False,
        "test_ledger_writes": [],
        "artifacts": {
            "light_v1_demo_scorecard": scorecard_record,
            "transport_objective_status": transport_record,
            "inferred_transport_transfer_scorecard": transfer_record,
        },
    }


def test_valid_durable_audit_inputs_pass(tmp_path):
    evidence = _evidence(tmp_path)
    assert validate_evidence(evidence, root=tmp_path) == []


def test_baseline_must_match_claim_documentation(tmp_path):
    evidence = _evidence(tmp_path)
    claim_evidence = {
        "claim_documentation": {
            "baseline_run_name": BASELINE_RUN,
            "baseline_metric_value": BASELINE_VALUE,
        }
    }
    assert validate_evidence(evidence, root=tmp_path, claim_evidence=claim_evidence) == []

    mismatched = {
        "claim_documentation": {
            "baseline_run_name": BASELINE_RUN,
            "baseline_metric_value": 0.9,
        }
    }
    errors = validate_evidence(evidence, root=tmp_path, claim_evidence=mismatched)
    assert any("claim_documentation" in error for error in errors)


def test_tampered_artifact_hash_fails(tmp_path):
    evidence = _evidence(tmp_path)
    artifact = tmp_path / "docs/claim_evidence/artifacts/transport_objective_status.json"
    artifact.write_text(
        json.dumps({"status": "literal_achieved", "blockers": [], "extra": True}),
        encoding="utf-8",
    )
    errors = validate_evidence(evidence, root=tmp_path)
    assert any("sha256 mismatch" in error for error in errors)


def test_non_achieved_transport_status_fails(tmp_path):
    evidence = _evidence(tmp_path, transport={"status": "literal_blocked", "blockers": ["dns"]})
    errors = validate_evidence(evidence, root=tmp_path)
    assert any("literal_achieved" in error for error in errors)
    assert any("blockers" in error for error in errors)


def test_insufficient_transfer_tasks_fail(tmp_path):
    transfer = _transfer_payload()
    transfer["evaluated_task_count"] = 1
    evidence = _evidence(tmp_path, transfer=transfer)
    errors = validate_evidence(evidence, root=tmp_path)
    assert any("evaluated_task_count" in error for error in errors)


def test_missing_baseline_row_fails(tmp_path):
    scorecard = _scorecard_payload()
    scorecard["rows"] = [row for row in scorecard["rows"] if row["run_name"] != BASELINE_RUN]
    evidence = _evidence(tmp_path, scorecard=scorecard)
    errors = validate_evidence(evidence, root=tmp_path)
    assert any(BASELINE_RUN in error for error in errors)


def test_held_out_flags_are_required(tmp_path):
    evidence = _evidence(tmp_path)
    evidence["held_out_test_data_read"] = True
    evidence["test_ledger_writes"] = ["unexpected"]
    errors = validate_evidence(evidence, root=tmp_path)
    assert any("held_out_test_data_read" in error for error in errors)
    assert any("test_ledger_writes" in error for error in errors)
