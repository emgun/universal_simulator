#!/usr/bin/env python
from __future__ import annotations

"""Audit the lagged observed-transition transport-shift result artifact."""

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from scripts.audit_transport_shift_goal import (
    PASS_STATUSES_BY_MODE,
    _data_identity_mismatches,
    _gate_source_mismatches,
    _inspect_data_schema,
    _load_optional_json,
    _missing_required_identities,
    _parse_expected_hashes,
    _result_record_mismatches,
)


def _test_result_count(gate: Mapping[str, Any] | None) -> int:
    if not gate:
        return 0
    test_payload = gate.get("test")
    if not test_payload:
        return 0
    if isinstance(test_payload, list):
        return len(test_payload)
    return 1


def _record_tokens(status: str, gate: Mapping[str, Any] | None) -> list[str]:
    tokens = [status]
    validation_nrmse = ((gate or {}).get("validation") or {}).get("nrmse")
    test_nrmse = ((gate or {}).get("test") or {}).get("nrmse") if isinstance((gate or {}).get("test"), Mapping) else None
    for value in (validation_nrmse, test_nrmse):
        if value is not None:
            tokens.append(str(value))
    return tokens


def audit_observed_result(args: argparse.Namespace) -> dict[str, Any]:
    gate, gate_path = _load_optional_json(args.observed_gate_json)
    data_schema = _inspect_data_schema(args.data_root, args.task, args.schema_splits)
    expected_hashes = _parse_expected_hashes(getattr(args, "expected_data_sha256", None))
    require_all_identities = bool(getattr(args, "require_data_identity", False))
    data_identity_mismatches = _data_identity_mismatches(data_schema, expected_hashes)
    missing_data_identities = (
        _missing_required_identities(data_schema, expected_hashes) if require_all_identities else []
    )
    gate_source_mismatches = _gate_source_mismatches(gate, data_schema)
    data_identity_blockers = [*missing_data_identities, *data_identity_mismatches, *gate_source_mismatches]

    validation_guard = ((gate or {}).get("validation_guard") or {})
    guard_passed = bool(validation_guard.get("passed"))
    test_eligible = bool((gate or {}).get("test_eligible"))
    test_result_count = _test_result_count(gate)
    has_test_result = test_result_count > 0
    leaked_test_result = has_test_result and not test_eligible
    multiple_test_results = test_result_count > 1

    if gate is None:
        status = "missing_evidence"
        blockers = [f"missing observed gate JSON: {args.observed_gate_json}"]
    elif data_identity_blockers:
        status = "invalid_data_identity"
        blockers = data_identity_blockers
    elif leaked_test_result:
        status = "invalid_test_leakage"
        blockers = ["held-out test result is present even though validation did not authorize test evaluation"]
    elif multiple_test_results:
        status = "invalid_multiple_tests"
        blockers = ["more than one held-out test result is present"]
    elif not guard_passed:
        status = "validation_failed"
        blockers = ["observed transport validation failed the SOTA guard"]
    elif test_eligible and not has_test_result:
        status = "test_ready"
        blockers = ["validation passed and exactly one held-out test still needs to be run"]
    elif test_eligible and has_test_result:
        status = "achieved"
        blockers = []
    else:
        status = "blocked"
        blockers = ["observed transport gate did not authorize held-out test"]

    result_records = list(getattr(args, "result_record", None) or [])
    require_result_records = bool(getattr(args, "require_result_records", False))
    required_record_tokens = _record_tokens(status, gate)
    result_record_mismatches = (
        _result_record_mismatches(result_records, required_record_tokens) if require_result_records else []
    )
    if result_record_mismatches and status not in {
        "missing_evidence",
        "invalid_data_identity",
        "invalid_test_leakage",
        "invalid_multiple_tests",
    }:
        status = "invalid_result_record"
        blockers = result_record_mismatches

    return {
        "status": status,
        "blockers": blockers,
        "observed_transport": {
            "gate_json": gate_path,
            "estimator": (gate or {}).get("estimator"),
            "train": (gate or {}).get("train"),
            "validation": (gate or {}).get("validation"),
            "validation_guard": validation_guard,
            "test_eligible": test_eligible,
            "test": (gate or {}).get("test"),
        },
        "held_out_test_policy": {
            "test_eligible": test_eligible,
            "test_result_count": test_result_count,
            "leaked_test_result": leaked_test_result,
            "exactly_one_test_after_validation": bool(test_eligible and test_result_count == 1),
            "ledger": (gate or {}).get("held_out_test_policy"),
        },
        "data_schema": data_schema,
        "data_identity_policy": {
            "expected_sha256": expected_hashes,
            "require_all_inspected_splits": require_all_identities,
            "missing": missing_data_identities,
            "mismatches": data_identity_mismatches,
            "gate_source_mismatches": gate_source_mismatches,
            "passed": not data_identity_blockers,
        },
        "result_record_policy": {
            "required": require_result_records,
            "records": result_records,
            "required_tokens": required_record_tokens,
            "mismatches": result_record_mismatches,
            "passed": not result_record_mismatches,
        },
        "recommendation": (
            "Promote this result only if the benchmark policy accepts a two-frame observed context; "
            "otherwise treat it as an upper-bound signal for a causal train-fitted head."
        ),
    }


def exit_code_for_status(status: str, mode: str) -> int:
    allowed = PASS_STATUSES_BY_MODE[mode]
    if allowed is None:
        return 0
    return 0 if status in allowed else 2


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit observed transport-shift result evidence")
    parser.add_argument(
        "--observed-gate-json",
        default="reports/research/sota_loop/observed_transport_shift_gate_real_light_v1.json",
        help="Result from scripts/run_observed_transport_shift_gate.py",
    )
    parser.add_argument("--data-root", default="data/pdebench")
    parser.add_argument("--task", default="advection1d")
    parser.add_argument("--schema-splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--expected-data-sha256", action="append", default=None)
    parser.add_argument("--require-data-identity", action="store_true")
    parser.add_argument("--result-record", action="append", default=None)
    parser.add_argument("--require-result-records", action="store_true")
    parser.add_argument("--require-status", choices=tuple(PASS_STATUSES_BY_MODE), default="report")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    record = audit_observed_result(args)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))
    raise SystemExit(exit_code_for_status(str(record["status"]), args.require_status))


if __name__ == "__main__":
    main()
