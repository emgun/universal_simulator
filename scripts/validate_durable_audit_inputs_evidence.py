#!/usr/bin/env python
from __future__ import annotations

"""Validate durable audit-input artifacts for the universal SOTA audit.

The universal SOTA audit reads three machine-local, gitignored report files:
the light-v1 demo scorecard, the official transport objective status, and the
inferred transport transfer scorecard. This validator checks the durable
committed copies under docs/claim_evidence/artifacts/ against the evidence
record so a clean checkout can audit without the original working tree.
"""

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

EXPECTED_MEASUREMENT_TYPE = "durable_audit_inputs_evidence"
DEFAULT_EVIDENCE_JSON = "docs/claim_evidence/durable_audit_inputs_evidence.json"
DEFAULT_CLAIM_EVIDENCE_JSON = "docs/claim_evidence/universal_sota_claim_evidence.json"
REQUIRED_ARTIFACT_KEYS = (
    "light_v1_demo_scorecard",
    "inferred_transport_transfer_scorecard",
    "transport_objective_status",
)
ACCEPTED_TRANSFER_STATUSES = {"transfer_validated", "partial_transfer_validated"}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return data


def _as_mapping(value: Any, label: str, errors: list[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(f"{label} must be an object")
        return {}
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_close(actual: Any, expected: Any) -> bool:
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return abs(float(actual) - float(expected)) <= 1e-12
    return actual == expected


def _validate_artifact_record(
    name: str,
    record: Mapping[str, Any],
    *,
    root: Path,
    errors: list[str],
) -> dict[str, Any]:
    artifact_path = str(record.get("artifact_path", ""))
    if not artifact_path.startswith("docs/claim_evidence/artifacts/"):
        errors.append(f"{name}.artifact_path must live under docs/claim_evidence/artifacts/")
        return {}
    resolved = root / artifact_path
    if not resolved.exists():
        errors.append(f"{name}.artifact_path does not exist: {artifact_path}")
        return {}
    actual_bytes = resolved.stat().st_size
    if record.get("bytes") != actual_bytes:
        errors.append(
            f"{name}.bytes mismatch: recorded {record.get('bytes')}, actual {actual_bytes}"
        )
    actual_sha = _sha256(resolved)
    if record.get("sha256") != actual_sha:
        errors.append(
            f"{name}.sha256 mismatch: recorded {record.get('sha256')}, actual {actual_sha}"
        )
    if not str(record.get("original_report_path", "")).startswith("reports/"):
        errors.append(f"{name}.original_report_path must reference the original reports/ path")
    try:
        return load_json(resolved)
    except (json.JSONDecodeError, TypeError):
        errors.append(f"{name} artifact is not a JSON object: {artifact_path}")
        return {}


def _validate_transport(payload: Mapping[str, Any], errors: list[str]) -> None:
    if payload.get("status") != "literal_achieved":
        errors.append("transport_objective_status artifact status must be literal_achieved")
    blockers = payload.get("blockers")
    if blockers != []:
        errors.append("transport_objective_status artifact blockers must be empty")


def _validate_transfer(payload: Mapping[str, Any], errors: list[str]) -> None:
    if payload.get("status") not in ACCEPTED_TRANSFER_STATUSES:
        errors.append(
            "inferred_transport_transfer_scorecard artifact status must be one of "
            f"{sorted(ACCEPTED_TRANSFER_STATUSES)}"
        )
    evaluated = payload.get("evaluated_task_count")
    if not isinstance(evaluated, int) or evaluated < 2:
        errors.append(
            "inferred_transport_transfer_scorecard artifact evaluated_task_count must be >= 2"
        )
    if "no held-out test" not in str(payload.get("held_out_policy", "")):
        errors.append(
            "inferred_transport_transfer_scorecard artifact held_out_policy must state that "
            "no held-out test split is used"
        )


def _validate_scorecard(
    payload: Mapping[str, Any],
    record: Mapping[str, Any],
    claim_evidence: Mapping[str, Any],
    errors: list[str],
) -> None:
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        errors.append("light_v1_demo_scorecard artifact must contain non-empty rows")
        return
    key_fields = _as_mapping(
        record.get("key_recorded_fields"), "light_v1_demo_scorecard.key_recorded_fields", errors
    )
    baseline_run_name = str(key_fields.get("baseline_run_name", ""))
    metric_name = str(key_fields.get("baseline_metric_name", ""))
    if not baseline_run_name or not metric_name:
        errors.append(
            "light_v1_demo_scorecard.key_recorded_fields must record baseline_run_name "
            "and baseline_metric_name"
        )
        return
    baseline_rows = [
        row for row in rows if isinstance(row, Mapping) and row.get("run_name") == baseline_run_name
    ]
    if not baseline_rows:
        errors.append(f"light_v1_demo_scorecard artifact is missing the {baseline_run_name} row")
        return
    baseline_value = baseline_rows[0].get(f"metric:{metric_name}")
    if not _is_close(baseline_value, key_fields.get("baseline_metric_value")):
        errors.append("light_v1_demo_scorecard baseline metric does not match key_recorded_fields")
    claim_documentation = claim_evidence.get("claim_documentation")
    if isinstance(claim_documentation, Mapping):
        documented_value = claim_documentation.get("baseline_metric_value")
        documented_name = claim_documentation.get("baseline_run_name")
        if documented_name == baseline_run_name and not _is_close(baseline_value, documented_value):
            errors.append(
                "light_v1_demo_scorecard baseline metric does not match "
                "universal_sota_claim_evidence claim_documentation.baseline_metric_value"
            )


def validate_evidence(
    evidence: Mapping[str, Any],
    *,
    root: Path,
    claim_evidence: Mapping[str, Any] | None = None,
) -> list[str]:
    errors: list[str] = []
    if evidence.get("measurement_type") != EXPECTED_MEASUREMENT_TYPE:
        errors.append(f"measurement_type must be {EXPECTED_MEASUREMENT_TYPE}")
    if evidence.get("status") != "complete":
        errors.append("status must be complete")
    if evidence.get("held_out_test_data_read") is not False:
        errors.append("held_out_test_data_read must be false")
    if evidence.get("test_ledger_writes") != []:
        errors.append("test_ledger_writes must be empty")

    artifacts = _as_mapping(evidence.get("artifacts"), "artifacts", errors)
    payloads: dict[str, dict[str, Any]] = {}
    for name in REQUIRED_ARTIFACT_KEYS:
        record = _as_mapping(artifacts.get(name), f"artifacts.{name}", errors)
        if record:
            payloads[name] = _validate_artifact_record(name, record, root=root, errors=errors)

    if payloads.get("transport_objective_status"):
        _validate_transport(payloads["transport_objective_status"], errors)
    if payloads.get("inferred_transport_transfer_scorecard"):
        _validate_transfer(payloads["inferred_transport_transfer_scorecard"], errors)
    if payloads.get("light_v1_demo_scorecard"):
        scorecard_record = _as_mapping(
            artifacts.get("light_v1_demo_scorecard"), "artifacts.light_v1_demo_scorecard", errors
        )
        _validate_scorecard(
            payloads["light_v1_demo_scorecard"],
            scorecard_record,
            claim_evidence or {},
            errors,
        )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-json", type=Path, default=Path(DEFAULT_EVIDENCE_JSON))
    parser.add_argument(
        "--claim-evidence-json", type=Path, default=Path(DEFAULT_CLAIM_EVIDENCE_JSON)
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)

    evidence = load_json(args.repo_root / args.evidence_json)
    claim_evidence_path = args.repo_root / args.claim_evidence_json
    claim_evidence = load_json(claim_evidence_path) if claim_evidence_path.exists() else {}
    errors = validate_evidence(evidence, root=args.repo_root, claim_evidence=claim_evidence)
    record = {
        "errors": errors,
        "evidence_json": str(args.evidence_json),
        "status": "valid" if not errors else "invalid",
    }
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
