#!/usr/bin/env python
from __future__ import annotations

"""Validate P2 parameter-conditioned transport sidecar evidence."""

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.validate_next_validation_contracts import load_json, validate_contract

DEFAULT_EVIDENCE_JSON = (
    "docs/claim_evidence/ups_advection_p2_parameter_conditioned_sidecar_val_evidence.json"
)
EXPECTED_MEASUREMENT_TYPE = "ups_advection_p2_parameter_conditioned_sidecar_validation"
EXPECTED_DECISION_STATUS = "validation_sidecar_supports_p2_reduced_context_path"
EXPECTED_ESTIMATOR_NAME = "parameter_conditioned_periodic_shift"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_mapping(value: Any, label: str, errors: list[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(f"{label} must be an object")
        return {}
    return value


def _validate_file_ref(
    source: Mapping[str, Any],
    *,
    label: str,
    root: Path,
    errors: list[str],
) -> Path | None:
    raw_path = source.get("path")
    if not raw_path:
        errors.append(f"{label}.path is required")
        return None
    path = root / str(raw_path)
    if not path.exists():
        errors.append(f"{label}.path does not exist: {path}")
        return None
    if source.get("sha256") != _sha256(path):
        errors.append(f"{label}.sha256 must match file bytes")
    if isinstance(source.get("bytes"), int) and path.stat().st_size != source.get("bytes"):
        errors.append(f"{label}.bytes must match file size")
    return path


def _report_validation_metric(report: Mapping[str, Any]) -> float | None:
    fit = report.get("fit")
    if not isinstance(fit, Mapping):
        return None
    selected = fit.get("selected_validation")
    if not isinstance(selected, Mapping):
        return None
    value = selected.get("metric_value", selected.get("nrmse"))
    if not isinstance(value, (int, float)):
        return None
    return float(value)


def validate_evidence(evidence: Mapping[str, Any], *, root: Path | None = None) -> list[str]:
    errors: list[str] = []
    repo_root = root or Path.cwd()
    if evidence.get("measurement_type") != EXPECTED_MEASUREMENT_TYPE:
        errors.append(f"measurement_type must be {EXPECTED_MEASUREMENT_TYPE}")
    if evidence.get("split") != "val":
        errors.append("split must be val")
    if evidence.get("held_out_test_used") is not False:
        errors.append("held_out_test_used must be false")
    if evidence.get("held_out_test_data_read") is not False:
        errors.append("held_out_test_data_read must be false")
    if evidence.get("test_ledger_writes") != []:
        errors.append("test_ledger_writes must be empty")

    contract_ref = _as_mapping(evidence.get("contract"), "contract", errors)
    contract_path = _validate_file_ref(
        contract_ref,
        label="contract",
        root=repo_root,
        errors=errors,
    )
    if contract_path is not None:
        errors.extend(validate_contract(load_json(contract_path), root=repo_root))

    report_ref = _as_mapping(evidence.get("candidate_report"), "candidate_report", errors)
    report_path = _validate_file_ref(
        report_ref,
        label="candidate_report",
        root=repo_root,
        errors=errors,
    )
    report: dict[str, Any] = {}
    if report_path is not None:
        report = load_json(report_path)

    comparison_reports = evidence.get("comparison_reports", {})
    if isinstance(comparison_reports, Mapping):
        for comparison_id, comparison_ref in comparison_reports.items():
            _validate_file_ref(
                _as_mapping(
                    comparison_ref,
                    f"comparison_reports.{comparison_id}",
                    errors,
                ),
                label=f"comparison_reports.{comparison_id}",
                root=repo_root,
                errors=errors,
            )
    elif comparison_reports not in ({}, None):
        errors.append("comparison_reports must be an object")

    if report:
        if report.get("test") is not None:
            errors.append("candidate report must not contain a held-out test record")
        policy = _as_mapping(
            report.get("held_out_test_policy"), "report.held_out_test_policy", errors
        )
        if policy.get("recorded") is not False:
            errors.append("candidate report held_out_test_policy.recorded must be false")
        if policy.get("measurement_key") is not None:
            errors.append("candidate report held_out_test_policy.measurement_key must be empty")
        estimator = _as_mapping(report.get("estimator"), "report.estimator", errors)
        if estimator.get("name") != EXPECTED_ESTIMATOR_NAME:
            errors.append(f"report.estimator.name must be {EXPECTED_ESTIMATOR_NAME}")
        if "does not use held-out transitions" not in str(estimator.get("causality_note", "")):
            errors.append("report.estimator.causality_note must rule out held-out transitions")
        if "does not use beta metadata" in str(estimator.get("causality_note", "")):
            errors.append("candidate must be parameter-conditioned, not metadata-free")
        if report.get("data_root") != evidence.get("data_root"):
            errors.append("data_root must match candidate report")
        if report.get("val_split") != evidence.get("split"):
            errors.append("split must match candidate report val_split")
        report_metric = _report_validation_metric(report)
        if report_metric is None:
            errors.append("candidate report validation metric is required")
        elif evidence.get("metrics", {}).get("validation_nrmse") != report_metric:
            errors.append("metrics.validation_nrmse must match candidate report")

    metrics = _as_mapping(evidence.get("metrics"), "metrics", errors)
    gate = _as_mapping(evidence.get("acceptance_gate"), "acceptance_gate", errors)
    validation_nrmse = metrics.get("validation_nrmse")
    threshold = gate.get("validation_nrmse_must_be_below")
    if not isinstance(validation_nrmse, (int, float)):
        errors.append("metrics.validation_nrmse must be numeric")
    if not isinstance(threshold, (int, float)):
        errors.append("acceptance_gate.validation_nrmse_must_be_below must be numeric")
    if isinstance(validation_nrmse, (int, float)) and isinstance(threshold, (int, float)):
        expected_passed = float(validation_nrmse) < float(threshold)
        if gate.get("passed") is not expected_passed:
            errors.append("acceptance_gate.passed mismatch")

    teacher = _as_mapping(
        evidence.get("teacher_context_dependency"),
        "teacher_context_dependency",
        errors,
    )
    if teacher.get("uses_observed_context_transitions") is not False:
        errors.append("teacher_context_dependency.uses_observed_context_transitions must be false")
    if teacher.get("uses_beta_metadata") is not True:
        errors.append("teacher_context_dependency.uses_beta_metadata must be true")
    if teacher.get("uses_source_identity_as_learned_key") is not False:
        errors.append(
            "teacher_context_dependency.uses_source_identity_as_learned_key must be false"
        )

    decision = _as_mapping(evidence.get("decision"), "decision", errors)
    if decision.get("status") != EXPECTED_DECISION_STATUS:
        errors.append(f"decision.status must be {EXPECTED_DECISION_STATUS}")
    if decision.get("held_out_test_allowed_by_this_evidence") is not False:
        errors.append("decision.held_out_test_allowed_by_this_evidence must be false")
    if decision.get("primary_claim_replacement") is not False:
        errors.append("decision.primary_claim_replacement must be false")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-json", type=Path, default=Path(DEFAULT_EVIDENCE_JSON))
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)

    evidence = load_json(args.repo_root / args.evidence_json)
    errors = validate_evidence(evidence, root=args.repo_root)
    record = {
        "evidence_json": str(args.evidence_json),
        "errors": errors,
        "status": "valid" if not errors else "invalid",
    }
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
