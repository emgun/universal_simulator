#!/usr/bin/env python
from __future__ import annotations

"""Validate data-conditioned advection ablation evidence."""

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.validate_next_validation_contracts import (
    REQUIRED_ABLATIONS,
    load_json,
    validate_contract,
)

DEFAULT_EVIDENCE_JSON = (
    "docs/claim_evidence/ups_advection_data_conditioned_ablation_val_evidence.json"
)
EXPECTED_MEASUREMENT_TYPE = "ups_advection_data_conditioned_ablation_validation"
EXPECTED_DECISION_STATUS = "validation_ablation_completed_p2_required"
CONTEXT_FEATURES = {"context_shift", "context_shift_abs"}


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


def _metric(record: Mapping[str, Any]) -> float:
    metrics = record.get("metrics", {})
    if not isinstance(metrics, Mapping):
        return float("nan")
    return float(metrics.get("validation_metric_value", float("nan")))


def _validate_variant_report_ref(
    *,
    variant_id: str,
    record: Mapping[str, Any],
    root: Path,
    errors: list[str],
) -> None:
    raw_path = record.get("report_path")
    if not raw_path:
        errors.append(f"variants.{variant_id}.report_path is required")
        return
    path = root / str(raw_path)
    if not path.exists():
        errors.append(f"variants.{variant_id}.report_path does not exist: {path}")
        return
    if record.get("report_sha256") != _sha256(path):
        errors.append(f"variants.{variant_id}.report_sha256 must match file bytes")
    if isinstance(record.get("report_bytes"), int) and path.stat().st_size != record.get(
        "report_bytes"
    ):
        errors.append(f"variants.{variant_id}.report_bytes must match file size")


def _validate_variant_shapes(
    variants: Mapping[str, Any],
    *,
    root: Path,
    errors: list[str],
) -> None:
    for required in sorted(REQUIRED_ABLATIONS - set(variants)):
        errors.append(f"variants must include {required}")
    if not REQUIRED_ABLATIONS.issubset(set(variants)):
        return

    full = _as_mapping(variants.get("full_context_shift"), "variants.full_context_shift", errors)
    weak = _as_mapping(
        variants.get("weaker_context_shift"), "variants.weaker_context_shift", errors
    )
    no_data = _as_mapping(
        variants.get("no_data_conditioning"), "variants.no_data_conditioning", errors
    )

    for variant_id, variant in variants.items():
        record = _as_mapping(variant, f"variants.{variant_id}", errors)
        metrics = _as_mapping(record.get("metrics"), f"variants.{variant_id}.metrics", errors)
        _validate_variant_report_ref(
            variant_id=str(variant_id),
            record=record,
            root=root,
            errors=errors,
        )
        if record.get("held_out_test_used") is not False:
            errors.append(f"{variant_id}.held_out_test_used must be false")
        if record.get("held_out_test_data_read") is not False:
            errors.append(f"{variant_id}.held_out_test_data_read must be false")
        if not isinstance(metrics.get("validation_metric_value"), (int, float)):
            errors.append(f"{variant_id}.metrics.validation_metric_value must be numeric")

    full_features = set(str(name) for name in full.get("feature_names", []))
    weak_features = set(str(name) for name in weak.get("feature_names", []))
    no_data_features = set(str(name) for name in no_data.get("feature_names", []))
    if "context_shift" not in full_features:
        errors.append("full_context_shift must include context_shift")
    if "context_shift" not in weak_features:
        errors.append("weaker_context_shift must include context_shift")
    if CONTEXT_FEATURES.intersection(no_data_features):
        errors.append("no_data_conditioning must not include context features")
    if not isinstance(full.get("context_transitions"), int) or full.get("context_transitions") <= 0:
        errors.append("full_context_shift.context_transitions must be positive")
    if not isinstance(weak.get("context_transitions"), int) or weak.get("context_transitions") <= 0:
        errors.append("weaker_context_shift.context_transitions must be positive")
    if no_data.get("context_transitions") not in (None, 0):
        errors.append("no_data_conditioning.context_transitions must be empty")
    if float(weak.get("candidate_shift_max_abs", 0.0)) >= float(
        full.get("candidate_shift_max_abs", 0.0)
    ):
        errors.append(
            "weaker_context_shift candidate range must be narrower than full_context_shift"
        )


def _validate_deltas(
    variants: Mapping[str, Any],
    deltas: Mapping[str, Any],
    errors: list[str],
) -> None:
    if "full_context_shift" not in variants:
        return
    full_metric = _metric(_as_mapping(variants["full_context_shift"], "full_context_shift", errors))
    for variant_id, raw_variant in variants.items():
        variant = _as_mapping(raw_variant, f"variants.{variant_id}", errors)
        expected = _metric(variant) - full_metric
        delta = _as_mapping(
            deltas.get(variant_id),
            f"deltas_vs_full_context_shift.{variant_id}",
            errors,
        )
        if abs(float(delta.get("absolute", float("nan"))) - expected) > 1e-12:
            errors.append(f"deltas_vs_full_context_shift.{variant_id}.absolute mismatch")
        expected_relative = 0.0 if full_metric == 0.0 else expected / full_metric
        if abs(float(delta.get("relative_to_full", float("nan"))) - expected_relative) > 1e-12:
            errors.append(f"deltas_vs_full_context_shift.{variant_id}.relative_to_full mismatch")


def _validate_interpretation(
    variants: Mapping[str, Any],
    deltas: Mapping[str, Any],
    interpretation: Mapping[str, Any],
    errors: list[str],
) -> None:
    if not REQUIRED_ABLATIONS.issubset(set(variants)):
        return
    full_metric = _metric(_as_mapping(variants["full_context_shift"], "full_context_shift", errors))
    full_is_best = all(
        full_metric <= _metric(_as_mapping(variant, str(variant_id), errors)) + 1e-12
        for variant_id, variant in variants.items()
    )
    if interpretation.get("full_context_shift_is_best") is not full_is_best:
        errors.append("context_dependency_interpretation.full_context_shift_is_best mismatch")
    weak_delta = _as_mapping(
        deltas.get("weaker_context_shift"),
        "deltas_vs_full_context_shift.weaker_context_shift",
        errors,
    )
    no_data_delta = _as_mapping(
        deltas.get("no_data_conditioning"),
        "deltas_vs_full_context_shift.no_data_conditioning",
        errors,
    )
    expected_supported = (
        full_is_best
        and float(weak_delta.get("absolute", 0.0)) > 0.0
        and float(no_data_delta.get("absolute", 0.0)) > 0.0
    )
    if interpretation.get("context_dependency_supported") is not expected_supported:
        errors.append("context_dependency_interpretation.context_dependency_supported mismatch")
    if not isinstance(interpretation.get("summary"), str) or not interpretation.get("summary"):
        errors.append("context_dependency_interpretation.summary is required")


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
        contract_ref, label="contract", root=repo_root, errors=errors
    )
    if contract_path is not None:
        errors.extend(validate_contract(load_json(contract_path), root=repo_root))

    matrix_ref = _as_mapping(evidence.get("matrix_report"), "matrix_report", errors)
    matrix_path = _validate_file_ref(
        matrix_ref,
        label="matrix_report",
        root=repo_root,
        errors=errors,
    )
    if matrix_path is not None:
        matrix = load_json(matrix_path)
        if matrix.get("variants") != evidence.get("variants"):
            errors.append("matrix_report.variants must match evidence variants")
        if matrix.get("deltas_vs_full_context_shift") != evidence.get(
            "deltas_vs_full_context_shift"
        ):
            errors.append("matrix_report.deltas_vs_full_context_shift must match evidence")

    variants = _as_mapping(evidence.get("variants"), "variants", errors)
    deltas = _as_mapping(
        evidence.get("deltas_vs_full_context_shift"),
        "deltas_vs_full_context_shift",
        errors,
    )
    interpretation = _as_mapping(
        evidence.get("context_dependency_interpretation"),
        "context_dependency_interpretation",
        errors,
    )
    _validate_variant_shapes(variants, root=repo_root, errors=errors)
    _validate_deltas(variants, deltas, errors)
    _validate_interpretation(variants, deltas, interpretation, errors)

    decision = _as_mapping(evidence.get("decision"), "decision", errors)
    if decision.get("status") != EXPECTED_DECISION_STATUS:
        errors.append(f"decision.status must be {EXPECTED_DECISION_STATUS}")
    if decision.get("held_out_test_allowed_by_this_evidence") is not False:
        errors.append("decision.held_out_test_allowed_by_this_evidence must be false")
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
