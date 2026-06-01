#!/usr/bin/env python
from __future__ import annotations

"""Validate the UPS advection phase-tracking validation gate contract."""

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

from scripts.validate_ups_advection_model_gate_evidence import load_json

DEFAULT_CONTRACT_JSON = (
    "docs/claim_evidence/ups_advection_phase_tracking_validation_gate_contract.json"
)
EXPECTED_MEASUREMENT_TYPE = "ups_advection_phase_tracking_validation_gate_contract"
EXPECTED_THRESHOLD_METRICS = {
    "decoded_rollout_nrmse",
    "task_advection1d_decoded_rollout_nrmse",
    "task_advection1d_decoded_h16_nrmse",
}
EMPTY_ROLL_SHIFT_FIELDS = (
    "decoded_decoded_context_roll_shift_estimator",
    "decoded_decoded_observed_roll_shift_estimator",
    "decoded_decoded_prediction_roll_shift_estimator",
)


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


def _thresholds(contract: Mapping[str, Any], errors: list[str]) -> list[Mapping[str, Any]]:
    raw_thresholds = contract.get("candidate_pass_thresholds")
    if not isinstance(raw_thresholds, list):
        errors.append("candidate_pass_thresholds must be a list")
        return []
    thresholds = [
        _as_mapping(item, f"candidate_pass_thresholds[{index}]", errors)
        for index, item in enumerate(raw_thresholds)
    ]
    metrics = {str(item.get("metric")) for item in thresholds}
    if metrics != EXPECTED_THRESHOLD_METRICS:
        errors.append("candidate_pass_thresholds metrics must match expected phase gate metrics")
    return thresholds


def _compare(value: float, comparison: str, threshold: float) -> bool:
    if comparison == "<":
        return value < threshold
    if comparison == "<=":
        return value <= threshold
    raise ValueError(f"unsupported threshold comparison: {comparison}")


def validate_contract(contract: Mapping[str, Any], *, root: Path | None = None) -> list[str]:
    errors: list[str] = []
    repo_root = root or Path.cwd()
    if contract.get("measurement_type") != EXPECTED_MEASUREMENT_TYPE:
        errors.append(f"measurement_type must be {EXPECTED_MEASUREMENT_TYPE}")
    if contract.get("contract_scope") != (
        "validation_only_pre_heldout_gate_for_no_context_primary_candidates"
    ):
        errors.append("contract_scope must target validation-only no-context primary candidates")

    source_files = _as_mapping(contract.get("source_files"), "source_files", errors)
    for label, source in source_files.items():
        source_map = _as_mapping(source, f"source_files.{label}", errors)
        path_value = source_map.get("path")
        if not path_value:
            errors.append(f"source_files.{label}.path is required")
            continue
        path = repo_root / str(path_value)
        if not path.exists():
            errors.append(f"source_files.{label}.path does not exist: {path}")
            continue
        if source_map.get("sha256") != _sha256(path):
            errors.append(f"source_files.{label}.sha256 must match file bytes")
        if isinstance(source_map.get("bytes"), int) and path.stat().st_size != source_map.get(
            "bytes"
        ):
            errors.append(f"source_files.{label}.bytes must match file size")

    held_out = _as_mapping(contract.get("held_out_test_access"), "held_out_test_access", errors)
    if held_out.get("allowed_by_this_contract") is not False:
        errors.append("held_out_test_access.allowed_by_this_contract must be false")
    if held_out.get("validation_gate_does_not_authorize_test_access") is not True:
        errors.append("held_out_test_access.validation_gate_does_not_authorize_test_access true")
    forbidden = held_out.get("candidate_command_forbidden_tokens")
    if not isinstance(forbidden, list) or "--extra-eval-split test" not in forbidden:
        errors.append("held_out_test_access must forbid --extra-eval-split test")

    protocol = _as_mapping(
        contract.get("required_candidate_protocol"),
        "required_candidate_protocol",
        errors,
    )
    if protocol.get("split") != "val":
        errors.append("required_candidate_protocol.split must be val")
    if protocol.get("must_keep_context_roll_shift_estimator_empty") is not True:
        errors.append("required_candidate_protocol must require empty context roll-shift")

    _thresholds(contract, errors)
    decision = _as_mapping(contract.get("decision"), "decision", errors)
    if decision.get("status") != "active_validation_only_gate":
        errors.append("decision.status must be active_validation_only_gate")
    if decision.get("candidate_can_run_held_out_test_from_this_contract") is not False:
        errors.append("decision.candidate_can_run_held_out_test_from_this_contract must be false")
    return errors


def evaluate_candidate_summary(
    summary: Mapping[str, Any], contract: Mapping[str, Any]
) -> list[str]:
    errors: list[str] = []
    metrics = _as_mapping(summary.get("metrics"), "summary.metrics", errors)
    extra = _as_mapping(summary.get("extra"), "summary.extra", errors)
    protocol = _as_mapping(
        contract.get("required_candidate_protocol"),
        "required_candidate_protocol",
        errors,
    )
    if extra.get("decoded_split") != protocol.get("split"):
        errors.append("summary.extra.decoded_split must match required validation split")
    for field in EMPTY_ROLL_SHIFT_FIELDS:
        if extra.get(field):
            errors.append(f"summary.extra.{field} must be empty")

    for threshold in _thresholds(contract, errors):
        metric = str(threshold.get("metric"))
        comparison = str(threshold.get("comparison"))
        raw_threshold = threshold.get("threshold")
        raw_value = metrics.get(metric)
        if not isinstance(raw_threshold, (int, float)):
            errors.append(f"threshold for {metric} must be numeric")
            continue
        if not isinstance(raw_value, (int, float)):
            errors.append(f"summary.metrics.{metric} must be numeric")
            continue
        threshold_value = float(raw_threshold)
        metric_value = float(raw_value)
        if not _compare(metric_value, comparison, threshold_value):
            errors.append(
                f"summary.metrics.{metric}={metric_value} must be {comparison} {threshold_value}"
            )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-json", type=Path, default=Path(DEFAULT_CONTRACT_JSON))
    parser.add_argument("--candidate-summary-json", type=Path)
    args = parser.parse_args(argv)

    contract = load_json(args.contract_json)
    errors = validate_contract(contract, root=Path.cwd())
    if args.candidate_summary_json is not None:
        summary = load_json(args.candidate_summary_json)
        errors.extend(evaluate_candidate_summary(summary, contract))
    result = {
        "status": "valid" if not errors else "invalid",
        "contract_json": str(args.contract_json),
        "candidate_summary_json": (
            str(args.candidate_summary_json) if args.candidate_summary_json else None
        ),
        "errors": errors,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
