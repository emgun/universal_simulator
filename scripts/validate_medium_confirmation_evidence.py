#!/usr/bin/env python
from __future__ import annotations

"""Validate medium-or-larger confirmation evidence for the universal SOTA audit."""

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

EXPECTED_MEASUREMENT_TYPE = "medium_or_larger_confirmation_evidence"
DEFAULT_EVIDENCE_JSON = "docs/claim_evidence/medium_v1_confirmation_evidence.json"
MIN_TRAIN_SAMPLES = 512
MIN_VAL_SAMPLES = 128
MIN_TEST_SAMPLES = 128
MIN_IMPROVEMENT_FRACTION = 0.2


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


def _is_close(actual: Any, expected: Any) -> bool:
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return abs(float(actual) - float(expected)) <= 1e-12
    return actual == expected


def _non_empty(value: Any) -> bool:
    return bool(str(value or "").strip())


def _validate_scope(scope: Mapping[str, Any], errors: list[str]) -> None:
    if scope.get("version") != "medium-v1":
        errors.append("confirmation_scope.version must be medium-v1")
    if scope.get("split") != "test":
        errors.append("confirmation_scope.split must be test")
    if scope.get("data_root") != "data/pdebench_medium_v1":
        errors.append("confirmation_scope.data_root must be data/pdebench_medium_v1")
    if scope.get("decoded_rollout_steps") != 16:
        errors.append("confirmation_scope.decoded_rollout_steps must be 16")
    sample_thresholds = {
        "train_samples": MIN_TRAIN_SAMPLES,
        "val_samples": MIN_VAL_SAMPLES,
        "test_samples": MIN_TEST_SAMPLES,
    }
    for key, threshold in sample_thresholds.items():
        value = scope.get(key)
        if not isinstance(value, int) or value < threshold:
            errors.append(f"confirmation_scope.{key} must be at least {threshold}")


def validate_evidence(evidence: Mapping[str, Any], *, root: Path | None = None) -> list[str]:
    del root
    errors: list[str] = []
    if evidence.get("measurement_type") != EXPECTED_MEASUREMENT_TYPE:
        errors.append(f"measurement_type must be {EXPECTED_MEASUREMENT_TYPE}")
    if evidence.get("status") != "complete":
        errors.append("status must be complete")

    scope = _as_mapping(evidence.get("confirmation_scope"), "confirmation_scope", errors)
    _validate_scope(scope, errors)

    policy = _as_mapping(evidence.get("selection_policy"), "selection_policy", errors)
    if policy.get("selected_from_light_v1") is not True:
        errors.append("selection_policy.selected_from_light_v1 must be true")
    if policy.get("test_tuned") is not False:
        errors.append("selection_policy.test_tuned must be false")

    candidate = _as_mapping(evidence.get("candidate"), "candidate", errors)
    comparison = _as_mapping(
        evidence.get("comparison_to_persistence"),
        "comparison_to_persistence",
        errors,
    )
    if not _non_empty(candidate.get("run_name")):
        errors.append("candidate.run_name is required")
    if not _non_empty(comparison.get("persistence_run_name")):
        errors.append("comparison_to_persistence.persistence_run_name is required")
    if comparison.get("metric_name") != "decoded_rollout_nrmse":
        errors.append("comparison_to_persistence.metric_name must be decoded_rollout_nrmse")

    candidate_metric = comparison.get("candidate_metric_value")
    baseline_metric = comparison.get("baseline_metric_value")
    if not isinstance(candidate_metric, (int, float)):
        errors.append("comparison_to_persistence.candidate_metric_value must be numeric")
    if not isinstance(baseline_metric, (int, float)):
        errors.append("comparison_to_persistence.baseline_metric_value must be numeric")
    if isinstance(candidate_metric, (int, float)) and isinstance(baseline_metric, (int, float)):
        if not _is_close(candidate.get("metric_value"), candidate_metric):
            errors.append("candidate.metric_value must match comparison candidate metric")
        expected_absolute = float(baseline_metric) - float(candidate_metric)
        expected_fraction = expected_absolute / abs(float(baseline_metric))
        if not _is_close(comparison.get("absolute_improvement"), expected_absolute):
            errors.append("comparison_to_persistence.absolute_improvement mismatch")
        if not _is_close(comparison.get("improvement_fraction"), expected_fraction):
            errors.append("comparison_to_persistence.improvement_fraction mismatch")
        reported_fraction = comparison.get("improvement_fraction")
        if (
            isinstance(reported_fraction, (int, float))
            and float(reported_fraction) < MIN_IMPROVEMENT_FRACTION
        ):
            errors.append(
                "comparison_to_persistence.improvement_fraction must be at least "
                f"{MIN_IMPROVEMENT_FRACTION}"
            )
        if expected_fraction < MIN_IMPROVEMENT_FRACTION:
            errors.append(
                "comparison_to_persistence.improvement_fraction must be at least "
                f"{MIN_IMPROVEMENT_FRACTION}"
            )

    artifact = _as_mapping(evidence.get("artifact"), "artifact", errors)
    artifact_handle = str(artifact.get("handle", ""))
    if not (
        artifact_handle.startswith("b2://")
        or artifact_handle.startswith("repo:docs/claim_evidence/")
    ):
        errors.append("artifact.handle must be a b2:// or repo:docs/claim_evidence/ handle")
    if artifact.get("published") is not True:
        errors.append("artifact.published must be true")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-json", type=Path, default=Path(DEFAULT_EVIDENCE_JSON))
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)

    evidence = load_json(args.repo_root / args.evidence_json)
    errors = validate_evidence(evidence, root=args.repo_root)
    record = {
        "errors": errors,
        "evidence_json": str(args.evidence_json),
        "status": "valid" if not errors else "invalid",
    }
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
