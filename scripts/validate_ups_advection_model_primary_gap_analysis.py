#!/usr/bin/env python
from __future__ import annotations

"""Validate the UPS advection primary-candidate gap analysis evidence."""

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.build_ups_advection_model_primary_gap_analysis import (
    DEFAULT_CLAIM_EVIDENCE_JSON,
    DEFAULT_HELDOUT_EVIDENCE_JSON,
    EXPECTED_MEASUREMENT_TYPE,
    build_analysis,
)
from scripts.validate_ups_advection_model_gate_evidence import load_json

DEFAULT_ANALYSIS_JSON = "docs/claim_evidence/ups_advection_model_primary_gap_analysis.json"


def _is_close(left: Any, right: Any) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-15)
    return left == right


def _diff_paths(actual: Any, expected: Any, path: str = "$") -> list[str]:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            return [f"{path} must be an object"]
        errors: list[str] = []
        for key, expected_value in expected.items():
            next_path = f"{path}.{key}"
            if key not in actual:
                errors.append(f"{next_path} is missing")
            else:
                errors.extend(_diff_paths(actual[key], expected_value, next_path))
        for key in actual:
            if key not in expected:
                errors.append(f"{path}.{key} is unexpected")
        return errors
    if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes, bytearray)):
        if not isinstance(actual, Sequence) or isinstance(actual, (str, bytes, bytearray)):
            return [f"{path} must be a list"]
        if len(actual) != len(expected):
            return [f"{path} length mismatch"]
        errors = []
        for index, expected_value in enumerate(expected):
            errors.extend(_diff_paths(actual[index], expected_value, f"{path}[{index}]"))
        return errors
    if not _is_close(actual, expected):
        return [f"{path} mismatch: expected {expected!r}, got {actual!r}"]
    return []


def _as_mapping(value: Any, label: str, errors: list[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(f"{label} must be an object")
        return {}
    return value


def validate_analysis(analysis: Mapping[str, Any], *, root: Path | None = None) -> list[str]:
    repo_root = root or Path.cwd()
    errors: list[str] = []
    if analysis.get("measurement_type") != EXPECTED_MEASUREMENT_TYPE:
        errors.append(f"measurement_type must be {EXPECTED_MEASUREMENT_TYPE}")
    if analysis.get("new_held_out_test_command_executed") is not False:
        errors.append("new_held_out_test_command_executed must be false")
    if analysis.get("held_out_test_data_reaccessed") is not False:
        errors.append("held_out_test_data_reaccessed must be false")
    if analysis.get("uses_existing_held_out_test_summary") is not True:
        errors.append("uses_existing_held_out_test_summary must be true")

    comparison = _as_mapping(
        analysis.get("candidate_vs_current_ct8_primary"),
        "candidate_vs_current_ct8_primary",
        errors,
    )
    if comparison.get("candidate_beats_current_ct8_primary") is not False:
        errors.append(
            "candidate_vs_current_ct8_primary.candidate_beats_current_ct8_primary must be false"
        )
    dominant = _as_mapping(
        comparison.get("dominant_test_regression"),
        "candidate_vs_current_ct8_primary.dominant_test_regression",
        errors,
    )
    if dominant.get("metric") != "task_advection1d_decoded_rollout_nrmse":
        errors.append("dominant_test_regression.metric must be advection rollout NRMSE")

    decision = _as_mapping(analysis.get("decision"), "decision", errors)
    if decision.get("status") != "gap_analysis_complete_no_promotion_no_rerun":
        errors.append("decision.status must be gap_analysis_complete_no_promotion_no_rerun")
    if decision.get("candidate_promoted") is not False:
        errors.append("decision.candidate_promoted must be false")
    if decision.get("do_not_repeat_held_out_test") is not True:
        errors.append("decision.do_not_repeat_held_out_test must be true")

    expected = build_analysis(
        root=repo_root,
        heldout_evidence_json=Path(DEFAULT_HELDOUT_EVIDENCE_JSON),
        claim_evidence_json=Path(DEFAULT_CLAIM_EVIDENCE_JSON),
    )
    errors.extend(_diff_paths(analysis, expected)[:20])
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-json", type=Path, default=Path(DEFAULT_ANALYSIS_JSON))
    args = parser.parse_args(argv)

    analysis = load_json(args.analysis_json)
    errors = validate_analysis(analysis, root=Path.cwd())
    result = {
        "status": "valid" if not errors else "invalid",
        "analysis_json": str(args.analysis_json),
        "errors": errors,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
