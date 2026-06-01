#!/usr/bin/env python
from __future__ import annotations

"""Validate UPS model-side advection validation-gate evidence."""

import argparse
import hashlib
import json
import math
import sys
import tarfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

DEFAULT_EVIDENCE_JSON = "docs/claim_evidence/ups_advection_model_gate_val_evidence.json"
GATE_MEASUREMENT_TYPE = "ups_advection_model_gate_validation"
STABILITY_MEASUREMENT_TYPE = "ups_advection_model_stability_validation"
SUPPORTED_MEASUREMENT_TYPES = {
    GATE_MEASUREMENT_TYPE,
    STABILITY_MEASUREMENT_TYPE,
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def _as_mapping(value: Any, label: str, errors: list[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(f"{label} must be an object")
        return {}
    return value


def _as_list(value: Any, label: str, errors: list[str]) -> list[Any]:
    if not isinstance(value, list):
        errors.append(f"{label} must be a list")
        return []
    return value


def _close_enough(left: Any, right: Any) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-15)
    return left == right


def _artifact_path(evidence: Mapping[str, Any], root: Path, errors: list[str]) -> Path | None:
    handle = str(evidence.get("artifact_handle", ""))
    prefix = "repo:"
    if not handle.startswith(prefix):
        errors.append("artifact_handle must use repo: prefix")
        return None
    path = root / handle[len(prefix) :]
    if not path.exists():
        errors.append(f"artifact_handle path does not exist: {path}")
        return None
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_artifact(evidence: Mapping[str, Any], root: Path, errors: list[str]) -> None:
    path = _artifact_path(evidence, root, errors)
    if path is None:
        return
    expected_sha = str(evidence.get("artifact_sha256", ""))
    if _sha256(path) != expected_sha:
        errors.append("artifact_sha256 must match artifact bytes")
    expected_bytes = evidence.get("artifact_bytes")
    if isinstance(expected_bytes, int) and path.stat().st_size != expected_bytes:
        errors.append("artifact_bytes must match artifact size")

    with tarfile.open(path, mode="r:gz") as archive:
        members = archive.getnames()
    if any(Path(member).name.startswith("._") for member in members):
        errors.append("artifact must not contain AppleDouble members")
    selected = _as_mapping(
        evidence.get("selected_validation_candidate"),
        "selected_validation_candidate",
        errors,
    )
    selected_summary = selected.get("summary_json")
    if selected_summary and not any(
        member == selected_summary or str(selected_summary).endswith(member) for member in members
    ):
        errors.append("artifact must include selected_validation_candidate.summary_json")


def _best_by_metric(items: list[Mapping[str, Any]]) -> Mapping[str, Any]:
    return min(items, key=lambda item: float(item.get("metric_value", float("inf"))))


def _validate_stability_replicates(
    evidence: Mapping[str, Any],
    *,
    selected: Mapping[str, Any],
    baseline_metric: Any,
    errors: list[str],
) -> None:
    replicates = [
        _as_mapping(item, f"stability_replicates[{index}]", errors)
        for index, item in enumerate(
            _as_list(evidence.get("stability_replicates"), "stability_replicates", errors)
        )
    ]
    if len(replicates) < 2:
        errors.append("stability_replicates must include at least two validation runs")
    seeds = {replicate.get("seed") for replicate in replicates if "seed" in replicate}
    if len(seeds) < 2:
        errors.append("stability_replicates must include at least two distinct seeds")
    if isinstance(baseline_metric, (int, float)):
        for index, replicate in enumerate(replicates):
            metric = replicate.get("metric_value")
            if isinstance(metric, (int, float)) and float(metric) >= float(baseline_metric):
                errors.append(f"stability_replicates[{index}].metric_value must improve baseline")
    if replicates:
        best = _best_by_metric(replicates)
        if not _close_enough(selected.get("metric_value"), best.get("metric_value")):
            errors.append(
                "selected_validation_candidate.metric_value must match stability_replicates best"
            )
        if selected.get("run_name") != best.get("run_name"):
            errors.append(
                "selected_validation_candidate.run_name must match stability_replicates best"
            )


def validate_evidence(evidence: Mapping[str, Any], *, root: Path | None = None) -> list[str]:
    errors: list[str] = []
    repo_root = root or Path.cwd()

    if evidence.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    measurement_type = evidence.get("measurement_type")
    if measurement_type not in SUPPORTED_MEASUREMENT_TYPES:
        errors.append(
            "measurement_type must be one of " + ", ".join(sorted(SUPPORTED_MEASUREMENT_TYPES))
        )
    if evidence.get("held_out_test_used") is not False:
        errors.append("held_out_test_used must be false")
    if evidence.get("held_out_test_data_read") is not False:
        errors.append("held_out_test_data_read must be false")
    if evidence.get("claim_comparable") is not False:
        errors.append("claim_comparable must be false for validation-only evidence")
    if evidence.get("published_numbers_directly_comparable") is not False:
        errors.append("published_numbers_directly_comparable must be false")
    if evidence.get("external_paper_reproduction") is not False:
        errors.append("external_paper_reproduction must be false")

    protocol = _as_mapping(evidence.get("data_protocol"), "data_protocol", errors)
    forbidden = set(
        str(item)
        for item in _as_list(
            protocol.get("forbidden_eval_features"), "forbidden_eval_features", errors
        )
    )
    for required in (
        "evaluation.decoded_context_roll_shift_estimator",
        "evaluation.decoded_observed_roll_shift_estimator",
        "evaluation.decoded_prediction_roll_shift_estimator",
        "--extra-eval-split test",
    ):
        if required not in forbidden:
            errors.append(f"data_protocol.forbidden_eval_features missing {required}")

    baseline = _as_mapping(evidence.get("baseline_validation"), "baseline_validation", errors)
    selected = _as_mapping(
        evidence.get("selected_validation_candidate"),
        "selected_validation_candidate",
        errors,
    )
    baseline_metric = baseline.get("metric_value")
    selected_metric = selected.get("metric_value")
    if isinstance(baseline_metric, (int, float)) and isinstance(selected_metric, (int, float)):
        if float(selected_metric) >= float(baseline_metric):
            errors.append("selected_validation_candidate.metric_value must improve baseline")

    selected_eval = _as_mapping(
        selected.get("evaluation"),
        "selected_validation_candidate.evaluation",
        errors,
    )
    for key in (
        "decoded_context_roll_shift_estimator",
        "decoded_observed_roll_shift_estimator",
        "decoded_prediction_roll_shift_estimator",
    ):
        if selected_eval.get(key) not in ({}, None):
            errors.append(f"selected_validation_candidate.evaluation.{key} must be empty")

    alpha_sweep: list[Mapping[str, Any]] = []
    if "alpha_sweep" in evidence:
        alpha_sweep = [
            _as_mapping(item, f"alpha_sweep[{index}]", errors)
            for index, item in enumerate(
                _as_list(evidence.get("alpha_sweep"), "alpha_sweep", errors)
            )
        ]
    elif measurement_type == GATE_MEASUREMENT_TYPE:
        errors.append("alpha_sweep must be a list")
    if alpha_sweep:
        best = _best_by_metric(alpha_sweep)
        if not _close_enough(selected.get("metric_value"), best.get("metric_value")):
            errors.append("selected_validation_candidate.metric_value must match alpha_sweep best")
        if selected.get("run_name") != best.get("run_name"):
            errors.append("selected_validation_candidate.run_name must match alpha_sweep best")

    if measurement_type == STABILITY_MEASUREMENT_TYPE:
        _validate_stability_replicates(
            evidence,
            selected=selected,
            baseline_metric=baseline_metric,
            errors=errors,
        )

    improvements = _as_mapping(
        selected.get("improvement_vs_baseline"),
        "selected_validation_candidate.improvement_vs_baseline",
        errors,
    )
    if float(improvements.get("decoded_rollout_nrmse_absolute", 0.0)) <= 0.0:
        errors.append("decoded_rollout_nrmse improvement must be positive")
    if float(improvements.get("task_advection1d_decoded_rollout_nrmse_absolute", 0.0)) <= 0.0:
        errors.append("task_advection1d improvement must be positive")

    _validate_artifact(evidence, repo_root, errors)
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-json", type=Path, default=Path(DEFAULT_EVIDENCE_JSON))
    args = parser.parse_args(argv)

    evidence = load_json(args.evidence_json)
    errors = validate_evidence(evidence, root=Path.cwd())
    result = {
        "status": "valid" if not errors else "invalid",
        "evidence_json": str(args.evidence_json),
        "errors": errors,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
