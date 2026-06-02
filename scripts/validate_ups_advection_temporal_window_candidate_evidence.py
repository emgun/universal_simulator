#!/usr/bin/env python
from __future__ import annotations

"""Validate validation-only UPS advection temporal-window candidate evidence."""

import argparse
import hashlib
import json
import sys
import tarfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.validate_ups_advection_model_gate_evidence import load_json
from scripts.validate_ups_advection_phase_tracking_gate_contract import (
    evaluate_candidate_summary,
    validate_contract,
)

DEFAULT_EVIDENCE_JSON = (
    "docs/claim_evidence/ups_advection_temporal_window_candidate_val_evidence.json"
)
EXPECTED_MEASUREMENT_TYPE = "ups_advection_temporal_window_training_candidate_validation"
EXPECTED_DECISION_STATUS = "phase_gate_not_cleared_temporal_window_candidate"
METRICS_TO_RECORD = (
    "decoded_rollout_nrmse",
    "task_advection1d_decoded_rollout_nrmse",
    "task_advection1d_decoded_h16_nrmse",
    "task_burgers1d_decoded_rollout_nrmse",
    "task_darcy2d_decoded_rollout_nrmse",
    "decoded_rollout_spectral_energy_error",
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


def _artifact_path(evidence: Mapping[str, Any], root: Path, errors: list[str]) -> Path | None:
    artifact = _as_mapping(evidence.get("artifact"), "artifact", errors)
    raw_path = artifact.get("path")
    if not raw_path:
        errors.append("artifact.path is required")
        return None
    path = root / str(raw_path)
    if not path.exists():
        errors.append(f"artifact.path does not exist: {path}")
        return None
    if artifact.get("sha256") != _sha256(path):
        errors.append("artifact.sha256 must match artifact bytes")
    if isinstance(artifact.get("bytes"), int) and path.stat().st_size != artifact.get("bytes"):
        errors.append("artifact.bytes must match artifact size")
    return path


def _load_tar_json(path: Path, member: str) -> dict[str, Any]:
    with tarfile.open(path, mode="r:gz") as archive:
        extracted = archive.extractfile(member)
        if extracted is None:
            raise FileNotFoundError(member)
        payload = json.load(extracted)
    if not isinstance(payload, dict):
        raise TypeError(f"{member} must contain a JSON object")
    return payload


def _contract(evidence: Mapping[str, Any], root: Path, errors: list[str]) -> Mapping[str, Any]:
    source = _as_mapping(evidence.get("phase_gate_contract"), "phase_gate_contract", errors)
    raw_path = source.get("path")
    if not raw_path:
        errors.append("phase_gate_contract.path is required")
        return {}
    path = root / str(raw_path)
    if not path.exists():
        errors.append(f"phase_gate_contract.path does not exist: {path}")
        return {}
    if source.get("sha256") != _sha256(path):
        errors.append("phase_gate_contract.sha256 must match file bytes")
    contract = load_json(path)
    errors.extend(validate_contract(contract, root=root))
    return contract


def validate_evidence(evidence: Mapping[str, Any], *, root: Path | None = None) -> list[str]:
    errors: list[str] = []
    repo_root = root or Path.cwd()
    if evidence.get("measurement_type") != EXPECTED_MEASUREMENT_TYPE:
        errors.append(f"measurement_type must be {EXPECTED_MEASUREMENT_TYPE}")
    if evidence.get("held_out_test_used") is not False:
        errors.append("held_out_test_used must be false")
    if evidence.get("held_out_test_data_read") is not False:
        errors.append("held_out_test_data_read must be false")
    if evidence.get("split") != "val":
        errors.append("split must be val")
    if evidence.get("checkpoint_preference_stage") != "operator_decoded":
        errors.append("checkpoint_preference_stage must be operator_decoded")

    code_change = _as_mapping(evidence.get("code_change"), "code_change", errors)
    if code_change.get("new_stage_config_key") != "rollout_start_strategy":
        errors.append("code_change.new_stage_config_key must be rollout_start_strategy")
    if code_change.get("candidate_value") != "latest":
        errors.append("code_change.candidate_value must be latest")
    training = _as_mapping(evidence.get("training"), "training", errors)
    if training.get("training_rollout_steps") != 16:
        errors.append("training.training_rollout_steps must be 16")
    if training.get("rollout_start_strategy") != "latest":
        errors.append("training.rollout_start_strategy must be latest")
    if training.get("rollout_loss_horizon_power") != 2.0:
        errors.append("training.rollout_loss_horizon_power must be 2.0")
    if training.get("stage") != "operator_decoded":
        errors.append("training.stage must be operator_decoded")

    contract = _contract(evidence, repo_root, errors)
    artifact_path = _artifact_path(evidence, repo_root, errors)
    if artifact_path is None:
        return errors
    artifact = _as_mapping(evidence.get("artifact"), "artifact", errors)
    contents = [str(item) for item in artifact.get("contents", [])]
    with tarfile.open(artifact_path, mode="r:gz") as archive:
        members = archive.getnames()
    if any(Path(member).name.startswith("._") for member in members):
        errors.append("artifact must not contain AppleDouble members")
    missing = sorted(set(contents) - set(members))
    if missing:
        errors.append(f"artifact.contents missing members: {missing}")

    summary_member = f"{evidence.get('run_name')}/summary.json"
    summary = _load_tar_json(artifact_path, summary_member)
    summary_metrics = _as_mapping(summary.get("metrics"), "summary.metrics", errors)
    metrics = _as_mapping(evidence.get("metrics"), "metrics", errors)
    for metric in METRICS_TO_RECORD:
        if metrics.get(metric) != summary_metrics.get(metric):
            errors.append(f"metrics.{metric} must match artifact summary")
    expected_gate_errors = evaluate_candidate_summary(summary, contract)
    phase_gate = _as_mapping(evidence.get("phase_gate"), "phase_gate", errors)
    if phase_gate.get("passed") != (not expected_gate_errors):
        errors.append("phase_gate.passed mismatch")
    if phase_gate.get("errors") != expected_gate_errors:
        errors.append("phase_gate.errors mismatch")

    comparison = _as_mapping(
        evidence.get("comparison_to_horizon_weighted_candidate"),
        "comparison_to_horizon_weighted_candidate",
        errors,
    )
    for key in (
        "overall_absolute_delta",
        "advection_rollout_absolute_delta",
        "advection_h16_absolute_delta",
    ):
        if comparison.get(key, 0.0) <= 0.0:
            errors.append(f"comparison_to_horizon_weighted_candidate.{key} must be positive")

    decision = _as_mapping(evidence.get("decision"), "decision", errors)
    if decision.get("status") != EXPECTED_DECISION_STATUS:
        errors.append(f"decision.status must be {EXPECTED_DECISION_STATUS}")
    if decision.get("held_out_pretest_contract_allowed") is not False:
        errors.append("decision.held_out_pretest_contract_allowed must be false")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-json", type=Path, default=Path(DEFAULT_EVIDENCE_JSON))
    args = parser.parse_args(argv)

    evidence = load_json(args.evidence_json)
    errors = validate_evidence(evidence, root=Path.cwd())
    if errors:
        print("Evidence validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "passed", "evidence_json": str(args.evidence_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
