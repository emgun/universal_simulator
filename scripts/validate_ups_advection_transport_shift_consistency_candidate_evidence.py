#!/usr/bin/env python
from __future__ import annotations

"""Validate validation-only UPS advection transport shift-consistency evidence."""

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
    "docs/claim_evidence/" "ups_advection_transport_shift_consistency_candidate_val_evidence.json"
)
EXPECTED_MEASUREMENT_TYPE = (
    "ups_advection_transport_shift_consistency_training_candidate_validation"
)
EXPECTED_DECISION_STATUS = "phase_gate_not_cleared_transport_shift_consistency_candidate"
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


def _load_tar_json(path: Path, member: str) -> dict[str, Any]:
    payload = json.loads(_load_tar_bytes(path, member).decode("utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{member} must contain a JSON object")
    return payload


def _load_tar_bytes(path: Path, member: str) -> bytes:
    with tarfile.open(path, mode="r:gz") as archive:
        extracted = archive.extractfile(member)
        if extracted is None:
            raise FileNotFoundError(member)
        return extracted.read()


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
    if code_change.get("new_stage_config_key") != "transport_shift_consistency_lambda":
        errors.append("code_change.new_stage_config_key must be transport_shift_consistency_lambda")
    training = _as_mapping(evidence.get("training"), "training", errors)
    if training.get("transport_shift_consistency_lambda") != 1.0:
        errors.append("training.transport_shift_consistency_lambda must be 1.0")
    shifts = _as_mapping(
        training.get("transport_shift_consistency_by_task"),
        "training.transport_shift_consistency_by_task",
        errors,
    )
    if shifts.get("advection1d") != 1:
        errors.append("training.transport_shift_consistency_by_task.advection1d must be 1")
    if training.get("stage") != "operator_decoded":
        errors.append("training.stage must be operator_decoded")

    contract_source = _as_mapping(
        evidence.get("phase_gate_contract"), "phase_gate_contract", errors
    )
    contract_path = _validate_file_ref(
        contract_source, label="phase_gate_contract", root=repo_root, errors=errors
    )
    contract = {}
    if contract_path is not None:
        contract = load_json(contract_path)
        errors.extend(validate_contract(contract, root=repo_root))

    artifact = _as_mapping(evidence.get("artifact"), "artifact", errors)
    artifact_path = _validate_file_ref(artifact, label="artifact", root=repo_root, errors=errors)
    if artifact_path is None:
        return errors
    contents = [str(item) for item in artifact.get("contents", [])]
    with tarfile.open(artifact_path, mode="r:gz") as archive:
        members = archive.getnames()
    if any(Path(member).name.startswith("._") for member in members):
        errors.append("artifact must not contain AppleDouble members")
    missing = sorted(set(contents) - set(members))
    if missing:
        errors.append(f"artifact.contents missing members: {missing}")

    run_name = evidence.get("run_name")
    summary_member = (
        "reports/research/sota_loop/model_advection_transport_shift_consistency/"
        f"{run_name}/summary.json"
    )
    summary = _load_tar_json(artifact_path, summary_member)
    summary_metrics = _as_mapping(summary.get("metrics"), "summary.metrics", errors)
    metrics = _as_mapping(evidence.get("metrics"), "metrics", errors)
    for metric in METRICS_TO_RECORD:
        if metrics.get(metric) != summary_metrics.get(metric):
            errors.append(f"metrics.{metric} must match artifact summary")

    shift_fit = _as_mapping(
        evidence.get("train_fit_shift_diagnostic"),
        "train_fit_shift_diagnostic",
        errors,
    )
    shift_fit_member = (
        "reports/research/sota_loop/transport_phase_objective/shift_fit_train32_val32.json"
    )
    shift_fit_bytes = _load_tar_bytes(artifact_path, shift_fit_member)
    if shift_fit.get("sha256") != hashlib.sha256(shift_fit_bytes).hexdigest():
        errors.append("train_fit_shift_diagnostic.sha256 must match artifact member bytes")
    if isinstance(shift_fit.get("bytes"), int) and shift_fit.get("bytes") != len(shift_fit_bytes):
        errors.append("train_fit_shift_diagnostic.bytes must match artifact member size")
    shift_fit_record = json.loads(shift_fit_bytes.decode("utf-8"))
    if shift_fit.get("selected_train_shift") != shift_fit_record.get("selected_train_shift"):
        errors.append("train_fit_shift_diagnostic.selected_train_shift mismatch")
    if shift_fit.get("validation_oracle_shift") != shift_fit_record.get(
        "oracle_validation", {}
    ).get("shift"):
        errors.append("train_fit_shift_diagnostic.validation_oracle_shift mismatch")
    if shift_fit.get("selected_train_shift") != shifts.get("advection1d"):
        errors.append("training shift must match train-fitted selected shift")

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
