#!/usr/bin/env python
from __future__ import annotations

"""Validate validation-only UPS data-conditioned advection phase evidence."""

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
    "docs/claim_evidence/ups_advection_data_conditioned_phase_candidate_val_evidence.json"
)
EXPECTED_MEASUREMENT_TYPE = "ups_advection_data_conditioned_context_phase_validation"
EXPECTED_DECISION_STATUS = "validation_phase_gate_cleared_pretest_contract_required"
FORBIDDEN_COMMAND_TOKENS = (
    "--extra-eval-split test",
    "--eval-split test",
    "--allow-held-out-test-eval",
    "--held-out-test-ledger-json",
    "data.split=test",
)
METRICS_TO_RECORD = (
    "decoded_rollout_nrmse",
    "task_advection1d_decoded_rollout_nrmse",
    "task_advection1d_decoded_h1_nrmse",
    "task_advection1d_decoded_h16_nrmse",
    "task_burgers1d_decoded_rollout_nrmse",
    "task_darcy2d_decoded_rollout_nrmse",
    "decoded_data_conditioned_roll_shift_mean",
    "task_advection1d_decoded_data_conditioned_roll_shift_mean",
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


def _load_tar_bytes(path: Path, member: str) -> bytes:
    with tarfile.open(path, mode="r:gz") as archive:
        extracted = archive.extractfile(member)
        if extracted is None:
            raise FileNotFoundError(member)
        return extracted.read()


def _load_tar_json(path: Path, member: str) -> dict[str, Any]:
    payload = json.loads(_load_tar_bytes(path, member).decode("utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{member} must contain a JSON object")
    return payload


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


def _command_uses_test(command: str) -> bool:
    normalized = " ".join(command.split())
    return any(token in normalized for token in FORBIDDEN_COMMAND_TOKENS)


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
    if _command_uses_test(str(evidence.get("command", ""))):
        errors.append("command must not request held-out test evaluation")

    contract_source = _as_mapping(
        evidence.get("phase_gate_contract"), "phase_gate_contract", errors
    )
    contract_path = _validate_file_ref(
        contract_source, label="phase_gate_contract", root=repo_root, errors=errors
    )
    contract: dict[str, Any] = {}
    if contract_path is not None:
        contract = load_json(contract_path)
        errors.extend(validate_contract(contract, root=repo_root))

    artifact = _as_mapping(evidence.get("artifact"), "artifact", errors)
    artifact_path = _validate_file_ref(artifact, label="artifact", root=repo_root, errors=errors)
    if artifact_path is None:
        return errors

    contents = [str(item) for item in artifact.get("contents", [])]
    try:
        with tarfile.open(artifact_path, mode="r:gz") as archive:
            members = archive.getnames()
    except tarfile.TarError as exc:
        errors.append(f"artifact could not be read as tar.gz: {exc}")
        return errors
    if any(Path(member).name.startswith("._") for member in members):
        errors.append("artifact must not contain AppleDouble members")
    if sorted(contents) != sorted(members):
        errors.append("artifact.contents must match tar members")

    run_name = str(evidence.get("run_name"))
    summary_member = (
        "reports/research/sota_loop/data_conditioned_transport_phase/" f"{run_name}/summary.json"
    )
    gate = _as_mapping(evidence.get("train_fit_gate"), "train_fit_gate", errors)
    gate_member = str(gate.get("member", ""))
    try:
        summary = _load_tar_json(artifact_path, summary_member)
        gate_bytes = _load_tar_bytes(artifact_path, gate_member)
        gate_record = json.loads(gate_bytes.decode("utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError, KeyError) as exc:
        errors.append(f"artifact required member could not be loaded: {exc}")
        return errors

    if gate.get("sha256") != hashlib.sha256(gate_bytes).hexdigest():
        errors.append("train_fit_gate.sha256 must match artifact member bytes")
    if isinstance(gate.get("bytes"), int) and gate.get("bytes") != len(gate_bytes):
        errors.append("train_fit_gate.bytes must match artifact member size")
    if (
        gate.get("held_out_test_used") is not False
        or gate_record.get("held_out_test_used") is not False
    ):
        errors.append("train_fit_gate must not use held-out test")
    if (
        gate.get("held_out_test_data_read") is not False
        or gate_record.get("held_out_test_data_read") is not False
    ):
        errors.append("train_fit_gate must not read held-out test data")
    if gate.get("validation_guard_passed") is not True:
        errors.append("train_fit_gate.validation_guard_passed must be true")
    if gate_record.get("test_eligible") is not True:
        errors.append("train-fit gate record must be test_eligible after validation")

    selected_override = _as_mapping(
        gate.get("selected_override"), "train_fit_gate.selected_override", errors
    )
    selected_estimator = selected_override.get(
        "evaluation.decoded_data_conditioned_roll_shift_estimator"
    )
    summary_extra = _as_mapping(summary.get("extra"), "summary.extra", errors)
    summary_estimator = summary_extra.get("decoded_decoded_data_conditioned_roll_shift_estimator")
    normalized_selected = (
        dict(selected_estimator) if isinstance(selected_estimator, Mapping) else {}
    )
    normalized_selected.setdefault("families", [])
    if normalized_selected != summary_estimator:
        errors.append("train_fit_gate selected override must match artifact summary extra")
    if summary_extra.get("decoded_decoded_context_roll_shift_estimator"):
        errors.append("summary.extra.decoded_decoded_context_roll_shift_estimator must be empty")
    if summary_extra.get("decoded_decoded_observed_roll_shift_estimator"):
        errors.append("summary.extra.decoded_decoded_observed_roll_shift_estimator must be empty")
    if summary_extra.get("decoded_decoded_prediction_roll_shift_estimator"):
        errors.append("summary.extra.decoded_decoded_prediction_roll_shift_estimator must be empty")

    metrics = _as_mapping(evidence.get("metrics"), "metrics", errors)
    summary_metrics = _as_mapping(summary.get("metrics"), "summary.metrics", errors)
    for metric in METRICS_TO_RECORD:
        if metrics.get(metric) != summary_metrics.get(metric):
            errors.append(f"metrics.{metric} must match artifact summary")

    if contract:
        expected_gate_errors = evaluate_candidate_summary(summary, contract)
        phase_gate = _as_mapping(evidence.get("phase_gate"), "phase_gate", errors)
        if phase_gate.get("passed") != (not expected_gate_errors):
            errors.append("phase_gate.passed mismatch")
        if phase_gate.get("errors") != expected_gate_errors:
            errors.append("phase_gate.errors mismatch")

    decision = _as_mapping(evidence.get("decision"), "decision", errors)
    if decision.get("status") != EXPECTED_DECISION_STATUS:
        errors.append(f"decision.status must be {EXPECTED_DECISION_STATUS}")
    if decision.get("held_out_test_allowed_by_this_evidence") is not False:
        errors.append("decision.held_out_test_allowed_by_this_evidence must be false")
    if decision.get("held_out_pretest_contract_allowed") is not True:
        errors.append("decision.held_out_pretest_contract_allowed must be true")
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
