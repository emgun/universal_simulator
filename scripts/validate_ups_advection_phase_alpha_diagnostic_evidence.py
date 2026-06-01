#!/usr/bin/env python
from __future__ import annotations

"""Validate validation-only UPS advection phase-alpha diagnostic evidence."""

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

DEFAULT_EVIDENCE_JSON = "docs/claim_evidence/ups_advection_phase_alpha_diagnostic_val_evidence.json"
EXPECTED_MEASUREMENT_TYPE = "ups_advection_phase_alpha_diagnostic_validation"
EXPECTED_DECISION_STATUS = "phase_gate_not_cleared_alpha_diagnostic"
EXPECTED_ALPHAS = [0.0, 0.1, 0.21, 0.3, 0.4]
METRICS_TO_RECORD = (
    "decoded_rollout_nrmse",
    "task_advection1d_decoded_rollout_nrmse",
    "task_advection1d_decoded_h16_nrmse",
    "task_burgers1d_decoded_rollout_nrmse",
    "task_darcy2d_decoded_rollout_nrmse",
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


def _as_list(value: Any, label: str, errors: list[str]) -> list[Any]:
    if not isinstance(value, list):
        errors.append(f"{label} must be a list")
        return []
    return value


def _load_tar_json(archive: tarfile.TarFile, member: str) -> dict[str, Any]:
    extracted = archive.extractfile(member)
    if extracted is None:
        raise FileNotFoundError(member)
    payload = json.load(extracted)
    if not isinstance(payload, dict):
        raise TypeError(f"{member} must contain a JSON object")
    return payload


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
    if isinstance(source.get("bytes"), int) and path.stat().st_size != source.get("bytes"):
        errors.append("phase_gate_contract.bytes must match file size")
    contract = load_json(path)
    errors.extend(validate_contract(contract, root=root))
    return contract


def _validate_candidate(
    *,
    candidate: Mapping[str, Any],
    summary: Mapping[str, Any],
    contract: Mapping[str, Any],
    errors: list[str],
) -> None:
    metrics = _as_mapping(candidate.get("metrics"), "alpha_candidates[].metrics", errors)
    summary_metrics = _as_mapping(summary.get("metrics"), "summary.metrics", errors)
    for metric in METRICS_TO_RECORD:
        if metrics.get(metric) != summary_metrics.get(metric):
            errors.append(f"alpha_candidates metric drift for {candidate.get('run_name')} {metric}")
    expected_errors = evaluate_candidate_summary(summary, contract)
    if candidate.get("phase_gate_passed") != (not expected_errors):
        errors.append(
            f"alpha_candidates phase_gate_passed mismatch for {candidate.get('run_name')}"
        )
    if candidate.get("phase_gate_errors") != expected_errors:
        errors.append(
            f"alpha_candidates phase_gate_errors mismatch for {candidate.get('run_name')}"
        )


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
    if "--extra-eval-split test" in str(evidence.get("command_template", "")):
        errors.append("command_template must not include --extra-eval-split test")

    contract = _contract(evidence, repo_root, errors)
    artifact_path = _artifact_path(evidence, repo_root, errors)
    if artifact_path is None:
        return errors
    artifact = _as_mapping(evidence.get("artifact"), "artifact", errors)
    expected_contents = [
        str(item) for item in _as_list(artifact.get("contents"), "artifact.contents", errors)
    ]
    candidates = [
        _as_mapping(item, f"alpha_candidates[{index}]", errors)
        for index, item in enumerate(
            _as_list(evidence.get("alpha_candidates"), "alpha_candidates", errors)
        )
    ]
    alphas = [candidate.get("alpha") for candidate in candidates]
    if alphas != EXPECTED_ALPHAS:
        errors.append("alpha_candidates must record the expected alpha sweep")

    summaries_by_member: dict[str, Mapping[str, Any]] = {}
    with tarfile.open(artifact_path, mode="r:gz") as archive:
        members = archive.getnames()
        if any(Path(member).name.startswith("._") for member in members):
            errors.append("artifact must not contain AppleDouble members")
        missing = sorted(set(expected_contents) - set(members))
        if missing:
            errors.append(f"artifact.contents missing members: {missing}")
        for member in expected_contents:
            if member in members:
                summary = _load_tar_json(archive, member)
                summaries_by_member[member] = summary

    with tarfile.open(artifact_path, mode="r:gz") as archive:
        for candidate in candidates:
            member = str(candidate.get("summary_member"))
            extracted = archive.extractfile(member)
            if extracted is None:
                errors.append(f"alpha_candidates summary_member missing: {member}")
                continue
            if candidate.get("summary_sha256") != hashlib.sha256(extracted.read()).hexdigest():
                errors.append(f"alpha_candidates summary_sha256 mismatch for {member}")

    for candidate in candidates:
        member = str(candidate.get("summary_member"))
        summary = summaries_by_member.get(member)
        if summary is not None:
            _validate_candidate(
                candidate=candidate, summary=summary, contract=contract, errors=errors
            )

    selected = _as_mapping(
        evidence.get("selected_best_validation_candidate"),
        "selected_best_validation_candidate",
        errors,
    )
    if candidates:
        best = min(
            candidates,
            key=lambda candidate: float(
                _as_mapping(candidate.get("metrics"), "alpha_candidates[].metrics", errors).get(
                    "decoded_rollout_nrmse", float("inf")
                )
            ),
        )
        if selected.get("run_name") != best.get("run_name"):
            errors.append("selected_best_validation_candidate.run_name must match best overall")
        if selected.get("phase_gate_passed") is not False:
            errors.append("selected_best_validation_candidate.phase_gate_passed must be false")
    if any(candidate.get("phase_gate_passed") for candidate in candidates):
        errors.append("no alpha candidate should clear the phase gate in this evidence")

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
    result = {
        "status": "valid" if not errors else "invalid",
        "evidence_json": str(args.evidence_json),
        "errors": errors,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
