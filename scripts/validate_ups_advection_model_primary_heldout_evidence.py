#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
import tarfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.validate_ups_advection_ct1_pretest_contract import _command_measurement_key
from scripts.validate_ups_advection_model_gate_evidence import load_json

DEFAULT_EVIDENCE_JSON = (
    "docs/claim_evidence/ups_advection_model_primary_heldout_light_v1_evidence.json"
)
EXPECTED_MEASUREMENT_TYPE = "held_out_test_ups_model_primary_candidate_measurement"
CURRENT_CT8_METRIC = 0.4165820594268877
CURRENT_CT8_ADVECTION_METRIC = 0.5765863333379032


def _as_mapping(value: Any, label: str, errors: list[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(f"{label} must be an object")
        return {}
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _validate_artifact(evidence: Mapping[str, Any], root: Path, errors: list[str]) -> None:
    path = _artifact_path(evidence, root, errors)
    if path is None:
        return
    artifact = _as_mapping(evidence.get("artifact"), "artifact", errors)
    expected = set(str(item) for item in artifact.get("contents", []) if item)
    with tarfile.open(path, mode="r:gz") as archive:
        members = set(archive.getnames())
    missing = sorted(expected - members)
    if missing:
        errors.append(f"artifact.contents missing members: {missing}")
    if any(Path(member).name.startswith("._") for member in members):
        errors.append("artifact must not contain AppleDouble members")


def _validate_command(evidence: Mapping[str, Any], errors: list[str]) -> None:
    command = str(evidence.get("command", ""))
    normalized = " ".join(command.split())
    tokens = shlex.split(command)
    required_tokens = (
        "--skip-training",
        "--stage operator_decoded",
        "--extra-eval-split test",
        "--held-out-test-ledger-json",
        "evaluation.decoded_persistence_residual_alpha_by_family={transport: 0.21}",
    )
    for token in required_tokens:
        if token not in normalized:
            errors.append(f"command must include {token}")
    forbidden_tokens = (
        "--allow-repeat-held-out-test",
        "evaluation.decoded_context_roll_shift_estimator",
        "evaluation.decoded_observed_roll_shift_estimator",
        "evaluation.decoded_prediction_roll_shift_estimator",
    )
    for token in forbidden_tokens:
        if token in normalized:
            errors.append(f"command must not include {token}")
    if "--promotion-rule" not in tokens:
        errors.append("command must include --promotion-rule")

    policy = _as_mapping(evidence.get("held_out_test_policy"), "held_out_test_policy", errors)
    try:
        computed_key = _command_measurement_key(command)
    except Exception as exc:
        errors.append(str(exc))
        computed_key = None
    if computed_key and policy.get("measurement_key") != computed_key:
        errors.append("held_out_test_policy.measurement_key does not match command-derived key")


def validate_evidence(evidence: Mapping[str, Any], *, root: Path | None = None) -> list[str]:
    errors: list[str] = []
    repo_root = root or Path.cwd()
    if evidence.get("measurement_type") != EXPECTED_MEASUREMENT_TYPE:
        errors.append(f"measurement_type must be {EXPECTED_MEASUREMENT_TYPE}")
    if evidence.get("held_out_test_used") is not True:
        errors.append("held_out_test_used must be true")
    if evidence.get("test_split_accessed") is not True:
        errors.append("test_split_accessed must be true")
    if evidence.get("split") != "test":
        errors.append("split must be test")
    if evidence.get("checkpoint_preference_stage") != "operator_decoded":
        errors.append("checkpoint_preference_stage must be operator_decoded")

    _validate_command(evidence, errors)
    _validate_artifact(evidence, repo_root, errors)

    policy = _as_mapping(evidence.get("held_out_test_policy"), "held_out_test_policy", errors)
    if policy.get("exactly_one_test_after_validation") is not True:
        errors.append("held_out_test_policy.exactly_one_test_after_validation must be true")
    if policy.get("allow_repeat_held_out_test") is not False:
        errors.append("held_out_test_policy.allow_repeat_held_out_test must be false")

    validation = _as_mapping(evidence.get("validation_metrics"), "validation_metrics", errors)
    test_metrics = _as_mapping(evidence.get("test_metrics"), "test_metrics", errors)
    if validation.get("decoded_rollout_nrmse") != 0.35078329353213156:
        errors.append("validation_metrics.decoded_rollout_nrmse must match pretest selection")
    test_metric = test_metrics.get("decoded_rollout_nrmse")
    test_advection = test_metrics.get("task_advection1d_decoded_rollout_nrmse")
    if test_metric != 0.5226095521324494:
        errors.append("test_metrics.decoded_rollout_nrmse must match measured held-out result")
    comparison = _as_mapping(
        evidence.get("comparison_to_current_ct8_claim"),
        "comparison_to_current_ct8_claim",
        errors,
    )
    if comparison.get("candidate_beats_current_ct8_primary") is not False:
        errors.append(
            "comparison_to_current_ct8_claim.candidate_beats_current_ct8_primary must be false"
        )
    if isinstance(test_metric, (int, float)):
        expected = CURRENT_CT8_METRIC - float(test_metric)
        if comparison.get("absolute_overall_improvement") != expected:
            errors.append("comparison_to_current_ct8_claim.absolute_overall_improvement mismatch")
    if isinstance(test_advection, (int, float)):
        expected = CURRENT_CT8_ADVECTION_METRIC - float(test_advection)
        if comparison.get("absolute_advection_improvement") != expected:
            errors.append("comparison_to_current_ct8_claim.absolute_advection_improvement mismatch")

    decision = _as_mapping(evidence.get("decision"), "decision", errors)
    if decision.get("status") != "held_out_complete_not_promoted":
        errors.append("decision.status must be held_out_complete_not_promoted")
    if decision.get("do_not_repeat_held_out_test") is not True:
        errors.append("decision.do_not_repeat_held_out_test must be true")
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
