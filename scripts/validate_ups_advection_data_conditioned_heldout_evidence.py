#!/usr/bin/env python
from __future__ import annotations

"""Validate the data-conditioned UPS advection held-out evidence package."""

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

from scripts.validate_ups_advection_data_conditioned_phase_candidate_evidence import (
    load_json,
)
from scripts.validate_ups_advection_data_conditioned_phase_candidate_evidence import (
    validate_evidence as validate_validation_evidence,
)
from scripts.validate_ups_advection_data_conditioned_pretest_contract import (
    _command_measurement_key,
)

DEFAULT_EVIDENCE_JSON = (
    "docs/claim_evidence/ups_advection_data_conditioned_heldout_light_v1_evidence.json"
)
EXPECTED_MEASUREMENT_TYPE = "held_out_test_ups_data_conditioned_context_phase_measurement"
EXPECTED_PRETEST_CONTRACT_TYPE = "ups_advection_data_conditioned_pretest_contract"
CURRENT_CT8_METRIC = 0.4165820594268877
CURRENT_CT8_ADVECTION_METRIC = 0.5765863333379032
CURRENT_CT1_METRIC = 0.20177292896682064
CURRENT_CT1_ADVECTION_METRIC = 0.22508631227914033
REQUIRED_PROMOTION_RULES = {
    "decoded_rollout_nrmse<=0.35078329353213156",
    "task_advection1d_decoded_rollout_nrmse<=0.4866576789288726",
    "task_advection1d_decoded_h16_nrmse<=0.44444171136384397",
}
KEY_METRICS = (
    "decoded_rollout_nrmse",
    "decoded_rollout_spectral_energy_error",
    "task_advection1d_decoded_rollout_nrmse",
    "task_advection1d_decoded_h1_nrmse",
    "task_advection1d_decoded_h16_nrmse",
    "task_burgers1d_decoded_rollout_nrmse",
    "task_darcy2d_decoded_rollout_nrmse",
    "decoded_data_conditioned_roll_shift_mean",
    "task_advection1d_decoded_data_conditioned_roll_shift_mean",
)


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


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _close_enough(actual: Any, expected: Any) -> bool:
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return abs(float(actual) - float(expected)) <= 1e-12
    return actual == expected


def _command_ledger_path(command: str) -> str | None:
    tokens = shlex.split(command)
    for index, token in enumerate(tokens):
        if token == "--held-out-test-ledger-json" and index + 1 < len(tokens):
            return tokens[index + 1]
    return None


def _promotion_rules(command: str) -> set[str]:
    tokens = shlex.split(command)
    rules: set[str] = set()
    for index, token in enumerate(tokens):
        if token == "--promotion-rule" and index + 1 < len(tokens):
            rules.add(tokens[index + 1])
    return rules


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


def _load_artifact_members(
    evidence: Mapping[str, Any],
    root: Path,
    errors: list[str],
) -> dict[str, bytes]:
    path = _artifact_path(evidence, root, errors)
    if path is None:
        return {}

    artifact = _as_mapping(evidence.get("artifact"), "artifact", errors)
    expected = set(str(item) for item in artifact.get("contents", []) if item)
    members: dict[str, bytes] = {}
    with tarfile.open(path, mode="r:gz") as archive:
        names = archive.getnames()
        if any(Path(member).name.startswith("._") for member in names):
            errors.append("artifact must not contain AppleDouble members")
        missing = sorted(expected - set(names))
        if missing:
            errors.append(f"artifact.contents missing members: {missing}")
        for name in names:
            extracted = archive.extractfile(name)
            if extracted is not None:
                members[name] = extracted.read()

    for index, file_value in enumerate(
        _as_list(artifact.get("files", []), "artifact.files", errors)
    ):
        file_record = _as_mapping(file_value, f"artifact.files[{index}]", errors)
        member = str(file_record.get("member", ""))
        data = members.get(member)
        if data is None:
            errors.append(f"artifact.files[{index}].member missing from artifact")
            continue
        if file_record.get("sha256") != _sha256_bytes(data):
            errors.append(f"artifact.files[{index}].sha256 must match artifact member bytes")
        if isinstance(file_record.get("bytes"), int) and len(data) != file_record.get("bytes"):
            errors.append(f"artifact.files[{index}].bytes must match artifact member size")
    return members


def _json_member(members: Mapping[str, bytes], member: str, errors: list[str]) -> Mapping[str, Any]:
    data = members.get(member)
    if data is None:
        errors.append(f"artifact member missing: {member}")
        return {}
    try:
        value = json.loads(data.decode("utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"artifact member is not valid JSON: {member}: {exc}")
        return {}
    return _as_mapping(value, member, errors)


def _validate_command(evidence: Mapping[str, Any], errors: list[str]) -> None:
    command = str(evidence.get("command", ""))
    normalized = " ".join(command.split())
    required_tokens = (
        "--skip-training",
        "--stage operator_decoded",
        "--extra-eval-split test",
        "--held-out-test-ledger-json",
        "evaluation.decoded_data_conditioned_roll_shift_estimator=",
        "context_shift",
        "context_transitions",
        '"mode":"roll_persistence"',
        "evaluation.report_all_horizon_metrics=true",
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

    missing_rules = sorted(REQUIRED_PROMOTION_RULES - _promotion_rules(command))
    if missing_rules:
        errors.append(f"command must include phase-gate promotion rules: {missing_rules}")

    policy = _as_mapping(evidence.get("held_out_test_policy"), "held_out_test_policy", errors)
    try:
        computed_key = _command_measurement_key(command)
    except Exception as exc:
        errors.append(str(exc))
        computed_key = None
    if computed_key and policy.get("measurement_key") != computed_key:
        errors.append("held_out_test_policy.measurement_key does not match command-derived key")

    ledger_path = _command_ledger_path(command)
    if ledger_path is None:
        errors.append("command must include held-out ledger path")
    elif policy.get("ledger_json") != ledger_path:
        errors.append("held_out_test_policy.ledger_json must match command ledger path")


def _validate_contract_and_validation_evidence(
    evidence: Mapping[str, Any],
    *,
    root: Path,
    errors: list[str],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    contract_path = root / str(evidence.get("pretest_contract_json", ""))
    validation_path = root / str(evidence.get("validation_evidence_json", ""))
    contract: Mapping[str, Any] = {}
    validation_evidence: Mapping[str, Any] = {}

    if not contract_path.exists():
        errors.append(f"pretest_contract_json does not exist: {contract_path}")
    else:
        contract = load_json(contract_path)
        if evidence.get("pretest_contract_sha256") != _sha256(contract_path):
            errors.append("pretest_contract_sha256 must match pretest contract bytes")
        if contract.get("measurement_type") != EXPECTED_PRETEST_CONTRACT_TYPE:
            errors.append(
                f"pretest contract measurement_type must be {EXPECTED_PRETEST_CONTRACT_TYPE}"
            )
        if contract.get("held_out_test_used") is not False:
            errors.append("pretest contract held_out_test_used must remain false")
        if contract.get("test_split_accessed") is not False:
            errors.append("pretest contract test_split_accessed must remain false")
        intended = _as_mapping(
            contract.get("intended_held_out"), "pretest.intended_held_out", errors
        )
        if intended.get("command") != evidence.get("command"):
            errors.append("command must match pretest intended_held_out.command")
        if intended.get("measurement_key") != evidence.get("held_out_test_policy", {}).get(
            "measurement_key"
        ):
            errors.append("held_out_test_policy.measurement_key must match pretest contract")
        if intended.get("command_status") != "pre_registered_not_run":
            errors.append("pretest intended command_status must remain pre_registered_not_run")
        decision = _as_mapping(
            contract.get("protocol_decision"), "pretest.protocol_decision", errors
        )
        required_decision = {
            "status": "accepted_for_one_held_out_confirmation",
            "data_conditioned_context_shift_disclosed": True,
            "teacher_forced_previous_frame_dependency_disclosed": True,
            "not_autonomous_rollout_claim": True,
            "requires_claim_language_update": True,
            "external_paper_reproduction": False,
        }
        for key, expected in required_decision.items():
            if decision.get(key) != expected:
                errors.append(f"pretest.protocol_decision.{key} must be {expected}")

    if not validation_path.exists():
        errors.append(f"validation_evidence_json does not exist: {validation_path}")
    else:
        validation_evidence = load_json(validation_path)
        if evidence.get("validation_evidence_sha256") != _sha256(validation_path):
            errors.append("validation_evidence_sha256 must match validation evidence bytes")
        errors.extend(
            f"validation_evidence: {error}"
            for error in validate_validation_evidence(validation_evidence, root=root)
        )
        if validation_evidence.get("phase_gate", {}).get("passed") is not True:
            errors.append("validation evidence phase gate must be passed")
        if (
            validation_evidence.get("decision", {}).get("held_out_pretest_contract_allowed")
            is not True
        ):
            errors.append("validation evidence must allow pretest contract creation")

    return contract, validation_evidence


def _validate_metrics(
    *,
    evidence_metrics: Mapping[str, Any],
    summary_metrics: Mapping[str, Any],
    label: str,
    errors: list[str],
) -> None:
    for metric in KEY_METRICS:
        if metric not in evidence_metrics:
            errors.append(f"{label}.{metric} is required")
            continue
        if not _close_enough(evidence_metrics.get(metric), summary_metrics.get(metric)):
            errors.append(
                f"{label}.{metric} must match summary{'_test' if label == 'test_metrics' else ''} metrics"
            )


def _validate_ledger(
    *,
    evidence: Mapping[str, Any],
    ledger: Mapping[str, Any],
    summary_test: Mapping[str, Any],
    errors: list[str],
) -> None:
    policy = _as_mapping(evidence.get("held_out_test_policy"), "held_out_test_policy", errors)
    if policy.get("exactly_one_test_after_validation") is not True:
        errors.append("held_out_test_policy.exactly_one_test_after_validation must be true")
    if policy.get("allow_repeat_held_out_test") is not False:
        errors.append("held_out_test_policy.allow_repeat_held_out_test must be false")

    measurements = _as_list(ledger.get("measurements"), "ledger.measurements", errors)
    key = policy.get("measurement_key")
    matches = [
        item
        for item in measurements
        if isinstance(item, Mapping) and item.get("measurement_key") == key
    ]
    if len(matches) != 1:
        errors.append("ledger must contain exactly one measurement for measurement_key")
        return
    measurement = matches[0]
    held_out = _as_mapping(evidence.get("held_out_measurement"), "held_out_measurement", errors)
    expected_values = {
        "measurement_key": key,
        "run_name": evidence.get("run_name"),
        "summary_json": evidence.get("summary_test_json"),
        "test_metric_name": "decoded_rollout_nrmse",
        "test_metric_value": summary_test.get("metrics", {}).get("decoded_rollout_nrmse"),
        "test_split": "test",
        "validation_metric_name": "decoded_rollout_nrmse",
        "validation_metric_value": evidence.get("validation_metrics", {}).get(
            "decoded_rollout_nrmse"
        ),
    }
    for key_name, expected in expected_values.items():
        if not _close_enough(measurement.get(key_name), expected):
            errors.append(f"ledger measurement {key_name} must match held-out summary")
        if key_name in held_out and not _close_enough(held_out.get(key_name), expected):
            errors.append(f"held_out_measurement.{key_name} must match held-out summary")


def _validate_comparisons(evidence: Mapping[str, Any], errors: list[str]) -> None:
    test_metrics = _as_mapping(evidence.get("test_metrics"), "test_metrics", errors)
    test_metric = test_metrics.get("decoded_rollout_nrmse")
    test_advection = test_metrics.get("task_advection1d_decoded_rollout_nrmse")
    ct8 = _as_mapping(
        evidence.get("comparison_to_current_ct8_claim"),
        "comparison_to_current_ct8_claim",
        errors,
    )
    ct1 = _as_mapping(
        evidence.get("comparison_to_ct1_scoped_variant"),
        "comparison_to_ct1_scoped_variant",
        errors,
    )
    if ct8.get("candidate_beats_current_ct8_primary") is not True:
        errors.append(
            "comparison_to_current_ct8_claim.candidate_beats_current_ct8_primary must be true"
        )
    if ct8.get("same_exact_inference_contract_as_primary") is not False:
        errors.append(
            "comparison_to_current_ct8_claim.same_exact_inference_contract_as_primary must be false"
        )
    if ct1.get("candidate_beats_ct1_scoped_variant") is not True:
        errors.append(
            "comparison_to_ct1_scoped_variant.candidate_beats_ct1_scoped_variant must be true"
        )
    if isinstance(test_metric, (int, float)):
        expected_ct8 = CURRENT_CT8_METRIC - float(test_metric)
        expected_ct1 = CURRENT_CT1_METRIC - float(test_metric)
        if not _close_enough(ct8.get("absolute_overall_improvement"), expected_ct8):
            errors.append("comparison_to_current_ct8_claim.absolute_overall_improvement mismatch")
        if not _close_enough(ct1.get("absolute_overall_improvement"), expected_ct1):
            errors.append("comparison_to_ct1_scoped_variant.absolute_overall_improvement mismatch")
    if isinstance(test_advection, (int, float)):
        expected_ct8 = CURRENT_CT8_ADVECTION_METRIC - float(test_advection)
        expected_ct1 = CURRENT_CT1_ADVECTION_METRIC - float(test_advection)
        if not _close_enough(ct8.get("absolute_advection_improvement"), expected_ct8):
            errors.append("comparison_to_current_ct8_claim.absolute_advection_improvement mismatch")
        if not _close_enough(ct1.get("absolute_advection_improvement"), expected_ct1):
            errors.append(
                "comparison_to_ct1_scoped_variant.absolute_advection_improvement mismatch"
            )


def _validate_scoped_language(evidence: Mapping[str, Any], errors: list[str]) -> None:
    scoped = _as_mapping(evidence.get("scoped_claim_language"), "scoped_claim_language", errors)
    if scoped.get("claim_contract_label") != "light-v1 data-conditioned context-phase UPS variant":
        errors.append("scoped_claim_language.claim_contract_label mismatch")
    if scoped.get("same_exact_inference_contract_as_primary") is not False:
        errors.append(
            "scoped_claim_language.same_exact_inference_contract_as_primary must be false"
        )
    if scoped.get("not_autonomous_rollout_claim") is not True:
        errors.append("scoped_claim_language.not_autonomous_rollout_claim must be true")
    if scoped.get("published_numbers_directly_comparable") is not False:
        errors.append("scoped_claim_language.published_numbers_directly_comparable must be false")
    if scoped.get("external_paper_reproduction") is not False:
        errors.append("scoped_claim_language.external_paper_reproduction must be false")
    if scoped.get("data_conditioned_context_shift_disclosed") is not True:
        errors.append("scoped_claim_language.data_conditioned_context_shift_disclosed must be true")
    if scoped.get("teacher_forced_previous_frame_dependency_disclosed") is not True:
        errors.append(
            "scoped_claim_language.teacher_forced_previous_frame_dependency_disclosed must be true"
        )

    decision = _as_mapping(evidence.get("decision"), "decision", errors)
    if decision.get("status") != "scoped_data_conditioned_held_out_complete":
        errors.append("decision.status must be scoped_data_conditioned_held_out_complete")
    if decision.get("do_not_repeat_held_out_test") is not True:
        errors.append("decision.do_not_repeat_held_out_test must be true")


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
    if evidence.get("validation_split") != "val":
        errors.append("validation_split must be val")
    if evidence.get("checkpoint_preference_stage") != "operator_decoded":
        errors.append("checkpoint_preference_stage must be operator_decoded")

    _validate_command(evidence, errors)
    _, validation_evidence = _validate_contract_and_validation_evidence(
        evidence,
        root=repo_root,
        errors=errors,
    )
    members = _load_artifact_members(evidence, repo_root, errors)
    summary_member = f"{evidence.get('run_name')}/summary.json"
    summary_test_member = f"{evidence.get('run_name')}/summary_test.json"
    summary = _json_member(members, summary_member, errors)
    summary_test = _json_member(members, summary_test_member, errors)
    ledger = _json_member(members, "test_ledger.json", errors)

    if summary.get("run_name") != evidence.get("run_name"):
        errors.append("summary.run_name must match evidence run_name")
    if summary_test.get("run_name") != evidence.get("run_name"):
        errors.append("summary_test.run_name must match evidence run_name")
    if summary_test.get("split") != "test":
        errors.append("summary_test.split must be test")
    if summary.get("extra", {}).get("promotion_passed") is not True:
        errors.append("summary promotion_passed must be true")
    if summary_test.get("extra", {}).get("promotion_passed") is not True:
        errors.append("summary_test promotion_passed must be true")
    if summary.get("extra_evaluations", {}).get("test", {}).get("summary") != evidence.get(
        "summary_test_json"
    ):
        errors.append("summary.extra_evaluations.test.summary must match summary_test_json")

    validation_metrics = _as_mapping(
        evidence.get("validation_metrics"), "validation_metrics", errors
    )
    test_metrics = _as_mapping(evidence.get("test_metrics"), "test_metrics", errors)
    _validate_metrics(
        evidence_metrics=validation_metrics,
        summary_metrics=_as_mapping(summary.get("metrics"), "summary.metrics", errors),
        label="validation_metrics",
        errors=errors,
    )
    _validate_metrics(
        evidence_metrics=test_metrics,
        summary_metrics=_as_mapping(summary_test.get("metrics"), "summary_test.metrics", errors),
        label="test_metrics",
        errors=errors,
    )

    selected_estimator = (
        validation_evidence.get("train_fit_gate", {})
        .get("selected_override", {})
        .get("evaluation.decoded_data_conditioned_roll_shift_estimator")
    )
    if evidence.get("data_conditioned_roll_shift_estimator") != selected_estimator:
        errors.append(
            "data_conditioned_roll_shift_estimator must match validation selected override"
        )

    artifact = _as_mapping(evidence.get("artifact"), "artifact", errors)
    ledger_file = repo_root / str(evidence.get("held_out_test_policy", {}).get("ledger_json", ""))
    if ledger_file.exists() and artifact.get("ledger_sha256") != _sha256(ledger_file):
        errors.append("artifact.ledger_sha256 must match ledger file bytes")
    if artifact.get("ledger_sha256") != _sha256_bytes(members.get("test_ledger.json", b"")):
        errors.append("artifact.ledger_sha256 must match artifact ledger member bytes")

    _validate_ledger(
        evidence=evidence,
        ledger=ledger,
        summary_test=summary_test,
        errors=errors,
    )
    _validate_comparisons(evidence, errors)
    _validate_scoped_language(evidence, errors)
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-json", type=Path, default=Path(DEFAULT_EVIDENCE_JSON))
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)

    evidence = load_json(args.repo_root / args.evidence_json)
    errors = validate_evidence(evidence, root=args.repo_root)
    result = {
        "status": "valid" if not errors else "invalid",
        "evidence_json": str(args.evidence_json),
        "errors": errors,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
