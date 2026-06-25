#!/usr/bin/env python
from __future__ import annotations

"""Validate the model-side beta transport-head held-out pretest contract."""

import argparse
import json
import shlex
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.validate_ups_advection_ct1_pretest_contract import (
    _command_ledger_path,
    _command_measurement_key,
    _parse_light_command,
)

DEFAULT_CONTRACT_JSON = (
    "docs/research/2026-06-25-p2-model-side-beta-head-heldout-pretest-contract.json"
)
EXPECTED_MEASUREMENT_TYPE = "p2_model_side_beta_transport_head_heldout_pretest_contract"
EXPECTED_PROTOCOL_MAPPING = "docs/research/2026-06-25-p2-model-side-beta-head-protocol-mapping.md"
EXPECTED_ARTIFACT_SHA256 = "9778317b2942728e0d5e9bd503baadbecd66ee08ef44968e9ed60eb2dff9e905"
EXPECTED_LABEL = "light-v1 model-side beta-parameter transport-head UPS variant"
EXPECTED_RUN_NAME = "ups_light_p2_model_side_beta_transport_head_scoped_pretest"
EXPECTED_OUTPUT_ROOT = "reports/research/sota_loop/model_side_transport_head_heldout_pretest"
EXPECTED_PRETEST_ROOT = f"{EXPECTED_OUTPUT_ROOT}/full_task_beta_pretest_root"
EXPECTED_LEDGER = f"{EXPECTED_OUTPUT_ROOT}/test_ledger.json"
EXPECTED_CHECKPOINT_SOURCE = (
    "reports/research/sota_loop/learned_capacity_gate/"
    "ups_light_local_joint_rollout4_residual_ft_val"
)
GATES = {
    "decoded_rollout_nrmse": 0.35078329353213156,
    "task_advection1d_decoded_rollout_nrmse": 0.4866576789288726,
    "task_advection1d_decoded_h16_nrmse": 0.44444171136384397,
    "task_burgers1d_decoded_rollout_nrmse": 0.15674926288225416,
    "task_darcy2d_decoded_rollout_nrmse": 0.2071060212271272,
}
SELECTED_METRICS = {
    "decoded_rollout_nrmse": 0.11122069837659315,
    "task_advection1d_decoded_rollout_nrmse": 0.0017868115829009724,
    "task_advection1d_decoded_h16_nrmse": 0.001784282965734058,
    "task_burgers1d_decoded_rollout_nrmse": 0.14738121133726986,
    "task_darcy2d_decoded_rollout_nrmse": 0.18897951477635447,
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _as_mapping(value: Any, label: str, errors: list[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(f"{label} must be an object")
        return {}
    return value


def _as_sequence(value: Any, label: str, errors: list[str]) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        errors.append(f"{label} must be a list")
        return []
    return value


def _normalized(command: str) -> str:
    return " ".join(command.split())


def _validate_selected_validation_run(
    selected: Mapping[str, Any],
    errors: list[str],
) -> None:
    expected_values: dict[str, Any] = {
        "name": "ups_light_p2_model_side_beta_transport_head_val",
        "artifact_sha256": EXPECTED_ARTIFACT_SHA256,
        "split": "val",
        "held_out_test_used": False,
        "held_out_test_data_read": False,
    }
    for key, expected in expected_values.items():
        if selected.get(key) != expected:
            errors.append(f"selected_validation_run.{key} must be {expected!r}")
    for key, expected in SELECTED_METRICS.items():
        if selected.get(key) != expected:
            errors.append(f"selected_validation_run.{key} must be {expected!r}")

    head = _as_mapping(
        selected.get("model_side_transport_head"),
        "selected_validation_run.model_side_transport_head",
        errors,
    )
    if head.get("enabled") is not True:
        errors.append("selected_validation_run.model_side_transport_head.enabled must be true")
    if list(head.get("tasks", [])) != ["advection1d"]:
        errors.append(
            "selected_validation_run.model_side_transport_head.tasks must be ['advection1d']"
        )
    if "beta" not in set(str(value) for value in head.get("required_params", [])):
        errors.append(
            "selected_validation_run.model_side_transport_head.required_params must include beta"
        )
    if list(head.get("features", [])) != ["param:beta", "bias"]:
        errors.append(
            "selected_validation_run.model_side_transport_head.features must be "
            "['param:beta', 'bias']"
        )
    if head.get("mode") != "periodic_roll":
        errors.append(
            "selected_validation_run.model_side_transport_head.mode must be periodic_roll"
        )
    if head.get("apply_at") != "decoded_rollout":
        errors.append(
            "selected_validation_run.model_side_transport_head.apply_at must be decoded_rollout"
        )
    if head.get("missing_param_policy") != "skip":
        errors.append(
            "selected_validation_run.model_side_transport_head.missing_param_policy must be skip"
        )

    telemetry = _as_mapping(
        selected.get("model_side_transport_head_metrics"),
        "selected_validation_run.model_side_transport_head_metrics",
        errors,
    )
    if int(telemetry.get("applied_count", 0)) <= 0:
        errors.append("selected_validation_run.model_side_transport_head_metrics.applied_count > 0")
    if int(telemetry.get("skipped_count", 0)) != 0:
        errors.append(
            "selected_validation_run.model_side_transport_head_metrics.skipped_count must be 0"
        )
    if int(telemetry.get("beta_missing_count", 0)) != 0:
        errors.append(
            "selected_validation_run.model_side_transport_head_metrics.beta_missing_count must be 0"
        )


def _validate_protocol_decision(decision: Mapping[str, Any], errors: list[str]) -> None:
    expected_values: dict[str, Any] = {
        "status": "accepted_for_one_scoped_held_out_contract_draft",
        "claim_contract_label": EXPECTED_LABEL,
        "same_exact_inference_contract_as_ct8_primary": False,
        "same_exact_inference_contract_as_ct1_online_transport_context": False,
        "external_paper_reproduction": False,
        "public_claim_evidence_before_heldout_and_mapping": False,
        "requires_beta_provenance_at_inference": True,
        "requires_scoped_claim_language_review_after_success": True,
    }
    for key, expected in expected_values.items():
        if decision.get(key) != expected:
            errors.append(f"protocol_decision.{key} must be {expected!r}")
    rejected = set(
        str(item)
        for item in _as_sequence(
            decision.get("rejected_overclaims"), "protocol_decision.rejected_overclaims", errors
        )
    )
    for required in (
        "same exact inference contract as CT8/shared-context primary",
        "same exact inference contract as CT1 online transport-context",
        "public claim evidence from validation alone",
        "held-out execution without a separate user direction",
    ):
        if required not in rejected:
            errors.append(f"protocol_decision.rejected_overclaims must include {required!r}")


def _validate_pretest_root_requirements(requirements: Mapping[str, Any], errors: list[str]) -> None:
    expected_true = (
        "requires_remote_scratch",
        "local_hydration_forbidden_while_disk_below_sequential_requirement",
        "requires_new_or_updated_pretest_root_builder",
        "validation_only_full_task_root_builder_must_not_be_reused_for_test",
    )
    for key in expected_true:
        if requirements.get(key) is not True:
            errors.append(f"pretest_root_requirements.{key} must be true")
    if requirements.get("root") != EXPECTED_PRETEST_ROOT:
        errors.append(f"pretest_root_requirements.root must be {EXPECTED_PRETEST_ROOT}")
    if list(requirements.get("required_splits_before_run_light_experiment", [])) != [
        "val",
        "test",
    ]:
        errors.append("pretest_root_requirements must require val and test splits")
    provenance = set(
        str(item) for item in requirements.get("required_advection_beta_provenance", [])
    )
    for required in ("source_file_index", "source_paths", "params.beta"):
        if required not in provenance:
            errors.append(
                f"pretest_root_requirements.required_advection_beta_provenance needs {required}"
            )


def _validate_intended_command(
    *,
    intended: Mapping[str, Any],
    repo_root: Path,
    errors: list[str],
) -> None:
    command = str(intended.get("command", ""))
    normalized = _normalized(command)
    tokens = shlex.split(command)
    required_snippets = (
        "--extra-eval-split test",
        "--held-out-test-ledger-json",
        "--stage operator_decoded",
        "--skip-training",
        "--decoded",
        "--decoded-rollout-steps 16",
        f"--checkpoint-source {EXPECTED_CHECKPOINT_SOURCE}",
        f"--name {EXPECTED_RUN_NAME}",
        f"--output-root {EXPECTED_OUTPUT_ROOT}",
        f"--override data.root={EXPECTED_PRETEST_ROOT}",
        "--override data.split=val",
        "--override data.max_samples=32",
        "--override data.param_keys=[beta]",
        "operator.conditioning.sources={task_id: 3, equation_signature: 15}",
        "evaluation.skip_missing_tasks=false",
        "evaluation.decoded_persistence_residual_alpha=0.0",
        "evaluation.report_all_horizon_metrics=true",
        "model_side_transport_head={enabled: true",
        "required_params: [beta]",
        'features: ["param:beta", bias]',
        "mode: periodic_roll",
        "apply_at: decoded_rollout",
        "missing_param_policy: skip",
    )
    for snippet in required_snippets:
        if snippet not in normalized:
            errors.append(f"intended_held_out.command must include {snippet}")
    for forbidden in (
        "--allow-repeat-held-out-test",
        "decoded_context_roll_shift_estimator",
        "decoded_observed_roll_shift_estimator",
        "decoded_prediction_roll_shift_estimator",
        "decoded_data_conditioned_roll_shift_estimator",
    ):
        if forbidden in normalized:
            errors.append(f"intended_held_out.command must not include {forbidden}")

    parsed = _parse_light_command(command)
    rules = set(str(rule) for rule in parsed.promotion_rule)
    for key, gate in GATES.items():
        expected_rule = f"{key}<={gate}"
        if expected_rule not in rules:
            errors.append(f"intended_held_out.command must include promotion rule {expected_rule}")

    if intended.get("command_status") != "pre_registered_not_run":
        errors.append("intended_held_out.command_status must be pre_registered_not_run")
    if intended.get("validation_split") != "val":
        errors.append("intended_held_out.validation_split must be val")
    if intended.get("test_split") != "test":
        errors.append("intended_held_out.test_split must be test")
    if intended.get("ledger_json") != EXPECTED_LEDGER:
        errors.append(f"intended_held_out.ledger_json must be {EXPECTED_LEDGER}")

    try:
        computed_key = _command_measurement_key(command)
    except ValueError as exc:
        errors.append(str(exc))
        computed_key = None
    if computed_key and intended.get("measurement_key") != computed_key:
        errors.append("intended_held_out.measurement_key does not match command-derived key")

    ledger_path = _command_ledger_path(command)
    if ledger_path != intended.get("ledger_json"):
        errors.append("intended_held_out.ledger_json must match command ledger path")
    if "--allow-repeat-held-out-test" in tokens:
        errors.append("intended_held_out.command must not allow repeat held-out tests")
    if not ledger_path or computed_key is None:
        return
    ledger_file = repo_root / ledger_path
    if not ledger_file.exists():
        return
    ledger = load_json(ledger_file)
    measurements = ledger.get("measurements", [])
    if isinstance(measurements, list) and any(
        isinstance(item, Mapping) and item.get("measurement_key") == computed_key
        for item in measurements
    ):
        errors.append("intended_held_out.measurement_key is already present in ledger")


def validate_contract(contract: Mapping[str, Any], *, repo_root: Path) -> list[str]:
    errors: list[str] = []
    if contract.get("measurement_type") != EXPECTED_MEASUREMENT_TYPE:
        errors.append(f"measurement_type must be {EXPECTED_MEASUREMENT_TYPE}")
    if contract.get("status") != "pre_registered_not_run":
        errors.append("status must be pre_registered_not_run")
    if contract.get("held_out_test_used") is not False:
        errors.append("held_out_test_used must be false before execution")
    if contract.get("test_split_accessed") is not False:
        errors.append("test_split_accessed must be false before execution")
    if contract.get("protocol_mapping") != EXPECTED_PROTOCOL_MAPPING:
        errors.append(f"protocol_mapping must be {EXPECTED_PROTOCOL_MAPPING}")

    selected = _as_mapping(
        contract.get("selected_validation_run"), "selected_validation_run", errors
    )
    _validate_selected_validation_run(selected, errors)

    gate = _as_mapping(contract.get("validation_gate"), "validation_gate", errors)
    if gate.get("passed") is not True:
        errors.append("validation_gate.passed must be true")
    thresholds = _as_mapping(gate.get("thresholds"), "validation_gate.thresholds", errors)
    for key, expected in GATES.items():
        if thresholds.get(key) != expected:
            errors.append(f"validation_gate.thresholds.{key} must be {expected!r}")
        if selected.get(key) is not None and float(selected[key]) > expected:
            errors.append(f"selected_validation_run.{key} must clear validation gate")

    decision = _as_mapping(contract.get("protocol_decision"), "protocol_decision", errors)
    _validate_protocol_decision(decision, errors)

    requirements = _as_mapping(
        contract.get("pretest_root_requirements"),
        "pretest_root_requirements",
        errors,
    )
    _validate_pretest_root_requirements(requirements, errors)

    intended = _as_mapping(contract.get("intended_held_out"), "intended_held_out", errors)
    _validate_intended_command(intended=intended, repo_root=repo_root, errors=errors)

    checks = set(
        str(item)
        for item in _as_sequence(
            contract.get("required_pre_run_checks"),
            "required_pre_run_checks",
            errors,
        )
    )
    for required in (
        "python scripts/validate_p2_model_side_beta_head_pretest_contract.py",
        "python -m pytest tests/unit/test_validate_p2_model_side_beta_head_pretest_contract.py -q",
    ):
        if required not in checks:
            errors.append(f"required_pre_run_checks must include {required}")

    rules = _as_mapping(contract.get("interpretation_rules"), "interpretation_rules", errors)
    positive = set(
        str(item)
        for item in _as_sequence(
            rules.get("positive_transfer"),
            "interpretation_rules.positive_transfer",
            errors,
        )
    )
    if "decoded_rollout_nrmse <= 0.4165820594268877" not in positive:
        errors.append("positive_transfer must compare against the CT8 primary held-out metric")
    if "Do not rerun the same measurement key" not in str(rules.get("repeat_policy", "")):
        errors.append("interpretation_rules.repeat_policy must forbid rerun selection")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-json", default=DEFAULT_CONTRACT_JSON)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root)
    contract_path = repo_root / args.contract_json
    contract = load_json(contract_path)
    errors = validate_contract(contract, repo_root=repo_root)
    result = {
        "contract_json": args.contract_json,
        "errors": errors,
        "status": "valid" if not errors else "invalid",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
