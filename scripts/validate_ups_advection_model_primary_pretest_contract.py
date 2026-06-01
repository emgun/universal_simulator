#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.validate_ups_advection_ct1_pretest_contract import (
    _command_ledger_path,
    _command_measurement_key,
)
from scripts.validate_ups_advection_model_gate_evidence import (
    load_json,
    validate_evidence,
)

DEFAULT_CONTRACT_JSON = "docs/claim_evidence/ups_advection_model_primary_pretest_contract.json"
EXPECTED_MEASUREMENT_TYPE = "ups_advection_model_primary_pretest_contract"
EXPECTED_VALIDATION_EVIDENCE = "docs/claim_evidence/ups_advection_model_stability_val_evidence.json"
EXPECTED_CHECKPOINT_SOURCE = (
    "reports/research/sota_loop/model_advection_stability/"
    "ups_light_advection_weighted_operator_stability_seed23_w15_lr1e4_e8_r8_alpha21"
)


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


def _normalized(command: str) -> str:
    return " ".join(command.split())


def _command_tokens(command: str) -> list[str]:
    return shlex.split(command)


def _validate_command_tokens(command: str, errors: list[str]) -> None:
    normalized = _normalized(command)
    tokens = _command_tokens(command)
    required_tokens = (
        "--skip-training",
        f"--checkpoint-source {EXPECTED_CHECKPOINT_SOURCE}",
        "--extra-eval-split test",
        "--held-out-test-ledger-json",
        "--override data.split=val",
        "--eval-override evaluation.decoded_persistence_residual_alpha=0.0",
        "evaluation.decoded_persistence_residual_alpha_by_family={transport: 0.21}",
    )
    for token in required_tokens:
        if token not in normalized:
            errors.append(f"intended_held_out.command must include {token}")
    if (
        "--promotion-rule" not in tokens
        or "decoded_rollout_nrmse<=0.35078329353213156" not in tokens
    ):
        errors.append(
            "intended_held_out.command must include "
            "--promotion-rule decoded_rollout_nrmse<=0.35078329353213156"
        )

    forbidden_tokens = (
        "--allow-repeat-held-out-test",
        "evaluation.decoded_context_roll_shift_estimator",
        "evaluation.decoded_observed_roll_shift_estimator",
        "evaluation.decoded_prediction_roll_shift_estimator",
    )
    for token in forbidden_tokens:
        if token in normalized:
            errors.append(f"intended_held_out.command must not include {token}")

    if "--stage" in tokens:
        errors.append("intended_held_out.command must not retrain stages")


def _validate_intended_command(
    *,
    intended: Mapping[str, Any],
    repo_root: Path,
    errors: list[str],
) -> None:
    command = str(intended.get("command", ""))
    _validate_command_tokens(command, errors)

    if intended.get("command_status") != "pre_registered_not_run":
        errors.append("intended_held_out.command_status must be pre_registered_not_run")
    if intended.get("test_split") != "test":
        errors.append("intended_held_out.test_split must be test")
    if intended.get("validation_gate_threshold") != 0.35078329353213156:
        errors.append("intended_held_out.validation_gate_threshold must match selected validation")

    try:
        computed_key = _command_measurement_key(command)
    except Exception as exc:
        errors.append(str(exc))
        computed_key = None
    if computed_key and intended.get("measurement_key") != computed_key:
        errors.append("intended_held_out.measurement_key does not match command-derived key")

    ledger_path = _command_ledger_path(command)
    if not ledger_path:
        errors.append("intended_held_out.command must include a held-out ledger path")
        return
    if intended.get("ledger_json") != ledger_path:
        errors.append("intended_held_out.ledger_json must match command ledger path")
    ledger_file = repo_root / ledger_path
    if not ledger_file.exists() or computed_key is None:
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
    if contract.get("held_out_test_used") is not False:
        errors.append("held_out_test_used must be false before execution")
    if contract.get("test_split_accessed") is not False:
        errors.append("test_split_accessed must be false before execution")

    evidence_json = str(contract.get("validation_evidence_json", ""))
    if evidence_json != EXPECTED_VALIDATION_EVIDENCE:
        errors.append(f"validation_evidence_json must be {EXPECTED_VALIDATION_EVIDENCE}")
    evidence_path = repo_root / evidence_json
    if not evidence_path.exists():
        errors.append("validation_evidence_json does not exist")
    else:
        if contract.get("validation_evidence_sha256") != _sha256(evidence_path):
            errors.append("validation_evidence_sha256 must match validation evidence bytes")
        evidence = load_json(evidence_path)
        errors.extend(
            f"validation_evidence: {error}" for error in validate_evidence(evidence, root=repo_root)
        )
        selected = _as_mapping(
            contract.get("selected_validation_run"), "selected_validation_run", errors
        )
        evidence_selected = _as_mapping(
            evidence.get("selected_validation_candidate"),
            "validation_evidence.selected_validation_candidate",
            errors,
        )
        if selected.get("name") != evidence_selected.get("run_name"):
            errors.append("selected_validation_run.name must match validation evidence best run")
        if selected.get("decoded_rollout_nrmse") != evidence_selected.get("metric_value"):
            errors.append(
                "selected_validation_run.decoded_rollout_nrmse must match validation evidence"
            )
        evidence_metrics = _as_mapping(
            evidence_selected.get("metrics"),
            "validation_evidence.selected_validation_candidate.metrics",
            errors,
        )
        if selected.get("task_advection1d_decoded_rollout_nrmse") != evidence_metrics.get(
            "task_advection1d_decoded_rollout_nrmse"
        ):
            errors.append(
                "selected_validation_run.task_advection1d_decoded_rollout_nrmse must match "
                "validation evidence"
            )

    decision = _as_mapping(contract.get("protocol_decision"), "protocol_decision", errors)
    if decision.get("status") != "accepted_for_one_primary_contract_held_out_confirmation":
        errors.append(
            "protocol_decision.status must be accepted_for_one_primary_contract_held_out_confirmation"
        )
    if decision.get("not_online_context_variant") is not True:
        errors.append("protocol_decision.not_online_context_variant must be true")
    if decision.get("online_context_roll_shift_disabled") is not True:
        errors.append("protocol_decision.online_context_roll_shift_disabled must be true")
    if decision.get("requires_claim_audit_update") is not True:
        errors.append("protocol_decision.requires_claim_audit_update must be true")
    if decision.get("external_paper_reproduction") is not False:
        errors.append("protocol_decision.external_paper_reproduction must be false")

    intended = _as_mapping(contract.get("intended_held_out"), "intended_held_out", errors)
    _validate_intended_command(intended=intended, repo_root=repo_root, errors=errors)
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-json", type=Path, default=Path(DEFAULT_CONTRACT_JSON))
    args = parser.parse_args(argv)

    contract = load_json(args.contract_json)
    errors = validate_contract(contract, repo_root=Path.cwd())
    result = {
        "status": "valid" if not errors else "invalid",
        "contract_json": str(args.contract_json),
        "errors": errors,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
