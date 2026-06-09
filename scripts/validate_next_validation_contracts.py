#!/usr/bin/env python
from __future__ import annotations

"""Validate next-step validation-only UPS advection experiment contracts."""

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_CONTRACT_JSON = "docs/claim_evidence/ups_advection_next_validation_contracts.json"
EXPECTED_MEASUREMENT_TYPE = "ups_advection_next_validation_contracts"
FORBIDDEN_TEST_TOKENS = (
    "--extra-eval-split test",
    "--eval-split test",
    "--held-out-test-ledger-json",
    "--allow-held-out-test-eval",
    "--allow-repeat-held-out-test",
    "data.split=test",
)
REQUIRED_ABLATIONS = {
    "full_context_shift",
    "weaker_context_shift",
    "no_data_conditioning",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return data


def _as_mapping(value: Any, label: str, errors: list[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(f"{label} must be an object")
        return {}
    return value


def _command_uses_test(command: str) -> bool:
    normalized = " ".join(command.split())
    return any(token in normalized for token in FORBIDDEN_TEST_TOKENS)


def _validate_p2(contract: Mapping[str, Any], errors: list[str]) -> None:
    p2 = _as_mapping(contract.get("p2_learned_warp_sidecar"), "p2_learned_warp_sidecar", errors)
    if p2.get("split") != "val":
        errors.append("p2_learned_warp_sidecar.split must be val")
    if p2.get("data_root") != "data/pdebench":
        errors.append("p2_learned_warp_sidecar.data_root must be data/pdebench")
    if p2.get("max_samples") != 32:
        errors.append("p2_learned_warp_sidecar.max_samples must be 32")
    if p2.get("decoded_rollout_steps") != 16:
        errors.append("p2_learned_warp_sidecar.decoded_rollout_steps must be 16")
    if _command_uses_test(str(p2.get("candidate_command_template", ""))):
        errors.append("p2_learned_warp_sidecar.candidate_command_template must not request test")
    gate = _as_mapping(p2.get("acceptance_gate"), "p2_learned_warp_sidecar.acceptance_gate", errors)
    if gate.get("validation_only") is not True:
        errors.append("p2_learned_warp_sidecar.acceptance_gate.validation_only must be true")
    if gate.get("must_reduce_teacher_forced_context_dependency") is not True:
        errors.append(
            "p2_learned_warp_sidecar.acceptance_gate must require reduced context dependency"
        )
    for key in (
        "decoded_rollout_nrmse_must_be_below",
        "task_advection1d_decoded_rollout_nrmse_must_be_below",
    ):
        if not isinstance(gate.get(key), (int, float)):
            errors.append(f"p2_learned_warp_sidecar.acceptance_gate.{key} must be numeric")


def _validate_ablation(contract: Mapping[str, Any], errors: list[str]) -> None:
    ablation = _as_mapping(
        contract.get("data_conditioned_ablation"),
        "data_conditioned_ablation",
        errors,
    )
    if ablation.get("split") != "val":
        errors.append("data_conditioned_ablation.split must be val")
    variants = ablation.get("required_variants", [])
    if not isinstance(variants, list):
        errors.append("data_conditioned_ablation.required_variants must be a list")
        return
    variant_set = {str(variant) for variant in variants}
    for required in sorted(REQUIRED_ABLATIONS - variant_set):
        errors.append(f"data_conditioned_ablation.required_variants must include {required}")
    outputs = ablation.get("acceptance_outputs", [])
    if not isinstance(outputs, list) or len(outputs) < 3:
        errors.append("data_conditioned_ablation.acceptance_outputs must describe evidence outputs")


def validate_contract(contract: Mapping[str, Any], *, root: Path | None = None) -> list[str]:
    del root
    errors: list[str] = []
    if contract.get("measurement_type") != EXPECTED_MEASUREMENT_TYPE:
        errors.append(f"measurement_type must be {EXPECTED_MEASUREMENT_TYPE}")
    if contract.get("status") != "active":
        errors.append("status must be active")
    policy = _as_mapping(contract.get("held_out_test_policy"), "held_out_test_policy", errors)
    if policy.get("held_out_test_allowed_by_this_contract") is not False:
        errors.append("held_out_test_policy.held_out_test_allowed_by_this_contract must be false")
    if policy.get("future_held_out_requires_separate_pretest_contract") is not True:
        errors.append(
            "held_out_test_policy.future_held_out_requires_separate_pretest_contract must be true"
        )
    if policy.get("must_use_new_measurement_key_for_future_test") is not True:
        errors.append(
            "held_out_test_policy.must_use_new_measurement_key_for_future_test must be true"
        )
    if policy.get("repeat_existing_data_conditioned_key_allowed") is not False:
        errors.append(
            "held_out_test_policy.repeat_existing_data_conditioned_key_allowed must be false"
        )
    _validate_p2(contract, errors)
    _validate_ablation(contract, errors)
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-json", type=Path, default=Path(DEFAULT_CONTRACT_JSON))
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)

    contract = load_json(args.repo_root / args.contract_json)
    errors = validate_contract(contract, root=args.repo_root)
    record = {
        "contract_json": str(args.contract_json),
        "errors": errors,
        "status": "valid" if not errors else "invalid",
    }
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
