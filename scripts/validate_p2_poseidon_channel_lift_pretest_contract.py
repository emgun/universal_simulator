#!/usr/bin/env python
from __future__ import annotations

"""Validate the Poseidon channel-lift held-out pre-test contract."""

import argparse
import json
import shlex
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts import run_external_poseidon_scot_finetune as poseidon_runner

DEFAULT_CONTRACT_JSON = (
    "docs/research/2026-06-23-p2-poseidon-channel-lift-heldout-pretest-contract.json"
)
EXPECTED_MEASUREMENT_TYPE = "p2_poseidon_channel_lift_heldout_pretest_contract"
EXPECTED_VALIDATION_SUMMARY = (
    "reports/research/sota_loop/external_baselines/"
    "poseidon_scot_channel_lift_val_light_v1_e30_lr1e2_roll4/summary.json"
)
EXPECTED_TEST_SUMMARY = (
    "reports/research/sota_loop/external_baselines/"
    "poseidon_scot_channel_lift_test_light_v1_e30_lr1e2_roll4/summary.json"
)
EXPECTED_LEDGER = (
    "reports/research/sota_loop/external_baselines/"
    "poseidon_scot_channel_lift_test_light_v1_e30_lr1e2_roll4/test_ledger.json"
)
EXPECTED_TASKS = ["advection1d", "burgers1d", "darcy2d"]
EXPECTED_CHECKPOINT_SHA256 = (
    "e97428c93a16cbb52a41bc4794eb71be3aed436fb9cc547d9eeebb20f3940fb2"
)
EXPECTED_POSEIDON_COMMIT = "b8fa28f59bd7f7673323f28d11a12c6f3a215c61"
G2A_THRESHOLD = 0.363424243629033
UPS_PRIMARY_LIGHT_V1_HELDOUT = 0.4165820594268877


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


def _parse_poseidon_command(command: str) -> argparse.Namespace:
    tokens = shlex.split(command)
    if tokens and tokens[0].endswith("python"):
        tokens = tokens[1:]
    if tokens and tokens[0] == "scripts/run_external_poseidon_scot_finetune.py":
        tokens = tokens[1:]
    return poseidon_runner.build_parser().parse_args(tokens)


def _command_measurement_key(command: str) -> str:
    parsed = _parse_poseidon_command(command)
    tasks = list(parsed.tasks or parsed.task)
    return poseidon_runner._poseidon_test_measurement_key(args=parsed, tasks=tasks)


def _validate_selected_validation_run(
    selected: Mapping[str, Any],
    errors: list[str],
) -> None:
    expected_values: dict[str, Any] = {
        "name": "poseidon_scot_channel_lift_val_light_v1_e30_lr1e2_roll4",
        "summary_json": EXPECTED_VALIDATION_SUMMARY,
        "split": "val",
        "train_split": "train",
        "decoded_rollout_nrmse": 0.35782889238675264,
        "task_advection1d_decoded_rollout_nrmse": 0.4937043430599529,
        "task_burgers1d_decoded_rollout_nrmse": 0.15674926288225416,
        "task_darcy2d_decoded_rollout_nrmse": 0.2071060212271272,
        "held_out_test_used": False,
        "adapter_mode": "channel_lift",
        "trainable_parameter_count": 13,
        "poseidon_commit": EXPECTED_POSEIDON_COMMIT,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "embedding_recovery_replaced": False,
        "pretrained_embedding_recovery_intact": True,
    }
    for key, expected in expected_values.items():
        if selected.get(key) != expected:
            errors.append(f"selected_validation_run.{key} must be {expected!r}")
    observed = selected.get("decoded_rollout_nrmse")
    if not isinstance(observed, (int, float)) or float(observed) > G2A_THRESHOLD:
        errors.append("selected_validation_run.decoded_rollout_nrmse must clear G2a")


def _validate_intended_held_out(
    *,
    intended: Mapping[str, Any],
    repo_root: Path,
    errors: list[str],
) -> None:
    command = str(intended.get("command", ""))
    if not command:
        errors.append("intended_held_out.command is required")
        return
    try:
        parsed = _parse_poseidon_command(command)
    except SystemExit:
        errors.append("intended_held_out.command must parse with Poseidon finetune parser")
        return

    expected_values: dict[str, Any] = {
        "config": "configs/train_multitask_heterogeneous_light_best.yaml",
        "name": "poseidon_scot_channel_lift_test_light_v1_e30_lr1e2_roll4",
        "output_root": "reports/research/sota_loop/external_baselines",
        "data_root": "data/pdebench",
        "train_split": "train",
        "eval_split": "test",
        "max_train_samples": 32,
        "max_eval_samples": 32,
        "rollout_steps": 16,
        "poseidon_model_size": "T",
        "checkpoint_file": "model.safetensors",
        "expected_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "poseidon_repo": "/tmp/poseidon-official",
        "image_size": 0,
        "time_value": 1.0,
        "device": "cuda",
        "adapter_mode": "channel_lift",
        "rollout_loss_steps": 4,
        "rollout_loss_weight": 1.0,
        "epochs": 30,
        "learning_rate": 0.01,
        "weight_decay": 0.0001,
        "batch_size": 32,
        "grad_clip_norm": 1.0,
        "seed": 17,
        "held_out_ledger_json": EXPECTED_LEDGER,
        "allow_held_out_test_eval": True,
        "allow_repeat_test": False,
    }
    for attr, expected in expected_values.items():
        if getattr(parsed, attr) != expected:
            errors.append(f"intended_held_out.command {attr} must be {expected!r}")
    if list(parsed.tasks or parsed.task) != EXPECTED_TASKS:
        errors.append(f"intended_held_out.command tasks must be {EXPECTED_TASKS!r}")
    if "--allow-repeat-test" in shlex.split(command):
        errors.append("intended_held_out.command must not include --allow-repeat-test")

    if intended.get("command_status") != "pre_registered_not_run":
        errors.append("intended_held_out.command_status must be pre_registered_not_run")
    if intended.get("test_split") != "test":
        errors.append("intended_held_out.test_split must be test")
    if intended.get("train_split") != "train":
        errors.append("intended_held_out.train_split must be train")
    if intended.get("ledger_json") != EXPECTED_LEDGER:
        errors.append("intended_held_out.ledger_json must match expected ledger path")
    if intended.get("expected_summary_json") != EXPECTED_TEST_SUMMARY:
        errors.append("intended_held_out.expected_summary_json must match expected summary path")

    try:
        computed_key = _command_measurement_key(command)
    except Exception as exc:  # pragma: no cover - defensive contract diagnostics.
        errors.append(f"could not recompute intended measurement key: {exc}")
        computed_key = None
    if computed_key and intended.get("measurement_key") != computed_key:
        errors.append("intended_held_out.measurement_key does not match command-derived key")

    ledger_file = repo_root / str(intended.get("ledger_json", ""))
    if ledger_file.exists() and computed_key:
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

    user_scope = _as_mapping(contract.get("user_scope"), "user_scope", errors)
    if "no approval needed" not in str(user_scope.get("source", "")):
        errors.append("user_scope.source must record no-approval-needed scope")

    selected = _as_mapping(
        contract.get("selected_validation_run"),
        "selected_validation_run",
        errors,
    )
    _validate_selected_validation_run(selected, errors)

    gate = _as_mapping(contract.get("validation_gate"), "validation_gate", errors)
    if gate.get("gate") != "G2a" or gate.get("passed") is not True:
        errors.append("validation_gate must record passed G2a")
    if gate.get("threshold") != G2A_THRESHOLD or gate.get("observed") != selected.get(
        "decoded_rollout_nrmse"
    ):
        errors.append("validation_gate threshold/observed values are inconsistent")

    intended = _as_mapping(contract.get("intended_held_out"), "intended_held_out", errors)
    _validate_intended_held_out(intended=intended, repo_root=repo_root, errors=errors)

    checks = [
        str(item)
        for item in _as_sequence(
            contract.get("required_pre_run_checks"),
            "required_pre_run_checks",
            errors,
        )
    ]
    for required in (
        "python scripts/validate_p2_poseidon_channel_lift_pretest_contract.py",
        "python -m pytest tests/unit/test_validate_p2_poseidon_channel_lift_pretest_contract.py -q",
    ):
        if required not in checks:
            errors.append(f"required_pre_run_checks must include {required}")

    rules = _as_mapping(contract.get("interpretation_rules"), "interpretation_rules", errors)
    positive = [
        str(item)
        for item in _as_sequence(
            rules.get("positive_transfer"),
            "interpretation_rules.positive_transfer",
            errors,
        )
    ]
    if f"decoded_rollout_nrmse <= {UPS_PRIMARY_LIGHT_V1_HELDOUT}" not in positive:
        errors.append("positive_transfer must compare against the UPS primary held-out metric")
    if "Do not rerun the same measurement key" not in str(rules.get("repeat_policy", "")):
        errors.append("interpretation_rules.repeat_policy must forbid rerun selection")

    claim = _as_mapping(
        contract.get("claim_language_boundaries"),
        "claim_language_boundaries",
        errors,
    )
    if claim.get("not_public_claim_evidence_until_packaged") is not True:
        errors.append("claim boundary must block public claim evidence until packaged")
    if claim.get("external_paper_reproduction") is not False:
        errors.append("claim boundary must reject external paper reproduction")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-json", default=DEFAULT_CONTRACT_JSON)
    args = parser.parse_args(argv)
    contract_path = REPO_ROOT / args.contract_json
    contract = load_json(contract_path)
    errors = validate_contract(contract, repo_root=REPO_ROOT)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"OK: {args.contract_json} is a valid Poseidon channel-lift pretest contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
