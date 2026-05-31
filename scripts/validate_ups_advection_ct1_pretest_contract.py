#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shlex
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts import run_light_experiment as light_runner
from scripts.validate_ups_advection_context_gate_evidence import (
    load_json,
    validate_evidence,
)
from ups.utils.config_loader import load_config_with_includes

DEFAULT_CONTRACT_JSON = "docs/claim_evidence/ups_advection_ct1_pretest_contract.json"


def _as_mapping(value: Any, label: str, errors: list[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(f"{label} must be an object")
        return {}
    return value


def _parse_light_command(command: str) -> argparse.Namespace:
    tokens = shlex.split(command)
    if len(tokens) >= 2 and tokens[0].endswith("python"):
        tokens = tokens[1:]
    if tokens and tokens[0] == "scripts/run_light_experiment.py":
        tokens = tokens[1:]

    args: dict[str, Any] = {
        "checkpoint_source": None,
        "config": None,
        "decoded": False,
        "decoded_rollout_steps": None,
        "eval_config": None,
        "eval_override": [],
        "extra_eval_split": [],
        "name": None,
        "output_root": None,
        "override": [],
        "promotion_rule": [],
        "skip_training": False,
        "stage": None,
    }
    list_flags = {
        "--eval-override": "eval_override",
        "--extra-eval-split": "extra_eval_split",
        "--override": "override",
        "--promotion-rule": "promotion_rule",
        "--stage": "stage",
    }
    value_flags = {
        "--checkpoint-source": "checkpoint_source",
        "--config": "config",
        "--decoded-rollout-steps": "decoded_rollout_steps",
        "--eval-config": "eval_config",
        "--name": "name",
        "--output-root": "output_root",
    }
    bool_flags = {
        "--decoded": "decoded",
        "--skip-training": "skip_training",
    }

    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in bool_flags:
            args[bool_flags[token]] = True
            index += 1
            continue
        if token in list_flags:
            if index + 1 >= len(tokens):
                raise ValueError(f"missing value for {token}")
            key = list_flags[token]
            if args[key] is None:
                args[key] = []
            args[key].append(tokens[index + 1])
            index += 2
            continue
        if token in value_flags:
            if index + 1 >= len(tokens):
                raise ValueError(f"missing value for {token}")
            value: Any = tokens[index + 1]
            if token == "--decoded-rollout-steps":
                value = int(value)
            args[value_flags[token]] = value
            index += 2
            continue
        if token.startswith("--"):
            index += 2 if index + 1 < len(tokens) and not tokens[index + 1].startswith("--") else 1
            continue
        index += 1

    if args["stage"] == []:
        args["stage"] = None
    return argparse.Namespace(**args)


def _command_measurement_key(command: str) -> str:
    args = _parse_light_command(command)
    missing = [
        key
        for key in ("config", "checkpoint_source", "name", "output_root")
        if not getattr(args, key)
    ]
    if missing:
        raise ValueError(f"intended command is missing required fields: {', '.join(missing)}")
    if "test" not in list(args.extra_eval_split):
        raise ValueError("intended command must include --extra-eval-split test")

    run_dir = Path(args.output_root) / str(args.name)
    train_cfg = light_runner._apply_overrides(
        load_config_with_includes(str(args.config)),
        list(args.override),
    )
    eval_source = load_config_with_includes(str(args.eval_config)) if args.eval_config else {}
    eval_cfg = (
        light_runner._apply_overrides(eval_source, list(args.eval_override))
        if args.eval_config or args.eval_override
        else None
    )
    train_cfg = light_runner._configure_wandb(
        train_cfg,
        enabled=False,
        run_name=str(args.name),
        project="",
        entity="",
        group="",
        tags=[],
        job_type="light-experiment",
    )
    if eval_cfg is not None:
        eval_cfg = light_runner._configure_wandb(
            eval_cfg,
            enabled=False,
            run_name=str(args.name),
            project="",
            entity="",
            group="",
            tags=[],
            job_type="light-experiment",
        )
    train_cfg = light_runner._prepare_runtime_cfg(
        train_cfg,
        checkpoint_dir=run_dir / "checkpoints",
        log_dir=run_dir / "logs",
        disable_wandb=True,
    )
    eval_cfg = light_runner._prepare_eval_cfg(
        train_cfg,
        eval_cfg,
        log_dir=run_dir / "logs",
        disable_wandb=True,
    )
    split_cfg = light_runner._clone_eval_cfg(eval_cfg, split="test")
    return light_runner._held_out_measurement_key(
        args=args,
        split_name="test",
        split_cfg=split_cfg,
    )


def _command_ledger_path(command: str) -> str | None:
    tokens = shlex.split(command)
    for index, token in enumerate(tokens):
        if token == "--held-out-test-ledger-json" and index + 1 < len(tokens):
            return tokens[index + 1]
    return None


def _validate_intended_command(
    *,
    intended: Mapping[str, Any],
    repo_root: Path,
    errors: list[str],
) -> None:
    command = str(intended.get("command", ""))
    required_tokens = (
        "--extra-eval-split test",
        "--held-out-test-ledger-json",
        "--promotion-rule",
        "context_transitions: 1",
        "families: [transport]",
        "mode: roll_persistence",
        "--override data.split=val",
    )
    normalized = " ".join(command.split())
    for token in required_tokens:
        if token not in normalized:
            errors.append(f"intended_held_out.command must include {token}")
    if "--allow-repeat-held-out-test" in normalized:
        errors.append("intended_held_out.command must not allow repeat held-out tests")

    if intended.get("command_status") != "pre_registered_not_run":
        errors.append("intended_held_out.command_status must be pre_registered_not_run")
    if intended.get("test_split") != "test":
        errors.append("intended_held_out.test_split must be test")

    try:
        computed_key = _command_measurement_key(command)
    except ValueError as exc:
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
    if contract.get("measurement_type") != "ups_advection_ct1_pretest_contract":
        errors.append("measurement_type must be ups_advection_ct1_pretest_contract")
    if contract.get("held_out_test_used") is not False:
        errors.append("held_out_test_used must be false before execution")
    if contract.get("test_split_accessed") is not False:
        errors.append("test_split_accessed must be false before execution")

    evidence_json = contract.get("validation_evidence_json")
    if not evidence_json:
        errors.append("validation_evidence_json is required")
    else:
        evidence = load_json(repo_root / str(evidence_json))
        errors.extend(
            f"validation_evidence: {error}"
            for error in validate_evidence(evidence, repo_root=repo_root)
        )
        selected = _as_mapping(
            contract.get("selected_validation_run"), "selected_validation_run", errors
        )
        best = _as_mapping(
            evidence.get("best_validation_candidate"),
            "validation_evidence.best_validation_candidate",
            errors,
        )
        if selected.get("name") != best.get("name"):
            errors.append("selected_validation_run.name must match validation evidence best run")
        if selected.get("decoded_rollout_nrmse") != best.get("decoded_rollout_nrmse"):
            errors.append(
                "selected_validation_run.decoded_rollout_nrmse must match validation evidence"
            )

    decision = _as_mapping(contract.get("protocol_decision"), "protocol_decision", errors)
    if decision.get("status") != "accepted_for_one_held_out_confirmation":
        errors.append("protocol_decision.status must be accepted_for_one_held_out_confirmation")
    if decision.get("teacher_forced_previous_frame_dependency_disclosed") is not True:
        errors.append("protocol_decision must disclose teacher-forced previous-frame dependency")
    if decision.get("not_autonomous_rollout_claim") is not True:
        errors.append("protocol_decision.not_autonomous_rollout_claim must be true")
    if decision.get("requires_claim_language_update") is not True:
        errors.append("protocol_decision.requires_claim_language_update must be true")

    intended = _as_mapping(contract.get("intended_held_out"), "intended_held_out", errors)
    _validate_intended_command(intended=intended, repo_root=repo_root, errors=errors)
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate UPS CT1 pre-test contract")
    parser.add_argument("--contract-json", default=DEFAULT_CONTRACT_JSON)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    contract = load_json(repo_root / args.contract_json)
    errors = validate_contract(contract, repo_root=repo_root)
    record = {
        "contract_json": args.contract_json,
        "errors": errors,
        "status": "valid" if not errors else "invalid",
    }
    print(json.dumps(record, indent=2, sort_keys=True))
    raise SystemExit(0 if not errors else 2)


if __name__ == "__main__":
    main()
