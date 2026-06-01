from __future__ import annotations

import copy
import json
from pathlib import Path

from scripts.validate_ups_advection_model_primary_pretest_contract import validate_contract

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "docs/claim_evidence/ups_advection_model_primary_pretest_contract.json"


def _load_current_contract() -> dict:
    with CONTRACT_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _with_temp_ledger(contract: dict, tmp_path: Path) -> dict:
    ledger = str(tmp_path / "model-primary-test-ledger.json")
    original = contract["intended_held_out"]["ledger_json"]
    contract["intended_held_out"]["ledger_json"] = ledger
    contract["intended_held_out"]["command"] = contract["intended_held_out"]["command"].replace(
        original,
        ledger,
    )
    return contract


def test_current_model_primary_pretest_contract_validates(tmp_path):
    contract = _load_current_contract()
    contract = _with_temp_ledger(contract, tmp_path)

    assert validate_contract(contract, repo_root=ROOT) == []


def test_contract_rejects_held_out_result_or_repeat_flag(tmp_path):
    contract = _load_current_contract()
    mutated = _with_temp_ledger(copy.deepcopy(contract), tmp_path)
    mutated["held_out_test_used"] = True
    mutated["intended_held_out"]["command"] += " --allow-repeat-held-out-test"

    errors = validate_contract(mutated, repo_root=ROOT)

    assert "held_out_test_used must be false before execution" in errors
    assert "intended_held_out.command must not include --allow-repeat-held-out-test" in errors


def test_contract_recomputes_intended_measurement_key(tmp_path):
    contract = _load_current_contract()
    mutated = _with_temp_ledger(copy.deepcopy(contract), tmp_path)
    mutated["intended_held_out"]["measurement_key"] = "0" * 64

    errors = validate_contract(mutated, repo_root=ROOT)

    assert "intended_held_out.measurement_key does not match command-derived key" in errors


def test_contract_rejects_online_context_roll_shift(tmp_path):
    contract = _load_current_contract()
    mutated = _with_temp_ledger(copy.deepcopy(contract), tmp_path)
    mutated["intended_held_out"][
        "command"
    ] += " --eval-override evaluation.decoded_context_roll_shift_estimator={candidate_shifts: [1]}"
    mutated["protocol_decision"]["online_context_roll_shift_disabled"] = False

    errors = validate_contract(mutated, repo_root=ROOT)

    assert (
        "intended_held_out.command must not include "
        "evaluation.decoded_context_roll_shift_estimator"
    ) in errors
    assert "protocol_decision.online_context_roll_shift_disabled must be true" in errors


def test_contract_requires_registered_checkpoint_source(tmp_path):
    contract = _load_current_contract()
    mutated = _with_temp_ledger(copy.deepcopy(contract), tmp_path)
    mutated["intended_held_out"]["command"] = mutated["intended_held_out"]["command"].replace(
        "reports/research/sota_loop/model_advection_stability/"
        "ups_light_advection_weighted_operator_stability_seed23_w15_lr1e4_e8_r8_alpha21",
        "reports/research/sota_loop/model_advection_stability/wrong",
    )

    errors = validate_contract(mutated, repo_root=ROOT)

    assert any(
        error.startswith("intended_held_out.command must include --checkpoint-source")
        for error in errors
    )
