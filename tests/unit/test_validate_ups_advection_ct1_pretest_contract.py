from __future__ import annotations

import copy
import json
from pathlib import Path

from scripts.validate_ups_advection_ct1_pretest_contract import validate_contract

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "docs/claim_evidence/ups_advection_ct1_pretest_contract.json"


def _load_current_contract() -> dict:
    with CONTRACT_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _with_temp_ledger(contract: dict, tmp_path: Path) -> dict:
    ledger = str(tmp_path / "ct1-test-ledger.json")
    original = contract["intended_held_out"]["ledger_json"]
    contract["intended_held_out"]["ledger_json"] = ledger
    contract["intended_held_out"]["command"] = contract["intended_held_out"]["command"].replace(
        original,
        ledger,
    )
    return contract


def test_current_ct1_pretest_contract_validates(tmp_path):
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
    assert "intended_held_out.command must not allow repeat held-out tests" in errors


def test_contract_recomputes_intended_measurement_key(tmp_path):
    contract = _load_current_contract()
    mutated = _with_temp_ledger(copy.deepcopy(contract), tmp_path)
    mutated["intended_held_out"]["measurement_key"] = "0" * 64

    errors = validate_contract(mutated, repo_root=ROOT)

    assert "intended_held_out.measurement_key does not match command-derived key" in errors


def test_contract_requires_protocol_shift_disclosure(tmp_path):
    contract = _load_current_contract()
    mutated = _with_temp_ledger(copy.deepcopy(contract), tmp_path)
    mutated["protocol_decision"]["teacher_forced_previous_frame_dependency_disclosed"] = False

    errors = validate_contract(mutated, repo_root=ROOT)

    assert "protocol_decision must disclose teacher-forced previous-frame dependency" in errors
