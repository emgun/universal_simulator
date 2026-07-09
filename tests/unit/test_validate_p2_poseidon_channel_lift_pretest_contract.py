from __future__ import annotations

import copy
import json
from pathlib import Path

from scripts.validate_p2_poseidon_channel_lift_pretest_contract import (
    EXPECTED_LEDGER,
    validate_contract,
)

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    ROOT
    / "docs/research/2026-06-23-p2-poseidon-channel-lift-heldout-pretest-contract.json"
)


def _load_current_contract() -> dict:
    with CONTRACT_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_current_poseidon_pretest_contract_validates(tmp_path):
    contract = _load_current_contract()

    assert validate_contract(contract, repo_root=tmp_path) == []


def test_contract_rejects_repeat_flag_and_held_out_result(tmp_path):
    contract = _load_current_contract()
    contract["held_out_test_used"] = True
    contract["intended_held_out"]["command"] += " --allow-repeat-test"

    errors = validate_contract(contract, repo_root=tmp_path)

    assert "held_out_test_used must be false before execution" in errors
    assert "intended_held_out.command allow_repeat_test must be False" in errors
    assert "intended_held_out.command must not include --allow-repeat-test" in errors


def test_contract_recomputes_poseidon_measurement_key(tmp_path):
    contract = _load_current_contract()
    contract["intended_held_out"]["measurement_key"] = "0" * 64

    errors = validate_contract(contract, repo_root=tmp_path)

    assert "intended_held_out.measurement_key does not match command-derived key" in errors


def test_contract_rejects_existing_ledger_measurement(tmp_path):
    contract = _load_current_contract()
    ledger_path = tmp_path / EXPECTED_LEDGER
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(
        json.dumps(
            {
                "measurements": [
                    {"measurement_key": contract["intended_held_out"]["measurement_key"]}
                ]
            }
        ),
        encoding="utf-8",
    )

    errors = validate_contract(contract, repo_root=tmp_path)

    assert "intended_held_out.measurement_key is already present in ledger" in errors


def test_contract_requires_test_split_and_claim_boundary(tmp_path):
    contract = copy.deepcopy(_load_current_contract())
    contract["intended_held_out"]["command"] = contract["intended_held_out"][
        "command"
    ].replace(
        "--eval-split test",
        "--eval-split val",
    )
    contract["claim_language_boundaries"]["external_paper_reproduction"] = True

    errors = validate_contract(contract, repo_root=tmp_path)

    assert "intended_held_out.command eval_split must be 'test'" in errors
    assert "claim boundary must reject external paper reproduction" in errors
