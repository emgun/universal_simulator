from __future__ import annotations

import copy
import json
from pathlib import Path

from scripts.validate_p2_model_side_beta_head_pretest_contract import (
    EXPECTED_LEDGER,
    validate_contract,
)

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    ROOT / "docs/research/2026-06-25-p2-model-side-beta-head-heldout-pretest-contract.json"
)


def _load_current_contract() -> dict:
    with CONTRACT_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_current_model_side_beta_head_pretest_contract_validates():
    contract = _load_current_contract()

    assert validate_contract(contract, repo_root=ROOT) == []


def test_contract_rejects_held_out_result_and_repeat_flag():
    contract = copy.deepcopy(_load_current_contract())
    contract["held_out_test_used"] = True
    contract["intended_held_out"]["command"] += " --allow-repeat-held-out-test"

    errors = validate_contract(contract, repo_root=ROOT)

    assert "held_out_test_used must be false before execution" in errors
    assert "intended_held_out.command must not include --allow-repeat-held-out-test" in errors
    assert "intended_held_out.command must not allow repeat held-out tests" in errors


def test_contract_recomputes_measurement_key():
    contract = copy.deepcopy(_load_current_contract())
    contract["intended_held_out"]["measurement_key"] = "0" * 64

    errors = validate_contract(contract, repo_root=ROOT)

    assert "intended_held_out.measurement_key does not match command-derived key" in errors


def test_contract_rejects_existing_ledger_measurement(tmp_path):
    contract = copy.deepcopy(_load_current_contract())
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


def test_contract_requires_beta_and_scoped_language():
    contract = copy.deepcopy(_load_current_contract())
    contract["protocol_decision"]["same_exact_inference_contract_as_ct8_primary"] = True
    contract["intended_held_out"]["command"] = contract["intended_held_out"]["command"].replace(
        "required_params: [beta]", "required_params: []"
    )

    errors = validate_contract(contract, repo_root=ROOT)

    assert "protocol_decision.same_exact_inference_contract_as_ct8_primary must be False" in errors
    assert "intended_held_out.command must include required_params: [beta]" in errors


def test_contract_requires_separate_pretest_root_builder():
    contract = copy.deepcopy(_load_current_contract())
    contract["pretest_root_requirements"]["requires_new_or_updated_pretest_root_builder"] = False
    contract["pretest_root_requirements"][
        "validation_only_full_task_root_builder_must_not_be_reused_for_test"
    ] = False

    errors = validate_contract(contract, repo_root=ROOT)

    assert (
        "pretest_root_requirements.requires_new_or_updated_pretest_root_builder must be true"
        in errors
    )
    assert (
        "pretest_root_requirements.validation_only_full_task_root_builder_must_not_be_reused_for_test must be true"
        in errors
    )
