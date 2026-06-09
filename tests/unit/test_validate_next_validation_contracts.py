from __future__ import annotations

import copy
import json
from pathlib import Path

from scripts.validate_next_validation_contracts import validate_contract

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "docs/claim_evidence/ups_advection_next_validation_contracts.json"


def _load_current_contract() -> dict:
    with CONTRACT_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_next_validation_contracts_validate():
    contract = _load_current_contract()

    assert validate_contract(contract, root=ROOT) == []


def test_next_validation_contracts_reject_held_out_access():
    contract = _load_current_contract()
    mutated = copy.deepcopy(contract)
    mutated["held_out_test_policy"]["held_out_test_allowed_by_this_contract"] = True
    mutated["p2_learned_warp_sidecar"]["candidate_command_template"] += " --extra-eval-split test"

    errors = validate_contract(mutated, root=ROOT)

    assert "held_out_test_policy.held_out_test_allowed_by_this_contract must be false" in errors
    assert "p2_learned_warp_sidecar.candidate_command_template must not request test" in errors


def test_next_validation_contracts_reject_missing_ablation_matrix():
    contract = _load_current_contract()
    mutated = copy.deepcopy(contract)
    mutated["data_conditioned_ablation"]["required_variants"] = ["full_context_shift"]

    errors = validate_contract(mutated, root=ROOT)

    assert "data_conditioned_ablation.required_variants must include no_data_conditioning" in errors
    assert "data_conditioned_ablation.required_variants must include weaker_context_shift" in errors
