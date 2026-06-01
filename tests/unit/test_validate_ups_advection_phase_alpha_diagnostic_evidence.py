from __future__ import annotations

import copy
import json
from pathlib import Path

from scripts.validate_ups_advection_phase_alpha_diagnostic_evidence import validate_evidence

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_PATH = ROOT / "docs/claim_evidence/ups_advection_phase_alpha_diagnostic_val_evidence.json"


def _load_current_evidence() -> dict:
    with EVIDENCE_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_phase_alpha_diagnostic_evidence_validates():
    evidence = _load_current_evidence()

    assert validate_evidence(evidence, root=ROOT) == []


def test_phase_alpha_diagnostic_evidence_rejects_heldout_flags():
    evidence = _load_current_evidence()
    mutated = copy.deepcopy(evidence)
    mutated["held_out_test_used"] = True
    mutated["held_out_test_data_read"] = True

    errors = validate_evidence(mutated, root=ROOT)

    assert "held_out_test_used must be false" in errors
    assert "held_out_test_data_read must be false" in errors


def test_phase_alpha_diagnostic_evidence_rejects_gate_promotion():
    evidence = _load_current_evidence()
    mutated = copy.deepcopy(evidence)
    mutated["alpha_candidates"][0]["phase_gate_passed"] = True
    mutated["decision"]["held_out_pretest_contract_allowed"] = True

    errors = validate_evidence(mutated, root=ROOT)

    assert (
        "alpha_candidates phase_gate_passed mismatch for ups_light_advection_phase_gate_alpha_0p00"
        in errors
    )
    assert "no alpha candidate should clear the phase gate in this evidence" in errors
    assert "decision.held_out_pretest_contract_allowed must be false" in errors


def test_phase_alpha_diagnostic_evidence_recomputes_metrics():
    evidence = _load_current_evidence()
    mutated = copy.deepcopy(evidence)
    mutated["alpha_candidates"][2]["metrics"]["decoded_rollout_nrmse"] = 0.0

    errors = validate_evidence(mutated, root=ROOT)

    assert any("alpha_candidates metric drift" in error for error in errors)
