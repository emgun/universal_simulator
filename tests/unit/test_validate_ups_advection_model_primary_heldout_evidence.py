from __future__ import annotations

import copy
import json
from pathlib import Path

from scripts.validate_ups_advection_model_primary_heldout_evidence import validate_evidence

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_PATH = (
    ROOT / "docs/claim_evidence/ups_advection_model_primary_heldout_light_v1_evidence.json"
)


def _load_current_evidence() -> dict:
    with EVIDENCE_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_model_primary_heldout_evidence_validates():
    evidence = _load_current_evidence()

    assert validate_evidence(evidence, root=ROOT) == []


def test_model_primary_heldout_evidence_rejects_promotion_claim():
    evidence = _load_current_evidence()
    mutated = copy.deepcopy(evidence)
    mutated["comparison_to_current_ct8_claim"]["candidate_beats_current_ct8_primary"] = True
    mutated["decision"]["status"] = "held_out_complete_promoted"

    errors = validate_evidence(mutated, root=ROOT)

    assert (
        "comparison_to_current_ct8_claim.candidate_beats_current_ct8_primary must be false"
        in errors
    )
    assert "decision.status must be held_out_complete_not_promoted" in errors


def test_model_primary_heldout_evidence_recomputes_measurement_key():
    evidence = _load_current_evidence()
    mutated = copy.deepcopy(evidence)
    mutated["held_out_test_policy"]["measurement_key"] = "0" * 64

    errors = validate_evidence(mutated, root=ROOT)

    assert "held_out_test_policy.measurement_key does not match command-derived key" in errors


def test_model_primary_heldout_evidence_rejects_repeat_or_context_shift():
    evidence = _load_current_evidence()
    mutated = copy.deepcopy(evidence)
    mutated["command"] += (
        " --allow-repeat-held-out-test "
        "--eval-override evaluation.decoded_context_roll_shift_estimator={}"
    )

    errors = validate_evidence(mutated, root=ROOT)

    assert "command must not include --allow-repeat-held-out-test" in errors
    assert "command must not include evaluation.decoded_context_roll_shift_estimator" in errors
