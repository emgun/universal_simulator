from __future__ import annotations

import copy
import json
from pathlib import Path

from scripts.validate_ups_advection_model_primary_gap_analysis import validate_analysis

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_PATH = ROOT / "docs/claim_evidence/ups_advection_model_primary_gap_analysis.json"


def _load_current_analysis() -> dict:
    with ANALYSIS_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_gap_analysis_validates():
    analysis = _load_current_analysis()

    assert validate_analysis(analysis, root=ROOT) == []


def test_gap_analysis_rejects_new_heldout_rerun():
    analysis = _load_current_analysis()
    mutated = copy.deepcopy(analysis)
    mutated["new_held_out_test_command_executed"] = True
    mutated["held_out_test_data_reaccessed"] = True

    errors = validate_analysis(mutated, root=ROOT)

    assert "new_held_out_test_command_executed must be false" in errors
    assert "held_out_test_data_reaccessed must be false" in errors


def test_gap_analysis_rejects_promotion_claim():
    analysis = _load_current_analysis()
    mutated = copy.deepcopy(analysis)
    mutated["candidate_vs_current_ct8_primary"]["candidate_beats_current_ct8_primary"] = True
    mutated["decision"]["candidate_promoted"] = True

    errors = validate_analysis(mutated, root=ROOT)

    assert (
        "candidate_vs_current_ct8_primary.candidate_beats_current_ct8_primary must be false"
        in errors
    )
    assert "decision.candidate_promoted must be false" in errors


def test_gap_analysis_recomputes_metric_deltas():
    analysis = _load_current_analysis()
    mutated = copy.deepcopy(analysis)
    mutated["candidate_vs_current_ct8_primary"]["test_delta_candidate_minus_ct8"][
        "decoded_rollout_nrmse"
    ]["absolute"] = 0.0

    errors = validate_analysis(mutated, root=ROOT)

    assert any(
        "$.candidate_vs_current_ct8_primary.test_delta_candidate_minus_ct8."
        "decoded_rollout_nrmse.absolute mismatch" in error
        for error in errors
    )
