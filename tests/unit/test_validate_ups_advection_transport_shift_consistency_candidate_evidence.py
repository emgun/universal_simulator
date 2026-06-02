from __future__ import annotations

import copy
import json
from pathlib import Path

from scripts.validate_ups_advection_transport_shift_consistency_candidate_evidence import (
    validate_evidence,
)

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_PATH = (
    ROOT
    / "docs/claim_evidence/ups_advection_transport_shift_consistency_candidate_val_evidence.json"
)


def _load_current_evidence() -> dict:
    with EVIDENCE_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_transport_shift_consistency_candidate_evidence_validates():
    evidence = _load_current_evidence()

    assert validate_evidence(evidence, root=ROOT) == []


def test_transport_shift_consistency_candidate_evidence_rejects_heldout_flags():
    evidence = _load_current_evidence()
    mutated = copy.deepcopy(evidence)
    mutated["held_out_test_used"] = True
    mutated["held_out_test_data_read"] = True

    errors = validate_evidence(mutated, root=ROOT)

    assert "held_out_test_used must be false" in errors
    assert "held_out_test_data_read must be false" in errors


def test_transport_shift_consistency_candidate_evidence_recomputes_phase_gate():
    evidence = _load_current_evidence()
    mutated = copy.deepcopy(evidence)
    mutated["phase_gate"]["passed"] = True

    errors = validate_evidence(mutated, root=ROOT)

    assert "phase_gate.passed mismatch" in errors


def test_transport_shift_consistency_candidate_evidence_recomputes_metrics():
    evidence = _load_current_evidence()
    mutated = copy.deepcopy(evidence)
    mutated["metrics"]["task_advection1d_decoded_h16_nrmse"] = 0.0

    errors = validate_evidence(mutated, root=ROOT)

    assert "metrics.task_advection1d_decoded_h16_nrmse must match artifact summary" in errors


def test_transport_shift_consistency_candidate_evidence_requires_train_shift():
    evidence = _load_current_evidence()
    mutated = copy.deepcopy(evidence)
    mutated["training"]["transport_shift_consistency_by_task"]["advection1d"] = 40

    errors = validate_evidence(mutated, root=ROOT)

    assert "training.transport_shift_consistency_by_task.advection1d must be 1" in errors
    assert "training shift must match train-fitted selected shift" in errors
