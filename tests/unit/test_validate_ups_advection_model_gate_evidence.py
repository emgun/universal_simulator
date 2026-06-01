from __future__ import annotations

import copy
import json
from pathlib import Path

from scripts.validate_ups_advection_model_gate_evidence import validate_evidence

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_PATH = ROOT / "docs/claim_evidence/ups_advection_model_gate_val_evidence.json"


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_ups_advection_model_gate_evidence_matches_artifact():
    evidence = _load(EVIDENCE_PATH)

    assert validate_evidence(evidence, root=ROOT) == []


def test_ups_advection_model_gate_evidence_rejects_held_out_use():
    evidence = _load(EVIDENCE_PATH)
    mutated = copy.deepcopy(evidence)
    mutated["held_out_test_used"] = True

    errors = validate_evidence(mutated, root=ROOT)

    assert "held_out_test_used must be false" in errors


def test_ups_advection_model_gate_evidence_rejects_non_best_selected_candidate():
    evidence = _load(EVIDENCE_PATH)
    mutated = copy.deepcopy(evidence)
    mutated["selected_validation_candidate"]["metric_value"] = 999.0

    errors = validate_evidence(mutated, root=ROOT)

    assert "selected_validation_candidate.metric_value must match alpha_sweep best" in errors
