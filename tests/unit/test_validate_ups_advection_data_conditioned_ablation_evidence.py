from __future__ import annotations

import copy
import json
from pathlib import Path

from scripts.validate_ups_advection_data_conditioned_ablation_evidence import (
    validate_evidence,
)

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_PATH = (
    ROOT / "docs/claim_evidence/ups_advection_data_conditioned_ablation_val_evidence.json"
)


def _load_current_evidence() -> dict:
    with EVIDENCE_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_current_data_conditioned_ablation_evidence_validates():
    evidence = _load_current_evidence()

    assert validate_evidence(evidence, root=ROOT) == []


def test_ablation_evidence_rejects_held_out_or_missing_variants():
    evidence = _load_current_evidence()
    mutated = copy.deepcopy(evidence)
    mutated["held_out_test_used"] = True
    del mutated["variants"]["no_data_conditioning"]

    errors = validate_evidence(mutated, root=ROOT)

    assert "held_out_test_used must be false" in errors
    assert "variants must include no_data_conditioning" in errors


def test_ablation_evidence_recomputes_deltas_and_dependency_boundary():
    evidence = _load_current_evidence()
    mutated = copy.deepcopy(evidence)
    mutated["deltas_vs_full_context_shift"]["no_data_conditioning"]["absolute"] = 0.0
    mutated["variants"]["no_data_conditioning"]["feature_names"] = ["context_shift"]

    errors = validate_evidence(mutated, root=ROOT)

    assert "deltas_vs_full_context_shift.no_data_conditioning.absolute mismatch" in errors
    assert "no_data_conditioning must not include context features" in errors


def test_ablation_evidence_rejects_variant_report_hash_drift():
    evidence = _load_current_evidence()
    mutated = copy.deepcopy(evidence)
    mutated["variants"]["full_context_shift"]["report_sha256"] = "bad"

    errors = validate_evidence(mutated, root=ROOT)

    assert "variants.full_context_shift.report_sha256 must match file bytes" in errors
