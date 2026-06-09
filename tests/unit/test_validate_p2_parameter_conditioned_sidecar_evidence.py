from __future__ import annotations

import copy
import json
from pathlib import Path

from scripts.validate_p2_parameter_conditioned_sidecar_evidence import validate_evidence

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_PATH = (
    ROOT / "docs/claim_evidence/ups_advection_p2_parameter_conditioned_sidecar_val_evidence.json"
)


def _load_current_evidence() -> dict:
    with EVIDENCE_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_current_p2_parameter_conditioned_sidecar_evidence_validates():
    evidence = _load_current_evidence()

    assert validate_evidence(evidence, root=ROOT) == []


def test_p2_sidecar_evidence_rejects_held_out_or_context_dependency():
    evidence = _load_current_evidence()
    mutated = copy.deepcopy(evidence)
    mutated["held_out_test_used"] = True
    mutated["teacher_context_dependency"]["uses_observed_context_transitions"] = True

    errors = validate_evidence(mutated, root=ROOT)

    assert "held_out_test_used must be false" in errors
    assert "teacher_context_dependency.uses_observed_context_transitions must be false" in errors


def test_p2_sidecar_evidence_recomputes_gate_and_report_hash():
    evidence = _load_current_evidence()
    mutated = copy.deepcopy(evidence)
    mutated["candidate_report"]["sha256"] = "bad"
    mutated["acceptance_gate"]["passed"] = False

    errors = validate_evidence(mutated, root=ROOT)

    assert "candidate_report.sha256 must match file bytes" in errors
    assert "acceptance_gate.passed mismatch" in errors


def test_p2_sidecar_evidence_rejects_comparison_report_hash_drift():
    evidence = _load_current_evidence()
    mutated = copy.deepcopy(evidence)
    mutated["comparison_reports"]["dense_context_inferred_data_pdebench_val"]["sha256"] = "bad"

    errors = validate_evidence(mutated, root=ROOT)

    assert (
        "comparison_reports.dense_context_inferred_data_pdebench_val.sha256 must match file bytes"
        in errors
    )
