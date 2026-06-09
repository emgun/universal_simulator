from __future__ import annotations

import copy
import json
from pathlib import Path

from scripts.validate_medium_confirmation_evidence import validate_evidence

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_PATH = ROOT / "docs/claim_evidence/medium_v1_confirmation_evidence.json"


def _load_current_evidence() -> dict:
    with EVIDENCE_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_medium_confirmation_evidence_validates():
    evidence = _load_current_evidence()

    assert validate_evidence(evidence, root=ROOT) == []


def test_medium_confirmation_evidence_rejects_small_or_test_tuned_claims():
    evidence = _load_current_evidence()
    mutated = copy.deepcopy(evidence)
    mutated["confirmation_scope"]["test_samples"] = 32
    mutated["selection_policy"]["test_tuned"] = True

    errors = validate_evidence(mutated, root=ROOT)

    assert "confirmation_scope.test_samples must be at least 128" in errors
    assert "selection_policy.test_tuned must be false" in errors


def test_medium_confirmation_evidence_rejects_failed_improvement_gate():
    evidence = _load_current_evidence()
    mutated = copy.deepcopy(evidence)
    mutated["comparison_to_persistence"]["improvement_fraction"] = 0.1

    errors = validate_evidence(mutated, root=ROOT)

    assert "comparison_to_persistence.improvement_fraction must be at least 0.2" in errors


def test_medium_confirmation_evidence_rejects_missing_remote_artifact():
    evidence = _load_current_evidence()
    mutated = copy.deepcopy(evidence)
    mutated["artifact"]["handle"] = "local.tar.gz"

    errors = validate_evidence(mutated, root=ROOT)

    assert "artifact.handle must be a b2:// or repo:docs/claim_evidence/ handle" in errors
