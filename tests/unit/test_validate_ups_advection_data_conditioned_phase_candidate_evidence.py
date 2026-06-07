from __future__ import annotations

import copy
import json
from pathlib import Path

from scripts.validate_ups_advection_data_conditioned_phase_candidate_evidence import (
    validate_evidence,
)

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_PATH = (
    ROOT / "docs/claim_evidence/ups_advection_data_conditioned_phase_candidate_val_evidence.json"
)


def _load_current_evidence() -> dict:
    with EVIDENCE_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_current_data_conditioned_phase_candidate_evidence_validates():
    evidence = _load_current_evidence()

    assert validate_evidence(evidence, root=ROOT) == []


def test_validator_rejects_held_out_use_and_test_split_command():
    evidence = _load_current_evidence()
    mutated = copy.deepcopy(evidence)
    mutated["held_out_test_used"] = True
    mutated["command"] += " --extra-eval-split test"

    errors = validate_evidence(mutated, root=ROOT)

    assert "held_out_test_used must be false" in errors
    assert "command must not request held-out test evaluation" in errors


def test_validator_rejects_artifact_hash_drift():
    evidence = _load_current_evidence()
    mutated = copy.deepcopy(evidence)
    mutated["artifact"]["sha256"] = "0" * 64

    errors = validate_evidence(mutated, root=ROOT)

    assert "artifact.sha256 must match file bytes" in errors


def test_validator_rejects_metric_drift():
    evidence = _load_current_evidence()
    mutated = copy.deepcopy(evidence)
    mutated["metrics"]["task_advection1d_decoded_h16_nrmse"] = 1.0

    errors = validate_evidence(mutated, root=ROOT)

    assert "metrics.task_advection1d_decoded_h16_nrmse must match artifact summary" in errors


def test_validator_rejects_selected_override_drift():
    evidence = _load_current_evidence()
    mutated = copy.deepcopy(evidence)
    estimator = mutated["train_fit_gate"]["selected_override"][
        "evaluation.decoded_data_conditioned_roll_shift_estimator"
    ]
    estimator["coefficients"]["context_shift"] = 0.0

    errors = validate_evidence(mutated, root=ROOT)

    assert "train_fit_gate selected override must match artifact summary extra" in errors
