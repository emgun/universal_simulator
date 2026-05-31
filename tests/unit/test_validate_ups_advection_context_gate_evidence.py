from __future__ import annotations

import copy
import json
from pathlib import Path

from scripts.validate_ups_advection_context_gate_evidence import validate_evidence

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_PATH = ROOT / "docs/claim_evidence/ups_advection_context_delay_val_gate_evidence.json"


def _load_current_evidence() -> dict:
    with EVIDENCE_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_current_ups_advection_context_gate_evidence_validates():
    evidence = _load_current_evidence()

    assert validate_evidence(evidence, repo_root=ROOT) == []


def test_validator_rejects_held_out_use_and_test_split_commands():
    evidence = _load_current_evidence()
    mutated = copy.deepcopy(evidence)
    mutated["held_out_test_used"] = True
    mutated["runs"][0]["command"] += " --extra-eval-split test"

    errors = validate_evidence(mutated, repo_root=ROOT)

    assert "held_out_test_used must be false" in errors
    assert "runs[0].command must not request held-out test evaluation" in errors


def test_validator_rejects_artifact_hash_drift():
    evidence = _load_current_evidence()
    mutated = copy.deepcopy(evidence)
    mutated["artifact"]["sha256"] = "0" * 64

    errors = validate_evidence(mutated, repo_root=ROOT)

    assert "artifact.sha256 does not match artifact bytes" in errors


def test_validator_requires_best_candidate_to_clear_improvement_gate():
    evidence = _load_current_evidence()
    mutated = copy.deepcopy(evidence)
    mutated["best_validation_candidate"][
        "name"
    ] = "ups_light_advection_context_transport_only_ct8_val"
    mutated["best_validation_candidate"]["relative_improvement_vs_current"] = 0.00005421520667444186

    errors = validate_evidence(mutated, repo_root=ROOT)

    assert "best_validation_candidate.name must identify the lowest validation metric run" in errors
    assert (
        "best_validation_candidate.relative_improvement_vs_current must clear the selection rule"
        in errors
    )


def test_validator_requires_context_delay_metrics_to_be_monotonic():
    evidence = _load_current_evidence()
    mutated = copy.deepcopy(evidence)
    for run in mutated["runs"]:
        if run["name"] == "ups_light_advection_context_transport_only_ct4_val":
            run["decoded_rollout_nrmse"] = 0.1

    errors = validate_evidence(mutated, repo_root=ROOT)

    assert (
        "transport-only context-delay metrics must be non-decreasing as delay increases" in errors
    )
