from __future__ import annotations

import copy
import json
from pathlib import Path

from scripts.validate_p2_parameter_decoded_sidecar_evidence import validate_evidence

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_PATH = (
    ROOT / "docs/claim_evidence/ups_advection_p2_parameter_decoded_sidecar_val_evidence.json"
)


def _load_current_evidence() -> dict:
    with EVIDENCE_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_current_p2_parameter_decoded_sidecar_evidence_validates():
    evidence = _load_current_evidence()

    assert validate_evidence(evidence, root=ROOT) == []


def test_p2_parameter_decoded_sidecar_evidence_rejects_heldout_or_test_command():
    evidence = _load_current_evidence()
    mutated = copy.deepcopy(evidence)
    mutated["held_out_test_used"] = True
    mutated["command_args"].extend(["--extra-eval-split", "test"])

    errors = validate_evidence(mutated, root=ROOT)

    assert "held_out_test_used must be false" in errors
    assert "command_args must not include --extra-eval-split" in errors


def test_p2_parameter_decoded_sidecar_evidence_rejects_summary_hash_and_metric_drift():
    evidence = _load_current_evidence()
    mutated = copy.deepcopy(evidence)
    mutated["summary_report"]["sha256"] = "bad"
    mutated["metrics"]["decoded_rollout_nrmse"] = 0.0

    errors = validate_evidence(mutated, root=ROOT)

    assert "summary_report.sha256 must match file bytes" in errors
    assert "metrics.decoded_rollout_nrmse must match summary" in errors


def test_p2_parameter_decoded_sidecar_evidence_rejects_estimator_drift():
    evidence = _load_current_evidence()
    mutated = copy.deepcopy(evidence)
    mutated["estimator"]["coefficients"]["param:beta"] = 0.0

    errors = validate_evidence(mutated, root=ROOT)

    assert "estimator.coefficients must match locked train-fit beta coefficients" in errors
    assert "summary estimator must match evidence estimator" in errors


def test_p2_parameter_decoded_sidecar_evidence_rejects_scope_overclaim():
    evidence = _load_current_evidence()
    mutated = copy.deepcopy(evidence)
    mutated["decoded_scope"]["full_multitask_light_v1_candidate"] = True
    mutated["decision"]["primary_claim_replacement"] = True

    errors = validate_evidence(mutated, root=ROOT)

    assert "decoded_scope.full_multitask_light_v1_candidate must be false" in errors
    assert "decision.primary_claim_replacement must be false" in errors
