from __future__ import annotations

import copy
import json
from pathlib import Path

from scripts.validate_ups_advection_data_conditioned_heldout_evidence import (
    validate_evidence,
)

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_PATH = (
    ROOT / "docs/claim_evidence/ups_advection_data_conditioned_heldout_light_v1_evidence.json"
)


def _load_current_evidence() -> dict:
    with EVIDENCE_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_data_conditioned_heldout_evidence_validates():
    evidence = _load_current_evidence()

    assert validate_evidence(evidence, root=ROOT) == []


def test_data_conditioned_heldout_evidence_rejects_measurement_key_drift():
    evidence = _load_current_evidence()
    mutated = copy.deepcopy(evidence)
    mutated["held_out_test_policy"]["measurement_key"] = "0" * 64

    errors = validate_evidence(mutated, root=ROOT)

    assert "held_out_test_policy.measurement_key does not match command-derived key" in errors


def test_data_conditioned_heldout_evidence_rejects_repeat_or_legacy_context_shift():
    evidence = _load_current_evidence()
    mutated = copy.deepcopy(evidence)
    mutated["command"] += (
        " --allow-repeat-held-out-test "
        "--eval-override evaluation.decoded_context_roll_shift_estimator={}"
    )

    errors = validate_evidence(mutated, root=ROOT)

    assert "command must not include --allow-repeat-held-out-test" in errors
    assert "command must not include evaluation.decoded_context_roll_shift_estimator" in errors


def test_data_conditioned_heldout_evidence_rejects_artifact_hash_drift():
    evidence = _load_current_evidence()
    mutated = copy.deepcopy(evidence)
    mutated["artifact"]["sha256"] = "0" * 64

    errors = validate_evidence(mutated, root=ROOT)

    assert "artifact.sha256 must match artifact bytes" in errors


def test_data_conditioned_heldout_evidence_rejects_metric_drift():
    evidence = _load_current_evidence()
    mutated = copy.deepcopy(evidence)
    mutated["test_metrics"]["decoded_rollout_nrmse"] = 0.0

    errors = validate_evidence(mutated, root=ROOT)

    assert "test_metrics.decoded_rollout_nrmse must match summary_test metrics" in errors


def test_data_conditioned_heldout_evidence_rejects_scoped_overclaim_language():
    evidence = _load_current_evidence()
    mutated = copy.deepcopy(evidence)
    scoped = mutated["scoped_claim_language"]
    scoped["same_exact_inference_contract_as_primary"] = True
    scoped["not_autonomous_rollout_claim"] = False
    scoped["published_numbers_directly_comparable"] = True
    scoped["external_paper_reproduction"] = True

    errors = validate_evidence(mutated, root=ROOT)

    assert "scoped_claim_language.same_exact_inference_contract_as_primary must be false" in errors
    assert "scoped_claim_language.not_autonomous_rollout_claim must be true" in errors
    assert "scoped_claim_language.published_numbers_directly_comparable must be false" in errors
    assert "scoped_claim_language.external_paper_reproduction must be false" in errors
