from __future__ import annotations

import copy
import json
import tarfile
from pathlib import Path

from scripts.validate_ups_advection_phase_tracking_gate_contract import (
    evaluate_candidate_summary,
    validate_contract,
)

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    ROOT / "docs/claim_evidence/ups_advection_phase_tracking_validation_gate_contract.json"
)
FAILED_CANDIDATE_ARTIFACT = (
    ROOT / "docs/claim_evidence/artifacts/ups_advection_model_primary_heldout_light_v1.tar.gz"
)
FAILED_CANDIDATE_VAL_MEMBER = (
    "ups_light_advection_weighted_operator_stability_seed23_primary_guarded/summary.json"
)


def _load_contract() -> dict:
    with CONTRACT_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_failed_candidate_validation_summary() -> dict:
    with tarfile.open(FAILED_CANDIDATE_ARTIFACT, mode="r:gz") as archive:
        extracted = archive.extractfile(FAILED_CANDIDATE_VAL_MEMBER)
        assert extracted is not None
        return json.load(extracted)


def test_phase_tracking_contract_validates():
    contract = _load_contract()

    assert validate_contract(contract, root=ROOT) == []


def test_failed_candidate_validation_summary_does_not_clear_gate():
    contract = _load_contract()
    summary = _load_failed_candidate_validation_summary()

    errors = evaluate_candidate_summary(summary, contract)

    assert (
        "summary.metrics.decoded_rollout_nrmse=0.35078329353213156 must be < " "0.35078329353213156"
    ) in errors
    assert (
        "summary.metrics.task_advection1d_decoded_h16_nrmse=0.4938241237376044 must be "
        "<= 0.44444171136384397"
    ) in errors


def test_synthetic_improved_validation_summary_clears_gate():
    contract = _load_contract()
    summary = _load_failed_candidate_validation_summary()
    improved = copy.deepcopy(summary)
    improved["metrics"]["decoded_rollout_nrmse"] = 0.34
    improved["metrics"]["task_advection1d_decoded_rollout_nrmse"] = 0.45
    improved["metrics"]["task_advection1d_decoded_h16_nrmse"] = 0.4

    assert evaluate_candidate_summary(improved, contract) == []


def test_candidate_summary_must_stay_validation_only_and_no_context_shift():
    contract = _load_contract()
    summary = _load_failed_candidate_validation_summary()
    mutated = copy.deepcopy(summary)
    mutated["extra"]["decoded_split"] = "test"
    mutated["extra"]["decoded_decoded_context_roll_shift_estimator"] = {"mode": "roll"}

    errors = evaluate_candidate_summary(mutated, contract)

    assert "summary.extra.decoded_split must match required validation split" in errors
    assert "summary.extra.decoded_decoded_context_roll_shift_estimator must be empty" in errors
