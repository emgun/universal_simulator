from __future__ import annotations

import copy
import json
from pathlib import Path

from scripts.validate_external_baseline_mapping import validate_mapping

ROOT = Path(__file__).resolve().parents[2]
MAPPING_PATH = ROOT / "docs/claim_evidence/external_baseline_mapping.json"
CLAIM_EVIDENCE_PATH = ROOT / "docs/claim_evidence/universal_sota_claim_evidence.json"


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_external_baseline_mapping_matches_current_claim_evidence():
    mapping = _load(MAPPING_PATH)
    claim_evidence = _load(CLAIM_EVIDENCE_PATH)

    assert validate_mapping(mapping, claim_evidence) == []


def test_external_baseline_mapping_rejects_scoped_variant_metric_mismatch():
    mapping = _load(MAPPING_PATH)
    claim_evidence = _load(CLAIM_EVIDENCE_PATH)
    mutated = copy.deepcopy(mapping)
    mutated["scoped_claim_variants"][0]["metric_value"] = 999.0

    errors = validate_mapping(mutated, claim_evidence)

    assert (
        "scoped_claim_variants[light_v1_ct1_online_transport_context].metric_value "
        "must match claim evidence" in errors
    )


def test_external_baseline_mapping_rejects_scoped_variant_overclaim():
    mapping = _load(MAPPING_PATH)
    claim_evidence = _load(CLAIM_EVIDENCE_PATH)
    mutated = copy.deepcopy(mapping)
    variant = mutated["scoped_claim_variants"][0]
    variant["same_exact_inference_contract_as_primary"] = True
    variant["not_autonomous_rollout_claim"] = False
    variant["published_numbers_directly_comparable"] = True
    variant["external_paper_reproduction"] = True

    errors = validate_mapping(mutated, claim_evidence)

    assert (
        "scoped_claim_variants[light_v1_ct1_online_transport_context]."
        "same_exact_inference_contract_as_primary must be false" in errors
    )
    assert (
        "scoped_claim_variants[light_v1_ct1_online_transport_context]."
        "not_autonomous_rollout_claim must be true" in errors
    )
    assert (
        "scoped_claim_variants[light_v1_ct1_online_transport_context]."
        "published_numbers_directly_comparable must be false" in errors
    )
    assert (
        "scoped_claim_variants[light_v1_ct1_online_transport_context]."
        "external_paper_reproduction must be false" in errors
    )


def test_external_baseline_mapping_rejects_claim_metric_mismatch():
    mapping = _load(MAPPING_PATH)
    claim_evidence = _load(CLAIM_EVIDENCE_PATH)
    mutated = copy.deepcopy(mapping)
    mutated["claim_protocol"]["metric_value"] = 999.0

    errors = validate_mapping(mutated, claim_evidence)

    assert "claim_protocol.metric_value must match claim_documentation.metric_value" in errors


def test_external_baseline_mapping_rejects_unknown_source_reference():
    mapping = _load(MAPPING_PATH)
    claim_evidence = _load(CLAIM_EVIDENCE_PATH)
    mutated = copy.deepcopy(mapping)
    mutated["selected_reproduction_path"]["source_refs"].append("missing_source")

    errors = validate_mapping(mutated, claim_evidence)

    assert "selected_reproduction_path references unknown source_ref: missing_source" in errors


def test_external_baseline_mapping_rejects_unknown_ecosystem_source_reference():
    mapping = _load(MAPPING_PATH)
    claim_evidence = _load(CLAIM_EVIDENCE_PATH)
    mutated = copy.deepcopy(mapping)
    mutated.setdefault("ecosystem_compatibility", []).append(
        {
            "surface": "Example",
            "candidate_id": "example_ecosystem_gate",
            "status": "planned",
            "readiness_lane": "ecosystem compatibility",
            "source_refs": ["missing_source"],
            "claim_boundary": "Compatibility surface; no current UPS metric.",
            "next_step": "Add adapter.",
        }
    )

    errors = validate_mapping(mutated, claim_evidence)

    assert (
        "ecosystem_compatibility[example_ecosystem_gate] references unknown source_ref: missing_source"
        in errors
    )


def test_external_baseline_mapping_rejects_physicsnemo_smoke_metric_overclaim():
    mapping = _load(MAPPING_PATH)
    claim_evidence = _load(CLAIM_EVIDENCE_PATH)
    mutated = copy.deepcopy(mapping)
    physicsnemo = next(
        row
        for row in mutated["ecosystem_compatibility"]
        if row["candidate_id"] == "physicsnemo_compatibility_gate"
    )
    physicsnemo["status"] = "compatibility_smoke_ready"
    physicsnemo["metric_name"] = "decoded_rollout_nrmse"
    physicsnemo["metric_value"] = 0.1
    physicsnemo["evidence_json"] = (
        "docs/claim_evidence/physicsnemo_compatibility_smoke_light_v1.json"
    )
    physicsnemo["adapter_entrypoint"] = "scripts/run_physicsnemo_compatibility_smoke.py"
    physicsnemo["validation_command"] = (
        "python scripts/run_physicsnemo_compatibility_smoke.py --check"
    )

    errors = validate_mapping(mutated, claim_evidence)

    assert (
        "ecosystem_compatibility[physicsnemo_compatibility_gate] compatibility smoke "
        "must not set metric_name, metric_value, or test_metric_value"
    ) in errors


def test_external_baseline_mapping_rejects_published_number_overclaim():
    mapping = _load(MAPPING_PATH)
    claim_evidence = _load(CLAIM_EVIDENCE_PATH)
    mutated = copy.deepcopy(mapping)
    mutated["comparability_decision"]["published_numbers_directly_comparable"] = True

    errors = validate_mapping(mutated, claim_evidence)

    assert "comparability_decision.published_numbers_directly_comparable must be false" in errors


def test_external_baseline_mapping_rejects_measured_status_without_test_evidence():
    mapping = _load(MAPPING_PATH)
    claim_evidence = _load(CLAIM_EVIDENCE_PATH)
    mutated = copy.deepcopy(mapping)
    primary = next(
        candidate
        for candidate in mutated["baseline_candidates"]
        if candidate["status"] == "selected_primary_reproduction_path"
    )
    primary["test_measurements"] = []

    errors = validate_mapping(mutated, claim_evidence)

    assert "external_reproduction_measured requires selected primary test_measurements" in errors


def test_external_baseline_mapping_rejects_secondary_measured_without_test_evidence():
    mapping = _load(MAPPING_PATH)
    claim_evidence = _load(CLAIM_EVIDENCE_PATH)
    mutated = copy.deepcopy(mapping)
    secondary = next(
        candidate
        for candidate in mutated["baseline_candidates"]
        if candidate["candidate_id"] == "neuraloperator_uno_light_v1"
    )
    secondary["test_measurements"] = []

    errors = validate_mapping(mutated, claim_evidence)

    assert (
        "neuraloperator_uno_light_v1.test_measurements is required for measured candidate" in errors
    )


def test_external_baseline_mapping_rejects_foundation_transfer_test_budget_use():
    mapping = _load(MAPPING_PATH)
    claim_evidence = _load(CLAIM_EVIDENCE_PATH)
    mutated = copy.deepcopy(mapping)
    mutated["foundation_transfer_contract"]["held_out_test_used"] = True

    errors = validate_mapping(mutated, claim_evidence)

    assert "foundation_transfer_contract.held_out_test_used must be false" in errors


def test_external_baseline_mapping_rejects_poseidon_adapter_model_metric_overclaim():
    mapping = _load(MAPPING_PATH)
    claim_evidence = _load(CLAIM_EVIDENCE_PATH)
    mutated = copy.deepcopy(mapping)
    mutated["foundation_transfer_adapter_gate"]["metrics"]["decoded_rollout_nrmse"] = 0.1

    errors = validate_mapping(mutated, claim_evidence)

    assert "foundation_transfer_adapter_gate must not report decoded_rollout_nrmse" in errors


def test_external_baseline_mapping_rejects_poseidon_validation_test_budget_use():
    mapping = _load(MAPPING_PATH)
    claim_evidence = _load(CLAIM_EVIDENCE_PATH)
    mutated = copy.deepcopy(mapping)
    mutated["foundation_transfer_validation_measurement"]["split"] = "test"
    mutated["foundation_transfer_validation_measurement"]["held_out_test_used"] = True

    errors = validate_mapping(mutated, claim_evidence)

    assert "foundation_transfer_validation_measurement.split must not be test" in errors
    assert "foundation_transfer_validation_measurement.held_out_test_used must be false" in errors


def test_external_baseline_mapping_rejects_poseidon_finetune_test_budget_use():
    mapping = _load(MAPPING_PATH)
    claim_evidence = _load(CLAIM_EVIDENCE_PATH)
    mutated = copy.deepcopy(mapping)
    mutated["foundation_transfer_finetune_validation_measurement"]["split"] = "test"
    mutated["foundation_transfer_finetune_validation_measurement"]["held_out_test_used"] = True

    errors = validate_mapping(mutated, claim_evidence)

    assert "foundation_transfer_finetune_validation_measurement.split must not be test" in errors
    assert (
        "foundation_transfer_finetune_validation_measurement.held_out_test_used must be false"
        in errors
    )
