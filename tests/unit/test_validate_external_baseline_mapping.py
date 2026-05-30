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
