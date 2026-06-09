from __future__ import annotations

import copy
import json
from pathlib import Path

from scripts.validate_p2_parameter_canonical_root_sidecar_evidence import validate_evidence

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_PATH = (
    ROOT / "docs/claim_evidence/ups_advection_p2_parameter_canonical_root_sidecar_val_evidence.json"
)


def _load_current_evidence() -> dict:
    with EVIDENCE_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_current_p2_parameter_canonical_root_sidecar_evidence_validates():
    evidence = _load_current_evidence()

    assert validate_evidence(evidence, root=ROOT) == []


def test_canonical_root_sidecar_evidence_rejects_heldout_or_test_command():
    evidence = _load_current_evidence()
    mutated = copy.deepcopy(evidence)
    mutated["held_out_test_data_read"] = True
    mutated["command_args"].extend(["--extra-eval-split", "test"])

    errors = validate_evidence(mutated, root=ROOT)

    assert "held_out_test_data_read must be false" in errors
    assert "command_args must not include --extra-eval-split" in errors


def test_canonical_root_sidecar_evidence_rejects_task_roots_or_summary_drift():
    evidence = _load_current_evidence()
    mutated = copy.deepcopy(evidence)
    mutated["command_args"].append(
        "data.task_roots={advection1d: data/pdebench_official_advection_light}"
    )
    mutated["summary_report"]["sha256"] = "bad"

    errors = validate_evidence(mutated, root=ROOT)

    assert "command_args must not include data.task_roots" in errors
    assert "summary_report.sha256 must match file bytes" in errors


def test_canonical_root_sidecar_evidence_rejects_manifest_or_scope_drift():
    evidence = _load_current_evidence()
    mutated = copy.deepcopy(evidence)
    mutated["root_manifest"]["sha256"] = "bad"
    mutated["decoded_scope"]["single_data_root_validation"] = False
    mutated["decoded_scope"]["uses_task_roots"] = True

    errors = validate_evidence(mutated, root=ROOT)

    assert "root_manifest.sha256 must match file bytes" in errors
    assert "decoded_scope.single_data_root_validation must be true" in errors
    assert "decoded_scope.uses_task_roots must be false" in errors


def test_canonical_root_sidecar_evidence_rejects_primary_claim_overclaim():
    evidence = _load_current_evidence()
    mutated = copy.deepcopy(evidence)
    mutated["decision"]["primary_claim_replacement"] = True
    mutated["decoded_scope"]["full_multitask_primary_claim_replacement"] = True

    errors = validate_evidence(mutated, root=ROOT)

    assert "decision.primary_claim_replacement must be false" in errors
    assert "decoded_scope.full_multitask_primary_claim_replacement must be false" in errors
