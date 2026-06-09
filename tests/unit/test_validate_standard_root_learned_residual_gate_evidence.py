from __future__ import annotations

import copy
import json
from pathlib import Path

from scripts.validate_standard_root_learned_residual_gate_evidence import validate_evidence

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_PATH = (
    ROOT / "docs/claim_evidence/ups_standard_root_learned_residual_gate_val_evidence.json"
)


def _load_current_evidence() -> dict:
    with EVIDENCE_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_current_standard_root_learned_residual_gate_evidence_validates():
    evidence = _load_current_evidence()

    assert validate_evidence(evidence, root=ROOT) == []


def test_standard_root_gate_evidence_rejects_heldout_or_test_command():
    evidence = _load_current_evidence()
    mutated = copy.deepcopy(evidence)
    mutated["held_out_test_data_read"] = True
    mutated["fit_command_args"].extend(["--extra-eval-split", "test"])

    errors = validate_evidence(mutated, root=ROOT)

    assert "held_out_test_data_read must be false" in errors
    assert "fit_command_args must not include --extra-eval-split" in errors


def test_standard_root_gate_evidence_rejects_task_roots_beta_or_generated_root_scope():
    evidence = _load_current_evidence()
    mutated = copy.deepcopy(evidence)
    mutated["fit_command_args"].append(
        "data.task_roots={advection1d: data/pdebench_official_advection_light}"
    )
    mutated["fit_command_args"].append("data.param_keys=[beta]")
    mutated["model_scope"]["uses_generated_root"] = True
    mutated["model_scope"]["uses_beta_metadata"] = True

    errors = validate_evidence(mutated, root=ROOT)

    assert "fit_command_args must not include data.task_roots" in errors
    assert "fit_command_args must not request beta param_keys" in errors
    assert "model_scope.uses_generated_root must be false" in errors
    assert "model_scope.uses_beta_metadata must be false" in errors


def test_standard_root_gate_evidence_rejects_metric_and_hash_drift():
    evidence = _load_current_evidence()
    mutated = copy.deepcopy(evidence)
    mutated["fit_report"]["sha256"] = "bad"
    mutated["metrics"]["learned_gate_val_decoded_rollout_nrmse"] = 0.0

    errors = validate_evidence(mutated, root=ROOT)

    assert "fit_report.sha256 must match file bytes" in errors
    assert "metrics.learned_gate_val_decoded_rollout_nrmse must match fit report" in errors


def test_standard_root_gate_evidence_rejects_positive_overclaim():
    evidence = _load_current_evidence()
    mutated = copy.deepcopy(evidence)
    mutated["decision"]["primary_claim_replacement"] = True
    mutated["model_scope"]["full_multitask_primary_claim_replacement"] = True

    errors = validate_evidence(mutated, root=ROOT)

    assert "decision.primary_claim_replacement must be false" in errors
    assert "model_scope.full_multitask_primary_claim_replacement must be false" in errors
