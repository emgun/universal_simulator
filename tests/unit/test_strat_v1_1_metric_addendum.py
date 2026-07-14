from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from ups.data.manifests import canonical_sha256

ROOT = Path(__file__).resolve().parents[2]
ADDENDUM_PATH = ROOT / "docs/data/protocols/strat_v1_1_metric_addendum.yaml"
DIAGNOSTICS_PATH = ROOT / "docs/research/artifacts/strat_v1_1_validation_regime_diagnostics.json"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_addendum() -> dict[str, object]:
    payload = yaml.safe_load(ADDENDUM_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_strat_v1_1_addendum_is_self_hashed_and_binds_frozen_inputs() -> None:
    payload = _load_addendum()
    self_hash = payload["self_hash"]
    assert isinstance(self_hash, dict)
    recorded = self_hash.pop("value")
    assert self_hash == {
        "algorithm": "sha256",
        "canonicalization": "canonical_json_sorted_keys_utf8",
        "excluded_field": "self_hash.value",
    }
    assert recorded == canonical_sha256(payload)

    base = payload["base_protocol"]
    assert isinstance(base, dict)
    protocol_path = ROOT / str(base["protocol_manifest_path"])
    training_lock_path = ROOT / str(base["training_lock_path"])
    assert _file_sha256(protocol_path) == base["protocol_manifest_file_sha256"]
    assert _file_sha256(training_lock_path) == base["training_lock_file_sha256"]

    training_lock = json.loads(training_lock_path.read_text(encoding="utf-8"))
    assert training_lock["lock_sha256"] == base["training_lock_sha256"]
    assert training_lock["protocol_manifest_sha256"] == base["protocol_manifest_sha256"]
    assert training_lock["source_revision"] == base["source_revision"]
    assert canonical_sha256(training_lock["selection"]) == base["selection_sha256"]

    evidence = payload["reference_evidence"]
    assert isinstance(evidence, dict)
    scorecard_path = ROOT / str(evidence["calibration_scorecard_path"])
    scorecard = json.loads(scorecard_path.read_text(encoding="utf-8"))
    assert _file_sha256(scorecard_path) == evidence["calibration_scorecard_file_sha256"]
    assert scorecard["scorecard_sha256"] == evidence["calibration_scorecard_sha256"]


def test_strat_v1_1_addendum_freezes_metric_and_validation_only_gate() -> None:
    payload = _load_addendum()
    numeric = payload["numeric_contract"]
    assert numeric == {
        "accumulation_dtype": "float64",
        "epsilon": 1.0e-8,
        "epsilon_placement": "task_target_mean_square_before_square_root",
        "finite_values_required": True,
    }

    metrics = payload["metrics"]
    assert metrics["global_scale_regime_nrmse"]["report_key_suffix"] == ("global_scale_nrmse")
    assert metrics["global_scale_regime_nrmse"]["selection_role"] == "promotion_gate"
    assert metrics["regime_error_ratio_to_persistence"]["report_key_suffix"] == (
        "error_ratio_to_persistence"
    )
    assert metrics["regime_error_ratio_to_persistence"]["selection_role"] == ("reporting_only")
    assert metrics["slice_normalized_regime_nrmse"]["selection_role"] == "diagnostic_only"

    gate = payload["promotion_gate"]
    assert gate == {
        "metric": "global_scale_regime_nrmse_to_task_primary_nrmse",
        "formula": "global_scale_regime_nrmse/task_primary_nrmse",
        "aggregation": "maximum_over_declared_regimes_within_each_task",
        "operator": "less_than_or_equal",
        "maximum": 1.5,
        "required_scope": "every_task",
        "persistence_ratio_role": "reporting_only",
    }

    access = payload["freeze_access"]
    assert access == {
        "derivation_split": "valid",
        "allowed_roles": ["valid"],
        "forbidden_roles": ["test"],
        "measurement_lock_access": "forbidden",
        "heldout_reads": "forbidden",
        "heldout_metrics": "forbidden",
    }

    constraints = payload["base_protocol"]["immutable_constraints"]
    assert set(constraints.values()) == {"unchanged"}


def test_committed_diagnostics_are_self_hashed_and_bound_to_addendum() -> None:
    addendum = _load_addendum()
    diagnostics = json.loads(DIAGNOSTICS_PATH.read_text(encoding="utf-8"))
    recorded = diagnostics.pop("artifact_sha256")
    assert recorded == canonical_sha256(diagnostics)
    assert diagnostics["addendum_sha256"] == addendum["self_hash"]["value"]
    assert diagnostics["held_out_measurements"] == 0
    assert diagnostics["status"] == "complete_validation_only_metric_reprojection"
    assert max(abs(row["reconstruction_delta"]) for row in diagnostics["rows"]) < 3e-9
