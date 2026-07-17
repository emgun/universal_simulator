import json
from pathlib import Path

import yaml

from scripts.audit_universal_latent_contract import audit_contract

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_config(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_d6_contract_fails_closed_on_universal_encoder_claim() -> None:
    report = audit_contract(
        REPO_ROOT / "configs/d6_strat_v1_modular_shared_trunk.yaml",
        repo_root=REPO_ROOT,
    )

    assert report["config"]["data_kind"] == "grid"
    assert report["architecture_facts"]["observed_modalities"] == ["grid"]
    assert report["architecture_facts"]["missing_universal_modalities"] == [
        "mesh",
        "particle",
    ]
    assert report["architecture_facts"]["operator_stage"]["epochs"] == 12
    assert not report["architecture_facts"]["operator_stage"]["encoder_is_optimizer_owned"]
    assert report["architecture_facts"]["joint_codec_operator_stage"]["epochs"] == 4
    assert report["architecture_facts"]["task_router"]["route_source"] == "task_id"
    assert report["classification"]["universal_encoder_claim"] == "not_tested"
    assert report["classification"]["codec_vs_dynamics_causality"] == "unresolved"
    assert not report["classification"]["family_router_authorized"]


def test_equal_latent_shape_never_counts_as_alignment(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path,
        {
            "data": {"kind": "mesh", "task": "elasticity"},
            "latent": {"tokens": 64, "dim": 64},
            "stages": {"operator": {"epochs": 3}},
        },
    )

    report = audit_contract(config, repo_root=REPO_ROOT)

    assert report["config"]["latent_shape"] == {"tokens": 64, "dim": 64}
    assert not report["architecture_facts"]["equal_output_shape_is_alignment_evidence"]
    assert (
        report["required_measurements"]["paired_cross_discretization_alignment"]["status"]
        == "unmeasured"
    )
    assert report["classification"]["common_latent_space_claim"] == "not_tested"


def test_report_is_json_serializable_and_source_bound(tmp_path: Path) -> None:
    config = _write_config(tmp_path, {"data": {"kind": "particle", "task": ["sph"]}})

    report = audit_contract(config, repo_root=REPO_ROOT)
    encoded = json.dumps(report, sort_keys=True)

    assert "ups.universal-latent-contract-audit.v1" in encoded
    assert len(report["source_bindings"]["training_pipeline"]["sha256"]) == 64
