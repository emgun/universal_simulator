from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import pytest
import yaml

from scripts import materialize_strat_v1_1_regime_diagnostics as diagnostics
from ups.data.manifests import canonical_sha256


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, monkeypatch) -> tuple[Path, Path, Path, Path]:
    repo = tmp_path / "repo"
    validation = tmp_path / "validation"
    repo.mkdir()
    validation.mkdir()
    monkeypatch.setattr(diagnostics, "REGIME_COUNTS", {task: 1 for task in diagnostics.TASKS})

    valid_objects = []
    rows = []
    for task in diagnostics.TASKS:
        shard = validation / f"{task}_val.h5"
        with h5py.File(shard, "w") as handle:
            if task == "darcy2d":
                handle.create_dataset("data", data=[[[[[0.0]]]]])
                handle.create_dataset("targets", data=[[[[[2.0]]]]])
                handle.create_dataset("beta", data=[1.0])
                primary = "decoded_solution_nrmse"
            else:
                handle.create_dataset("data", data=[[[[float(i)]] for i in range(17)]])
                handle.create_dataset(diagnostics.REGIME_KEYS[task], data=[0.1])
                primary = "decoded_rollout_nrmse"
        valid_objects.append({"path": shard.name, "role": "valid", "sha256": _sha256(shard)})
        summary_rel = Path("reports") / task / "summary.json"
        summary_path = repo / summary_rel
        summary_path.parent.mkdir(parents=True)
        summary = {
            "split": "val",
            "metrics": {
                f"task_{task}_{primary}": 0.5,
                f"task_{task}_regime_{diagnostics._slug(1.0 if task == 'darcy2d' else 0.1)}_{primary}": 0.5,
            },
            "extra": {"allow_held_out_test_eval": False},
            "held_out_test_policy": {"enabled": False},
        }
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        rows.append(
            {
                "row_id": f"{task}-persistence",
                "model": "persistence",
                "task": task,
                "source_summary": str(summary_rel),
                "summary_sha256": _sha256(summary_path),
                "primary_nrmse": 0.5,
            }
        )

    scorecard = {
        "scorecard_sha256": "scorecard-internal",
        "held_out_measurements": 0,
        "training_lock": {"lock_sha256": "lock", "objects": valid_objects},
        "rows": rows,
    }
    scorecard_path = repo / "scorecard.json"
    scorecard_path.write_text(json.dumps(scorecard), encoding="utf-8")
    addendum = {
        "addendum_id": "fixture",
        "reference_evidence": {
            "calibration_scorecard_file_sha256": _sha256(scorecard_path),
            "calibration_scorecard_sha256": "scorecard-internal",
        },
        "numeric_contract": {"epsilon": 1e-8},
        "promotion_gate": {"maximum": 1.5},
        "freeze_access": {"derivation_split": "valid", "heldout_reads": "forbidden"},
        "self_hash": {
            "algorithm": "sha256",
            "canonicalization": "canonical_json_sorted_keys_utf8",
            "excluded_field": "self_hash.value",
        },
    }
    addendum["self_hash"]["value"] = canonical_sha256(addendum)
    addendum_path = repo / "addendum.yaml"
    addendum_path.write_text(yaml.safe_dump(addendum), encoding="utf-8")
    return repo, validation, scorecard_path, addendum_path


def test_build_diagnostics_is_validation_only_and_self_hashed(tmp_path, monkeypatch):
    repo, validation, scorecard, addendum = _fixture(tmp_path, monkeypatch)

    result = diagnostics.build_diagnostics(
        addendum_path=addendum,
        scorecard_path=scorecard,
        validation_root=validation,
        repo_root=repo,
    )

    assert result["held_out_measurements"] == 0
    assert result["status"] == "complete_validation_only_metric_reprojection"
    assert len(result["rows"]) == 3
    assert result["artifact_sha256"] == canonical_sha256(
        {key: value for key, value in result.items() if key != "artifact_sha256"}
    )
    assert all(row["regimes"][0]["error_ratio_to_persistence"] == 1 for row in result["rows"])


def test_build_diagnostics_rejects_test_objects_and_tampered_summaries(tmp_path, monkeypatch):
    repo, validation, scorecard, addendum = _fixture(tmp_path, monkeypatch)
    with h5py.File(validation / "darcy2d_test.h5", "w") as handle:
        handle.create_dataset("data", data=[0.0])
    with pytest.raises(PermissionError, match="test object"):
        diagnostics.build_diagnostics(
            addendum_path=addendum,
            scorecard_path=scorecard,
            validation_root=validation,
            repo_root=repo,
        )
    (validation / "darcy2d_test.h5").unlink()
    summary = repo / "reports/advection1d/summary.json"
    summary.write_text(summary.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="source summary hash"):
        diagnostics.build_diagnostics(
            addendum_path=addendum,
            scorecard_path=scorecard,
            validation_root=validation,
            repo_root=repo,
        )


def test_validate_addendum_rejects_tampering(tmp_path, monkeypatch):
    _, _, _, addendum_path = _fixture(tmp_path, monkeypatch)
    payload = yaml.safe_load(addendum_path.read_text(encoding="utf-8"))
    payload["promotion_gate"]["maximum"] = 2.0
    with pytest.raises(ValueError, match="self hash"):
        diagnostics._validate_addendum(payload)
