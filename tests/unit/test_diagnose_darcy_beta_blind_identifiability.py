from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts import diagnose_darcy_beta_blind_identifiability as diagnostic
from ups.data.manifests import canonical_sha256


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(
    tmp_path: Path,
    *,
    missing_last_row: bool = False,
    unequal_input: bool = False,
    nonfinite_target: bool = False,
    validation_name: str = "darcy2d_val.h5",
) -> tuple[Path, Path, str, str, str]:
    root = tmp_path / "fixture"
    root.mkdir()
    shard = root / validation_name
    rows = []
    # Interleave groups to prove grouping is by provenance, not row adjacency.
    for beta_index, beta in enumerate(diagnostic.EXPECTED_BETAS):
        for group in range(2):
            coefficient = np.full((1, 2, 2, 1), group + 1.0, dtype=np.float32)
            target = coefficient + np.float32(beta_index + 1)
            rows.append((coefficient, target, beta, beta_index, 100 + group))
    if missing_last_row:
        rows.pop()
    inputs = np.stack([row[0] for row in rows])
    if unequal_input:
        inputs[3, 0, 0, 0, 0] += 1.0
    targets = np.stack([row[1] for row in rows])
    if nonfinite_target:
        targets[0, 0, 0, 0, 0] = np.nan
    with h5py.File(shard, "w") as handle:
        handle.create_dataset("data", data=inputs)
        handle.create_dataset("targets", data=targets)
        handle.create_dataset("beta", data=[row[2] for row in rows])
        handle.create_dataset("source_file_id", data=[row[3] for row in rows])
        handle.create_dataset("source_sample_index", data=[row[4] for row in rows])

    validation_sha = _sha256(shard)
    selection = {
        "algorithm": "sha256-protocol-seed-provenance-v1",
        "protocol": "strat-v1",
        "seed": 0,
    }
    lock_payload = {
        "schema_version": 1,
        "dataset_id": "pdebench",
        "source_revision": "sha256:" + "1" * 64,
        "source_manifest_sha256": "2" * 64,
        "protocol_id": "pdebench-strat-v1",
        "protocol_manifest_sha256": "3" * 64,
        "adapter": "pdebench_hdf5",
        "adapter_revision": "1.0.0",
        "purpose": "training",
        "requested_roles": ["valid"],
        "measurement_contract_id": None,
        "objects": [
            {
                "object_id": "darcy2d-valid",
                "role": "valid",
                "path": validation_name,
                "size_bytes": shard.stat().st_size,
                "checksums": {"sha256": validation_sha},
                "uris": [f"b2://pdebench/valid/{validation_name}"],
                "media_type": "application/x-hdf5",
            }
        ],
        "selection": selection,
        "normalization": {"fit_role": "train", "method": "zscore"},
    }
    lock = {**lock_payload, "lock_sha256": canonical_sha256(lock_payload)}
    lock_path = root / "training.lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    return (
        lock_path,
        shard,
        lock["lock_sha256"],
        canonical_sha256(selection),
        validation_sha,
    )


def _build(fixture: tuple[Path, Path, str, str, str]):
    lock, shard, lock_sha, selection_sha, validation_sha = fixture
    return diagnostic.build_diagnostic(
        training_lock_path=lock,
        validation_shard_path=shard,
        expected_lock_sha256=lock_sha,
        expected_selection_sha256=selection_sha,
        expected_validation_sha256=validation_sha,
    )


def test_builds_validation_only_self_hashed_beta_blind_oracle(tmp_path):
    result = _build(_fixture(tmp_path))

    assert result["status"] == "complete_validation_only"
    assert result["access"] == {
        "split": "valid",
        "read_roles": ["valid"],
        "heldout_reads": 0,
        "held_out_measurements": 0,
    }
    assert result["coverage"]["group_count"] == 2
    assert result["coverage"]["rows"] == 10
    assert set(result["coverage"]["groups_per_beta"].values()) == {2}
    oracle = result["beta_blind_oracle"]
    assert oracle["pooled_global_scale_nrmse"] > 0
    assert oracle["reconstructed_global_scale_nrmse"] == pytest.approx(
        oracle["pooled_global_scale_nrmse"], rel=1e-12
    )
    assert len(oracle["regimes"]) == 5
    assert len(result["target_separation"]["pairs"]) == 10
    assert result["target_separation"]["minimum_global_scale_nrmse"] > 0
    assert len(result["provenance_groups"]) == 2
    assert result["artifact_sha256"] == canonical_sha256(
        {key: value for key, value in result.items() if key != "artifact_sha256"}
    )


@pytest.mark.parametrize(
    ("fixture_kwargs", "message"),
    [
        ({"missing_last_row": True}, "cover all five"),
        ({"unequal_input": True}, "coefficient inputs differ"),
        ({"nonfinite_target": True}, "solution targets.*finite"),
    ],
)
def test_fails_closed_on_invalid_group_coverage_inputs_or_values(
    tmp_path, fixture_kwargs, message
):
    with pytest.raises(ValueError, match=message):
        _build(_fixture(tmp_path, **fixture_kwargs))


@pytest.mark.parametrize("identity", ["lock", "selection", "validation"])
def test_requires_exact_caller_pinned_lock_selection_and_validation_sha(tmp_path, identity):
    fixture = list(_fixture(tmp_path))
    fixture[{"lock": 2, "selection": 3, "validation": 4}[identity]] = "f" * 64
    with pytest.raises(ValueError, match={
        "lock": "training lock SHA",
        "selection": "training lock selection",
        "validation": "locked validation object SHA",
    }[identity]):
        _build(tuple(fixture))


def test_rejects_test_named_validation_path_before_opening_bytes(tmp_path):
    fixture = _fixture(tmp_path, validation_name="darcy2d_test.h5")
    with pytest.raises(PermissionError, match="validation shard contains a test path"):
        _build(fixture)
