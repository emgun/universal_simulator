from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts.build_p2_model_side_beta_head_pretest_root import build_pretest_root
from scripts.validate_p2_model_side_beta_head_pretest_contract import (
    DEFAULT_CONTRACT_JSON,
)

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_JSON = ROOT / DEFAULT_CONTRACT_JSON
MEASUREMENT_KEY = "9c028afbfb85328fd21fc7de4cffb277fbde274aa042ad63e6499abc562addc3"


def _write_h5(path: Path, value: float, *, with_beta_provenance: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=np.full((3, 2), value, dtype=np.float32))
        if with_beta_provenance:
            handle.create_dataset("source_file_index", data=np.asarray([0, 1, 1], dtype=np.int32))
            handle.attrs["source_paths"] = [
                "1D/Advection/Train/1D_Advection_Sols_beta0.1.hdf5",
                "1D/Advection/Train/1D_Advection_Sols_beta0.2.hdf5",
            ]


def _write_roots(base_root: Path, advection_root: Path, *, missing_test_beta: bool = False) -> None:
    for split in ("val", "test"):
        _write_h5(base_root / f"burgers1d_{split}.h5", 1.0)
        _write_h5(base_root / f"darcy2d_{split}.h5", 2.0)
        _write_h5(
            advection_root / f"advection1d_{split}.h5",
            3.0,
            with_beta_provenance=not (split == "test" and missing_test_beta),
        )


def test_pretest_root_refuses_without_explicit_flag(tmp_path):
    base_root = tmp_path / "base"
    advection_root = tmp_path / "official_advection"
    _write_roots(base_root, advection_root)

    with pytest.raises(ValueError, match="allow-heldout-pretest-root"):
        build_pretest_root(
            base_root=base_root,
            advection_root=advection_root,
            out_root=tmp_path / "pretest_root",
            manifest_json=tmp_path / "manifest.json",
            contract_json=CONTRACT_JSON,
            measurement_key=MEASUREMENT_KEY,
            allow_heldout_pretest_root=False,
        )


def test_pretest_root_builds_val_and_test_under_contract(tmp_path):
    base_root = tmp_path / "base"
    advection_root = tmp_path / "official_advection"
    out_root = tmp_path / "pretest_root"
    manifest_json = tmp_path / "manifest.json"
    _write_roots(base_root, advection_root)

    manifest = build_pretest_root(
        base_root=base_root,
        advection_root=advection_root,
        out_root=out_root,
        manifest_json=manifest_json,
        contract_json=CONTRACT_JSON,
        measurement_key=MEASUREMENT_KEY,
        allow_heldout_pretest_root=True,
    )

    assert sorted(path.name for path in out_root.iterdir()) == [
        "advection1d_test.h5",
        "advection1d_val.h5",
        "burgers1d_test.h5",
        "burgers1d_val.h5",
        "darcy2d_test.h5",
        "darcy2d_val.h5",
    ]
    assert manifest == json.loads(manifest_json.read_text(encoding="utf-8"))
    assert manifest["measurement_type"] == "p2_model_side_beta_head_pretest_root_manifest"
    assert manifest["contract"]["measurement_key"] == MEASUREMENT_KEY
    assert manifest["held_out_test_data_materialized"] is True
    assert manifest["held_out_test_used"] is False
    assert manifest["test_ledger_writes"] == []
    assert manifest["sources"]["test"]["advection1d"]["beta_provenance"] == {
        "required": True,
        "has_source_file_index": True,
        "has_source_paths": True,
    }


def test_pretest_root_rejects_measurement_key_mismatch(tmp_path):
    base_root = tmp_path / "base"
    advection_root = tmp_path / "official_advection"
    _write_roots(base_root, advection_root)

    with pytest.raises(ValueError, match="measurement key mismatch"):
        build_pretest_root(
            base_root=base_root,
            advection_root=advection_root,
            out_root=tmp_path / "pretest_root",
            manifest_json=tmp_path / "manifest.json",
            contract_json=CONTRACT_JSON,
            measurement_key="0" * 64,
            allow_heldout_pretest_root=True,
        )


def test_pretest_root_rejects_test_advection_without_beta_provenance(tmp_path):
    base_root = tmp_path / "base"
    advection_root = tmp_path / "official_advection"
    _write_roots(base_root, advection_root, missing_test_beta=True)

    with pytest.raises(ValueError, match="source_file_index.*source_paths"):
        build_pretest_root(
            base_root=base_root,
            advection_root=advection_root,
            out_root=tmp_path / "pretest_root",
            manifest_json=tmp_path / "manifest.json",
            contract_json=CONTRACT_JSON,
            measurement_key=MEASUREMENT_KEY,
            allow_heldout_pretest_root=True,
        )
