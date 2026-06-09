from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts.build_p2_parameter_full_task_root import build_full_task_root


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


def test_build_full_task_root_links_validation_mix_and_manifest(tmp_path):
    base_root = tmp_path / "base"
    advection_root = tmp_path / "official_advection"
    out_root = tmp_path / "canonical"
    manifest_json = tmp_path / "manifest.json"

    _write_h5(base_root / "burgers1d_val.h5", 1.0)
    _write_h5(base_root / "darcy2d_val.h5", 2.0)
    _write_h5(base_root / "burgers1d_test.h5", 99.0)
    _write_h5(advection_root / "advection1d_val.h5", 3.0, with_beta_provenance=True)
    _write_h5(advection_root / "advection1d_test.h5", 100.0, with_beta_provenance=True)

    manifest = build_full_task_root(
        base_root=base_root,
        advection_root=advection_root,
        out_root=out_root,
        manifest_json=manifest_json,
        split="val",
        overwrite=False,
    )

    assert sorted(path.name for path in out_root.iterdir()) == [
        "advection1d_val.h5",
        "burgers1d_val.h5",
        "darcy2d_val.h5",
    ]
    assert not (out_root / "advection1d_test.h5").exists()
    assert not (out_root / "burgers1d_test.h5").exists()
    with h5py.File(out_root / "advection1d_val.h5", "r") as handle:
        assert handle["source_file_index"][:].tolist() == [0, 1, 1]
        assert list(handle.attrs["source_paths"]) == [
            "1D/Advection/Train/1D_Advection_Sols_beta0.1.hdf5",
            "1D/Advection/Train/1D_Advection_Sols_beta0.2.hdf5",
        ]

    disk_manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
    assert manifest == disk_manifest
    assert manifest["measurement_type"] == "ups_p2_parameter_full_task_root_manifest"
    assert manifest["split"] == "val"
    assert manifest["tasks"] == ["burgers1d", "advection1d", "darcy2d"]
    assert manifest["held_out_test_data_read"] is False
    assert manifest["test_ledger_writes"] == []
    assert manifest["sources"]["advection1d"]["beta_provenance"] == {
        "required": True,
        "has_source_file_index": True,
        "has_source_paths": True,
    }
    assert all(record["split"] == "val" for record in manifest["sources"].values())
    assert all("_test." not in record["source_path"] for record in manifest["sources"].values())


def test_build_full_task_root_rejects_advection_without_beta_provenance(tmp_path):
    base_root = tmp_path / "base"
    advection_root = tmp_path / "official_advection"
    _write_h5(base_root / "burgers1d_val.h5", 1.0)
    _write_h5(base_root / "darcy2d_val.h5", 2.0)
    _write_h5(advection_root / "advection1d_val.h5", 3.0, with_beta_provenance=False)

    with pytest.raises(ValueError, match="source_file_index.*source_paths"):
        build_full_task_root(
            base_root=base_root,
            advection_root=advection_root,
            out_root=tmp_path / "canonical",
            manifest_json=tmp_path / "manifest.json",
            split="val",
            overwrite=False,
        )
