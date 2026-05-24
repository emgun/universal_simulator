from __future__ import annotations

from argparse import Namespace

import h5py
import numpy as np
import yaml

from scripts.audit_transport_data_hydration_options import audit_hydration_options


def _write_split(
    root, *, split: str, shift: int, samples: int = 2, steps: int = 5, width: int = 8
) -> None:
    path = root / f"advection1d_{split}.h5"
    root.mkdir(parents=True, exist_ok=True)
    base = np.zeros((samples, width), dtype=np.float32)
    base[:, 1] = 1.0
    data = []
    for sample_idx in range(samples):
        trajectory = [np.roll(base[sample_idx], shift * step_idx) for step_idx in range(steps)]
        data.append(trajectory)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=np.asarray(data, dtype=np.float32)[..., None])


def _write_manifest(path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "files": [
                    {
                        "path": "1D/Advection/Train/1D_Advection_Sols_beta0.7.hdf5",
                        "file_id": 123,
                        "size_bytes": 100,
                        "checksum": "abc",
                        "checksum_type": "MD5",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def _args(tmp_path, *, data_root):
    manifest = tmp_path / "manifest.yaml"
    _write_manifest(manifest)
    return Namespace(
        data_root=str(data_root),
        task="advection1d",
        train_split="train",
        val_split="val",
        max_samples=-1,
        val_max_samples=-1,
        rollout_steps=3,
        shift=[0, 1, 2],
        metric="nrmse",
        manifest=str(manifest),
        synthetic_root=str(tmp_path / "reports"),
        synthetic_limit=10,
        output_json=str(tmp_path / "audit.json"),
    )


def test_hydration_audit_requires_remote_when_local_train_lacks_val_support(tmp_path):
    data_root = tmp_path / "data"
    _write_split(data_root, split="train", shift=0)
    _write_split(data_root, split="val", shift=2)
    synthetic_root = tmp_path / "reports" / "run" / "synthetic_pdebench"
    _write_split(synthetic_root, split="train", shift=2)

    args = _args(tmp_path, data_root=data_root)
    record = audit_hydration_options(args)

    assert record["status"] == "remote_official_hydration_required"
    assert record["canonical_local"]["unsupported_val_shifts"] == [2]
    assert record["remote_official_manifest"]["advection_train_file_count"] == 1
    assert record["synthetic_report_artifacts"]["returned_count"] == 1


def test_hydration_audit_accepts_local_support_when_train_covers_validation(tmp_path):
    data_root = tmp_path / "data"
    _write_split(data_root, split="train", shift=1)
    _write_split(data_root, split="val", shift=1)

    record = audit_hydration_options(_args(tmp_path, data_root=data_root))

    assert record["status"] == "local_benchmark_clean_support_available"
    assert record["blockers"] == []
