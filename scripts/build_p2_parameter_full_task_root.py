#!/usr/bin/env python
from __future__ import annotations

"""Build a validation-only full-task PDEBench root with Advection beta provenance."""

import argparse
import errno
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import h5py

TASKS = ("burgers1d", "advection1d", "darcy2d")
MEASUREMENT_TYPE = "ups_p2_parameter_full_task_root_manifest"


def _json_safe(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, dict):
        return {str(key): _json_safe(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(inner) for inner in value]
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _h5_summary(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        datasets: dict[str, Any] = {}
        for key, value in handle.items():
            if not isinstance(value, h5py.Dataset):
                continue
            datasets[key] = {
                "shape": [int(dim) for dim in value.shape],
                "dtype": str(value.dtype),
                "attrs": _json_safe(dict(value.attrs.items())),
            }
        file_attrs = _json_safe(dict(handle.attrs.items()))
    return {
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "file_attrs": file_attrs,
        "datasets": datasets,
    }


def _beta_provenance(path: Path) -> dict[str, bool]:
    with h5py.File(path, "r") as handle:
        source_paths = _json_safe(handle.attrs.get("source_paths", []))
        return {
            "required": True,
            "has_source_file_index": "source_file_index" in handle,
            "has_source_paths": bool(source_paths),
        }


def _validate_advection_beta_provenance(path: Path) -> dict[str, bool]:
    with h5py.File(path, "r") as handle:
        if "sequential_hydration_complete" in handle.attrs and not bool(
            handle.attrs["sequential_hydration_complete"]
        ):
            raise ValueError(f"{path} is marked sequential_hydration_complete=False")
        source_paths = _json_safe(handle.attrs.get("source_paths", []))
        has_source_file_index = "source_file_index" in handle
        if not has_source_file_index or not source_paths:
            raise ValueError(
                f"{path} must contain source_file_index dataset and source_paths attrs "
                "for Advection beta provenance"
            )
        if "data" in handle:
            data_rows = int(handle["data"].shape[0])
            source_rows = int(handle["source_file_index"].shape[0])
            if source_rows != data_rows:
                raise ValueError(
                    f"{path} data/source_file_index row mismatch: {data_rows} != {source_rows}"
                )
        known_sources = set(range(len(source_paths)))
        observed_sources = set(int(value) for value in handle["source_file_index"][:].tolist())
        missing_sources = sorted(observed_sources - known_sources)
        if missing_sources:
            raise ValueError(
                f"{path} has source_file_index values without source_paths entries: "
                f"{missing_sources}"
            )
    return {
        "required": True,
        "has_source_file_index": True,
        "has_source_paths": True,
    }


def _link_or_copy(source: Path, destination: Path, *, overwrite: bool) -> str:
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"{destination} already exists; pass --overwrite to replace it")
        destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError as exc:
        if exc.errno not in {errno.EXDEV, errno.EPERM, errno.EACCES, errno.EOPNOTSUPP}:
            raise
    shutil.copy2(source, destination)
    return "copy"


def _source_root_for_task(task: str, *, base_root: Path, advection_root: Path) -> tuple[str, Path]:
    if task == "advection1d":
        return "official_advection_beta_provenance", advection_root
    return "base_light_v1", base_root


def build_full_task_root(
    *,
    base_root: Path,
    advection_root: Path,
    out_root: Path,
    manifest_json: Path,
    split: str = "val",
    overwrite: bool = False,
) -> dict[str, Any]:
    if split == "test":
        raise ValueError("Refusing to build a test split root in validation-only workflow")
    expected_names = {f"{task}_{split}.h5" for task in TASKS}
    out_root.mkdir(parents=True, exist_ok=True)
    unexpected_h5 = sorted(
        path for path in out_root.glob("*.h5") if path.name not in expected_names
    )
    if unexpected_h5 and not overwrite:
        names = ", ".join(path.name for path in unexpected_h5)
        raise ValueError(f"{out_root} contains unexpected HDF5 files: {names}")
    for path in unexpected_h5:
        path.unlink()

    sources: dict[str, dict[str, Any]] = {}
    for task in TASKS:
        source_root_kind, source_root = _source_root_for_task(
            task, base_root=base_root, advection_root=advection_root
        )
        source_path = source_root / f"{task}_{split}.h5"
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        beta_provenance = (
            _validate_advection_beta_provenance(source_path)
            if task == "advection1d"
            else {"required": False, "has_source_file_index": False, "has_source_paths": False}
        )
        destination = out_root / source_path.name
        transfer = _link_or_copy(source_path, destination, overwrite=overwrite)
        source_summary = _h5_summary(source_path)
        destination_summary = _h5_summary(destination)
        if source_summary["sha256"] != destination_summary["sha256"]:
            raise ValueError(f"{destination} hash differs from source {source_path}")
        sources[task] = {
            "task": task,
            "split": split,
            "source_root_kind": source_root_kind,
            "source_root": str(source_root),
            "source_path": str(source_path),
            "output_path": str(destination),
            "transfer": transfer,
            "bytes": destination_summary["bytes"],
            "sha256": destination_summary["sha256"],
            "source_sha256": source_summary["sha256"],
            "datasets": destination_summary["datasets"],
            "file_attrs": destination_summary["file_attrs"],
            "beta_provenance": beta_provenance,
        }

    manifest = {
        "measurement_type": MEASUREMENT_TYPE,
        "version": 1,
        "tasks": list(TASKS),
        "split": split,
        "base_root": str(base_root),
        "advection_root": str(advection_root),
        "out_root": str(out_root),
        "source_policy": (
            "Burgers and Darcy from base_root; Advection from official beta-provenance root; "
            "copies only the requested non-test split."
        ),
        "held_out_test_data_read": False,
        "test_ledger_writes": [],
        "sources": sources,
    }
    manifest_json.parent.mkdir(parents=True, exist_ok=True)
    manifest_json.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, default=Path("data/pdebench"))
    parser.add_argument(
        "--advection-root", type=Path, default=Path("data/pdebench_official_advection_light")
    )
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest = build_full_task_root(
        base_root=args.base_root,
        advection_root=args.advection_root,
        out_root=args.out_root,
        manifest_json=args.manifest_json,
        split=args.split,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
