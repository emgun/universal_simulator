#!/usr/bin/env python3
"""
Convert raw PDEBench datasets (grid, mesh, particle) into UPS-friendly artifacts.

Grid-based tasks (e.g. Burgers, Advection, Navier–Stokes) are consolidated into
single HDF5 files using the streaming converter from `ups.data.convert_pdebench`.
Mesh and particle tasks are exported to Zarr stores compatible with the loaders
implemented in `ups.data.datasets`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional

import numpy as np
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:  # pragma: no cover
    import h5py  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit("h5py is required: install with `pip install h5py`.") from exc

try:  # pragma: no cover
    import zarr  # type: ignore
except ImportError:  # pragma: no cover
    zarr = None  # type: ignore

try:  # pragma: no cover
    from scipy.sparse import csr_matrix  # type: ignore
except ImportError:  # pragma: no cover
    csr_matrix = None  # type: ignore

from ups.data.convert_pdebench import convert_files

if TYPE_CHECKING:  # pragma: no cover - typing only
    import zarr as zarr_mod


TaskConfig = Dict[str, object]


DEFAULT_TASKS: Dict[str, TaskConfig] = {
    "burgers1d": {
        "kind": "grid",
        "pattern": "1D/Burgers/{split}/*.h5",
    },
    "advection1d": {
        "kind": "grid",
        "pattern": "1D/Advection/{split}/*.h5",
    },
    "navier_stokes2d": {
        "kind": "grid",
        "pattern": "2D/NavierStokes/{split}/*.h5",
    },
    "darcy2d": {
        "kind": "grid",
        "pattern": "2D/DarcyFlow/regular/{split}/*.h5",
    },
    "darcy2d_mesh": {
        "kind": "mesh",
        "pattern": "2D/DarcyFlow/irregular/{split}/*.npz",
    },
    "particles_advect": {
        "kind": "particles",
        "pattern": "Particles/Advection/{split}/*.npz",
    },
}


def _glob(root: Path, pattern: str) -> List[Path]:
    return sorted(root.glob(pattern))


def _select_inputs(root: Path, pattern: str, limit: Optional[int]) -> List[Path]:
    files = _glob(root, pattern)
    if limit is not None:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"No input files matched '{pattern}' under {root}")
    return files


def _convert_grid_task(
    *,
    task: str,
    split: str,
    root: Path,
    out_dir: Path,
    pattern: str,
    limit: Optional[int],
    sample_size: Optional[int],
    chunk_size: Optional[int],
) -> Path:
    files = _select_inputs(root, pattern.format(split=split), limit)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{task}_{split}.h5"
    convert_files(
        files,
        out_path,
        limit=limit,
        sample_size=sample_size,
        chunk_size=chunk_size,
    )
    return out_path


def _write_mesh_sample(group: "zarr_mod.Group", data: Dict[str, np.ndarray], index: int) -> None:
    if csr_matrix is None:
        raise RuntimeError("SciPy is required to convert mesh datasets.")
    sample = group.create_group(f"sample_{index:05d}")
    sample.create_dataset("coords", data=data["coords"].astype(np.float32), dtype="f4")
    sample.create_dataset("edges", data=data["edges"].astype(np.int32), dtype="i4")
    sample.create_dataset("cells", data=data["cells"].astype(np.int32), dtype="i4")

    laplacian = csr_matrix(
        (data["laplacian_data"], data["laplacian_indices"], data["laplacian_indptr"]),
        shape=(data["coords"].shape[0], data["coords"].shape[0]),
    )
    lap_group = sample.create_group("laplacian")
    lap_group.create_dataset("data", data=laplacian.data.astype(np.float32), dtype="f4")
    lap_group.create_dataset("indices", data=laplacian.indices.astype(np.int32), dtype="i4")
    lap_group.create_dataset("indptr", data=laplacian.indptr.astype(np.int32), dtype="i4")


def _convert_mesh_task(
    *,
    task: str,
    split: str,
    root: Path,
    out_dir: Path,
    pattern: str,
    limit: Optional[int],
) -> Path:
    files = _select_inputs(root, pattern.format(split=split), limit)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{task}_{split}.zarr"

    if zarr is None:
        raise RuntimeError("The 'zarr' package is required to convert mesh datasets.")
    store = zarr.open(str(out_path), mode="w")
    group = store.create_group(task)
    group.attrs["kind"] = "mesh"

    for idx, file_path in enumerate(files):
        with np.load(file_path) as npz:
            expected_keys = {"coords", "edges", "cells", "laplacian_data", "laplacian_indices", "laplacian_indptr"}
            missing = expected_keys - set(npz.keys())
            if missing:
                raise KeyError(f"File {file_path} missing required keys: {sorted(missing)}")
            sample_data = {key: npz[key] for key in expected_keys}
            _write_mesh_sample(group, sample_data, idx)

    return out_path


def _write_particle_sample(group: "zarr_mod.Group", data: Dict[str, np.ndarray], index: int, radius: float) -> None:
    sample = group.create_group(f"sample_{index:05d}")
    sample.create_dataset("positions", data=data["positions"].astype(np.float32), dtype="f4")
    sample.create_dataset("velocities", data=data["velocities"].astype(np.float32), dtype="f4")

    neighbours = sample.create_group("neighbors")
    neighbours.create_dataset("indices", data=data["indices"].astype(np.int32), dtype="i4")
    neighbours.create_dataset("indptr", data=data["indptr"].astype(np.int32), dtype="i4")
    neighbours.create_dataset("edges", data=data["edges"].astype(np.int32), dtype="i4")
    neighbours.attrs["radius"] = float(radius)


def _convert_particle_task(
    *,
    task: str,
    split: str,
    root: Path,
    out_dir: Path,
    pattern: str,
    limit: Optional[int],
    radius: float,
) -> Path:
    files = _select_inputs(root, pattern.format(split=split), limit)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{task}_{split}.zarr"

    if zarr is None:
        raise RuntimeError("The 'zarr' package is required to convert particle datasets.")
    store = zarr.open(str(out_path), mode="w")
    group = store.create_group(task)
    group.attrs["kind"] = "particles"

    expected_keys = {"positions", "velocities", "indices", "indptr", "edges"}

    for idx, file_path in enumerate(files):
        with np.load(file_path) as npz:
            missing = expected_keys - set(npz.keys())
            if missing:
                raise KeyError(f"File {file_path} missing required keys: {sorted(missing)}")
            sample_data = {key: npz[key] for key in expected_keys}
            _write_particle_sample(group, sample_data, idx, radius)

    return out_path


def convert_task(
    *,
    task: str,
    split: str,
    root: Path,
    out_dir: Path,
    config: TaskConfig,
    limit: Optional[int],
    sample_size: Optional[int],
    chunk_size: Optional[int],
    radius: float,
) -> Path:
    kind = config["kind"]
    pattern = config["pattern"]
    if kind == "grid":
        return _convert_grid_task(
            task=task,
            split=split,
            root=root,
            out_dir=out_dir,
            pattern=str(pattern),
            limit=limit,
            sample_size=sample_size,
            chunk_size=chunk_size,
        )
    if kind == "mesh":
        return _convert_mesh_task(
            task=task,
            split=split,
            root=root,
            out_dir=out_dir,
            pattern=str(pattern),
            limit=limit,
        )
    if kind == "particles":
        return _convert_particle_task(
            task=task,
            split=split,
            root=root,
            out_dir=out_dir,
            pattern=str(pattern),
            limit=limit,
            radius=radius,
        )
    raise ValueError(f"Unsupported task kind '{kind}' for task '{task}'")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert PDEBench datasets into UPS-friendly artifacts.")
    parser.add_argument("task", help="Task name (e.g. burgers1d, darcy2d_mesh, particles_advect).")
    parser.add_argument("--root", required=True, help="Directory containing raw PDEBench files.")
    parser.add_argument("--out", required=True, help="Output directory for converted artifacts.")
    parser.add_argument("--split", default="train", help="Dataset split to process (train/val/test).")
    parser.add_argument("--limit", type=int, help="Optional limit on the number of shards/files to process.")
    parser.add_argument("--samples", type=int, help="Optional limit on samples per file (grid tasks only).")
    parser.add_argument("--chunk-size", type=int, help="Chunk size override for grid conversion.")
    parser.add_argument("--radius", type=float, default=0.2, help="Neighbourhood radius for particle datasets.")
    parser.add_argument("--pattern", help="Override glob pattern if defaults do not match.")
    parser.add_argument("--tasks-json", help="Optional JSON file describing task configs.")
    return parser


def _load_task_configs(tasks_json: Optional[str]) -> Dict[str, TaskConfig]:
    if tasks_json is None:
        return dict(DEFAULT_TASKS)
    with open(tasks_json, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("Task configuration JSON must define an object at the top level.")
    return {str(k): dict(v) for k, v in data.items()}


def main(namespace: Optional[argparse.Namespace] = None) -> Path:
    parser = _build_parser()
    args = namespace or parser.parse_args()
    tasks = _load_task_configs(args.tasks_json)

    task_name = args.task
    if task_name not in tasks:
        raise KeyError(f"Unknown task '{task_name}'. Provide --pattern or a JSON config file.")

    config = dict(tasks[task_name])
    if args.pattern:
        config["pattern"] = args.pattern

    if "kind" not in config or "pattern" not in config:
        raise ValueError(f"Task configuration for '{task_name}' must include 'kind' and 'pattern' fields.")

    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()

    return convert_task(
        task=task_name,
        split=args.split,
        root=root,
        out_dir=out_dir,
        config=config,
        limit=args.limit,
        sample_size=args.samples,
        chunk_size=args.chunk_size,
        radius=args.radius,
    )


if __name__ == "__main__":
    output_path = main()
    print(f"✅ Converted dataset written to {output_path}")
