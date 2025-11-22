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
        "pattern": "1D/Advection/Train/*.hdf5",  # All data in Train/
    },
    "diffusion_sorption1d": {
        "kind": "grid",
        "pattern": "1D/diffusion-sorption/*.hdf5",
    },
    "reaction_diffusion1d": {
        "kind": "grid",
        "pattern": "1D/ReactionDiffusion/{split}/*.hdf5",
    },
    "navier_stokes2d": {
        "kind": "grid",
        "pattern": "2D/NS_incom/{split}/*.hdf5",
    },
    "darcy2d": {
        "kind": "grid",
        "pattern": "2D/DarcyFlow/*.hdf5",  # All data in root
    },
    "reaction_diffusion2d": {
        "kind": "grid",
        "pattern": "2D/diffusion-reaction/{split}/*.hdf5",
    },
    "shallow_water2d": {
        "kind": "grid",
        "pattern": "2D/shallow-water/{split}/*.hdf5",
    },
    "cfd1d_shocktube": {
        "kind": "grid",
        "pattern": "1D/CFD/{split}/**/*.hdf5",  # Train and Test/ShockTube
    },
    "cfd2d_rand": {
        "kind": "grid",
        "pattern": "2D/CFD/2D_Train_Rand/*.hdf5",  # Train only, synthesize val/test
    },
    "cfd2d_turb": {
        "kind": "grid",
        "pattern": "2D/CFD/2D_Train_Turb/*.hdf5",  # Train only, synthesize val/test
    },
    "cfd3d": {
        "kind": "grid",
        "pattern": "3D/Train/*.hdf5",  # Train only, synthesize val/test
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


def _select_inputs(root: Path, pattern: str, limit: Optional[int], strict: bool = True) -> List[Path]:
    files = _glob(root, pattern)
    if not files and strict:
        raise FileNotFoundError(f"No input files matched '{pattern}' under {root}")
    if limit is not None:
        files = files[:limit]
    return files


def _resolve_files(root: Path, pattern: str, split: str, limit: Optional[int]) -> List[Path]:
    """Find files for a split, handling case-sensitivity (Train vs train)."""
    # Try exact match first
    try:
        return _select_inputs(root, pattern.format(split=split), limit)
    except FileNotFoundError:
        pass

    # Try Capitalized match
    try:
        return _select_inputs(root, pattern.format(split=split.capitalize()), limit)
    except FileNotFoundError:
        pass
        
    # If we are looking for validation/test and found nothing, it might be 
    # that the pattern doesn't support splits (e.g. hardcoded 'Train').
    # We return empty list to let the caller decide if this is an error or 
    # a signal to use synthetic splitting.
    return []


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
    allow_synth: bool = True,
) -> Path:
    # 1. Find files for the requested split
    files = _resolve_files(root, pattern, split, limit=None) # Don't limit yet, we need full list for splitting
    
    # 2. Find files for the 'train' split (reference)
    train_files = _resolve_files(root, pattern, "train", limit=None)
    
    if not train_files:
         # If we can't even find train files, something is wrong with the pattern or root
         raise FileNotFoundError(f"No training files found for task '{task}' with pattern '{pattern}' at {root}")

    # 3. Detect if we need synthetic splitting
    # If the requested files are identical to train files (and we aren't asking for train),
    # OR if we found no files for the requested split but we have train files (implied fallback),
    # then we slice the train_files.
    use_synthetic_split = (split != "train") and (
        not files or files == train_files
    )
    
    if use_synthetic_split:
        # Deterministic split: 80% Train, 10% Val, 10% Test
        # We sort to ensure determinism across runs
        all_files = sorted(train_files)
        n_files = len(all_files)
        
        if split == "val":
            start = int(n_files * 0.8)
            end = int(n_files * 0.9)
            files = all_files[start:end]
        elif split == "test":
            start = int(n_files * 0.9)
            end = n_files
            files = all_files[start:end]
        else:
            # Should not happen given the condition, but safe fallback
            files = []
            
        print(f"  ℹ️  Synthetic split for '{task} {split}': using files {start} to {end} (of {n_files})")
        if not allow_synth:
            raise FileNotFoundError(
                f"Synthetic split required for '{task}' ({split}); rerun without --no-synth-splits or provide explicit split files."
            )
        
    elif split == "train" and (files == train_files):
         # We are processing train, and it matches the train set (obviously).
         # But if we are in a synthetic split scenario (implied by the patterns being hardcoded),
         # we should ONLY take the first 80%.
         # How do we know? We check if "val" would return the same files.
         val_files = _resolve_files(root, pattern, "val", limit=None)
         if not val_files or val_files == train_files:
             # Synthetic split applies to train too!
             n_files = len(train_files)
             end = int(n_files * 0.8)
             files = train_files[:end]
             print(f"  ℹ️  Synthetic split for '{task} {split}': using files 0 to {end} (of {n_files})")
             if not allow_synth:
                 raise FileNotFoundError(
                     f"Synthetic split required for '{task}' (train); rerun without --no-synth-splits or provide explicit split files."
                 )

    if not files:
        raise FileNotFoundError(f"No files found for split '{split}' (and synthetic fallback failed)")

    # Apply limit after splitting
    if limit is not None:
        files = files[:limit]

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
    allow_synth: bool = True,
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
            allow_synth=allow_synth,
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
    parser.add_argument(
        "--no-synth-splits",
        action="store_true",
        help="Disable synthetic val/test splitting; error if splits are missing.",
    )
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
        allow_synth=not args.no_synth_splits,
    )


if __name__ == "__main__":
    output_path = main()
    print(f"✅ Converted dataset written to {output_path}")
