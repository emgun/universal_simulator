#!/usr/bin/env python
from __future__ import annotations

"""CLI wrappers to convert various PDEBench modalities into UPS-ready datasets."""

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import zarr

from ups.data.convert_pdebench import convert_files


DATASETS = {
    "burgers1d": {
        "pattern": "1D/Burgers/Train/*.hdf5",
        "out": "burgers1d_train.h5",
        "type": "grid",
        "description": "1D Burgers grid trajectories",
    },
    "advection1d": {
        "pattern": "1D/Advection/Train/*.hdf5",
        "out": "advection1d_train.h5",
        "type": "grid",
        "description": "1D Advection grid trajectories",
    },
    "navier_stokes2d": {
        "pattern": "2D/NavierStokes/Train/*.hdf5",
        "out": "navier_stokes2d_train.h5",
        "type": "grid",
        "description": "2D Navier–Stokes grid trajectories",
    },
    "darcy2d_mesh": {
        "pattern": "2D/Darcy/**/*.{npz,npy}",
        "out": "darcy2d_train.zarr",
        "type": "npz",
        "description": "Darcy flow mesh samples (npz).",
    },
    "particles_advect": {
        "pattern": "Particles/Advection/**/*.npz",
        "out": "particles_advect_train.zarr",
        "type": "npz",
        "description": "Particle advection trajectories (npz).",
    },
}


def _glob_files(root: Path, pattern: str, limit: int | None) -> list[Path]:
    files = sorted(p for p in root.glob(pattern) if p.is_file())
    if limit is not None:
        files = files[:limit]
    return files


def convert_grid(pattern: str, out_path: Path, *, limit: int | None, samples: int | None, root: Path) -> int:
    files = _glob_files(root, pattern, limit)
    if not files:
        raise SystemExit(f"No files matched pattern: {root / pattern}")
    return convert_files(files, out_path, sample_size=samples)


def convert_npz(pattern: str, out_path: Path, *, limit: int | None, root: Path) -> int:
    files = _glob_files(root, pattern, limit)
    if not files:
        raise SystemExit(f"No files matched pattern: {root / pattern}")
    if out_path.exists():
        import shutil

        if out_path.is_dir():
            shutil.rmtree(out_path)
        else:
            out_path.unlink()
    store = zarr.open(out_path, mode="w")
    compressor = zarr.Blosc(cname="zstd", clevel=5, shuffle=zarr.Blosc.BITSHUFFLE)
    keys: list[str] | None = None
    count = 0
    for file in files:
        with np.load(file) as data:
            if keys is None:
                keys = list(data.keys())
                for key in keys:
                    store.create_dataset(key, shape=(0, *data[key].shape), maxshape=(None, *data[key].shape), chunks=(1, *data[key].shape), dtype=data[key].dtype, compressor=compressor)
            else:
                missing = set(keys) - set(data.keys())
                if missing:
                    raise ValueError(f"File {file} missing keys {missing}")
            for key in keys:
                arr = data[key]
                dset = store[key]
                new_len = dset.shape[0] + 1
                dset.resize(new_len, axis=0)
                dset[new_len - 1] = arr
        count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert multiple PDEBench datasets into UPS format")
    parser.add_argument("dataset", choices=DATASETS.keys(), help="Dataset key")
    parser.add_argument("--root", default="data/pdebench/raw", help="Root directory containing raw PDEBench data")
    parser.add_argument("--out", default="data/pdebench", help="Output directory for converted datasets")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of files")
    parser.add_argument("--samples", type=int, default=None, help="Optional limit on samples per file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entry = DATASETS[args.dataset]
    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / entry["out"]
    if entry["type"] == "grid":
        written = convert_grid(entry["pattern"], out_path, limit=args.limit, samples=args.samples, root=root)
    elif entry["type"] == "npz":
        written = convert_npz(entry["pattern"], out_path, limit=args.limit, root=root)
    else:
        raise SystemExit(f"Unsupported dataset type: {entry['type']}")
    desc = entry.get("description", "")
    suffix = f" — {desc}" if desc else ""
    print(f"Converted {args.dataset}{suffix} → {out_path} ({written} files)")


if __name__ == "__main__":
    main()
