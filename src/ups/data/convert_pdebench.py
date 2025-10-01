from __future__ import annotations

"""Utilities to convert raw PDEBench HDF5 shards into UPS-ready HDF5 files."""

from pathlib import Path
from typing import Iterable, List, Tuple

import h5py
import numpy as np


def _find_largest_dataset(h5: h5py.File) -> Tuple[str, h5py.Dataset]:
    best_name = None
    best_size = -1
    best_data = None

    def _visit(name, obj):
        nonlocal best_name, best_size, best_data
        if isinstance(obj, h5py.Dataset) and obj.dtype.kind in ("f", "i") and obj.ndim >= 2:
            size = int(np.prod(obj.shape))
            if size > best_size:
                best_size = size
                best_name = name
                best_data = obj

    h5.visititems(_visit)
    if best_data is None:
        raise ValueError("No suitable numeric dataset (ndim>=2) found in file")
    return best_name, best_data


def _normalise_batch(arr: np.ndarray) -> np.ndarray:
    """Ensure batch arrays have an explicit channel dimension."""

    if arr.ndim == 2:
        # (samples, features) → (samples, features, 1)
        return arr[:, :, None]
    if arr.ndim == 3:
        # (samples, T, X) → (samples, T, X, 1)
        return arr[..., None]
    if arr.ndim == 4:
        # (samples, T, H, W) → (samples, T, H, W, 1)
        return arr[..., None]
    return arr


def convert_files(
    input_paths: Iterable[Path],
    out_path: Path,
    *,
    limit: int | None = None,
    sample_size: int | None = None,
    chunk_size: int | None = None,
) -> int:
    """Write a consolidated HDF5 file and return the number of samples stored."""

    files: List[Path] = [Path(p) for p in input_paths]
    if not files:
        raise ValueError("No input files provided")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_written = 0
    target_shape: Tuple[int, ...] | None = None
    dset: h5py.Dataset | None = None

    with h5py.File(out_path, "w") as out_h5:
        for index, path in enumerate(files):
            if limit is not None and index >= limit:
                break

            with h5py.File(path, "r") as in_h5:
                _, src = _find_largest_dataset(in_h5)
                num_samples = src.shape[0]
                stop = min(sample_size, num_samples) if sample_size is not None else num_samples
                if stop == 0:
                    continue

                if chunk_size is not None:
                    step = max(1, min(chunk_size, stop))
                else:
                    inferred = src.chunks[0] if src.chunks else 64
                    step = max(1, min(inferred, stop))
                for start in range(0, stop, step):
                    end = min(start + step, stop)
                    chunk = np.asarray(src[start:end])
                    chunk = _normalise_batch(chunk).astype(np.float32, copy=False)

                    if target_shape is None:
                        target_shape = chunk.shape[1:]
                        dset = out_h5.create_dataset(
                            "data",
                            shape=(0, *target_shape),
                            maxshape=(None, *target_shape),
                            dtype=np.float32,
                            chunks=(chunk.shape[0], *target_shape),
                        )
                    elif chunk.shape[1:] != target_shape:
                        raise ValueError(
                            "Inconsistent sample shape: "
                            f"{chunk.shape[1:]} vs {target_shape} from {path}"
                        )

                    assert dset is not None  # mypy guard
                    next_size = total_written + chunk.shape[0]
                    dset.resize(next_size, axis=0)
                    dset[total_written:next_size] = chunk
                    total_written = next_size

    if total_written == 0:
        # Remove the empty file to signal the caller something went wrong.
        out_path.unlink(missing_ok=True)
        raise ValueError("No samples written; check inputs and sample_size")

    return total_written
