#!/usr/bin/env python
from __future__ import annotations

"""Build small PDEBench-style HDF5 train/val/test shards from larger local HDF5 files."""

import argparse
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


def _source_paths(root: Path, task: str, split: str) -> list[Path]:
    exact = root / f"{task}_{split}.h5"
    if exact.exists():
        return [exact]
    return sorted(root.glob(f"{task}_{split}_*.h5"))


def _dataset_keys(handle: h5py.File, sample_count: int) -> list[str]:
    keys: list[str] = []
    for key, value in handle.items():
        if isinstance(value, h5py.Dataset) and value.shape and int(value.shape[0]) == sample_count:
            keys.append(key)
    return keys


def _read_window(paths: Iterable[Path], start: int, count: int) -> dict[str, np.ndarray]:
    remaining = int(count)
    offset = int(start)
    chunks: dict[str, list[np.ndarray]] = {}
    attrs: dict[str, dict[str, object]] = {}

    for path in paths:
        if remaining <= 0:
            break
        with h5py.File(path, "r") as handle:
            if "data" not in handle:
                raise KeyError(f"{path} does not contain a 'data' dataset")
            total = int(handle["data"].shape[0])
            if offset >= total:
                offset -= total
                continue
            take = min(remaining, total - offset)
            keys = _dataset_keys(handle, total)
            for key in keys:
                dataset = handle[key]
                chunks.setdefault(key, []).append(dataset[offset : offset + take])
                attrs.setdefault(key, dict(dataset.attrs.items()))
            remaining -= take
            offset = 0

    if remaining > 0:
        raise ValueError(f"Requested {count} samples from offset {start}, but source files ran short by {remaining}")

    merged = {key: np.concatenate(values, axis=0) for key, values in chunks.items()}
    for key, values in attrs.items():
        merged[f"__attrs__:{key}"] = values  # type: ignore[assignment]
    return merged


def _write_h5(path: Path, arrays: dict[str, np.ndarray], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    attrs = {key.removeprefix("__attrs__:"): value for key, value in arrays.items() if key.startswith("__attrs__:")}
    with h5py.File(path, "w") as handle:
        for key, array in arrays.items():
            if key.startswith("__attrs__:"):
                continue
            dataset = handle.create_dataset(key, data=array, compression="gzip", compression_opts=3)
            for attr_key, attr_value in attrs.get(key, {}).items():
                dataset.attrs[attr_key] = attr_value


def build_task_shards(
    *,
    root: Path,
    out_root: Path,
    task: str,
    source_split: str,
    train_count: int,
    val_count: int,
    test_count: int,
    start_index: int,
    overwrite: bool,
) -> list[Path]:
    paths = _source_paths(root, task, source_split)
    if not paths:
        raise FileNotFoundError(root / f"{task}_{source_split}.h5")

    outputs: list[Path] = []
    cursor = int(start_index)
    for split, count in (("train", train_count), ("val", val_count), ("test", test_count)):
        if count <= 0:
            continue
        arrays = _read_window(paths, cursor, count)
        out_path = out_root / f"{task}_{split}.h5"
        _write_h5(out_path, arrays, overwrite=overwrite)
        outputs.append(out_path)
        cursor += count
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Create small HDF5 train/val/test shards for cheap UPS experiments")
    parser.add_argument("--root", default="data/pdebench", help="Directory containing source <task>_<split>.h5 files")
    parser.add_argument("--out-root", default="data/pdebench_light", help="Directory to write small shards")
    parser.add_argument("--tasks", nargs="+", required=True, help="Task names, e.g. burgers1d advection1d darcy2d")
    parser.add_argument("--source-split", default="train", help="Source split to slice, default train")
    parser.add_argument("--train-count", type=int, default=16)
    parser.add_argument("--val-count", type=int, default=8)
    parser.add_argument("--test-count", type=int, default=8)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    out_root = Path(args.out_root)
    written: list[Path] = []
    for task in args.tasks:
        written.extend(
            build_task_shards(
                root=root,
                out_root=out_root,
                task=str(task),
                source_split=args.source_split,
                train_count=args.train_count,
                val_count=args.val_count,
                test_count=args.test_count,
                start_index=args.start_index,
                overwrite=args.overwrite,
            )
        )

    for path in written:
        print(path)


if __name__ == "__main__":
    main()
