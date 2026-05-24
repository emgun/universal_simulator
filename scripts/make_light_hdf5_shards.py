#!/usr/bin/env python
from __future__ import annotations

"""Build small PDEBench-style HDF5 train/val/test shards from larger local HDF5 files."""

import argparse
import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml


def _source_paths(root: Path, task: str, split: str) -> list[Path]:
    exact = root / f"{task}_{split}.h5"
    if exact.exists():
        return [exact]
    return sorted(root.glob(f"{task}_{split}_*.h5"))


def _resolve_source_split(root: Path, task: str, preferred_split: str, fallback_split: str) -> str:
    if _source_paths(root, task, preferred_split):
        return preferred_split
    if preferred_split != fallback_split and _source_paths(root, task, fallback_split):
        return fallback_split
    raise FileNotFoundError(root / f"{task}_{preferred_split}.h5")


def _dataset_keys(handle: h5py.File, sample_count: int) -> list[str]:
    keys: list[str] = []
    for key, value in handle.items():
        if isinstance(value, h5py.Dataset) and value.shape and int(value.shape[0]) == sample_count:
            keys.append(key)
    return keys


def _validate_source_complete(path: Path, handle: h5py.File) -> None:
    if "sequential_hydration_complete" not in handle.attrs:
        return
    if not bool(handle.attrs["sequential_hydration_complete"]):
        raise ValueError(
            f"{path} is marked sequential_hydration_complete=False; "
            "refusing to build light shards from partial official hydration"
        )


def _read_window(paths: Iterable[Path], start: int, count: int) -> dict[str, np.ndarray]:
    remaining = int(count)
    offset = int(start)
    chunks: dict[str, list[np.ndarray]] = {}
    attrs: dict[str, dict[str, object]] = {}
    file_attrs: dict[str, object] | None = None

    for path in paths:
        if remaining <= 0:
            break
        with h5py.File(path, "r") as handle:
            _validate_source_complete(path, handle)
            if file_attrs is None:
                file_attrs = dict(handle.attrs.items())
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
        raise ValueError(
            f"Requested {count} samples from offset {start}, but source files ran short by {remaining}"
        )

    merged = {key: np.concatenate(values, axis=0) for key, values in chunks.items()}
    for key, values in attrs.items():
        merged[f"__attrs__:{key}"] = values  # type: ignore[assignment]
    if file_attrs is not None:
        merged["__file_attrs__"] = file_attrs  # type: ignore[assignment]
    return merged


def _read_indices(paths: Iterable[Path], indices: list[int]) -> dict[str, np.ndarray]:
    if not indices:
        raise ValueError("Requested indexed shard with no indices")

    remaining = sorted((int(index), position) for position, index in enumerate(indices))
    chunks: dict[str, list[np.ndarray | None]] = {}
    attrs: dict[str, dict[str, object]] = {}
    file_attrs: dict[str, object] | None = None
    base = 0

    for path in paths:
        if not remaining:
            break
        with h5py.File(path, "r") as handle:
            _validate_source_complete(path, handle)
            if file_attrs is None:
                file_attrs = dict(handle.attrs.items())
            if "data" not in handle:
                raise KeyError(f"{path} does not contain a 'data' dataset")
            total = int(handle["data"].shape[0])
            local_rows = [
                (index - base, position)
                for index, position in remaining
                if base <= index < base + total
            ]
            if local_rows:
                keys = _dataset_keys(handle, total)
                local_indices = [row for row, _ in local_rows]
                order_positions = [position for _, position in local_rows]
                for key in keys:
                    dataset = handle[key]
                    selected = np.asarray(dataset[local_indices])
                    slots = chunks.setdefault(key, [None] * len(indices))
                    attrs.setdefault(key, dict(dataset.attrs.items()))
                    for source_row, target_position in enumerate(order_positions):
                        slots[target_position] = selected[source_row : source_row + 1]
            remaining = [
                (index, position) for index, position in remaining if index >= base + total
            ]
            base += total

    if remaining:
        missing = ", ".join(str(index) for index, _ in remaining[:5])
        raise ValueError(f"Requested sample indices beyond available source rows: {missing}")

    merged: dict[str, np.ndarray] = {}
    for key, values in chunks.items():
        if any(value is None for value in values):
            raise ValueError(f"Internal indexed read error for dataset {key}")
        merged[key] = np.concatenate([value for value in values if value is not None], axis=0)
    for key, values in attrs.items():
        merged[f"__attrs__:{key}"] = values  # type: ignore[assignment]
    if file_attrs is not None:
        merged["__file_attrs__"] = file_attrs  # type: ignore[assignment]
    return merged


def _source_sample_count(paths: Iterable[Path]) -> int:
    total = 0
    for path in paths:
        with h5py.File(path, "r") as handle:
            _validate_source_complete(path, handle)
            if "data" not in handle:
                raise KeyError(f"{path} does not contain a 'data' dataset")
            total += int(handle["data"].shape[0])
    return total


def _stratified_indices(*, block_size: int, block_count: int, offset: int, count: int) -> list[int]:
    if block_size <= 0:
        raise ValueError("Stratified block size must be positive")
    if offset < 0 or offset >= block_size:
        raise ValueError(f"Stratified offset {offset} must be in [0, {block_size})")
    if count <= 0:
        return []
    if block_count <= 0:
        raise ValueError("Stratified block count must be positive")
    per_block, remainder = divmod(count, block_count)
    if remainder:
        # This helper is intentionally strict because the official Advection path
        # uses equal per-beta blocks; partial final blocks would silently skew beta mix.
        raise ValueError(f"Stratified count {count} is not divisible by block count {block_count}")
    if per_block <= 0 or offset + per_block > block_size:
        raise ValueError(
            f"Cannot take {count} rows with offset {offset} from block size {block_size}"
        )
    return [
        block_start + row
        for block_start in range(0, block_count * block_size, block_size)
        for row in range(offset, offset + per_block)
    ]


def _write_h5(path: Path, arrays: dict[str, np.ndarray], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    attrs = {
        key.removeprefix("__attrs__:"): value
        for key, value in arrays.items()
        if key.startswith("__attrs__:")
    }
    file_attrs = arrays.get("__file_attrs__", {})
    with h5py.File(path, "w") as handle:
        for attr_key, attr_value in file_attrs.items():  # type: ignore[union-attr]
            handle.attrs[attr_key] = attr_value
        for key, array in arrays.items():
            if key.startswith("__attrs__:") or key == "__file_attrs__":
                continue
            dataset = handle.create_dataset(key, data=array, compression="gzip", compression_opts=3)
            for attr_key, attr_value in attrs.get(key, {}).items():
                dataset.attrs[attr_key] = attr_value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _h5_summary(path: Path) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    sample_count = 0
    with h5py.File(path, "r") as handle:
        for key, value in handle.items():
            if isinstance(value, h5py.Dataset):
                shape = [int(dim) for dim in value.shape]
                datasets[key] = {
                    "shape": shape,
                    "dtype": str(value.dtype),
                }
                if key == "data" and shape:
                    sample_count = shape[0]
    return {
        "sample_count": sample_count,
        "datasets": datasets,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _parse_split_sources(values: list[str] | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"Invalid --split-source '{item}'. Expected SPLIT=SOURCE_SPLIT")
        split, source = item.split("=", 1)
        split = split.strip()
        source = source.strip()
        if not split or not source:
            raise ValueError(f"Invalid --split-source '{item}'. Expected SPLIT=SOURCE_SPLIT")
        mapping[split] = source
    return mapping


def _parse_split_ints(values: list[str] | None, *, setting: str) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"Invalid {setting} '{item}'. Expected SPLIT=INTEGER")
        split, raw_value = item.split("=", 1)
        split = split.strip()
        raw_value = raw_value.strip()
        if not split or not raw_value:
            raise ValueError(f"Invalid {setting} '{item}'. Expected SPLIT=INTEGER")
        mapping[split] = int(raw_value)
    return mapping


def build_task_shard_records(
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
    split_sources: dict[str, str] | None = None,
    split_start_indices: dict[str, int] | None = None,
    split_block_size: int | None = None,
    split_block_offsets: dict[str, int] | None = None,
    fallback_source_split: str = "train",
    remote_prefix: str | None = None,
) -> list[dict[str, Any]]:
    split_sources = dict(split_sources or {})
    split_start_indices = dict(split_start_indices or {})
    split_block_offsets = dict(split_block_offsets or {})
    records: list[dict[str, Any]] = []
    offsets: dict[str, int] = {}

    for split, count in (("train", train_count), ("val", val_count), ("test", test_count)):
        if count <= 0:
            continue
        preferred_source = split_sources.get(split, split if split != "train" else source_split)
        resolved_source = _resolve_source_split(root, task, preferred_source, fallback_source_split)
        paths = _source_paths(root, task, resolved_source)
        if not paths:
            raise FileNotFoundError(root / f"{task}_{resolved_source}.h5")

        stratified_offset = split_block_offsets.get(split)
        cursor = split_start_indices.get(split)
        selected_indices: list[int] | None = None
        if stratified_offset is not None:
            if split_block_size is None:
                raise ValueError(f"--split-block-offset for {split} requires --split-block-size")
            total_rows = _source_sample_count(paths)
            block_count, remainder = divmod(total_rows, split_block_size)
            if remainder:
                raise ValueError(
                    f"Source rows {total_rows} are not divisible by stratified block size {split_block_size}"
                )
            selected_indices = _stratified_indices(
                block_size=split_block_size,
                block_count=block_count,
                offset=stratified_offset,
                count=count,
            )
            arrays = _read_indices(paths, selected_indices)
            cursor = stratified_offset
        else:
            if cursor is None:
                cursor = offsets.setdefault(resolved_source, int(start_index))
            arrays = _read_window(paths, cursor, count)
        out_path = out_root / f"{task}_{split}.h5"
        _write_h5(out_path, arrays, overwrite=overwrite)
        summary = _h5_summary(out_path)
        remote_key = (
            f"{remote_prefix.rstrip('/')}/{task}/{out_path.name}" if remote_prefix else None
        )
        records.append(
            {
                "task": task,
                "split": split,
                "source_split": resolved_source,
                "preferred_source_split": preferred_source,
                "derived_from_source_split": resolved_source != split,
                "source_paths": [str(path) for path in paths],
                "start_index": cursor,
                "stratified_block_size": (
                    split_block_size if stratified_offset is not None else None
                ),
                "stratified_block_offset": stratified_offset,
                "stratified_indices": selected_indices,
                "sample_count": summary["sample_count"],
                "output_path": str(out_path),
                "remote_key": remote_key,
                "bytes": summary["bytes"],
                "sha256": summary["sha256"],
                "datasets": summary["datasets"],
            }
        )
        if split not in split_start_indices and stratified_offset is None:
            offsets[resolved_source] = cursor + count
    return records


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
    split_sources: dict[str, str] | None = None,
    split_start_indices: dict[str, int] | None = None,
    split_block_size: int | None = None,
    split_block_offsets: dict[str, int] | None = None,
    fallback_source_split: str = "train",
) -> list[Path]:
    records = build_task_shard_records(
        root=root,
        out_root=out_root,
        task=task,
        source_split=source_split,
        train_count=train_count,
        val_count=val_count,
        test_count=test_count,
        start_index=start_index,
        overwrite=overwrite,
        split_sources=split_sources,
        split_start_indices=split_start_indices,
        split_block_size=split_block_size,
        split_block_offsets=split_block_offsets,
        fallback_source_split=fallback_source_split,
    )
    return [Path(str(record["output_path"])) for record in records]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create small HDF5 train/val/test shards for cheap UPS experiments"
    )
    parser.add_argument(
        "--root",
        default="data/pdebench",
        help="Directory containing source <task>_<split>.h5 files",
    )
    parser.add_argument(
        "--out-root", default="data/pdebench_light", help="Directory to write small shards"
    )
    parser.add_argument(
        "--tasks", nargs="+", required=True, help="Task names, e.g. burgers1d advection1d darcy2d"
    )
    parser.add_argument(
        "--source-split", default="train", help="Source split to slice, default train"
    )
    parser.add_argument(
        "--split-source",
        action="append",
        default=[],
        help="Optional split source mapping like val=train. Defaults to native split when present.",
    )
    parser.add_argument(
        "--split-start-index",
        action="append",
        default=[],
        help="Optional split-specific start index like train=1024. Overrides --start-index for that split.",
    )
    parser.add_argument(
        "--split-block-size",
        type=int,
        help="Optional ordered-source block size for stratified split slicing.",
    )
    parser.add_argument(
        "--split-block-offset",
        action="append",
        default=[],
        help="Optional stratified split offset like val=32. Requires --split-block-size.",
    )
    parser.add_argument(
        "--fallback-source-split",
        default="train",
        help="Fallback source split when native split is missing",
    )
    parser.add_argument("--train-count", type=int, default=16)
    parser.add_argument("--val-count", type=int, default=8)
    parser.add_argument("--test-count", type=int, default=8)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--manifest", help="Optional YAML manifest path to write")
    parser.add_argument("--version", default="light-local", help="Manifest/data version label")
    parser.add_argument(
        "--remote-prefix", help="Optional remote key prefix to record for each output"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    out_root = Path(args.out_root)
    split_sources = _parse_split_sources(args.split_source)
    split_start_indices = _parse_split_ints(args.split_start_index, setting="--split-start-index")
    split_block_offsets = _parse_split_ints(args.split_block_offset, setting="--split-block-offset")
    written: list[Path] = []
    records: list[dict[str, Any]] = []
    for task in args.tasks:
        task_records = build_task_shard_records(
            root=root,
            out_root=out_root,
            task=str(task),
            source_split=args.source_split,
            train_count=args.train_count,
            val_count=args.val_count,
            test_count=args.test_count,
            start_index=args.start_index,
            overwrite=args.overwrite,
            split_sources=split_sources,
            split_start_indices=split_start_indices,
            split_block_size=args.split_block_size,
            split_block_offsets=split_block_offsets,
            fallback_source_split=args.fallback_source_split,
            remote_prefix=args.remote_prefix,
        )
        records.extend(task_records)
        written.extend(Path(str(record["output_path"])) for record in task_records)

    for path in written:
        print(path)

    if args.manifest:
        manifest = {
            "version": args.version,
            "source_root": str(root),
            "out_root": str(out_root),
            "remote_prefix": args.remote_prefix,
            "tasks": [str(task) for task in args.tasks],
            "splits": {
                "train": {
                    "samples": args.train_count,
                    "preferred_source_split": split_sources.get("train", args.source_split),
                },
                "val": {
                    "samples": args.val_count,
                    "preferred_source_split": split_sources.get("val", "val"),
                    "fallback_source_split": args.fallback_source_split,
                },
                "test": {
                    "samples": args.test_count,
                    "preferred_source_split": split_sources.get("test", "test"),
                    "fallback_source_split": args.fallback_source_split,
                },
            },
            "split_start_indices": split_start_indices,
            "split_block_size": args.split_block_size,
            "split_block_offsets": split_block_offsets,
            "records": records,
        }
        manifest_path = Path(args.manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
        print(manifest_path)


if __name__ == "__main__":
    main()
