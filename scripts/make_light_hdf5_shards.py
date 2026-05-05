#!/usr/bin/env python
from __future__ import annotations

"""Build small PDEBench-style HDF5 train/val/test shards from larger local HDF5 files."""

import argparse
import hashlib
from pathlib import Path
from typing import Any, Iterable

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
    fallback_source_split: str = "train",
    remote_prefix: str | None = None,
) -> list[dict[str, Any]]:
    split_sources = dict(split_sources or {})
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

        cursor = offsets.setdefault(resolved_source, int(start_index))
        arrays = _read_window(paths, cursor, count)
        out_path = out_root / f"{task}_{split}.h5"
        _write_h5(out_path, arrays, overwrite=overwrite)
        summary = _h5_summary(out_path)
        remote_key = f"{remote_prefix.rstrip('/')}/{task}/{out_path.name}" if remote_prefix else None
        records.append(
            {
                "task": task,
                "split": split,
                "source_split": resolved_source,
                "preferred_source_split": preferred_source,
                "derived_from_source_split": resolved_source != split,
                "source_paths": [str(path) for path in paths],
                "start_index": cursor,
                "sample_count": summary["sample_count"],
                "output_path": str(out_path),
                "remote_key": remote_key,
                "bytes": summary["bytes"],
                "sha256": summary["sha256"],
                "datasets": summary["datasets"],
            }
        )
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
        fallback_source_split=fallback_source_split,
    )
    return [Path(str(record["output_path"])) for record in records]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create small HDF5 train/val/test shards for cheap UPS experiments")
    parser.add_argument("--root", default="data/pdebench", help="Directory containing source <task>_<split>.h5 files")
    parser.add_argument("--out-root", default="data/pdebench_light", help="Directory to write small shards")
    parser.add_argument("--tasks", nargs="+", required=True, help="Task names, e.g. burgers1d advection1d darcy2d")
    parser.add_argument("--source-split", default="train", help="Source split to slice, default train")
    parser.add_argument(
        "--split-source",
        action="append",
        default=[],
        help="Optional split source mapping like val=train. Defaults to native split when present.",
    )
    parser.add_argument("--fallback-source-split", default="train", help="Fallback source split when native split is missing")
    parser.add_argument("--train-count", type=int, default=16)
    parser.add_argument("--val-count", type=int, default=8)
    parser.add_argument("--test-count", type=int, default=8)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--manifest", help="Optional YAML manifest path to write")
    parser.add_argument("--version", default="light-local", help="Manifest/data version label")
    parser.add_argument("--remote-prefix", help="Optional remote key prefix to record for each output")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    out_root = Path(args.out_root)
    split_sources = _parse_split_sources(args.split_source)
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
                "train": {"samples": args.train_count, "preferred_source_split": split_sources.get("train", args.source_split)},
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
            "records": records,
        }
        manifest_path = Path(args.manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
        print(manifest_path)


if __name__ == "__main__":
    main()
