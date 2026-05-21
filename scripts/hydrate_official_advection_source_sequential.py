#!/usr/bin/env python
from __future__ import annotations

"""Sequentially hydrate a sampled official Advection source shard.

The regular official plan downloads all raw train files before conversion.
This entrypoint downloads one official file, appends the sampled rows to the
UPS-ready source HDF5, then optionally removes the raw file before moving to the
next beta file. It preserves the train-only provenance fields used by the
source-conditioned transport gate while reducing scratch disk requirements.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ups.data.convert_pdebench import _find_largest_dataset, _normalise_batch


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _raw_path(raw_root: Path, logical_path: str) -> Path:
    return raw_root / logical_path


def _source_paths(entries: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [str(entry.get("path") or "") for entry in entries],
        dtype=h5py.string_dtype("utf-8"),
    )


def _initialize_source_attrs(out_path: Path, entries: list[dict[str, Any]], *, overwrite: bool) -> None:
    if out_path.exists() and overwrite:
        out_path.unlink()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "a") as out_h5:
        out_h5.attrs["source_paths"] = _source_paths(entries)
        out_h5.attrs["sequential_hydration_complete"] = False


def _mark_complete(out_path: Path, entries: list[dict[str, Any]]) -> None:
    with h5py.File(out_path, "a") as out_h5:
        out_h5.attrs["source_paths"] = _source_paths(entries)
        out_h5.attrs["sequential_hydration_complete"] = True


def _append_samples(
    *,
    source_path: Path,
    out_path: Path,
    source_file_index: int,
    sample_count: int,
) -> int:
    with h5py.File(source_path, "r") as in_h5:
        _, src = _find_largest_dataset(in_h5)
        stop = min(int(sample_count), int(src.shape[0]))
        if stop <= 0:
            return 0
        chunk = _normalise_batch(np.asarray(src[:stop])).astype(np.float32, copy=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "a") as out_h5:
        if "data" not in out_h5:
            data = out_h5.create_dataset(
                "data",
                shape=(0, *chunk.shape[1:]),
                maxshape=(None, *chunk.shape[1:]),
                dtype=np.float32,
                chunks=(chunk.shape[0], *chunk.shape[1:]),
            )
            source_file = out_h5.create_dataset(
                "source_file_index",
                shape=(0,),
                maxshape=(None,),
                dtype=np.int32,
                chunks=(chunk.shape[0],),
            )
            source_sample = out_h5.create_dataset(
                "source_sample_index",
                shape=(0,),
                maxshape=(None,),
                dtype=np.int64,
                chunks=(chunk.shape[0],),
            )
        else:
            data = out_h5["data"]
            source_file = out_h5["source_file_index"]
            source_sample = out_h5["source_sample_index"]
            if tuple(data.shape[1:]) != tuple(chunk.shape[1:]):
                raise ValueError(
                    f"Inconsistent sample shape from {source_path}: {chunk.shape[1:]} vs {data.shape[1:]}"
                )

        start = int(data.shape[0])
        end = start + int(chunk.shape[0])
        data.resize(end, axis=0)
        source_file.resize(end, axis=0)
        source_sample.resize(end, axis=0)
        data[start:end] = chunk
        source_file[start:end] = int(source_file_index)
        source_sample[start:end] = np.arange(0, int(chunk.shape[0]), dtype=np.int64)
        return int(chunk.shape[0])


def hydrate_sequential(args: argparse.Namespace) -> dict[str, Any]:
    plan = _load_json(args.plan_json)
    entries = list(plan.get("remote_entries") or [])
    raw_root = Path(args.raw_out or plan.get("raw_out") or "data/pdebench/raw")
    out_root = Path(args.hydrated_source_root or plan.get("hydrated_source_root") or "data/pdebench_official_advection_hydrated")
    out_path = out_root / "advection1d_train.h5"
    sample_count = int(args.samples_per_file or plan.get("samples_per_file") or 0)
    blockers: list[str] = []
    records: list[dict[str, Any]] = []

    if not entries:
        blockers.append("plan has no remote_entries")
    if sample_count <= 0:
        blockers.append("plan has no positive samples_per_file")
    if out_path.exists() and not args.overwrite and args.execute:
        blockers.append(f"{out_path} exists; pass --overwrite to replace it")
    if not args.execute_downloads:
        blockers.append("sequential hydration requires --execute-downloads")

    should_execute = bool(args.execute and not blockers)
    source_paths_initialized = False
    if should_execute and args.overwrite:
        _initialize_source_attrs(out_path, entries, overwrite=True)
        source_paths_initialized = True
    elif should_execute:
        _initialize_source_attrs(out_path, entries, overwrite=False)
        source_paths_initialized = True

    for index, entry in enumerate(entries):
        logical_path = str(entry.get("path") or "")
        raw_path = _raw_path(raw_root, logical_path)
        record: dict[str, Any] = {
            "source_file_index": index,
            "logical_path": logical_path,
            "raw_path": str(raw_path),
            "download_executed": False,
            "append_executed": False,
            "samples_appended": 0,
            "raw_removed": False,
        }
        if should_execute:
            download_cmd = [
                "python",
                "scripts/download_pdebench_file.py",
                logical_path,
                "--out",
                str(raw_root),
            ]
            completed = subprocess.run(download_cmd, check=False)
            record["download_executed"] = True
            record["download_returncode"] = completed.returncode
            if completed.returncode != 0:
                blockers.append(f"download failed for {logical_path} with exit code {completed.returncode}")
                should_execute = False
            else:
                written = _append_samples(
                    source_path=raw_path,
                    out_path=out_path,
                    source_file_index=index,
                    sample_count=sample_count,
                )
                record["append_executed"] = True
                record["samples_appended"] = written
                if args.cleanup_raw:
                    raw_path.unlink(missing_ok=True)
                    record["raw_removed"] = True
        records.append(record)

    if should_execute and not blockers:
        _mark_complete(out_path, entries)

    status = "executed" if args.execute and not blockers else "dry_run" if not args.execute else "blocked"
    return {
        "status": status,
        "blockers": blockers,
        "plan_json": args.plan_json,
        "raw_out": str(raw_root),
        "hydrated_source_path": str(out_path),
        "samples_per_file": sample_count,
        "cleanup_raw": bool(args.cleanup_raw),
        "execute_requested": bool(args.execute),
        "execute_downloads": bool(args.execute_downloads),
        "source_paths_initialized_before_download": source_paths_initialized,
        "records": records,
        "disk_strategy": "download one official file, append sampled rows, optionally remove raw file",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequentially hydrate official Advection source data")
    parser.add_argument("--plan-json", default="reports/research/sota_loop/official_advection_hydration_plan.json")
    parser.add_argument("--raw-out")
    parser.add_argument("--hydrated-source-root")
    parser.add_argument("--samples-per-file", type=int)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--execute-downloads", action="store_true")
    parser.add_argument("--cleanup-raw", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--output-json",
        default="reports/research/sota_loop/official_advection_sequential_hydration_run.json",
    )
    args = parser.parse_args()

    record = hydrate_sequential(args)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))
    raise SystemExit(0 if record["status"] in {"dry_run", "executed"} else 2)


if __name__ == "__main__":
    main()
