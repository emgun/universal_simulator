#!/usr/bin/env python
from __future__ import annotations

"""Protocol-level diagnostic: exact-overlap audit across train/val/test splits.

Extends the 2026-07-08 advection regime diagnosis: beyond regime composition,
this audits whether splits share identical trajectories. For time-dependent
tasks the key is the initial-condition frame; for steady tasks the key is the
full field. Matches are exact byte-level equality of float32 arrays.

This is NOT a candidate measurement: nothing is trained, tuned, selected, or
scored, and no ledger is written. Reading the held-out test split requires the
explicit --include-test flag and is recorded truthfully in the output JSON.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from ups.data.manifests import load_data_lock

TIME_DEPENDENT_TASKS = {"advection1d", "burgers1d"}


def _keys_for_split(path: Path, task: str) -> tuple[list[bytes], list[tuple[bytes, str]]]:
    with h5py.File(path, "r") as handle:
        data = np.asarray(handle["data"], dtype=np.float32)
        regimes = np.asarray(handle["beta"]) if "beta" in handle else None
    if task in TIME_DEPENDENT_TASKS:
        frames = data[:, 0]
    else:
        frames = data.reshape(data.shape[0], -1)
    field_keys = [np.ascontiguousarray(frame).tobytes() for frame in frames]
    if regimes is None:
        pair_keys = [(field, "") for field in field_keys]
    else:
        if regimes.ndim != 1 or len(regimes) != len(field_keys):
            raise ValueError(f"{path} beta dataset is not aligned to physical fields")
        pair_keys = [
            (field, repr(value.item() if isinstance(value, np.generic) else value))
            for field, value in zip(field_keys, regimes, strict=True)
        ]
    return field_keys, pair_keys


def _overlap(a: list[bytes], b: list[bytes]) -> int:
    b_set = set(b)
    return sum(1 for key in a if key in b_set)


def audit_task(root: Path, task: str, splits: list[str]) -> dict[str, Any]:
    keys: dict[str, list[bytes]] = {}
    field_regime_keys: dict[str, list[tuple[bytes, str]]] = {}
    missing: list[str] = []
    for split in splits:
        path = root / f"{task}_{split}.h5"
        if not path.exists():
            missing.append(split)
            continue
        keys[split], field_regime_keys[split] = _keys_for_split(path, task)

    unique_counts = {split: len(set(values)) for split, values in field_regime_keys.items()}
    unique_field_groups = {split: len(set(values)) for split, values in keys.items()}
    record: dict[str, Any] = {
        "task": task,
        "key": (
            "initial_condition_frame_plus_regime"
            if task in TIME_DEPENDENT_TASKS
            else "full_field_plus_regime"
        ),
        "split_sizes": {split: len(k) for split, k in keys.items()},
        "missing_splits": missing,
        "overlaps": {},
        "unique_keys_per_split": unique_counts,
        "unique_field_groups_per_split": unique_field_groups,
        "duplicate_count_per_split": {
            split: len(keys[split]) - unique_counts[split] for split in keys
        },
    }
    names = list(keys)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            record["overlaps"][f"{first}<->{second}"] = _overlap(keys[first], keys[second])
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--tasks", nargs="+", default=["advection1d", "burgers1d", "darcy2d"])
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Explicitly allow reading *_test.h5 shards for this protocol diagnostic",
    )
    parser.add_argument(
        "--data-lock",
        help="Required measurement lock when --include-test authorizes test bytes",
    )
    parser.add_argument(
        "--require-unique-within-split",
        action="store_true",
        help="Return nonzero when any audited split contains duplicate field identities",
    )
    parser.add_argument("--output-json", default="")
    args = parser.parse_args(argv)

    splits = list(args.splits)
    if "test" in splits and not args.include_test:
        raise SystemExit(
            "Reading held-out test splits requires --include-test. This diagnostic "
            "audits split construction only; it never scores a candidate."
        )
    measurement_lock = None
    if "test" in splits:
        if not args.data_lock:
            raise SystemExit(
                "Reading held-out test splits requires --data-lock with a measurement purpose"
            )
        measurement_lock = load_data_lock(args.data_lock)
        if (
            measurement_lock.purpose != "measurement"
            or "test" not in measurement_lock.requested_roles
        ):
            raise SystemExit("--data-lock must be a test-authorizing measurement lock")
        locked_names = {
            Path(item.path).name for item in measurement_lock.objects if item.role == "test"
        }
        requested_names = {f"{task}_test.h5" for task in args.tasks}
        unauthorized = sorted(requested_names - locked_names)
        if unauthorized:
            raise SystemExit(
                "Measurement lock does not authorize requested test files: "
                + ", ".join(unauthorized)
            )

    root = Path(args.root)
    results: dict[str, Any] = {
        "measurement_type": "split_integrity_overlap_audit",
        "purpose": "protocol_distribution_diagnostic",
        "candidate_scored": False,
        "test_ledger_writes": [],
        "held_out_test_data_read": "test" in splits,
        "measurement_data_lock_sha256": (
            measurement_lock.lock_sha256 if measurement_lock is not None else None
        ),
        "measurement_contract_id": (
            measurement_lock.measurement_contract_id if measurement_lock is not None else None
        ),
        "root": str(root),
        "root_digest": hashlib.sha256(str(root).encode()).hexdigest()[:12],
        "tasks": {task: audit_task(root, task, splits) for task in args.tasks},
    }
    payload = json.dumps(results, indent=2, sort_keys=True)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload)
    if args.require_unique_within_split and any(
        count > 0
        for task in results["tasks"].values()
        for count in task["duplicate_count_per_split"].values()
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
