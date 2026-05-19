#!/usr/bin/env python
from __future__ import annotations

"""Audit benchmark-clean data hydration options for the transport-shift objective."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_train_only_transport_identifiability import _per_sample_best_shifts
from scripts.fit_transport_shift_head import _candidate_shifts, _load_series


def _max_samples(value: int | None) -> int | None:
    return None if value is not None and value < 0 else value


def _histogram_from_labels(labels) -> dict[str, int]:
    hist: dict[str, int] = {}
    for value in labels.tolist():
        key = str(int(value))
        hist[key] = hist.get(key, 0) + 1
    return hist


def _local_split_support(
    *,
    root: Path,
    task: str,
    train_split: str,
    val_split: str,
    train_max_samples: int | None,
    val_max_samples: int | None,
    shifts: Sequence[int],
    rollout_steps: int,
    metric: str,
) -> dict[str, Any]:
    train_path = root / f"{task}_{train_split}.h5"
    val_path = root / f"{task}_{val_split}.h5"
    exists = {"train": train_path.exists(), "val": val_path.exists()}
    if not all(exists.values()):
        return {
            "root": str(root),
            "exists": exists,
            "status": "missing_required_local_splits",
        }

    train_fields = _load_series(root=root, task=task, split=train_split, max_samples=train_max_samples)
    val_fields = _load_series(root=root, task=task, split=val_split, max_samples=val_max_samples)
    train_labels, _ = _per_sample_best_shifts(train_fields, shifts, rollout_steps=rollout_steps, metric=metric)
    val_labels, _ = _per_sample_best_shifts(val_fields, shifts, rollout_steps=rollout_steps, metric=metric)
    train_support = sorted(set(int(value) for value in train_labels.tolist()))
    val_support = sorted(set(int(value) for value in val_labels.tolist()))
    unsupported_val = sorted(set(val_support) - set(train_support))
    return {
        "root": str(root),
        "exists": exists,
        "status": "local_support_covers_validation" if not unsupported_val else "local_train_support_missing_validation_shift",
        "train_path": str(train_path),
        "val_path": str(val_path),
        "train_shape": list(train_fields.shape),
        "val_shape": list(val_fields.shape),
        "train_shift_histogram": _histogram_from_labels(train_labels),
        "val_shift_histogram": _histogram_from_labels(val_labels),
        "train_shift_support": train_support,
        "val_shift_support": val_support,
        "unsupported_val_shifts": unsupported_val,
    }


def _remote_advection_entries(manifest_path: Path) -> list[dict[str, Any]]:
    if not manifest_path.exists():
        return []
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    entries = []
    for entry in payload.get("files", []):
        logical_path = str(entry.get("path", ""))
        if "1D/Advection/Train/" not in logical_path:
            continue
        entries.append(
            {
                "path": logical_path,
                "file_id": entry.get("file_id"),
                "size_bytes": entry.get("size_bytes"),
                "checksum": entry.get("checksum"),
                "checksum_type": entry.get("checksum_type"),
            }
        )
    return entries


def _synthetic_roots(root: Path, task: str, limit: int) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    rows = []
    for path in sorted(root.glob(f"**/{task}_train.h5"))[:limit]:
        sibling_val = path.with_name(f"{task}_val.h5")
        rows.append(
            {
                "root": str(path.parent),
                "train_path": str(path),
                "val_path": str(sibling_val) if sibling_val.exists() else None,
                "has_val": sibling_val.exists(),
                "classification": "synthetic_report_artifact_not_benchmark_clean",
            }
        )
    return rows


def audit_hydration_options(args: argparse.Namespace) -> dict[str, Any]:
    shifts = _candidate_shifts(args.shift)
    canonical_support = _local_split_support(
        root=Path(args.data_root),
        task=args.task,
        train_split=args.train_split,
        val_split=args.val_split,
        train_max_samples=_max_samples(args.max_samples),
        val_max_samples=_max_samples(args.val_max_samples if args.val_max_samples is not None else args.max_samples),
        shifts=shifts,
        rollout_steps=args.rollout_steps,
        metric=args.metric,
    )
    remote_entries = _remote_advection_entries(Path(args.manifest))
    remote_total_bytes = sum(int(entry.get("size_bytes") or 0) for entry in remote_entries)
    synthetic = _synthetic_roots(Path(args.synthetic_root), args.task, args.synthetic_limit)

    remote_available = bool(remote_entries)
    local_covers_validation = canonical_support.get("status") == "local_support_covers_validation"
    if local_covers_validation:
        status = "local_benchmark_clean_support_available"
        blockers: list[str] = []
    elif remote_available:
        status = "remote_official_hydration_required"
        blockers = [
            "canonical local train split does not cover validation shift support",
            "official raw Advection train files are listed in the manifest but are not hydrated locally",
            "synthetic report shards are not benchmark-clean substitutes",
        ]
    else:
        status = "no_benchmark_clean_hydration_source_found"
        blockers = [
            "canonical local train split does not cover validation shift support",
            "no official raw Advection entries found in the manifest",
            "synthetic report shards are not benchmark-clean substitutes",
        ]

    return {
        "status": status,
        "blockers": blockers,
        "task": args.task,
        "data_root": args.data_root,
        "metric": args.metric,
        "rollout_steps": args.rollout_steps,
        "candidate_shifts": shifts,
        "canonical_local": canonical_support,
        "remote_official_manifest": {
            "manifest": args.manifest,
            "advection_train_file_count": len(remote_entries),
            "total_size_bytes": remote_total_bytes,
            "total_size_gib": remote_total_bytes / float(1024**3),
            "entries": remote_entries,
            "hydration_command_template": (
                "python scripts/download_pdebench_file.py "
                "'1D/Advection/Train/1D_Advection_Sols_beta0.7.hdf5' --out data/pdebench/raw"
            ),
        },
        "synthetic_report_artifacts": {
            "root": args.synthetic_root,
            "returned_count": len(synthetic),
            "limit": args.synthetic_limit,
            "entries": synthetic,
        },
        "recommendation": (
            "Hydrate official raw Advection train files or build an explicitly approved split-compatible benchmark; "
            "do not use synthetic report shards as benchmark-clean evidence."
            if status == "remote_official_hydration_required"
            else "Use the canonical local support for a literal train-only gate."
            if status == "local_benchmark_clean_support_available"
            else "No local or manifest-backed benchmark-clean hydration path was found."
        ),
        "notes": [
            "Reads train/val only for local support; does not read or evaluate held-out test.",
            "Remote manifest entries are not downloaded by this audit.",
            "Report-generated synthetic shards are cataloged only to avoid mistaking them for real benchmark data.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit transport data hydration options")
    parser.add_argument("--data-root", default="data/pdebench")
    parser.add_argument("--task", default="advection1d")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--val-max-samples", type=int, default=-1)
    parser.add_argument("--rollout-steps", type=int, default=16)
    parser.add_argument("--shift", action="append", type=int, default=None)
    parser.add_argument("--metric", choices=("mse", "nrmse"), default="nrmse")
    parser.add_argument("--manifest", default="docs/pdebench_manifest.yaml")
    parser.add_argument("--synthetic-root", default="reports/light_experiments")
    parser.add_argument("--synthetic-limit", type=int, default=20)
    parser.add_argument(
        "--output-json",
        default="reports/research/sota_loop/transport_data_hydration_options.json",
    )
    args = parser.parse_args()

    record = audit_hydration_options(args)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
