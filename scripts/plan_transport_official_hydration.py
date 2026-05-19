#!/usr/bin/env python
from __future__ import annotations

"""Create an executable plan for hydrating official Advection data for the literal transport gate."""

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def _advection_entries(manifest_path: Path) -> list[dict[str, Any]]:
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    rows = []
    for entry in payload.get("files", []):
        logical_path = str(entry.get("path", ""))
        if "1D/Advection/Train/" not in logical_path:
            continue
        rows.append(
            {
                "path": logical_path,
                "file_id": entry.get("file_id"),
                "size_bytes": int(entry.get("size_bytes") or 0),
                "checksum": entry.get("checksum"),
                "checksum_type": entry.get("checksum_type"),
            }
        )
    return rows


def _download_commands(entries: list[dict[str, Any]], out_root: str) -> list[str]:
    return [
        f"python scripts/download_pdebench_file.py '{entry['path']}' --out {out_root}"
        for entry in entries
    ]


def _shard_command(args: argparse.Namespace, selected_entries: list[dict[str, Any]]) -> str:
    samples_per_file = max(1, (args.train_count + args.val_count + len(selected_entries) - 1) // len(selected_entries))
    return (
        "python scripts/convert_pdebench.py "
        f"--pattern '{args.raw_out}/1D/Advection/Train/*.hdf5' "
        f"--out {args.hydrated_source_root}/advection1d_train.h5 "
        f"--samples {samples_per_file}"
    )


def _gate_command(args: argparse.Namespace) -> str:
    shift_args = " ".join(f"--shift {shift}" for shift in args.shift)
    return (
        "python scripts/run_transport_shift_gate.py "
        f"--data-root {args.hydrated_light_root} "
        "--task advection1d --train-split train --val-split val "
        f"--max-samples {args.train_count} --rollout-steps {args.rollout_steps} "
        f"{shift_args} --metric nrmse "
        f"--reference-metric-value {args.reference_metric_value} "
        f"--val-min-relative-improvement {args.val_min_relative_improvement} "
        f"--output-json {args.output_root}/official_hydrated_transport_shift_gate.json"
    )


def create_plan(args: argparse.Namespace) -> dict[str, Any]:
    entries = _advection_entries(Path(args.manifest))
    if args.max_files is not None:
        entries = entries[: args.max_files]
    total_size = sum(int(entry["size_bytes"]) for entry in entries)
    selected_paths = [str(entry["path"]) for entry in entries]
    download_commands = _download_commands(entries, args.raw_out)
    return {
        "status": "ready_for_explicit_hydration" if entries else "missing_manifest_entries",
        "manifest": args.manifest,
        "selected_official_advection_train_files": selected_paths,
        "selected_file_count": len(entries),
        "estimated_download_bytes": total_size,
        "estimated_download_gib": total_size / float(1024**3),
        "raw_out": args.raw_out,
        "hydrated_source_root": args.hydrated_source_root,
        "hydrated_light_root": args.hydrated_light_root,
        "train_count": args.train_count,
        "val_count": args.val_count,
        "test_count": 0,
        "held_out_test_policy": {
            "test_split_downloaded": False,
            "test_split_sharded": False,
            "test_may_run_only_after_validation_guard": True,
        },
        "commands": {
            "download_official_train_files": download_commands,
            "build_train_val_source": _shard_command(args, entries) if entries else None,
            "build_light_train_val_shards": (
                "python scripts/make_light_hdf5_shards.py "
                f"--root {args.hydrated_source_root} --out-root {args.hydrated_light_root} "
                "--tasks advection1d --source-split train --split-source val=train "
                f"--split-start-index train=0 --split-start-index val={args.train_count} "
                f"--train-count {args.train_count} --val-count {args.val_count} --test-count 0 "
                f"--manifest {args.output_root}/official_hydrated_trainval_manifest.yaml --overwrite"
            ),
            "validate_without_test": _gate_command(args),
            "objective_audit_after_validation": "bash scripts/run_official_transport_objective_status.sh",
        },
        "decision_points": [
            "Run downloads only with explicit approval for network and disk use.",
            "Do not download or shard official test data during train/val hydration.",
            "Run the held-out test only through the gated transport command after validation passes.",
            "If train support still misses validation after hydration, stop and record the blocker before test.",
        ],
        "notes": [
            "This plan intentionally uses official manifest entries, not synthetic report shards.",
            "The convert command is a plan-level command and may need adaptation to the raw HDF5 schema after hydration.",
            "The current workspace has not performed these downloads.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan official Advection hydration for the literal transport objective")
    parser.add_argument("--manifest", default="docs/pdebench_manifest.yaml")
    parser.add_argument("--raw-out", default="data/pdebench/raw")
    parser.add_argument("--hydrated-source-root", default="data/pdebench_official_advection_hydrated")
    parser.add_argument("--hydrated-light-root", default="data/pdebench_official_advection_light")
    parser.add_argument("--output-root", default="reports/research/sota_loop")
    parser.add_argument("--train-count", type=int, default=256)
    parser.add_argument("--val-count", type=int, default=64)
    parser.add_argument("--rollout-steps", type=int, default=16)
    parser.add_argument("--shift", action="append", type=int, default=None)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--reference-metric-value", type=float, default=0.30780652221851373)
    parser.add_argument("--val-min-relative-improvement", type=float, default=0.0)
    parser.add_argument(
        "--output-json",
        default="reports/research/sota_loop/official_advection_hydration_plan.json",
    )
    args = parser.parse_args()
    if args.shift is None:
        args.shift = list(range(-96, 97, 8))

    record = create_plan(args)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
