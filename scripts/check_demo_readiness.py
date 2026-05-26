#!/usr/bin/env python
from __future__ import annotations

"""Summarize whether the UPS demo loop is ready for remote experiments."""

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any

import h5py
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.check_demo_b2_shards import (  # noqa: E402
    check_keys,
    configure_rclone_env,
    expected_keys_from_manifest,
    load_env_file,
)
from ups.eval.demo_scorecard import load_summary  # noqa: E402

DEFAULT_LIGHT_REPORT_DIR = "reports/demo/light_latest"


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest must contain a mapping: {path}")
    return payload


def _glob_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        if Path(pattern).is_absolute():
            paths.extend(Path(item) for item in glob.glob(pattern))
        else:
            paths.extend(Path().glob(pattern))
    return sorted({path.resolve() for path in paths if path.exists()})


def _summary_status(paths: list[Path], *, baseline_run: str, candidate_run: str) -> dict[str, Any]:
    runs: list[str] = []
    metrics_by_run: dict[str, list[str]] = {}
    for path in paths:
        summary = load_summary(path)
        run_name = str(summary.get("run_name") or path.parent.name)
        runs.append(run_name)
        metrics_by_run[run_name] = sorted(str(key) for key in dict(summary.get("metrics", {})))
    unique_runs = sorted(dict.fromkeys(runs))
    return {
        "summary_count": len(paths),
        "runs": unique_runs,
        "metrics_by_run": metrics_by_run,
        "baseline_run": baseline_run,
        "candidate_run": candidate_run,
        "has_baseline": bool(baseline_run and baseline_run in unique_runs),
        "has_candidate": bool(candidate_run and candidate_run in unique_runs),
    }


def _source_paths(root: Path, task: str, split: str) -> list[Path]:
    exact = root / f"{task}_{split}.h5"
    if exact.exists():
        return [exact]
    return sorted(root.glob(f"{task}_{split}_*.h5"))


def _candidate_source_path(root: Path, task: str, split: str) -> Path:
    return root / f"{task}_{split}.h5"


def _configured_data_root(manifest_payload: dict[str, Any], *, data_root: Path | None) -> Path:
    root = data_root or Path(str(manifest_payload.get("source_root") or "data/pdebench"))
    root = root.expanduser()
    if root.is_absolute():
        return root
    return REPO_ROOT / root


def _h5_data_summary(paths: list[Path]) -> dict[str, Any]:
    total_samples = 0
    shapes: dict[str, list[int]] = {}
    errors: dict[str, str] = {}
    for path in paths:
        try:
            with h5py.File(path, "r") as handle:
                if "data" not in handle:
                    raise KeyError("missing 'data' dataset")
                shape = [int(dim) for dim in handle["data"].shape]
                shapes[str(path)] = shape
                if shape:
                    total_samples += shape[0]
        except (KeyError, OSError, ValueError) as exc:
            errors[str(path)] = str(exc)
    return {
        "available_samples": total_samples,
        "data_shapes": shapes,
        "errors": errors,
    }


def _local_data_status(
    *, manifest_payload: dict[str, Any], data_root: Path | None
) -> dict[str, Any]:
    root = _configured_data_root(manifest_payload, data_root=data_root)
    tasks = [str(task) for task in manifest_payload.get("tasks", [])]
    splits_cfg = manifest_payload.get("splits", {}) or {}
    present: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    short: list[dict[str, Any]] = []

    for task in tasks:
        for split, raw_cfg in splits_cfg.items():
            cfg = dict(raw_cfg or {})
            required_samples = int(cfg.get("samples", 0) or 0)
            if required_samples <= 0:
                continue
            preferred_split = str(cfg.get("preferred_source_split") or split)
            fallback_raw = cfg.get("fallback_source_split")
            source_splits = [preferred_split]
            if fallback_raw:
                fallback_split = str(fallback_raw)
                if fallback_split not in source_splits:
                    source_splits.append(fallback_split)

            source_split: str | None = None
            paths: list[Path] = []
            for candidate_split in source_splits:
                paths = _source_paths(root, task, candidate_split)
                if paths:
                    source_split = candidate_split
                    break

            candidate_paths = [
                str(_candidate_source_path(root, task, candidate_split))
                for candidate_split in source_splits
            ]
            base_record = {
                "task": task,
                "split": str(split),
                "required_samples": required_samples,
                "preferred_source_split": preferred_split,
                "fallback_source_split": str(fallback_raw) if fallback_raw else None,
                "candidate_paths": candidate_paths,
            }
            if not paths or source_split is None:
                missing.append(base_record)
                continue

            h5_summary = _h5_data_summary(paths)
            record = {
                **base_record,
                "source_split": source_split,
                "derived_from_source_split": source_split != str(split),
                "paths": [str(path) for path in paths],
                **h5_summary,
            }
            present.append(record)
            if h5_summary["errors"] or int(h5_summary["available_samples"]) < required_samples:
                short.append(record)

    expected_count = len(present) + len(missing)
    return {
        "checked": True,
        "ok": not missing and not short,
        "root": str(root),
        "expected_count": expected_count,
        "present_count": len(present),
        "missing_count": len(missing),
        "short_count": len(short),
        "present": present,
        "missing": missing,
        "short": short,
    }


def readiness_payload(
    *,
    manifest: Path,
    summary_patterns: list[str],
    baseline_run: str,
    candidate_run: str,
    check_b2: bool,
    env_file: Path,
    check_local_data: bool = False,
    data_root: Path | None = None,
) -> dict[str, Any]:
    blockers: list[str] = []
    next_steps: list[str] = []

    manifest_status: dict[str, Any]
    manifest_payload: dict[str, Any] = {}
    expected_keys: list[str] = []
    if manifest.exists():
        manifest_payload = _load_manifest(manifest)
        expected_keys = expected_keys_from_manifest(manifest)
        manifest_status = {
            "ok": True,
            "path": str(manifest),
            "expected_key_count": len(expected_keys),
            "expected_keys": expected_keys,
        }
    else:
        manifest_status = {
            "ok": False,
            "path": str(manifest),
            "expected_key_count": 0,
            "expected_keys": [],
        }
        blockers.append(f"Manifest missing: {manifest}")
        next_steps.append("Create or restore docs/demo_data_manifest.yaml.")

    local_data_status: dict[str, Any] = {"checked": False, "ok": None}
    if check_local_data:
        if manifest_status["ok"]:
            local_data_status = _local_data_status(
                manifest_payload=manifest_payload, data_root=data_root
            )
            missing_count = int(local_data_status["missing_count"])
            short_count = int(local_data_status["short_count"])
            if missing_count:
                blockers.append(
                    f"Local data missing {missing_count} expected demo source shard(s)."
                )
                next_steps.append(
                    "Hydrate missing local source shards from B2 or rebuild them before "
                    "running full-task train selection."
                )
            if short_count:
                blockers.append(
                    f"Local data has {short_count} undersized or unreadable source shard(s)."
                )
                next_steps.append(
                    "Regenerate or replace local source shards whose 'data' dataset is "
                    "missing, unreadable, or below the manifest sample count."
                )
        else:
            local_data_status = {"checked": True, "ok": False}

    b2_status: dict[str, Any] = {"checked": False, "ok": None}
    if check_b2 and expected_keys:
        env, bucket = configure_rclone_env(load_env_file(env_file))
        b2_status = check_keys(expected_keys, bucket=bucket, env=env, dry_run=False)
        b2_status["checked"] = True
        b2_status["ok"] = b2_status["missing_count"] == 0
        if b2_status["missing_count"]:
            blockers.append(f"B2 missing {b2_status['missing_count']} expected demo shard keys.")
            next_steps.append(
                "Run scripts/run_remote_shard_prep_b2.sh on a remote/data-prep box, then re-check shards."
            )
    elif expected_keys:
        next_steps.append(
            "Run with --check-b2 after credentials are available to verify shard presence."
        )

    summary_paths = _glob_paths(summary_patterns)
    summary_status = _summary_status(
        summary_paths, baseline_run=baseline_run, candidate_run=candidate_run
    )
    if baseline_run and not summary_status["has_baseline"]:
        blockers.append(f"Missing baseline summary: {baseline_run}")
        next_steps.append("Run scripts/run_persistence_baseline.py on the held-out shards.")
    if candidate_run and not summary_status["has_candidate"]:
        blockers.append(f"Missing candidate summary: {candidate_run}")
        next_steps.append(
            "Run the UPS candidate via scripts/run_remote_light_promotion.sh or the generated queue."
        )

    report_ready = (
        manifest_status["ok"]
        and bool(summary_paths)
        and (not baseline_run or summary_status["has_baseline"])
        and (not candidate_run or summary_status["has_candidate"])
        and (not check_b2 or bool(b2_status.get("ok")))
        and (not check_local_data or bool(local_data_status.get("ok")))
    )
    if report_ready:
        next_steps.append(f"Build {DEFAULT_LIGHT_REPORT_DIR} with scripts/build_demo_report.py.")

    return {
        "ready": report_ready,
        "manifest": manifest_status,
        "local_data": local_data_status,
        "b2": b2_status,
        "summaries": summary_status,
        "blockers": blockers,
        "next_steps": list(dict.fromkeys(next_steps)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check UPS demo readiness")
    parser.add_argument("--manifest", default="docs/demo_data_manifest.yaml")
    parser.add_argument(
        "--summary-glob",
        action="append",
        default=["reports/light_experiments_remote/*/summary.json"],
    )
    parser.add_argument("--baseline-run", default="persistence_light_v1_test")
    parser.add_argument("--candidate-run", default="ups_light_v1_task_signature_only")
    parser.add_argument("--check-b2", action="store_true")
    parser.add_argument(
        "--check-local-data",
        action="store_true",
        help="Verify manifest-required local source HDF5 shards under source_root.",
    )
    parser.add_argument(
        "--data-root",
        default="",
        help="Optional override for the manifest source_root used by --check-local-data.",
    )
    parser.add_argument("--env-file", default=os.environ.get("ENV_FILE", ".env"))
    parser.add_argument("--json", default="", help="Optional output JSON path")
    parser.add_argument(
        "--strict", action="store_true", help="Exit nonzero when readiness blockers remain"
    )
    args = parser.parse_args()

    payload = readiness_payload(
        manifest=Path(args.manifest),
        summary_patterns=args.summary_glob,
        baseline_run=args.baseline_run,
        candidate_run=args.candidate_run,
        check_b2=args.check_b2,
        env_file=Path(args.env_file),
        check_local_data=args.check_local_data,
        data_root=Path(args.data_root) if args.data_root else None,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.json:
        output = Path(args.json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    if args.strict and not payload["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
