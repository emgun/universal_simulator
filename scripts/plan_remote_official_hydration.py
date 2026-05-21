#!/usr/bin/env python
from __future__ import annotations

"""Plan a remote run for official Advection hydration when local disk is insufficient."""

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def create_remote_plan(args: argparse.Namespace) -> dict[str, Any]:
    hydration_plan = _load_json(args.hydration_plan_json)
    preflight = _load_json(args.preflight_json)
    storage = _load_json(args.storage_json)
    estimated_gib = float(hydration_plan.get("estimated_download_gib") or 0.0)
    sequential_hydration = bool(getattr(args, "sequential_hydration", True))
    remote_entries = list(hydration_plan.get("remote_entries") or [])
    max_file_gib = max((float(entry.get("size_bytes") or 0) / float(1024**3) for entry in remote_entries), default=estimated_gib)
    if sequential_hydration:
        required_disk_gb = max(
            int(getattr(args, "sequential_min_disk_gb", 32)),
            int(max_file_gib * float(args.disk_multiplier)) + int(getattr(args, "sequential_disk_padding_gb", 16)),
        )
    else:
        required_disk_gb = max(
            int(args.min_disk_gb),
            int(estimated_gib * float(args.disk_multiplier)) + int(args.disk_padding_gb),
        )
    blockers = []
    if preflight.get("status") != "blocked_insufficient_disk":
        blockers.append(f"local preflight status is {preflight.get('status')}; remote plan may not be needed")
    if storage.get("status") != "external_or_freed_space_required":
        blockers.append(f"storage recommendation status is {storage.get('status')}; remote plan may not be needed")

    offer_arg = f"OFFER_ID={args.offer_id} " if args.offer_id else ""
    launcher = (
        "DRY_RUN=1 "
        f"{offer_arg}"
        f"GIT_REF={args.git_ref} "
        f"LAUNCH_RETRIES={args.launch_retries} "
        f"LAUNCH_RETRY_BACKOFF={args.launch_retry_backoff} "
        f"DISK_GB={required_disk_gb} "
        "GPU=RTX_4090 "
        "REMOTE_SCRIPT=scripts/run_remote_official_hydration.sh "
        "EXTRA_PIPELINE_ARGS='"
        f"PLAN_JSON={args.remote_plan_json} "
        f"VALIDATION_JSON={args.remote_validation_json} "
        f"RUN_JSON={args.remote_run_json} "
        f"POST_VALIDATION_TEST_JSON={args.remote_post_validation_test_json} "
        "EXECUTE=1 EXECUTE_DOWNLOADS=1 "
        f"SEQUENTIAL_HYDRATION={1 if sequential_hydration else 0} "
        "SEQUENTIAL_CLEANUP_RAW=1 "
        "RUN_POST_VALIDATION_TEST=1 EXECUTE_TEST=1 "
        "PUBLISH_ARTIFACTS=1 "
        "MIN_DOWNLOAD_BYTES=60000000000 "
        f"PDEBENCH_DOWNLOAD_WORKERS={args.download_workers} "
        f"PDEBENCH_DOWNLOAD_PART_SIZE_MIB={args.download_part_size_mib} "
        f"PDEBENCH_DOWNLOAD_RETRIES={args.download_retries} "
        f"PDEBENCH_DOWNLOAD_PART_TIMEOUT={args.download_part_timeout} "
        f"PDEBENCH_DOWNLOAD_RETRY_BACKOFF={args.download_retry_backoff} "
        f"PDEBENCH_DOWNLOAD_SPLIT_AFTER_RETRIES={args.download_split_after_retries} "
        f"PDEBENCH_DOWNLOAD_MIN_SPLIT_SIZE_MIB={args.download_min_split_size_mib}' "
        "bash scripts/launch_remote_transport_shift_candidate_vast.sh"
    )
    return {
        "status": "ready_for_remote_hydration" if not blockers else "blocked_remote_plan_not_needed",
        "blockers": blockers,
        "hydration_plan_json": args.hydration_plan_json,
        "local_preflight_status": preflight.get("status"),
        "storage_recommendation_status": storage.get("status"),
        "estimated_download_gib": estimated_gib,
        "max_official_file_gib": max_file_gib,
        "required_disk_gb": required_disk_gb,
        "sequential_hydration": sequential_hydration,
        "git_ref": args.git_ref,
        "remote_plan_json": args.remote_plan_json,
        "remote_validation_json": args.remote_validation_json,
        "remote_run_json": args.remote_run_json,
        "remote_post_validation_test_json": args.remote_post_validation_test_json,
        "download_runtime": {
            "workers": args.download_workers,
            "part_size_mib": args.download_part_size_mib,
            "retries": args.download_retries,
            "part_timeout": args.download_part_timeout,
            "retry_backoff": args.download_retry_backoff,
            "split_after_retries": args.download_split_after_retries,
            "min_split_size_mib": args.download_min_split_size_mib,
        },
        "launch_runtime": {
            "retries": args.launch_retries,
            "retry_backoff": args.launch_retry_backoff,
        },
        "commands": {
            "dry_run_launcher": launcher,
            "actual_launcher": launcher.replace("DRY_RUN=1", "DRY_RUN=0", 1),
        },
        "preferred_offer_id": args.offer_id or None,
        "held_out_test_policy": hydration_plan.get("held_out_test_policy"),
        "notes": [
            "This is a launch plan only; it does not start paid compute.",
            "The remote run uses sequential hydrate-convert-delete by default and requires --execute-downloads.",
            "The Vast launcher invokes a bash wrapper; do not use a Python file as REMOTE_SCRIPT.",
            "The hydration plan downloads official train files only and builds train/val shards with test_count=0.",
            "Sequential hydration lowers scratch disk by keeping at most one raw official train file before appending sampled rows.",
            "The launcher pins GIT_REF so the remote host checks out the intended benchmark branch instead of relying on launcher defaults.",
            "Launch retries are included for transient Vast CLI DNS/connectivity failures before the remote instance is created.",
            "The post-validation test stage is chained but gated on literal_test_ready before it can build or read the held-out test shard.",
            "Downloader runtime knobs are included so remote retries use bounded reads, exponential backoff, adaptive range splitting, and same-host resume sidecars.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan remote official Advection hydration")
    parser.add_argument("--hydration-plan-json", default="reports/research/sota_loop/official_advection_hydration_plan.json")
    parser.add_argument(
        "--preflight-json",
        default="reports/research/sota_loop/official_advection_hydration_preflight.json",
    )
    parser.add_argument(
        "--storage-json",
        default="reports/research/sota_loop/official_advection_hydration_storage_recommendation.json",
    )
    parser.add_argument("--remote-plan-json", default="reports/research/sota_loop/official_advection_hydration_plan.json")
    parser.add_argument(
        "--remote-validation-json",
        default="reports/research/sota_loop/official_advection_hydration_plan_validation.json",
    )
    parser.add_argument(
        "--remote-run-json",
        default="reports/research/sota_loop/official_advection_hydration_plan_run.json",
    )
    parser.add_argument(
        "--remote-post-validation-test-json",
        default="reports/research/sota_loop/official_hydrated_post_validation_test_run.json",
    )
    parser.add_argument("--min-disk-gb", type=int, default=120)
    parser.add_argument("--disk-multiplier", type=float, default=1.3)
    parser.add_argument("--disk-padding-gb", type=int, default=40)
    parser.add_argument("--sequential-min-disk-gb", type=int, default=32)
    parser.add_argument("--sequential-disk-padding-gb", type=int, default=16)
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--download-part-size-mib", type=int, default=128)
    parser.add_argument("--download-retries", type=int, default=6)
    parser.add_argument("--download-part-timeout", type=int, default=180)
    parser.add_argument("--download-retry-backoff", type=float, default=20.0)
    parser.add_argument("--download-split-after-retries", type=int, default=2)
    parser.add_argument("--download-min-split-size-mib", type=int, default=8)
    parser.add_argument("--sequential-hydration", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--git-ref", default="codex/sota-learned-gate", help="Git ref the Vast launcher should checkout")
    parser.add_argument("--launch-retries", type=int, default=3)
    parser.add_argument("--launch-retry-backoff", type=float, default=10.0)
    parser.add_argument("--offer-id", default="", help="Optional explicit Vast offer ID for direct relaunch")
    parser.add_argument(
        "--output-json",
        default="reports/research/sota_loop/remote_official_advection_hydration_plan.json",
    )
    args = parser.parse_args()

    record = create_remote_plan(args)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
