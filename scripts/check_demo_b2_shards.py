#!/usr/bin/env python
from __future__ import annotations

"""Check whether demo HDF5 shard keys exist in Backblaze B2."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, value = line.split("=", 1)
        elif ":" in line:
            key, value = line.split(":", 1)
        else:
            continue
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            values[key] = value
    return values


def expected_keys_from_manifest(path: Path) -> list[str]:
    manifest = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    records = manifest.get("records") or []
    keys = [str(record["remote_key"]) for record in records if record.get("remote_key")]
    if keys:
        return sorted(dict.fromkeys(keys))

    remote_prefix = str(manifest.get("remote_prefix") or manifest.get("version") or "").strip("/")
    if not remote_prefix:
        raise ValueError("Manifest must define remote_prefix or records[].remote_key")
    tasks = [str(task) for task in manifest.get("tasks", [])]
    splits_cfg = manifest.get("splits", {})
    splits = [
        split for split, cfg in splits_cfg.items() if int((cfg or {}).get("samples", 0) or 0) > 0
    ]
    if not tasks or not splits:
        raise ValueError("Manifest must define tasks and non-empty splits")
    keys = []
    for task in tasks:
        for split in splits:
            keys.append(f"{remote_prefix}/{task}/{task}_{split}.h5")
    return sorted(keys)


def configure_rclone_env(values: dict[str, str]) -> tuple[dict[str, str], str]:
    env = os.environ.copy()
    merged = {
        **values,
        **{key: value for key, value in os.environ.items() if key.startswith("B2_")},
    }
    bucket = merged.get("B2_BUCKET") or merged.get("B2_BUCKET_NAME")
    if not bucket:
        raise ValueError("B2_BUCKET must be set in env or --env-file")
    key_id = merged.get("B2_KEY_ID") or merged.get("B2_ACCOUNT_ID")
    app_key = merged.get("B2_APP_KEY") or merged.get("B2_APPLICATION_KEY")
    if not key_id or not app_key:
        raise ValueError("B2_KEY_ID and B2_APP_KEY must be set in env or --env-file")

    if merged.get("B2_S3_ENDPOINT") or merged.get("B2_S3_REGION"):
        env["RCLONE_CONFIG_UPSB2_TYPE"] = "s3"
        env["RCLONE_CONFIG_UPSB2_PROVIDER"] = "B2"
        env["RCLONE_CONFIG_UPSB2_ACCESS_KEY_ID"] = key_id
        env["RCLONE_CONFIG_UPSB2_SECRET_ACCESS_KEY"] = app_key
        if merged.get("B2_S3_ENDPOINT"):
            env["RCLONE_CONFIG_UPSB2_ENDPOINT"] = merged["B2_S3_ENDPOINT"]
        if merged.get("B2_S3_REGION"):
            env["RCLONE_CONFIG_UPSB2_REGION"] = merged["B2_S3_REGION"]
    else:
        env["RCLONE_CONFIG_UPSB2_TYPE"] = "b2"
        env["RCLONE_CONFIG_UPSB2_ACCOUNT"] = key_id
        env["RCLONE_CONFIG_UPSB2_KEY"] = app_key
    return env, bucket


def check_keys(
    keys: list[str], *, bucket: str, env: dict[str, str], dry_run: bool
) -> dict[str, Any]:
    present: list[str] = []
    missing: list[str] = []
    for key in keys:
        remote = f"UPSB2:{bucket}/{key}"
        if dry_run:
            missing.append(key)
            continue
        proc = subprocess.run(
            ["rclone", "size", remote, "--json"],
            env=env,
            capture_output=True,
            text=True,
        )
        exists = False
        if proc.returncode == 0:
            try:
                size_payload = json.loads(proc.stdout or "{}")
                exists = int(size_payload.get("count", 0) or 0) > 0
            except json.JSONDecodeError:
                exists = False
        if exists:
            present.append(key)
        else:
            missing.append(key)
    return {
        "bucket": bucket,
        "expected_count": len(keys),
        "present_count": len(present),
        "missing_count": len(missing),
        "present": present,
        "missing": missing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check demo B2 shard availability")
    parser.add_argument("--manifest", default="docs/demo_data_manifest.yaml")
    parser.add_argument("--env-file", default=os.environ.get("ENV_FILE", ".env"))
    parser.add_argument(
        "--dry-run", action="store_true", help="Print expected keys without contacting B2"
    )
    parser.add_argument("--json", default=None, help="Optional output JSON path")
    args = parser.parse_args()

    keys = expected_keys_from_manifest(Path(args.manifest))
    if args.dry_run:
        payload = {
            "bucket": os.environ.get("B2_BUCKET", "dry-run-bucket"),
            "expected_count": len(keys),
            "present_count": 0,
            "missing_count": len(keys),
            "present": [],
            "missing": keys,
        }
    else:
        env, bucket = configure_rclone_env(load_env_file(Path(args.env_file)))
        payload = check_keys(keys, bucket=bucket, env=env, dry_run=False)
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.json:
        output = Path(args.json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    if payload["missing_count"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
