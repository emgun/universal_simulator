#!/usr/bin/env python
from __future__ import annotations

"""
Publish a processed dataset shard to Backblaze B2 and log a lightweight W&B
artifact that references the remote shard, then delete local shard files.

Supports:
- Upload via rclone (default; no Python deps)
- Optional S3-style URL reference if B2_S3_REGION is provided

Environment variables (or pass via .env and `scripts/load_env.sh`):
- B2_KEY_ID, B2_APP_KEY, B2_BUCKET [required]
- B2_PREFIX [optional, default: "pdebench"]
- B2_S3_REGION [optional, enables https://s3.<region>.backblazeb2.com refs]

Usage examples:
  python scripts/publish_shard_b2.py artifacts/burgers1d_subset_v1.tar.gz \
    --artifact-name burgers1d_subset_v1 --artifact-type dataset

  # Publish a directory recursively
  python scripts/publish_shard_b2.py data/pdebench/burgers1d_train \
    --prefix pdebench/burgers1d/train

Set --dry-run to validate without network or deletions.
"""

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple


def _load_env_file(path: Path) -> None:
    """Load simple .env files with KEY=value or KEY: value into os.environ.

    Only sets keys not already present in the environment.
    """
    if not path.exists():
        return
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.lstrip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                key, val = line.split(":", 1)
            elif "=" in line:
                key, val = line.split("=", 1)
            else:
                continue
            key = key.strip()
            # strip surrounding quotes if present
            v = val.strip()
            if (v.startswith("\"") and v.endswith("\"")) or (v.startswith("'") and v.endswith("'")):
                v = v[1:-1]
            val = v
            if key and key not in os.environ:
                os.environ[key] = val
    except Exception:
        # Best-effort; fall back to env
        pass


def _which(cmd: str) -> Optional[str]:
    path = shutil.which(cmd)
    return path


def _run(cmd: List[str], *, env: Optional[dict] = None, dry_run: bool = False) -> None:
    if dry_run:
        print("DRY_RUN:", " ".join(shlex.quote(c) for c in cmd))
        return
    subprocess.run(cmd, check=True, env=env)


def _rclone_copyto(
    local: Path,
    bucket: str,
    key: str,
    *,
    b2_key_id: str,
    b2_app_key: str,
    dry_run: bool,
) -> str:
    """Upload to B2. Prefer rclone; fallback to boto3 S3-compatible API if available."""
    rclone_path = _which("rclone")
    if rclone_path is None and not dry_run:
        try:
            import boto3  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Neither rclone nor boto3 are available. Install rclone (brew install rclone) "
                "or install boto3 (pip install boto3)"
            ) from e
        # boto3 fallback
        endpoint = os.environ.get("B2_S3_ENDPOINT")
        region = os.environ.get("B2_S3_REGION")
        if not endpoint:
            if region:
                endpoint = f"https://s3.{region}.backblazeb2.com"
            else:
                # Generic default; Backblaze recommends region endpoint
                endpoint = "https://s3.us-west-000.backblazeb2.com"
        session = boto3.session.Session()
        s3 = session.client(
            "s3",
            region_name=region or "us-west-000",
            endpoint_url=endpoint,
            aws_access_key_id=b2_key_id,
            aws_secret_access_key=b2_app_key,
        )
        if local.is_dir():
            # Upload contents recursively under the given key prefix
            for root, _, files in os.walk(local):
                for fname in files:
                    fpath = Path(root) / fname
                    rel = fpath.relative_to(local)
                    dest = key.rstrip("/") + "/" + str(rel).replace("\\", "/")
                    s3.upload_file(str(fpath), bucket, dest)
        else:
            s3.upload_file(str(local), bucket, key)
        return f"s3:{bucket}/{key}"

    if rclone_path is None and dry_run:
        print("DRY_RUN: rclone not found; skipping presence check")

    env = os.environ.copy()
    env.setdefault("RCLONE_CONFIG_UPSB2_TYPE", "b2")
    env.setdefault("RCLONE_CONFIG_UPSB2_ACCOUNT", b2_key_id)
    env.setdefault("RCLONE_CONFIG_UPSB2_KEY", b2_app_key)

    remote = f"UPSB2:{bucket}/{key}"

    if local.is_dir():
        # Copy directory contents into prefix (remote is a directory)
        cmd = [
            "rclone",
            "copy",
            str(local),
            remote,
            "--ignore-existing",
        ]
    else:
        # Copy file to exact key
        cmd = [
            "rclone",
            "copyto",
            str(local),
            remote,
        ]

    _run(cmd, env=env, dry_run=dry_run)
    return remote


def _build_b2_refs(
    bucket: str,
    key: str,
    *,
    s3_region: Optional[str],
) -> Tuple[Optional[str], str]:
    """Return (public_url_if_known, opaque_uri)."""
    opaque = f"b2://{bucket}/{key}"
    if s3_region:
        # Virtual hosted-style URL is also common, but path-style is broadly supported
        public = f"https://s3.{s3_region}.backblazeb2.com/{bucket}/{key}"
        return public, opaque
    return None, opaque


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Publish dataset shard to B2 and log W&B reference artifact")
    p.add_argument("paths", nargs="+", help="File(s) or directory to upload")
    p.add_argument("--prefix", default=os.environ.get("B2_PREFIX", "pdebench"), help="Remote prefix inside the bucket")
    p.add_argument("--bucket", default=os.environ.get("B2_BUCKET"), help="Target B2 bucket (env B2_BUCKET)")
    p.add_argument("--artifact-name", default=None, help="W&B artifact name (default derives from first file)")
    p.add_argument("--artifact-type", default="dataset-shard", help="W&B artifact type")
    p.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "universal-simulator"), help="W&B project")
    p.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"), help="W&B entity")
    p.add_argument("--dry-run", action="store_true", help="Print commands and skip network and deletion")
    return p.parse_args()


def main() -> None:
    # Load .env best-effort
    _load_env_file(Path(".env"))

    args = parse_args()
    bucket = args.bucket or os.environ.get("B2_BUCKET")
    if not bucket:
        print("Error: B2_BUCKET must be set (env or --bucket)", file=sys.stderr)
        sys.exit(2)

    b2_key_id = os.environ.get("B2_KEY_ID")
    b2_app_key = os.environ.get("B2_APP_KEY")
    if not (b2_key_id and b2_app_key):
        print("Error: B2_KEY_ID and B2_APP_KEY must be set in env/.env", file=sys.stderr)
        sys.exit(2)

    s3_region = os.environ.get("B2_S3_REGION")

    inputs: List[Path] = [Path(p).expanduser().resolve() for p in args.paths]
    for p in inputs:
        if not p.exists():
            print(f"Missing path: {p}", file=sys.stderr)
            sys.exit(2)

    # Derive artifact name
    if args.artifact_name:
        art_name = args.artifact_name
    else:
        base = inputs[0].name
        art_name = base.replace(".tar.gz", "").replace(".tar", "")

    uploaded: List[Tuple[Path, str, Optional[str], str]] = []
    # Each tuple: (local_path, remote_key, public_url, opaque_uri)

    for local in inputs:
        # Destination key: <prefix>/<basename or dirname>
        if local.is_dir():
            dest_key = f"{args.prefix.rstrip('/')}/{local.name}/"
        else:
            dest_key = f"{args.prefix.rstrip('/')}/{local.name}"

        remote = _rclone_copyto(local, bucket, dest_key, b2_key_id=b2_key_id, b2_app_key=b2_app_key, dry_run=args.dry_run)
        public, opaque = _build_b2_refs(bucket, dest_key, s3_region=s3_region)
        print(f"Uploaded: {local} -> {remote}")
        uploaded.append((local, dest_key, public, opaque))

    # Log W&B reference artifact
    try:
        import wandb  # type: ignore
    except Exception as e:  # pragma: no cover
        print(f"Error: wandb not installed: {e}", file=sys.stderr)
        sys.exit(2)

    if args.dry_run:
        print(f"DRY_RUN: would log W&B artifact name={art_name} type={args.artifact_type} project={args.project}")
    else:
        run = wandb.init(project=args.project, entity=args.entity, job_type="dataset-publish")
        artifact = wandb.Artifact(name=art_name, type=args.artifact_type, metadata={
            "storage": "b2",
            "bucket": bucket,
            "prefix": args.prefix,
        })
        for local, dest_key, public, opaque in uploaded:
            ref = public or opaque
            artifact.add_reference(ref, name=Path(dest_key).name.rstrip("/"))
        wandb.log_artifact(artifact)
        run.finish()
        print(f"Logged W&B artifact: {art_name}")

    # Delete local shard files
    for local, _, _, _ in uploaded:
        if args.dry_run:
            print(f"DRY_RUN: would delete {local}")
            continue
        try:
            if local.is_dir():
                shutil.rmtree(local)
            else:
                local.unlink()
            print(f"Deleted local: {local}")
        except Exception as e:
            print(f"Warning: could not delete {local}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()


