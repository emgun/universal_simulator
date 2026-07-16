#!/usr/bin/env python
"""Finalize a verified D5 ingress object using credentials only on the local host."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.d5_presigned_io import checked_manifest
from scripts.generate_b2_presigned_bundle import _load_literal_env, _setting

_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_ARCHIVE_STEM = re.compile(r"^[a-z0-9][a-z0-9_-]{0,127}$")
_WORKFLOW_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")


def _stream_sha256(body: Any) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    while block := body.read(8 << 20):
        digest.update(block)
        size += len(block)
    return digest.hexdigest(), size


def _client(env_file: Path) -> tuple[Any, str]:
    values = _load_literal_env(env_file)
    key_id = _setting(values, "B2_KEY_ID", "B2_ACCOUNT_ID")
    app_key = _setting(values, "B2_APP_KEY", "B2_APPLICATION_KEY")
    bucket = _setting(values, "B2_BUCKET", "B2_BUCKET_NAME")
    endpoint = _setting(values, "B2_S3_ENDPOINT")
    region = _setting(values, "B2_S3_REGION") or "us-west-000"
    if not key_id or not app_key or not bucket or not endpoint:
        raise ValueError("complete local B2 S3 configuration is required")
    import boto3  # type: ignore
    from botocore.config import Config  # type: ignore

    return (
        boto3.client(
            "s3",
            endpoint_url=endpoint,
            region_name=region,
            aws_access_key_id=key_id,
            aws_secret_access_key=app_key,
            config=Config(signature_version="s3v4"),
        ),
        bucket,
    )


def finalize(
    manifest_path: Path,
    *,
    env_file: Path,
    client: Any | None = None,
    bucket: str | None = None,
    receipt_path: Path | None = None,
    archive_stem: str = "strat_v1_shared_tier_b",
    workflow_label: str = "D5",
) -> str:
    if not _ARCHIVE_STEM.fullmatch(archive_stem):
        raise ValueError("archive stem must be a safe lowercase artifact name")
    if not _WORKFLOW_LABEL.fullmatch(workflow_label):
        raise ValueError("workflow label must be a short safe identifier")
    if receipt_path is not None:
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("managed Vast receipt is unavailable") from exc
        if receipt.get("status") != "succeeded" or receipt.get("destroyed") is not True:
            raise ValueError(
                f"refusing {workflow_label} finalization without a succeeded destroyed Vast receipt"
            )
    manifest = checked_manifest(manifest_path)
    if client is None:
        client, bucket = _client(env_file)
    if not bucket:
        raise ValueError("artifact bucket is required")
    slots = manifest["slots"]
    success_key = slots["success"]["key"]
    digest_key = slots["success_sha256"]["key"]
    try:
        digest_body = client.get_object(Bucket=bucket, Key=digest_key)["Body"].read()
        digest = digest_body.decode("ascii").strip()
        if not _HEX_64.fullmatch(digest):
            raise ValueError(f"{workflow_label} ingress digest sidecar is malformed")
        source_digest, source_size = _stream_sha256(
            client.get_object(Bucket=bucket, Key=success_key)["Body"]
        )
        if source_size <= 0:
            raise ValueError(f"{workflow_label} ingress artifact is empty")
        if source_digest != digest:
            raise ValueError(f"{workflow_label} ingress artifact differs from its digest sidecar")
        archive_name = f"{archive_stem}_{manifest['launch_id']}.tar.gz"
        immutable_key = f"{manifest['artifact_prefix']}/immutable/sha256/{digest}/{archive_name}"
        client.copy_object(
            Bucket=bucket,
            Key=immutable_key,
            CopySource={"Bucket": bucket, "Key": success_key},
            MetadataDirective="REPLACE",
            Metadata={"sha256": digest, "plan-sha256": manifest["plan_sha256"]},
            ContentType="application/gzip",
        )
        destination_digest, destination_size = _stream_sha256(
            client.get_object(Bucket=bucket, Key=immutable_key)["Body"]
        )
        if destination_size != source_size:
            raise ValueError(f"immutable {workflow_label} server-side copy size mismatch")
        if destination_digest != digest:
            raise ValueError(f"immutable {workflow_label} artifact read-back mismatch")
    except ValueError:
        raise
    except Exception:
        raise RuntimeError(
            f"{workflow_label} local finalization failed; ingress objects were retained"
        ) from None

    control_key = (
        f"{manifest['artifact_prefix']}/incoming/{manifest['plan_sha256']}/"
        f"{manifest['launch_id']}/transfer-manifest.json"
    )
    cleanup = [
        success_key,
        digest_key,
        control_key,
        slots["resume"]["key"],
        slots["log"]["key"],
    ]
    try:
        client.delete_objects(
            Bucket=bucket,
            Delete={"Objects": [{"Key": key} for key in cleanup], "Quiet": True},
        )
    except Exception:
        # Publication is already immutable and verified. Cleanup is best effort.
        pass
    return f"b2://{bucket}/{immutable_key}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--archive-stem", default="strat_v1_shared_tier_b")
    parser.add_argument("--workflow-label", default="D5")
    args = parser.parse_args()
    print(
        f"Published immutable {args.workflow_label} artifact: "
        f"{finalize(args.manifest, env_file=args.env_file, receipt_path=args.receipt, archive_stem=args.archive_stem, workflow_label=args.workflow_label)}"
    )


if __name__ == "__main__":
    main()
