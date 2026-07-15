#!/usr/bin/env python
from __future__ import annotations

"""Generate short-lived HTTPS GET capabilities for an exact training lock."""

import argparse
import json
import os
import secrets
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ups.data.manifests import load_data_lock

_B2_KEYS = {
    "B2_KEY_ID",
    "B2_ACCOUNT_ID",
    "B2_APP_KEY",
    "B2_APPLICATION_KEY",
    "B2_BUCKET",
    "B2_BUCKET_NAME",
    "B2_S3_ENDPOINT",
    "B2_S3_REGION",
}


def _load_literal_env(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
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
        if key not in _B2_KEYS:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def _setting(file_values: dict[str, str], *names: str) -> str | None:
    for name in names:
        if os.environ.get(name):
            return os.environ[name]
    for name in names:
        if file_values.get(name):
            return file_values[name]
    return None


def _b2_object(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "b2" or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError("training lock contains an invalid B2 object URI")
    key = parsed.path.removeprefix("/")
    if not key or any(part in {"", ".", ".."} for part in key.split("/")):
        raise ValueError("training lock contains an unsafe B2 object key")
    return parsed.netloc, key


def _atomic_private_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def generate_bundle(
    *,
    lock_path: Path,
    plan_path: Path,
    output_path: Path,
    env_file: Path,
    expires_in: int,
    artifact_prefix: str = "remote-runs/strat-v1-shared-tier-b",
    launch_id: str | None = None,
    client: Any | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    if expires_in < 60 or expires_in > 43200:
        raise ValueError("expires-in must be between 60 and 43200 seconds")
    lock = load_data_lock(lock_path)
    if lock.purpose != "training" or set(lock.requested_roles) - {"train", "valid"}:
        raise PermissionError("presigned input bundles require a train/valid-only training lock")

    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        plan_sha256 = plan["plan_sha256"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError("plan must contain its frozen plan_sha256") from exc
    if (
        not isinstance(plan_sha256, str)
        or len(plan_sha256) != 64
        or any(character not in "0123456789abcdef" for character in plan_sha256)
    ):
        raise ValueError("plan_sha256 must be lowercase hexadecimal")

    file_values = _load_literal_env(env_file)
    key_id = _setting(file_values, "B2_KEY_ID", "B2_ACCOUNT_ID")
    app_key = _setting(file_values, "B2_APP_KEY", "B2_APPLICATION_KEY")
    configured_bucket = _setting(file_values, "B2_BUCKET", "B2_BUCKET_NAME")
    endpoint = _setting(file_values, "B2_S3_ENDPOINT")
    region = _setting(file_values, "B2_S3_REGION") or "us-west-000"
    if not key_id or not app_key or not configured_bucket or not endpoint:
        raise ValueError("B2_KEY_ID, B2_APP_KEY, B2_BUCKET, and B2_S3_ENDPOINT are required")

    if client is None:
        import boto3  # type: ignore
        from botocore.config import Config  # type: ignore

        client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            region_name=region,
            aws_access_key_id=key_id,
            aws_secret_access_key=app_key,
            config=Config(signature_version="s3v4"),
        )

    objects: list[dict[str, str]] = []
    for item in lock.objects:
        if item.role not in {"train", "valid"}:
            raise PermissionError("presigned input bundle encountered a forbidden role")
        if len(item.uris) != 1:
            raise ValueError("each locked object must have exactly one B2 URI")
        uri = item.uris[0]
        bucket, key = _b2_object(uri)
        if bucket != configured_bucket:
            raise ValueError("training lock B2 bucket differs from configured B2_BUCKET")
        try:
            get_url = client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": key},
                ExpiresIn=expires_in,
            )
        except Exception:
            raise RuntimeError("presigned input capability generation failed") from None
        objects.append({"object_id": item.object_id, "uri": uri, "get_url": get_url})

    issued = now or datetime.now(timezone.utc)
    launch_id = launch_id or secrets.token_hex(16)
    prefix = artifact_prefix.strip("/")
    if (
        len(launch_id) != 32
        or any(character not in "0123456789abcdef" for character in launch_id)
        or not prefix
        or ".." in prefix.split("/")
    ):
        raise ValueError("launch identity or artifact prefix is unsafe")
    slot_keys = {
        "resume": f"{prefix}/resumable/{plan_sha256}/output.tar",
        "log": f"{prefix}/resumable/{plan_sha256}/remote.log",
        "success": f"{prefix}/incoming/{plan_sha256}/{launch_id}/artifact.tar.gz",
        "success_sha256": (f"{prefix}/incoming/{plan_sha256}/{launch_id}/artifact.tar.gz.sha256"),
    }
    slots: dict[str, dict[str, str]] = {}
    try:
        for name, key in slot_keys.items():
            slot = {
                "key": key,
                "put_url": client.generate_presigned_url(
                    "put_object",
                    Params={"Bucket": configured_bucket, "Key": key},
                    ExpiresIn=expires_in,
                ),
            }
            if name in {"resume", "success"}:
                slot["get_url"] = client.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": configured_bucket, "Key": key},
                    ExpiresIn=expires_in,
                )
            slots[name] = slot
    except Exception:
        raise RuntimeError("presigned output capability generation failed") from None
    url_host = urlparse(objects[0]["get_url"]).hostname if objects else None
    if not url_host:
        raise ValueError("presigned capability lacks an HTTPS host")
    payload = {
        "schema_version": 1,
        "lock_sha256": lock.lock_sha256,
        "plan_sha256": plan_sha256,
        "launch_id": launch_id,
        "artifact_prefix": prefix,
        "url_host": url_host,
        "issued_unix": issued.timestamp(),
        "expires_unix": (issued + timedelta(seconds=expires_in)).timestamp(),
        "issued_at": issued.astimezone(timezone.utc).isoformat(),
        "expires_at": (issued + timedelta(seconds=expires_in)).astimezone(timezone.utc).isoformat(),
        "objects": objects,
        "slots": slots,
    }
    _atomic_private_json(output_path, payload)
    return payload


def publish_control(
    *,
    manifest_path: Path,
    url_output: Path,
    env_file: Path,
    payload: dict[str, Any],
    expires_in: int,
    client: Any | None = None,
) -> None:
    file_values = _load_literal_env(env_file)
    key_id = _setting(file_values, "B2_KEY_ID", "B2_ACCOUNT_ID")
    app_key = _setting(file_values, "B2_APP_KEY", "B2_APPLICATION_KEY")
    bucket = _setting(file_values, "B2_BUCKET", "B2_BUCKET_NAME")
    endpoint = _setting(file_values, "B2_S3_ENDPOINT")
    region = _setting(file_values, "B2_S3_REGION") or "us-west-000"
    if not key_id or not app_key or not bucket or not endpoint:
        raise ValueError("complete B2 S3 configuration is required")
    if client is None:
        import boto3  # type: ignore
        from botocore.config import Config  # type: ignore

        client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            region_name=region,
            aws_access_key_id=key_id,
            aws_secret_access_key=app_key,
            config=Config(signature_version="s3v4"),
        )
    control_key = (
        f"{payload['artifact_prefix']}/incoming/{payload['plan_sha256']}/"
        f"{payload['launch_id']}/transfer-manifest.json"
    )
    try:
        client.put_object(
            Bucket=bucket,
            Key=control_key,
            Body=manifest_path.read_bytes(),
            ContentType="application/json",
        )
        control_url = client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": control_key},
            ExpiresIn=expires_in,
        )
    except Exception:
        raise RuntimeError("transfer control publication failed") from None
    _atomic_private_json(url_output, {"TRANSFER_MANIFEST_URL": control_url})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--artifact-prefix", default="remote-runs/strat-v1-shared-tier-b")
    parser.add_argument("--max-runtime-minutes", type=int, required=True)
    parser.add_argument("--url-output", type=Path)
    parser.add_argument("--upload-control", action="store_true")
    args = parser.parse_args()
    if args.max_runtime_minutes <= 0 or args.max_runtime_minutes > 600:
        raise SystemExit("max-runtime-minutes must be in 1..600")
    expires_in = args.max_runtime_minutes * 60 + 1800
    payload = generate_bundle(
        lock_path=args.lock,
        plan_path=args.plan,
        output_path=args.output,
        env_file=args.env_file,
        expires_in=expires_in,
        artifact_prefix=args.artifact_prefix,
    )
    if args.upload_control:
        if args.url_output is None:
            raise SystemExit("--upload-control requires --url-output")
        publish_control(
            manifest_path=args.output,
            url_output=args.url_output,
            env_file=args.env_file,
            payload=payload,
            expires_in=expires_in,
        )
    print(f"Wrote {len(payload['objects'])} presigned object capabilities to {args.output}")
    if args.upload_control:
        print(f"Wrote the private transfer URL receipt to {args.url_output}")


if __name__ == "__main__":
    main()
