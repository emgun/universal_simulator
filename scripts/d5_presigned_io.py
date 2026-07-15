#!/usr/bin/env python
"""Narrow, credential-free remote I/O for the frozen D5 Vast workflow.

The trusted launcher creates a short-lived manifest containing object-scoped
S3 presigned URLs.  This module deliberately understands only the four D5
output slots; it never accepts B2 credentials, bucket-wide remotes, or an
arbitrary destination URL from the command line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
MAX_MANIFEST_LIFETIME_SECONDS = 12 * 60 * 60
TRANSFER_CHUNK_BYTES = 8 << 20
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_LAUNCH_ID = re.compile(r"^[0-9a-f]{32}$")


class TransferError(RuntimeError):
    """A fail-closed manifest, transfer, or integrity error."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        raise TransferError("presigned transfer refused an HTTP redirect")


_OPENER = urllib.request.build_opener(_NoRedirect)


def _safe_url(value: str, *, expected_host: str | None = None) -> str:
    if len(value) > 8192:
        raise TransferError("presigned URL is unreasonably long")
    parsed = urllib.parse.urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or not parsed.query
    ):
        raise TransferError("presigned URL must be a query-signed HTTPS URL")
    if expected_host is not None and parsed.hostname.lower() != expected_host.lower():
        raise TransferError("presigned URL host differs from the sealed transfer host")
    return parsed.hostname.lower()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TransferError("could not read the D5 transfer manifest") from exc
    if not isinstance(payload, dict):
        raise TransferError("D5 transfer manifest must be a JSON object")
    return payload


def checked_manifest(path: Path, *, now: float | None = None) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise TransferError("unsupported D5 transfer manifest schema")
    try:
        issued = float(
            payload.get("issued_unix")
            or datetime.fromisoformat(str(payload["issued_at"])).timestamp()
        )
        expires = float(
            payload.get("expires_unix")
            or datetime.fromisoformat(str(payload["expires_at"])).timestamp()
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise TransferError("D5 transfer manifest timestamps are invalid") from exc
    current = time.time() if now is None else now
    if issued <= 0 or expires <= issued or expires - issued > MAX_MANIFEST_LIFETIME_SECONDS:
        raise TransferError("D5 transfer manifest lifetime is invalid")
    if current >= expires:
        raise TransferError("D5 transfer manifest has expired")
    plan_sha = str(payload.get("plan_sha256", ""))
    launch_id = str(payload.get("launch_id", ""))
    prefix = str(payload.get("artifact_prefix", "")).strip("/")
    host = str(payload.get("url_host", "")).lower()
    if not _HEX_64.fullmatch(plan_sha) or not _LAUNCH_ID.fullmatch(launch_id):
        raise TransferError("D5 transfer identity is invalid")
    if not prefix or prefix.startswith("/") or ".." in prefix.split("/"):
        raise TransferError("D5 artifact prefix is unsafe")
    if not host or "/" in host or "@" in host:
        raise TransferError("D5 transfer host is invalid")

    expected_keys = {
        "resume": f"{prefix}/resumable/{plan_sha}/output.tar",
        "log": f"{prefix}/resumable/{plan_sha}/remote.log",
        "success": f"{prefix}/incoming/{plan_sha}/{launch_id}/artifact.tar.gz",
        "success_sha256": (f"{prefix}/incoming/{plan_sha}/{launch_id}/artifact.tar.gz.sha256"),
    }
    slots = payload.get("slots")
    if not isinstance(slots, dict) or set(slots) != set(expected_keys):
        raise TransferError("D5 transfer manifest must seal exactly four output slots")
    for name, expected_key in expected_keys.items():
        slot = slots[name]
        if not isinstance(slot, dict) or slot.get("key") != expected_key:
            raise TransferError(f"D5 {name} slot key is not sealed to the run identity")
        required = {"put_url", "get_url"} if name in {"resume", "success"} else {"put_url"}
        if not required.issubset(slot):
            raise TransferError(f"D5 {name} slot lacks a required capability")
        for field in required:
            _safe_url(str(slot[field]), expected_host=host)
    return payload


def fetch_manifest(url: str, output: Path) -> dict[str, Any]:
    host = _safe_url(url)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    try:
        with _OPENER.open(url, timeout=60) as response, temporary.open("wb") as sink:
            shutil.copyfileobj(response, sink, length=TRANSFER_CHUNK_BYTES)
        os.chmod(temporary, 0o600)
        payload = checked_manifest(temporary)
        if payload["url_host"].lower() != host:
            raise TransferError("control URL host differs from the sealed transfer host")
        os.replace(temporary, output)
        return payload
    except (OSError, urllib.error.URLError) as exc:
        temporary.unlink(missing_ok=True)
        raise TransferError("failed to fetch the D5 transfer manifest") from exc


def _put_file(url: str, source: Path) -> None:
    handle = source.open("rb")
    request = urllib.request.Request(
        url,
        data=handle,
        headers={"Content-Length": str(source.stat().st_size)},
        method="PUT",
    )
    try:
        with _OPENER.open(request, timeout=300) as response:
            if not 200 <= response.status < 300:
                raise TransferError("presigned upload returned a non-success status")
    except (OSError, urllib.error.URLError) as exc:
        raise TransferError("presigned upload failed") from exc
    finally:
        handle.close()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(TRANSFER_CHUNK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _get_to_file(url: str, destination: Path, *, absent_ok: bool = False) -> bool:
    temporary = destination.with_suffix(destination.suffix + ".partial")
    try:
        with _OPENER.open(url, timeout=300) as response, temporary.open("wb") as sink:
            shutil.copyfileobj(response, sink, length=TRANSFER_CHUNK_BYTES)
        os.replace(temporary, destination)
        return True
    except urllib.error.HTTPError as exc:
        temporary.unlink(missing_ok=True)
        if absent_ok and exc.code == 404:
            return False
        raise TransferError("presigned download failed") from exc
    except (OSError, urllib.error.URLError) as exc:
        temporary.unlink(missing_ok=True)
        raise TransferError("presigned download failed") from exc


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:") as bundle:
        members = bundle.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if member.name.startswith("/") or root not in (target, *target.parents):
                raise TransferError("resume archive contains an unsafe path")
            if member.issym() or member.islnk() or member.isdev():
                raise TransferError("resume archive contains a forbidden member type")
        bundle.extractall(destination, members=members)


def fetch_resume(manifest_path: Path, output_dir: Path) -> bool:
    manifest = checked_manifest(manifest_path)
    if output_dir.exists():
        raise TransferError("refusing to overlay an existing D5 output directory")
    with tempfile.TemporaryDirectory() as temporary:
        archive = Path(temporary) / "resume.tar"
        if not _get_to_file(manifest["slots"]["resume"]["get_url"], archive, absent_ok=True):
            return False
        _safe_extract(archive, output_dir)
    if not (output_dir / "run_identity.json").is_file():
        shutil.rmtree(output_dir, ignore_errors=True)
        raise TransferError("resume archive lacks run_identity.json")
    return True


def preserve(manifest_path: Path, output_dir: Path, run_log: Path) -> None:
    manifest = checked_manifest(manifest_path)
    if output_dir.is_dir():
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "output.tar"
            with tarfile.open(archive, "w") as bundle:
                for child in sorted(output_dir.rglob("*")):
                    bundle.add(child, arcname=child.relative_to(output_dir), recursive=False)
            _put_file(manifest["slots"]["resume"]["put_url"], archive)
    if run_log.is_file():
        _put_file(manifest["slots"]["log"]["put_url"], run_log)


def publish(manifest_path: Path, archive: Path) -> tuple[str, str]:
    manifest = checked_manifest(manifest_path)
    digest = _sha256(archive)
    with tempfile.TemporaryDirectory() as temporary:
        digest_path = Path(temporary) / "artifact.tar.gz.sha256"
        digest_path.write_text(digest + "\n", encoding="ascii")
        _put_file(manifest["slots"]["success"]["put_url"], archive)
        _put_file(manifest["slots"]["success_sha256"]["put_url"], digest_path)
        readback = Path(temporary) / "readback.tar.gz"
        _get_to_file(manifest["slots"]["success"]["get_url"], readback)
        if _sha256(readback) != digest:
            raise TransferError("D5 success artifact read-back mismatch")
    return str(manifest["slots"]["success"]["key"]), digest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch-manifest")
    fetch.add_argument("--url", required=True)
    fetch.add_argument("--output", type=Path, required=True)
    resume = sub.add_parser("fetch-resume")
    resume.add_argument("--manifest", type=Path, required=True)
    resume.add_argument("--output-dir", type=Path, required=True)
    keep = sub.add_parser("preserve")
    keep.add_argument("--manifest", type=Path, required=True)
    keep.add_argument("--output-dir", type=Path, required=True)
    keep.add_argument("--run-log", type=Path, required=True)
    upload = sub.add_parser("publish")
    upload.add_argument("--manifest", type=Path, required=True)
    upload.add_argument("--archive", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "fetch-manifest":
        fetch_manifest(args.url, args.output)
    elif args.command == "fetch-resume":
        raise SystemExit(0 if fetch_resume(args.manifest, args.output_dir) else 3)
    elif args.command == "preserve":
        preserve(args.manifest, args.output_dir, args.run_log)
    elif args.command == "publish":
        key, digest = publish(args.manifest, args.archive)
        print(f"Uploaded verified D5 ingress artifact: b2://{key} sha256={digest}")


if __name__ == "__main__":
    main()
