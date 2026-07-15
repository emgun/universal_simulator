from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from scripts import d5_presigned_io as io_helper

PLAN = "a" * 64
PREFIX = "remote-runs/strat-v1-shared-tier-b"
HOST = "bucket.s3.us-west-000.backblazeb2.com"


def _url(key: str, verb: str) -> str:
    return f"https://{HOST}/{key}?X-Amz-Signature={verb}"


def _manifest(tmp_path: Path) -> Path:
    launch_id = "b" * 32
    resume = f"{PREFIX}/resumable/{PLAN}/output.tar"
    log = f"{PREFIX}/resumable/{PLAN}/remote.log"
    success = f"{PREFIX}/incoming/{PLAN}/{launch_id}/artifact.tar.gz"
    payload = {
        "schema_version": 1,
        "issued_unix": 1000,
        "expires_unix": 2000,
        "lock_sha256": "c" * 64,
        "plan_sha256": PLAN,
        "launch_id": launch_id,
        "artifact_prefix": PREFIX,
        "url_host": HOST,
        "objects": [],
        "slots": {
            "resume": {
                "key": resume,
                "put_url": _url(resume, "resume-put"),
                "get_url": _url(resume, "resume-get"),
            },
            "log": {"key": log, "put_url": _url(log, "log-put")},
            "success": {
                "key": success,
                "put_url": _url(success, "success-put"),
                "get_url": _url(success, "success-get"),
            },
            "success_sha256": {
                "key": f"{success}.sha256",
                "put_url": _url(f"{success}.sha256", "digest-put"),
            },
        },
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_manifest_seals_exact_output_keys_and_expiring_https_capabilities(tmp_path: Path) -> None:
    path = _manifest(tmp_path)
    payload = io_helper.checked_manifest(path, now=1500)
    assert set(payload["slots"]) == {"resume", "log", "success", "success_sha256"}

    payload["slots"]["success"]["key"] = f"{PREFIX}/arbitrary/upload.tar.gz"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(io_helper.TransferError, match="sealed"):
        io_helper.checked_manifest(path, now=1500)


def test_manifest_rejects_cross_host_and_overlong_lifetime(tmp_path: Path) -> None:
    path = _manifest(tmp_path)
    payload = json.loads(path.read_text())
    payload["slots"]["log"]["put_url"] = "https://evil.invalid/log?signature=x"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(io_helper.TransferError, match="host"):
        io_helper.checked_manifest(path, now=1500)

    path = _manifest(tmp_path)
    payload = json.loads(path.read_text())
    payload["expires_unix"] = payload["issued_unix"] + 12 * 60 * 60 + 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(io_helper.TransferError, match="lifetime"):
        io_helper.checked_manifest(path, now=1500)


def test_preserve_uses_only_resume_and_log_slots(tmp_path: Path, monkeypatch) -> None:
    manifest = _manifest(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    (output / "run_identity.json").write_text("{}")
    (output / "checkpoint.pt").write_bytes(b"weights")
    log = tmp_path / "remote.log"
    log.write_text("failure")
    uploads: list[tuple[str, bytes]] = []

    def fake_put(url: str, source: Path) -> None:
        uploads.append((url, source.read_bytes()))

    monkeypatch.setattr(io_helper, "_put_file", fake_put)
    monkeypatch.setattr(io_helper.time, "time", lambda: 1500)
    io_helper.preserve(manifest, output, log)

    assert [url for url, _ in uploads] == [
        json.loads(manifest.read_text())["slots"]["resume"]["put_url"],
        json.loads(manifest.read_text())["slots"]["log"]["put_url"],
    ]
    with tarfile.open(fileobj=io.BytesIO(uploads[0][1]), mode="r:") as bundle:
        assert {member.name for member in bundle.getmembers()} >= {
            "run_identity.json",
            "checkpoint.pt",
        }


def test_publish_uploads_digest_and_verifies_success_readback(tmp_path: Path, monkeypatch) -> None:
    manifest = _manifest(tmp_path)
    archive = tmp_path / "artifact.tar.gz"
    archive.write_bytes(b"sealed artifact")
    uploads: dict[str, bytes] = {}

    monkeypatch.setattr(io_helper.time, "time", lambda: 1500)
    monkeypatch.setattr(
        io_helper, "_put_file", lambda url, source: uploads.__setitem__(url, source.read_bytes())
    )

    def fake_get(url: str, destination: Path, *, absent_ok: bool = False) -> bool:
        destination.write_bytes(archive.read_bytes())
        return True

    monkeypatch.setattr(io_helper, "_get_to_file", fake_get)
    key, digest = io_helper.publish(manifest, archive)
    payload = json.loads(manifest.read_text())

    assert key == payload["slots"]["success"]["key"]
    assert digest == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert uploads[payload["slots"]["success_sha256"]["put_url"]] == (digest + "\n").encode()


def test_fetch_resume_refuses_link_members(tmp_path: Path, monkeypatch) -> None:
    manifest = _manifest(tmp_path)

    def fake_get(url: str, destination: Path, *, absent_ok: bool = False) -> bool:
        with tarfile.open(destination, "w") as bundle:
            member = tarfile.TarInfo("checkpoint-link")
            member.type = tarfile.SYMTYPE
            member.linkname = "/etc/passwd"
            bundle.addfile(member)
        return True

    monkeypatch.setattr(io_helper.time, "time", lambda: 1500)
    monkeypatch.setattr(io_helper, "_get_to_file", fake_get)
    with pytest.raises(io_helper.TransferError, match="forbidden"):
        io_helper.fetch_resume(manifest, tmp_path / "output")
