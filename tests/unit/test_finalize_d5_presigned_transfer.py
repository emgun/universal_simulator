from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest

from scripts import finalize_d5_presigned_transfer as finalizer
from tests.unit.test_d5_presigned_io import _manifest


class FakeS3:
    def __init__(self, manifest: dict, artifact: bytes, digest: str):
        self.manifest = manifest
        self.objects = {
            manifest["slots"]["success"]["key"]: artifact,
            manifest["slots"]["success_sha256"]["key"]: (digest + "\n").encode(),
        }
        self.deleted = None

    def get_object(self, *, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[Key])}

    def copy_object(self, *, Bucket, Key, CopySource, **kwargs):
        self.objects[Key] = self.objects[CopySource["Key"]]

    def delete_objects(self, *, Bucket, Delete):
        self.deleted = Delete


def _receipt(tmp_path: Path, status: str = "succeeded") -> Path:
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps({"status": status, "destroyed": True}))
    return path


def test_finalizer_hashes_ingress_and_immutable_before_cleanup(tmp_path: Path, monkeypatch) -> None:
    manifest_path = _manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    artifact = b"verified archive bytes"
    digest = hashlib.sha256(artifact).hexdigest()
    client = FakeS3(manifest, artifact, digest)
    monkeypatch.setattr("scripts.d5_presigned_io.time.time", lambda: 1500)

    handle = finalizer.finalize(
        manifest_path,
        env_file=tmp_path / "absent.env",
        client=client,
        bucket="pdebench",
        receipt_path=_receipt(tmp_path),
    )

    assert f"/immutable/sha256/{digest}/" in handle
    immutable_key = handle.removeprefix("b2://pdebench/")
    assert client.objects[immutable_key] == artifact
    assert client.deleted is not None


def test_finalizer_refuses_failed_receipt_without_reading_objects(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    client = FakeS3(manifest, b"archive", hashlib.sha256(b"archive").hexdigest())
    with pytest.raises(ValueError, match="succeeded"):
        finalizer.finalize(
            manifest_path,
            env_file=tmp_path / "absent.env",
            client=client,
            bucket="pdebench",
            receipt_path=_receipt(tmp_path, "remote_failed"),
        )
    assert client.deleted is None


def test_finalizer_retains_ingress_when_sidecar_is_false(tmp_path: Path, monkeypatch) -> None:
    manifest_path = _manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    client = FakeS3(manifest, b"archive", "0" * 64)
    monkeypatch.setattr("scripts.d5_presigned_io.time.time", lambda: 1500)
    with pytest.raises(ValueError, match="differs"):
        finalizer.finalize(
            manifest_path,
            env_file=tmp_path / "absent.env",
            client=client,
            bucket="pdebench",
            receipt_path=_receipt(tmp_path),
        )
    assert client.deleted is None
