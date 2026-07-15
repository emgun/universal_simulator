from __future__ import annotations

import hashlib
import json
import subprocess
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from ups.data.staging import IntegrityError, StagingError, plan_staging, stage_objects
from ups.data.transports import TransportError, transfer_to_partial


def _object(path: Path, *, name: str = "train.h5", role: str = "train") -> dict:
    data = path.read_bytes()
    return {
        "id": name,
        "name": name,
        "role": role,
        "uri": str(path),
        "size": len(data),
        "checksum": {"algorithm": "sha256", "value": hashlib.sha256(data).hexdigest()},
    }


def test_local_stage_verifies_promotes_links_and_reports(tmp_path: Path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"immutable physics bytes")
    cache = tmp_path / "cache"
    run = tmp_path / "run"
    report_path = tmp_path / "report.json"

    report = stage_objects([_object(source)], cache, run_dir=run, report_path=report_path)

    assert (run / "train.h5").read_bytes() == source.read_bytes()
    assert report["bytes_transferred"] == source.stat().st_size
    assert report["cache_hits"] == 0
    assert json.loads(report_path.read_text())["status"] == "complete"

    second = stage_objects([_object(source)], cache)
    assert second["cache_hits"] == 1
    assert second["bytes_transferred"] == 0


def test_bad_checksum_never_promotes_partial(tmp_path: Path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"wrong bytes")
    obj = _object(source)
    obj["checksum"]["value"] = "0" * 64
    cache = tmp_path / "cache"

    with pytest.raises(IntegrityError):
        stage_objects([obj], cache)

    assert not list((cache / "objects").rglob("0" * 64))
    assert not list((cache / "objects").rglob("*.partial"))


def test_local_partial_copy_resumes(tmp_path: Path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"abcdefghij")
    obj = _object(source)
    digest = obj["checksum"]["value"]
    partial = tmp_path / "cache" / "objects" / "sha256" / digest[:2] / f".{digest}.partial"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"abcd")

    report = stage_objects([obj], tmp_path / "cache")
    assert report["bytes_transferred"] == 6
    assert Path(report["objects"][0]["cache_path"]).read_bytes() == b"abcdefghij"


def test_complete_partial_is_promoted_without_source(tmp_path: Path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"already complete")
    obj = _object(source)
    digest = obj["checksum"]["value"]
    partial = tmp_path / "cache" / "objects" / "sha256" / digest[:2] / f".{digest}.partial"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(source.read_bytes())
    source.unlink()

    report = stage_objects([obj], tmp_path / "cache")
    assert report["bytes_transferred"] == 0
    assert report["objects"][0]["source_uri"] == "completed-partial"


def test_test_role_is_denied_by_default_and_explicit_when_allowed(tmp_path: Path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"held out")
    obj = _object(source, name="test.h5", role="test")
    with pytest.raises(PermissionError):
        stage_objects([obj], tmp_path / "cache")

    report = stage_objects([obj], tmp_path / "cache", allow_test=True)
    assert report["object_count"] == 1


def test_other_roles_are_filtered(tmp_path: Path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"metadata")
    report = stage_objects([_object(source, role="metadata")], tmp_path / "cache")
    assert report["object_count"] == 0


def test_run_view_supports_nested_paths_and_rejects_traversal(tmp_path: Path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"data")
    obj = _object(source)
    obj["name"] = "nested/train.h5"
    stage_objects([obj], tmp_path / "cache", run_dir=tmp_path / "run")
    assert (tmp_path / "run/nested/train.h5").read_bytes() == b"data"

    unsafe = _object(source, name="unsafe.h5")
    unsafe["name"] = "../unsafe.h5"
    with pytest.raises(StagingError, match="safe relative"):
        stage_objects([unsafe], tmp_path / "other-cache", run_dir=tmp_path / "run")


def test_http_transfer(tmp_path: Path):
    served = tmp_path / "served"
    served.mkdir()
    source = served / "source.bin"
    source.write_bytes(b"downloaded over HTTP")

    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, *_args):
            pass

    def handler(*args, **kwargs):
        return QuietHandler(*args, directory=served, **kwargs)

    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        obj = _object(source)
        obj["uri"] = f"http://127.0.0.1:{server.server_port}/source.bin"
        report = stage_objects([obj], tmp_path / "cache")
    finally:
        server.shutdown()
        thread.join()
    assert report["bytes_transferred"] == source.stat().st_size


def test_b2_transfer_uses_fixed_remote_and_atomically_replaces_partial(tmp_path, monkeypatch):
    destination = tmp_path / "object.partial"
    destination.write_bytes(b"existing partial")
    calls = []
    monkeypatch.setenv("B2_KEY_ID", "process-key-id")
    monkeypatch.setenv("B2_APP_KEY", "process-app-key")

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        Path(args[3]).write_bytes(b"complete object")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr("ups.data.transports.subprocess.run", fake_run)

    transferred = transfer_to_partial("b2://ups-datasets/objects/sha256/abc123", destination)

    assert transferred == len(b"complete object")
    assert destination.read_bytes() == b"complete object"
    assert calls[0][0][:3] == [
        "rclone",
        "copyto",
        "UPSB2:ups-datasets/objects/sha256/abc123",
    ]
    assert calls[0][1]["check"] is False
    assert calls[0][1]["capture_output"] is True
    assert calls[0][1]["text"] is True
    assert calls[0][1]["env"]["RCLONE_CONFIG_UPSB2_TYPE"] == "b2"
    assert calls[0][1]["env"]["RCLONE_CONFIG_UPSB2_ACCOUNT"] == "process-key-id"
    assert calls[0][1]["env"]["RCLONE_CONFIG_UPSB2_KEY"] == "process-app-key"
    assert "process-key-id" not in " ".join(calls[0][0])
    assert "process-app-key" not in " ".join(calls[0][0])


def test_b2_presigned_override_uses_https_without_credentials_or_url_in_report(
    tmp_path, monkeypatch
):
    uri = "b2://ups-datasets/objects/sha256/abc123"
    signed_url = "https://objects.example.invalid/object?signature=secret"
    bundle = tmp_path / "presigned.json"
    bundle.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "objects": [{"object_id": "object", "uri": uri, "get_url": signed_url}],
            }
        )
    )
    monkeypatch.setenv("UPS_B2_PRESIGNED_URLS_FILE", str(bundle))
    monkeypatch.delenv("B2_KEY_ID", raising=False)
    monkeypatch.delenv("B2_APP_KEY", raising=False)
    observed = []

    def fake_http(url, destination, **_kwargs):
        observed.append(url)
        destination.write_bytes(b"signed bytes")
        return len(b"signed bytes")

    monkeypatch.setattr("ups.data.transports.download_http_resumable", fake_http)
    monkeypatch.setattr(
        "ups.data.transports.subprocess.run",
        lambda *_args, **_kwargs: pytest.fail("rclone must not run for a presigned override"),
    )

    destination = tmp_path / "object.partial"
    transferred = transfer_to_partial(uri, destination)

    assert transferred == len(b"signed bytes")
    assert destination.read_bytes() == b"signed bytes"
    assert observed == [signed_url]


def test_b2_presigned_override_fails_closed_and_redacts_signed_url(tmp_path, monkeypatch):
    uri = "b2://ups-datasets/object.h5"
    signed_url = "https://objects.example.invalid/object?signature=secret"
    bundle = tmp_path / "presigned.json"
    bundle.write_text(json.dumps({"schema_version": 1, "urls": {uri: signed_url}}))
    monkeypatch.setenv("UPS_B2_PRESIGNED_URLS_FILE", str(bundle))

    def fail_http(url, *_args, **_kwargs):
        raise TransportError(f"network failure for {url}")

    monkeypatch.setattr("ups.data.transports.download_http_resumable", fail_http)
    with pytest.raises(TransportError) as error:
        transfer_to_partial(uri, tmp_path / "object.partial")

    assert signed_url not in str(error.value)
    assert "signature=secret" not in str(error.value)

    missing = tmp_path / "missing.json"
    missing.write_text(json.dumps({"schema_version": 1, "urls": {}}))
    monkeypatch.setenv("UPS_B2_PRESIGNED_URLS_FILE", str(missing))
    with pytest.raises(TransportError, match="does not authorize"):
        transfer_to_partial(uri, tmp_path / "other.partial")


def test_b2_failure_preserves_existing_partial_and_hides_process_output(tmp_path, monkeypatch):
    destination = tmp_path / "object.partial"
    destination.write_bytes(b"keep me")
    monkeypatch.setenv("B2_KEY_ID", "secret-key-id")
    monkeypatch.setenv("B2_APP_KEY", "secret-app-key")

    def fake_run(args, **_kwargs):
        Path(args[3]).write_bytes(b"incomplete replacement")
        return subprocess.CompletedProcess(
            args, 9, stdout="credential-like-stdout", stderr="credential-like-stderr"
        )

    monkeypatch.setattr("ups.data.transports.subprocess.run", fake_run)

    with pytest.raises(TransportError, match="exit code 9") as error:
        transfer_to_partial("b2://ups-datasets/object.h5", destination)

    assert destination.read_bytes() == b"keep me"
    assert "credential-like" not in str(error.value)
    assert "secret-key-id" not in str(error.value)
    assert "secret-app-key" not in str(error.value)
    assert not list(tmp_path.glob(".object.partial.rclone-*"))


@pytest.mark.parametrize(
    "uri",
    [
        "b2://short/object",
        "b2://ups-datasets",
        "b2://user@ups-datasets/object",
        "b2://ups-datasets/../object",
        "b2://ups-datasets/a//object",
        "b2://ups-datasets/a%2Fobject",
        "b2://ups-datasets/object?version=1",
        "b2://ups-datasets/object#fragment",
        "b2://ups_datasets/object",
    ],
)
def test_b2_uri_validation_fails_closed(uri, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "ups.data.transports.subprocess.run",
        lambda *_args, **_kwargs: pytest.fail("rclone must not run for an unsafe URI"),
    )
    with pytest.raises(TransportError):
        transfer_to_partial(uri, tmp_path / "object.partial")


def test_b2_env_file_is_literal_process_env_wins_and_s3_uses_other_provider(tmp_path, monkeypatch):
    marker = tmp_path / "must-not-exist"
    env_file = tmp_path / "b2.env"
    env_file.write_text(
        "\n".join(
            [
                "B2_KEY_ID=file-key-id",
                "B2_APP_KEY=file-app-key",
                "B2_BUCKET=ups-datasets",
                "B2_S3_ENDPOINT=https://s3.example.invalid",
                "B2_S3_REGION=us-west-999",
                f"touch {marker}",
                f"MALFORMED=$(touch {marker})",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ENV_FILE", str(env_file))
    monkeypatch.setenv("B2_KEY_ID", "process-key-id")
    observed_env = {}

    def fake_run(args, **kwargs):
        observed_env.update(kwargs["env"])
        Path(args[3]).write_bytes(b"object")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr("ups.data.transports.subprocess.run", fake_run)

    transfer_to_partial("b2://ups-datasets/object", tmp_path / "object.partial")

    assert not marker.exists()
    assert observed_env["RCLONE_CONFIG_UPSB2_TYPE"] == "s3"
    assert observed_env["RCLONE_CONFIG_UPSB2_PROVIDER"] == "Other"
    assert observed_env["RCLONE_CONFIG_UPSB2_ACCESS_KEY_ID"] == "process-key-id"
    assert observed_env["RCLONE_CONFIG_UPSB2_SECRET_ACCESS_KEY"] == "file-app-key"
    assert observed_env["RCLONE_CONFIG_UPSB2_ENDPOINT"] == "https://s3.example.invalid"
    assert observed_env["RCLONE_CONFIG_UPSB2_REGION"] == "us-west-999"


def test_plan_reports_cache_and_missing_bytes(tmp_path: Path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"123456")
    obj = _object(source)
    plan = plan_staging([obj], tmp_path / "cache", reserve_bytes=10)
    assert plan["missing_bytes"] == 6
    assert plan["largest_object_bytes"] == 6
    assert plan["reserve_bytes"] == 10


def test_generic_checksum_algorithm(tmp_path: Path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"generic hash")
    obj = _object(source)
    obj["checksum"] = {
        "algorithm": "sha512",
        "value": hashlib.sha512(source.read_bytes()).hexdigest(),
    }
    report = stage_objects([obj], tmp_path / "cache")
    assert "/sha512/" in report["objects"][0]["cache_path"]


def test_digest_must_be_safe_hex(tmp_path: Path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"data")
    obj = _object(source)
    obj["checksum"]["value"] = "../" + "0" * 61
    with pytest.raises(StagingError, match="Invalid sha256 digest"):
        stage_objects([obj], tmp_path / "cache")
