from __future__ import annotations

import json
import stat
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts import generate_b2_presigned_bundle as generator

LOCK = Path(
    "docs/data/releases/strat-v1/universal/"
    "9d43d283f04f5b8d17cf6126ad189075c53307e715d7d4f61af440c2fed155c1/"
    "training.lock.json"
)
PLAN = Path("docs/research/artifacts/strat_v1_shared_tier_b_plan.json")


class FakeS3:
    def __init__(self):
        self.calls = []
        self.put_calls = []

    def generate_presigned_url(self, operation, *, Params, ExpiresIn):
        self.calls.append((operation, Params, ExpiresIn))
        return f"https://signed.invalid/{Params['Key']}?secret=do-not-log"

    def put_object(self, **kwargs):
        self.put_calls.append(kwargs)


def test_generator_binds_exact_lock_objects_and_writes_private_file(tmp_path, monkeypatch):
    env_file = tmp_path / "b2.env"
    env_file.write_text(
        "B2_KEY_ID=id\nB2_APP_KEY=key\nB2_BUCKET=pdebench\n"
        "B2_S3_ENDPOINT=https://s3.example.invalid\n"
    )
    output = tmp_path / "bundle.json"
    client = FakeS3()
    now = datetime(2026, 7, 15, tzinfo=timezone.utc)
    monkeypatch.delenv("B2_BUCKET", raising=False)

    payload = generator.generate_bundle(
        lock_path=LOCK,
        plan_path=PLAN,
        output_path=output,
        env_file=env_file,
        expires_in=3600,
        client=client,
        now=now,
    )

    assert len(payload["objects"]) == 6
    assert len(client.calls) == 12
    assert {call[1]["Bucket"] for call in client.calls} == {"pdebench"}
    assert {call[0] for call in client.calls} == {"get_object", "put_object"}
    assert {call[2] for call in client.calls} == {3600}
    assert payload["expires_at"] == "2026-07-15T01:00:00+00:00"
    assert payload["plan_sha256"] == json.loads(PLAN.read_text())["plan_sha256"]
    assert len(payload["launch_id"]) == 32
    assert set(payload["slots"]) == {"resume", "log", "success", "success_sha256"}
    assert json.loads(output.read_text()) == payload
    assert stat.S_IMODE(output.stat().st_mode) == 0o600


@pytest.mark.parametrize("expires_in", [59, 43201])
def test_generator_rejects_unbounded_expiry(tmp_path, expires_in):
    with pytest.raises(ValueError, match="expires-in"):
        generator.generate_bundle(
            lock_path=LOCK,
            plan_path=PLAN,
            output_path=tmp_path / "bundle.json",
            env_file=tmp_path / "absent.env",
            expires_in=expires_in,
            client=FakeS3(),
        )


def test_publish_control_writes_private_url_receipt_without_exposing_url(tmp_path, monkeypatch):
    env_file = tmp_path / "b2.env"
    env_file.write_text(
        "B2_KEY_ID=id\nB2_APP_KEY=key\nB2_BUCKET=pdebench\n"
        "B2_S3_ENDPOINT=https://s3.example.invalid\n"
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    receipt = tmp_path / "receipt.json"
    payload = {
        "artifact_prefix": "remote-runs/strat-v1-shared-tier-b",
        "plan_sha256": "a" * 64,
        "launch_id": "b" * 32,
    }
    client = FakeS3()
    monkeypatch.delenv("B2_BUCKET", raising=False)

    generator.publish_control(
        manifest_path=manifest,
        url_output=receipt,
        env_file=env_file,
        payload=payload,
        expires_in=3600,
        client=client,
    )

    assert len(client.put_calls) == 1
    assert client.put_calls[0]["Key"].endswith("/transfer-manifest.json")
    assert "secret=do-not-log" in json.loads(receipt.read_text())["TRANSFER_MANIFEST_URL"]
    assert stat.S_IMODE(receipt.stat().st_mode) == 0o600
