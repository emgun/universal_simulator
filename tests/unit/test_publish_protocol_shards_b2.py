from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import h5py
import pytest
import yaml

SCRIPT = "scripts/publish_light_hdf5_shards_b2.sh"


def _canonical_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _publish_fixture(tmp_path):
    task = "advection1d"
    prefix = "strat-v1"
    bucket = "example-bucket"
    out_root = tmp_path / "shards"
    out_root.mkdir()
    shards = []
    for index, split in enumerate(("train", "val", "test")):
        shard = out_root / f"{task}_{split}.h5"
        with h5py.File(shard, "w") as handle:
            handle.create_dataset("data", data=[float(index)])
        digest = hashlib.sha256(shard.read_bytes()).hexdigest()
        key = f"{prefix}/immutable/sha256/{digest}/{shard.name}"
        shards.append((split, shard, digest, key))

    manifest_payload = {
        "version": "strat-v1",
        "remote_prefix": prefix,
        "protocol_mode": "strat-v1",
        "tasks": [task],
        "selection": {"algorithm": "sha256-protocol-seed-provenance-v1", "seed": 0},
        "protocol_gates": {task: {"status": "passed"}},
        "records": [
            {
                "task": task,
                "split": split,
                "output_path": str(shard),
                "bytes": shard.stat().st_size,
                "sha256": digest,
                "remote_key": key,
                "protocol_gate": {"status": "passed"},
            }
            for split, shard, digest, key in shards
        ],
    }
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(yaml.safe_dump(manifest_payload), encoding="utf-8")
    construction_digest = _canonical_sha256(manifest_payload)
    objects = []
    split_ids = {"train": [], "valid": [], "test": []}
    for split, shard, digest, key in shards:
        role = "valid" if split == "val" else split
        object_id = f"{task}-{role}"
        objects.append(
            {
                "object_id": object_id,
                "path": shard.name,
                "size_bytes": shard.stat().st_size,
                "checksums": {"sha256": digest},
                "uris": [f"b2://{bucket}/{key}", shard.as_uri()],
                "declared_roles": [role],
                "media_type": "application/x-hdf5",
            }
        )
        split_ids[role].append(object_id)
    source_payload = {
        "schema_version": 1,
        "dataset_id": "pdebench",
        "provider": "test",
        "revision": f"sha256:{construction_digest}",
        "native_format": "HDF5",
        "license": "CC BY 4.0",
        "citation": "test",
        "objects": objects,
        "metadata": {"construction_manifest_sha256": construction_digest},
    }
    protocol_payload = {
        "schema_version": 1,
        "protocol_id": "pdebench-strat-v1",
        "dataset_id": "pdebench",
        "source_revision": source_payload["revision"],
        "adapter": "pdebench_hdf5",
        "adapter_revision": "1.0.0",
        "split_authority": "constructed_trajectory_disjoint_parameter_stratified",
        "splits": split_ids,
        "identity_fields": ["source_file_id", "source_sample_index"],
        "selection": manifest_payload["selection"],
        "normalization": {"fit_role": "train", "method": "zscore"},
        "test_access": "measurement_contract_required",
        "metadata": {"construction_manifest_sha256": construction_digest},
    }
    source = tmp_path / "manifest.source.yaml"
    protocol = tmp_path / "manifest.protocol.yaml"
    source.write_text(yaml.safe_dump(source_payload), encoding="utf-8")
    protocol.write_text(yaml.safe_dump(protocol_payload), encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "OUT_ROOT": str(out_root),
            "MANIFEST": str(manifest),
            "SOURCE_MANIFEST": str(source),
            "PROTOCOL_MANIFEST": str(protocol),
            "REMOTE_PREFIX": prefix,
            "B2_BUCKET": bucket,
        }
    )
    return out_root, manifest, source, protocol, env


def _run(env):
    return subprocess.run(["bash", SCRIPT], capture_output=True, env=env, text=True)


def _write_lock(tmp_path, source_path, protocol_path, purpose):
    from ups.data.manifests import ProtocolManifest, SourceManifest

    source = yaml.safe_load(source_path.read_text())
    protocol = yaml.safe_load(protocol_path.read_text())
    roles = ["train", "valid"] if purpose == "training" else ["test"]
    source_by_id = {item["object_id"]: item for item in source["objects"]}
    objects = []
    for role in roles:
        for object_id in protocol["splits"][role]:
            item = source_by_id[object_id]
            objects.append(
                {
                    "object_id": object_id,
                    "role": role,
                    "path": item["path"],
                    "size_bytes": item["size_bytes"],
                    "checksums": item["checksums"],
                    "uris": item["uris"],
                    "media_type": item["media_type"],
                }
            )
    payload = {
        "schema_version": 1,
        "dataset_id": source["dataset_id"],
        "source_revision": source["revision"],
        "source_manifest_sha256": SourceManifest.from_dict(source).manifest_sha256,
        "protocol_id": protocol["protocol_id"],
        "protocol_manifest_sha256": ProtocolManifest.from_dict(protocol).manifest_sha256,
        "adapter": protocol["adapter"],
        "adapter_revision": protocol["adapter_revision"],
        "purpose": purpose,
        "requested_roles": roles,
        "measurement_contract_id": "test-measurement" if purpose == "measurement" else None,
        "objects": objects,
        "selection": protocol["selection"],
        "normalization": protocol["normalization"],
    }
    payload["lock_sha256"] = _canonical_sha256(payload)
    path = tmp_path / f"{purpose}.lock.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def test_publish_emits_immutable_objects_and_release_scoped_controls(tmp_path):
    _, manifest, source, _, env = _publish_fixture(tmp_path)
    release_id = yaml.safe_load(source.read_text())["revision"].removeprefix("sha256:")
    proc = _run(env)

    assert proc.returncode == 0, proc.stderr
    assert "/immutable/sha256/" in proc.stdout
    assert f"/releases/advection1d/{release_id}/construction.manifest.yaml" in proc.stdout
    assert f"/releases/advection1d/{release_id}/source.manifest.yaml" in proc.stdout
    assert f"/releases/advection1d/{release_id}/protocol.manifest.yaml" in proc.stdout
    assert f"{env['REMOTE_PREFIX']}/manifest.yaml" not in proc.stdout
    assert str(manifest) in proc.stdout


def test_publish_verification_hashes_artifacts_in_chunks():
    script = Path(SCRIPT).read_text(encoding="utf-8")
    assert "handle.read(1024 * 1024)" in script
    assert "candidate.read_bytes()" not in script


def test_publish_rejects_artifact_mutation(tmp_path):
    out_root, _, _, _, env = _publish_fixture(tmp_path)
    with (out_root / "advection1d_train.h5").open("ab") as handle:
        handle.write(b"mutated")
    proc = _run(env)
    assert proc.returncode != 0
    assert "bytes/hash do not match" in proc.stderr


def test_publish_rejects_multiple_tasks(tmp_path):
    _, manifest, _, _, env = _publish_fixture(tmp_path)
    payload = yaml.safe_load(manifest.read_text())
    payload["tasks"].append("burgers1d")
    payload["protocol_gates"]["burgers1d"] = {"status": "passed"}
    manifest.write_text(yaml.safe_dump(payload), encoding="utf-8")
    proc = _run(env)
    assert proc.returncode != 0
    assert "exactly one task" in proc.stderr


def test_publish_rejects_source_uri_for_wrong_bucket_or_key(tmp_path):
    _, _, source, _, env = _publish_fixture(tmp_path)
    payload = yaml.safe_load(source.read_text())
    payload["objects"][0]["uris"][0] = "b2://wrong-bucket/mutable/train.h5"
    source.write_text(yaml.safe_dump(payload), encoding="utf-8")
    proc = _run(env)
    assert proc.returncode != 0
    assert "first b2:// URI is not the immutable publish URI" in proc.stderr


def test_publish_rejects_mismatched_source_or_protocol_manifest(tmp_path):
    _, _, _, protocol, env = _publish_fixture(tmp_path)
    payload = yaml.safe_load(protocol.read_text())
    payload["metadata"]["construction_manifest_sha256"] = "0" * 64
    protocol.write_text(yaml.safe_dump(payload), encoding="utf-8")
    proc = _run(env)
    assert proc.returncode != 0
    assert "Protocol manifest does not match construction manifest" in proc.stderr


def test_publish_optional_canonical_pair_is_immutable_and_release_scoped(tmp_path):
    _, _, source, _, env = _publish_fixture(tmp_path)
    canonical = tmp_path / "advection1d_train.h5"
    canonical.write_bytes(b"canonical-source")
    digest = hashlib.sha256(canonical.read_bytes()).hexdigest()
    key = f"strat-v1/immutable/sha256/{digest}/{canonical.name}"
    record = tmp_path / "hydration.json"
    record.write_text(
        json.dumps(
            {
                "task": "advection1d",
                "output_bytes": canonical.stat().st_size,
                "output_sha256": digest,
                "remote_key": key,
                "uri": f"b2://example-bucket/{key}",
            }
        ),
        encoding="utf-8",
    )
    env.update({"CANONICAL_SOURCE": str(canonical), "HYDRATION_RECORD": str(record)})
    release_id = yaml.safe_load(source.read_text())["revision"].removeprefix("sha256:")
    proc = _run(env)
    assert proc.returncode == 0, proc.stderr
    assert key in proc.stdout
    assert f"releases/advection1d/{release_id}/canonical/source.json" in proc.stdout


def test_publish_validates_and_publishes_optional_locks_after_objects(tmp_path):
    _, _, source, protocol, env = _publish_fixture(tmp_path)
    training = _write_lock(tmp_path, source, protocol, "training")
    measurement = _write_lock(tmp_path, source, protocol, "measurement")
    env.update({"TRAINING_LOCK": str(training), "MEASUREMENT_LOCK": str(measurement)})
    proc = _run(env)
    assert proc.returncode == 0, proc.stderr
    assert "/training.lock.json" in proc.stdout
    assert "/measurement.lock.json" in proc.stdout
    assert proc.stdout.index("/immutable/sha256/") < proc.stdout.index("/training.lock.json")


def test_publish_rejects_lock_with_divergent_manifest_hash(tmp_path):
    _, _, source, protocol, env = _publish_fixture(tmp_path)
    training = _write_lock(tmp_path, source, protocol, "training")
    payload = json.loads(training.read_text())
    payload["source_manifest_sha256"] = "0" * 64
    payload_without_digest = dict(payload)
    payload_without_digest.pop("lock_sha256")
    payload["lock_sha256"] = _canonical_sha256(payload_without_digest)
    training.write_text(json.dumps(payload), encoding="utf-8")
    env["TRAINING_LOCK"] = str(training)
    proc = _run(env)
    assert proc.returncode != 0
    assert "training lock manifest hashes do not match" in proc.stderr


def test_publish_rejects_invalid_canonical_pair(tmp_path):
    _, _, _, _, env = _publish_fixture(tmp_path)
    canonical = tmp_path / "canonical.h5"
    canonical.write_bytes(b"canonical-source")
    record = tmp_path / "hydration.json"
    record.write_text(json.dumps({"task": "wrong", "output_bytes": 1, "output_sha256": "0" * 64}))
    env.update({"CANONICAL_SOURCE": str(canonical), "HYDRATION_RECORD": str(record)})
    proc = _run(env)
    assert proc.returncode != 0
    assert "Hydration record task does not match" in proc.stderr


@pytest.mark.parametrize("prefix", ["smoke-v1", "light-v1", "medium-v1"])
def test_publish_rejects_frozen_legacy_remote_prefix(tmp_path, prefix):
    _, manifest, _, _, env = _publish_fixture(tmp_path)
    env["REMOTE_PREFIX"] = prefix
    payload = yaml.safe_load(manifest.read_text())
    payload["remote_prefix"] = prefix
    manifest.write_text(yaml.safe_dump(payload), encoding="utf-8")
    proc = _run(env)
    assert proc.returncode != 0
    assert "frozen legacy prefix" in proc.stderr


def test_publish_rejects_missing_manifest_pair(tmp_path):
    _, _, source, _, env = _publish_fixture(tmp_path)
    source.unlink()
    proc = _run(env)
    assert proc.returncode != 0
    assert "Required publication manifest not found" in proc.stderr
