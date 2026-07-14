from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from ups.data.manifests import (
    ManifestError,
    ProtocolManifest,
    RunDataLock,
    SourceManifest,
    canonical_json_bytes,
    canonical_sha256,
    load_data_lock,
    load_protocol_manifest,
    load_source_manifest,
    resolve_data_lock,
    write_data_lock,
)


def _source_dict() -> dict:
    return {
        "schema_version": 1,
        "dataset_id": "fixture-pde",
        "provider": "fixture",
        "revision": "sha256:0123456789abcdef",
        "native_format": "hdf5",
        "license": "CC-BY-4.0",
        "citation": "Fixture et al.",
        "objects": [
            {
                "object_id": "valid-0",
                "path": "valid/part-0.h5",
                "size_bytes": 20,
                "checksums": {"sha256": "2" * 64},
                "uris": ["https://example.invalid/valid-0.h5"],
                "declared_roles": ["valid"],
            },
            {
                "object_id": "train-0",
                "path": "train/part-0.h5",
                "size_bytes": 10,
                "checksums": {"sha256": "1" * 64},
                "uris": ["https://example.invalid/train-0.h5"],
                "declared_roles": ["train"],
            },
            {
                "object_id": "test-0",
                "path": "test/part-0.h5",
                "size_bytes": 30,
                "checksums": {"sha256": "3" * 64},
                "uris": ["https://example.invalid/test-0.h5"],
                "declared_roles": ["test"],
            },
        ],
    }


def _protocol_dict() -> dict:
    return {
        "schema_version": 1,
        "protocol_id": "fixture-strat-v1",
        "dataset_id": "fixture-pde",
        "source_revision": "sha256:0123456789abcdef",
        "adapter": "fixture_hdf5",
        "adapter_revision": "1.0.0",
        "split_authority": "constructed_trajectory_disjoint",
        "splits": {"train": ["train-0"], "valid": ["valid-0"], "test": ["test-0"]},
        "identity_fields": ["source_file_id", "source_sample_index"],
        "selection": {"algorithm": "sha256_identity_rank", "seed": 17},
        "normalization": {"fit_role": "train", "method": "channel_standardization"},
        "test_access": "measurement_contract_required",
        "coverage_dimensions": ["coefficient"],
    }


def test_canonical_serialization_and_hash_ignore_mapping_order():
    left = {"b": [2, {"z": False, "a": None}], "a": 1}
    right = {"a": 1, "b": [2, {"a": None, "z": False}]}
    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert canonical_sha256(left) == canonical_sha256(right)


def test_source_manifest_rejects_mutable_revision_missing_checksums_and_duplicates():
    raw = _source_dict()
    raw["revision"] = "latest"
    with pytest.raises(ManifestError, match="must be immutable"):
        SourceManifest.from_dict(raw)

    raw = _source_dict()
    raw["revision"] = "refs/heads/main"
    with pytest.raises(ManifestError, match="must be immutable"):
        SourceManifest.from_dict(raw)

    raw = _source_dict()
    raw["objects"][0]["checksums"] = {}
    with pytest.raises(ManifestError, match="checksums must not be empty"):
        SourceManifest.from_dict(raw)

    raw = _source_dict()
    raw["objects"][1]["object_id"] = "valid-0"
    with pytest.raises(ManifestError, match="duplicate object_id"):
        SourceManifest.from_dict(raw)


def test_protocol_rejects_cross_split_overlap_and_non_train_statistics():
    raw = _protocol_dict()
    raw["splits"]["valid"] = ["train-0"]
    with pytest.raises(ManifestError, match="more than one protocol split"):
        ProtocolManifest.from_dict(raw)

    raw = _protocol_dict()
    raw["normalization"]["fit_role"] = "valid"
    with pytest.raises(ManifestError, match="fit_role must be 'train'"):
        ProtocolManifest.from_dict(raw)


def test_training_lock_is_deterministic_and_contains_only_requested_roles():
    source = SourceManifest.from_dict(_source_dict())
    protocol = ProtocolManifest.from_dict(_protocol_dict())
    lock = resolve_data_lock(source, protocol, requested_roles=("valid", "train"))

    assert lock.requested_roles == ("train", "valid")
    assert [(item.role, item.object_id) for item in lock.objects] == [
        ("train", "train-0"),
        ("valid", "valid-0"),
    ]
    assert not any(item.role == "test" for item in lock.objects)
    assert len(lock.lock_sha256) == 64
    lock.verify()

    # Source inventory and protocol split ordering are presentation details.
    source_reordered = _source_dict()
    source_reordered["objects"].reverse()
    protocol_reordered = _protocol_dict()
    protocol_reordered["splits"] = {
        "test": ["test-0"],
        "valid": ["valid-0"],
        "train": ["train-0"],
    }
    again = resolve_data_lock(
        SourceManifest.from_dict(source_reordered),
        ProtocolManifest.from_dict(protocol_reordered),
        requested_roles=("train", "valid"),
    )
    assert again.lock_sha256 == lock.lock_sha256
    assert again.to_dict() == lock.to_dict()


def test_test_bytes_require_measurement_purpose_and_contract():
    source = SourceManifest.from_dict(_source_dict())
    protocol = ProtocolManifest.from_dict(_protocol_dict())
    with pytest.raises(ManifestError, match="training run locks cannot contain test"):
        resolve_data_lock(source, protocol, requested_roles=("train", "test"))
    with pytest.raises(ManifestError, match="measurement_contract_id"):
        resolve_data_lock(source, protocol, requested_roles=("test",), purpose="measurement")

    lock = resolve_data_lock(
        source,
        protocol,
        requested_roles=("test",),
        purpose="measurement",
        measurement_contract_id="evaluation-2026-01",
    )
    assert lock.purpose == "measurement"
    assert lock.measurement_contract_id == "evaluation-2026-01"
    assert [item.object_id for item in lock.objects] == ["test-0"]


def test_resolution_rejects_unknown_or_role_mismatched_objects():
    source = SourceManifest.from_dict(_source_dict())
    raw = _protocol_dict()
    raw["splits"]["valid"] = ["missing"]
    with pytest.raises(ManifestError, match="unknown source object"):
        resolve_data_lock(source, ProtocolManifest.from_dict(raw), requested_roles=("valid",))

    raw = _protocol_dict()
    raw["splits"]["valid"] = ["train-0"]
    raw["splits"]["train"] = []
    with pytest.raises(ManifestError, match="not declared for protocol role"):
        resolve_data_lock(source, ProtocolManifest.from_dict(raw), requested_roles=("valid",))


def test_run_lock_detects_tampering():
    lock = resolve_data_lock(
        SourceManifest.from_dict(_source_dict()),
        ProtocolManifest.from_dict(_protocol_dict()),
    )
    tampered = deepcopy(lock.to_dict())
    tampered["objects"][0]["size_bytes"] += 1
    rebuilt = RunDataLock(
        **{
            **tampered,
            "objects": tuple(type(lock.objects[0])(**item) for item in tampered["objects"]),
        }
    )
    with pytest.raises(ManifestError, match="digest mismatch"):
        rebuilt.verify()


def test_run_lock_write_and_load_is_byte_deterministic(tmp_path):
    lock = resolve_data_lock(
        SourceManifest.from_dict(_source_dict()),
        ProtocolManifest.from_dict(_protocol_dict()),
    )
    first = tmp_path / "first.lock.json"
    second = tmp_path / "second.lock.json"
    write_data_lock(first, lock)
    write_data_lock(second, lock)
    assert first.read_bytes() == second.read_bytes()
    assert load_data_lock(first) == lock

    first.write_text(first.read_text().replace('"size_bytes":10', '"size_bytes":11'))
    with pytest.raises(ManifestError, match="digest mismatch"):
        load_data_lock(first)


def test_checked_in_catalog_readiness_matches_external_validation_state():
    pdebench_source = load_source_manifest("docs/data/catalog/pdebench.yaml")
    pdebench_protocol = load_protocol_manifest("docs/data/protocols/strat_v1.yaml")
    assert not pdebench_source.metadata_only
    assert not pdebench_protocol.metadata_only
    pdebench_lock = resolve_data_lock(pdebench_source, pdebench_protocol)
    assert pdebench_lock.lock_sha256 == (
        "5666fa8bb646b270d23077d1936c8ee80fa1cfeaf8f5870c55103fc58dd80afd"
    )
    assert sum(item.size_bytes for item in pdebench_lock.objects) == 427_029_641

    well_source = load_source_manifest("docs/data/catalog/the_well.yaml")
    well_protocol = load_protocol_manifest("docs/data/protocols/the_well_native_v1.yaml")
    assert not well_source.metadata_only
    assert not well_protocol.metadata_only
    lock = resolve_data_lock(well_source, well_protocol)
    assert lock.requested_roles == ("train", "valid")
    assert sum(item.size_bytes for item in lock.objects) == 763_363_328


def test_checked_in_darcy_source_control_matches_reviewed_inventory():
    source = load_source_manifest("docs/data/catalog/pdebench_darcy_source_v1.yaml")
    protocol = load_protocol_manifest("docs/data/protocols/pdebench_darcy_source_v1.yaml")
    inventory = yaml.safe_load(Path("docs/pdebench_manifest.yaml").read_text(encoding="utf-8"))
    reviewed_rows = {
        int(row["file_id"]): row
        for row in inventory["files"]
        if str(row["path"]).startswith("2D/DarcyFlow/")
    }

    assert set(reviewed_rows) == {133217, 133218, 133219, 133220, 133221}
    assert len(source.objects) == 5
    for item in source.objects:
        file_id = int(item.metadata["darus_file_id"])
        row = reviewed_rows[file_id]
        assert item.path == row["path"]
        assert item.size_bytes == row["size_bytes"]
        assert item.checksums == {"md5": row["checksum"]}
        assert item.declared_roles == ("train",)

    lock = resolve_data_lock(source, protocol, requested_roles=("train",))
    checked_lock = load_data_lock("docs/data/locks/pdebench_darcy_source_v1.training.lock.json")
    assert lock.requested_roles == ("train",)
    assert checked_lock == lock
    assert lock.lock_sha256 == ("d2e04f82ec63d9e6ebdac6f4773f7fb6fc32c6d68a54671fafd6218790745254")
    assert len(lock.objects) == 5
    assert sum(item.size_bytes for item in lock.objects) == 6_553_622_984
    assert protocol.test_access == "forbidden"
    assert protocol.selection["samples_per_beta"] == 78
    assert protocol.selection["downstream_protocol"]["per_beta"] == {
        "train": 52,
        "valid": 13,
        "test": 13,
    }
