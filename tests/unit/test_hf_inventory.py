from __future__ import annotations

import pytest

from ups.data.hf_inventory import HubInventoryError, build_well_manifests
from ups.data.manifests import (
    ProtocolManifest,
    SourceManifest,
    load_protocol_manifest,
    load_source_manifest,
    resolve_data_lock,
)
from ups.data.staging import staging_objects_from_lock

REVISION = "a" * 40
PACKAGE_COMMIT = "b" * 40


def _metadata(*, mismatched_size: bool = False) -> dict:
    siblings = []
    for index, role in enumerate(("train", "valid", "test"), start=1):
        size = index * 100
        siblings.append(
            {
                "rfilename": f"data/{role}/example_tcool_0.03.hdf5",
                "size": size + int(mismatched_size and role == "train"),
                "lfs": {"sha256": str(index) * 64, "size": size},
            }
        )
        siblings.append(
            {
                "rfilename": f"data/{role}/example_tcool_0.06.hdf5",
                "size": size,
                "lfs": {"sha256": str(index + 3) * 64, "size": size},
            }
        )
    siblings.append({"rfilename": "README.md", "size": 20, "blobId": "c" * 40})
    return {"sha": REVISION, "siblings": siblings}


def test_builds_complete_inventory_and_exact_upstream_pilot() -> None:
    source_raw, protocol_raw = build_well_manifests(
        _metadata(),
        repo_id="polymathic-ai/example",
        revision=REVISION,
        package_version="v1.2.0",
        package_commit=PACKAGE_COMMIT,
    )
    source = SourceManifest.from_dict(source_raw)
    protocol = ProtocolManifest.from_dict(protocol_raw)

    assert len(source.objects) == 6
    assert source.revision == f"hf:{REVISION}"
    assert {item.path for item in source.objects} == {
        f"example/data/{role}/example_tcool_{value}.hdf5"
        for role in ("train", "valid", "test")
        for value in ("0.03", "0.06")
    }
    assert all(len(protocol.splits[role]) == 1 for role in ("train", "valid", "test"))
    training_lock = resolve_data_lock(source, protocol)
    assert {item.role for item in training_lock.objects} == {"train", "valid"}
    assert all(REVISION in item.uris[0] for item in training_lock.objects)


def test_rejects_mutable_revision_and_inconsistent_lfs_metadata() -> None:
    with pytest.raises(HubInventoryError, match="exact commit"):
        build_well_manifests(
            {"sha": "main", "siblings": []},
            repo_id="polymathic-ai/example",
            revision="main",
            package_version="v1.2.0",
            package_commit=PACKAGE_COMMIT,
        )
    with pytest.raises(HubInventoryError, match="inconsistent sizes"):
        build_well_manifests(
            _metadata(mismatched_size=True),
            repo_id="polymathic-ai/example",
            revision=REVISION,
            package_version="v1.2.0",
            package_commit=PACKAGE_COMMIT,
        )


def test_requires_one_pilot_object_in_every_upstream_split() -> None:
    with pytest.raises(HubInventoryError, match="exactly one object per split"):
        build_well_manifests(
            _metadata(),
            repo_id="polymathic-ai/example",
            revision=REVISION,
            package_version="v1.2.0",
            package_commit=PACKAGE_COMMIT,
            pilot_parameter="9.99",
        )


def test_checked_well_pilot_preserves_distinct_native_split_paths() -> None:
    source = load_source_manifest("docs/data/catalog/the_well.yaml")
    protocol = load_protocol_manifest("docs/data/protocols/the_well_native_v1.yaml")
    lock = resolve_data_lock(source, protocol)

    objects = staging_objects_from_lock(lock)

    assert len(objects) == 2
    assert {item["name"] for item in objects} == {
        "turbulent_radiative_layer_2D/data/train/" "turbulent_radiative_layer_tcool_0.03.hdf5",
        "turbulent_radiative_layer_2D/data/valid/" "turbulent_radiative_layer_tcool_0.03.hdf5",
    }
