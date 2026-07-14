from __future__ import annotations

"""Metadata-only Hugging Face inventory construction for The Well.

The Hub model API exposes immutable Git LFS object sizes and SHA-256 values.
This module converts that metadata into the repository's source/protocol
manifest contracts without reading dataset payload bytes.
"""

import json
import re
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_ROLES = ("train", "valid", "test")


class HubInventoryError(RuntimeError):
    """The Hub metadata is incomplete, mutable, or internally inconsistent."""


def fetch_hub_inventory(repo_id: str, revision: str) -> Mapping[str, Any]:
    """Fetch repository metadata at an exact commit, never payload content."""

    if not _COMMIT_RE.fullmatch(revision):
        raise HubInventoryError("revision must be an exact 40-character commit SHA")
    encoded_repo = urllib.parse.quote(repo_id, safe="/")
    url = f"https://huggingface.co/api/datasets/{encoded_repo}/revision/{revision}?blobs=true"
    request = urllib.request.Request(url, headers={"User-Agent": "ups-data-inventory/1"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            raw = json.load(response)
    except (OSError, ValueError) as exc:
        raise HubInventoryError(f"could not retrieve Hub metadata: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise HubInventoryError("Hub metadata response must be an object")
    if raw.get("sha") != revision:
        raise HubInventoryError(
            f"Hub resolved revision to {raw.get('sha')!r}, expected exact commit {revision!r}"
        )
    return raw


def build_well_manifests(
    hub_metadata: Mapping[str, Any],
    *,
    repo_id: str,
    revision: str,
    package_version: str,
    package_commit: str,
    pilot_parameter: str = "0.03",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a complete HDF5 inventory and a one-file-per-split pilot protocol."""

    if not _COMMIT_RE.fullmatch(revision) or hub_metadata.get("sha") != revision:
        raise HubInventoryError("Hub metadata must be bound to the requested exact commit")
    if not _COMMIT_RE.fullmatch(package_commit):
        raise HubInventoryError("package_commit must be an exact 40-character commit SHA")
    if not re.fullmatch(r"v?\d+\.\d+\.\d+", package_version):
        raise HubInventoryError("package_version must be an exact semantic version")
    siblings = hub_metadata.get("siblings")
    if not isinstance(siblings, Sequence) or isinstance(siblings, (str, bytes)):
        raise HubInventoryError("Hub metadata does not contain a sibling inventory")

    dataset_name = repo_id.rsplit("/", 1)[-1]
    objects: list[dict[str, Any]] = []
    splits: dict[str, list[str]] = {role: [] for role in _ROLES}
    counts = {role: 0 for role in _ROLES}
    bytes_by_role = {role: 0 for role in _ROLES}
    pilot_suffix = f"tcool_{pilot_parameter}.hdf5"
    for sibling in siblings:
        if not isinstance(sibling, Mapping):
            continue
        source_path = sibling.get("rfilename")
        if not isinstance(source_path, str):
            continue
        parts = PurePosixPath(source_path).parts
        if len(parts) != 3 or parts[0] != "data" or parts[1] not in _ROLES:
            continue
        if PurePosixPath(source_path).suffix not in {".h5", ".hdf5"}:
            continue
        role = parts[1]
        lfs = sibling.get("lfs")
        if not isinstance(lfs, Mapping):
            raise HubInventoryError(f"dataset object {source_path!r} has no Git LFS identity")
        digest = lfs.get("sha256")
        size = lfs.get("size")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise HubInventoryError(f"dataset object {source_path!r} has no valid SHA-256")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise HubInventoryError(f"dataset object {source_path!r} has no valid size")
        if sibling.get("size") != size:
            raise HubInventoryError(f"dataset object {source_path!r} has inconsistent sizes")

        stem = PurePosixPath(source_path).stem.replace("_", "-")
        object_id = f"well-{dataset_name}-{role}-{stem}"
        manifest_path = f"{dataset_name}/{source_path}"
        uri_path = urllib.parse.quote(source_path, safe="/")
        objects.append(
            {
                "object_id": object_id,
                "path": manifest_path,
                "size_bytes": size,
                "checksums": {"sha256": digest},
                "uris": [
                    f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{uri_path}"
                ],
                "declared_roles": [role],
                "media_type": "application/x-hdf5",
                "metadata": {
                    "hub_path": source_path,
                    "git_lfs_sha256": digest,
                    "upstream_split": role,
                },
            }
        )
        counts[role] += 1
        bytes_by_role[role] += size
        if source_path.endswith(pilot_suffix):
            splits[role].append(object_id)

    if not objects:
        raise HubInventoryError("no native HDF5 data objects found in Hub metadata")
    missing_roles = [role for role in _ROLES if counts[role] == 0]
    if missing_roles:
        raise HubInventoryError(f"inventory is missing upstream splits: {missing_roles}")
    ambiguous_pilot = {role: ids for role, ids in splits.items() if len(ids) != 1}
    if ambiguous_pilot:
        raise HubInventoryError(
            f"pilot parameter must select exactly one object per split: {ambiguous_pilot}"
        )

    source_revision = f"hf:{revision}"
    source = {
        "schema_version": 1,
        "dataset_id": "the-well",
        "provider": "Polymathic AI via Hugging Face Hub",
        "revision": source_revision,
        "native_format": "The Well self-describing HDF5",
        "license": "CC-BY-4.0",
        "citation": "The Well, NeurIPS Datasets and Benchmarks 2024",
        "objects": sorted(objects, key=lambda item: item["object_id"]),
        "metadata": {
            "hub_repo": repo_id,
            "hub_commit": revision,
            "inventory_scope": "all native HDF5 split objects",
            "object_count_by_role": counts,
            "size_bytes_by_role": bytes_by_role,
            "the_well_package_version": package_version,
            "the_well_package_commit": package_commit,
        },
    }
    protocol = {
        "schema_version": 1,
        "protocol_id": "the-well-turbulent-radiative-layer-2d-pilot-v1",
        "dataset_id": "the-well",
        "source_revision": source_revision,
        "adapter": "the_well_native",
        "adapter_revision": "1.0.0",
        "split_authority": "upstream_trajectory_split",
        "splits": splits,
        "identity_fields": [
            "dataset_name",
            "source_file",
            "trajectory_index",
            "window_start",
        ],
        "selection": {
            "algorithm": "preserve_upstream_split_then_exact_parameter_file",
            "dataset_name": dataset_name,
            "pilot_parameter": {"tcool": pilot_parameter},
        },
        "normalization": {
            "fit_role": "train",
            "method": "channel_standardization",
            "preserve_physical_metadata": True,
            "artifact_required_for_roles": ["valid", "test"],
        },
        "test_access": "measurement_contract_required",
        "coverage_dimensions": ["dataset_name", "physical_system", "tcool"],
        "metadata": {
            "scope": "smallest one-file-per-upstream-split integration pilot",
            "not_a_benchmark_claim": True,
            "the_well_package_version": package_version,
            "the_well_package_commit": package_commit,
        },
    }
    return source, protocol
