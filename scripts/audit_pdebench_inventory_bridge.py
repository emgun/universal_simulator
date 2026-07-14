#!/usr/bin/env python
from __future__ import annotations

"""Validate the legacy PDEBench byte inventory and bridge it without guessing roles.

The checked-in DaRUS inventory is authoritative for byte identity, but it does
not encode the train/valid/test authorization required by ``SourceManifest``.
This tool therefore audits all byte-level fields unconditionally and emits a
non-metadata source manifest only when an explicit, complete role-assignment
sidecar is supplied.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import yaml

from ups.data.manifests import SourceManifest

DATAFILE_URL = "https://darus.uni-stuttgart.de/api/access/datafile/{file_id}?format=original"
VALID_ROLES = frozenset({"train", "valid", "test"})
MD5_RE = re.compile(r"^[0-9a-f]{32}$")


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return payload


def _object_id(file_id: int) -> str:
    return f"darus-datafile-{file_id}"


def _source_url(file_id: int) -> str:
    return DATAFILE_URL.format(file_id=file_id)


def _url_is_exact(url: str, file_id: int) -> bool:
    parsed = urlparse(url)
    return (
        parsed.scheme == "https"
        and parsed.netloc == "darus.uni-stuttgart.de"
        and parsed.path == f"/api/access/datafile/{file_id}"
        and parse_qs(parsed.query) == {"format": ["original"]}
    )


def _load_role_assignments(path: Path | None) -> dict[int, tuple[str, ...]]:
    if path is None:
        return {}
    payload = _load_yaml(path)
    raw_assignments = payload.get("files", payload)
    if not isinstance(raw_assignments, dict):
        raise ValueError("role assignments must be a mapping keyed by DaRUS file ID")
    assignments: dict[int, tuple[str, ...]] = {}
    for raw_id, raw_roles in raw_assignments.items():
        try:
            file_id = int(raw_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid role-assignment file ID: {raw_id!r}") from exc
        if (
            not isinstance(raw_roles, list)
            or not raw_roles
            or any(not isinstance(role, str) or role not in VALID_ROLES for role in raw_roles)
        ):
            raise ValueError(
                f"role assignment for file {file_id} must be a non-empty subset of "
                f"{sorted(VALID_ROLES)}"
            )
        roles = tuple(sorted(set(raw_roles)))
        if len(roles) != len(raw_roles):
            raise ValueError(f"role assignment for file {file_id} contains duplicates")
        assignments[file_id] = roles
    return assignments


def audit_inventory(
    inventory_path: Path,
    *,
    role_assignments_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Return an audit report and, only when complete, a source manifest mapping."""

    payload = _load_yaml(inventory_path)
    rows = payload.get("files")
    if not isinstance(rows, list) or not rows:
        raise ValueError("inventory.files must be a non-empty list")
    assignments = _load_role_assignments(role_assignments_path)

    errors: list[str] = []
    seen_ids: set[int] = set()
    seen_paths: set[str] = set()
    objects: list[dict[str, Any]] = []
    missing_role_ids: list[int] = []
    byte_total = 0

    for index, raw in enumerate(rows):
        if not isinstance(raw, dict):
            errors.append(f"files[{index}] must be a mapping")
            continue
        try:
            file_id = int(raw["file_id"])
            if file_id <= 0 or isinstance(raw["file_id"], bool):
                raise ValueError
        except (KeyError, TypeError, ValueError):
            errors.append(f"files[{index}].file_id must be a positive integer")
            continue
        path = raw.get("path")
        if (
            not isinstance(path, str)
            or not path
            or path.startswith("/")
            or ".." in Path(path).parts
        ):
            errors.append(f"file {file_id} has an unsafe or missing relative path")
            continue
        if file_id in seen_ids:
            errors.append(f"duplicate file_id {file_id}")
        if path in seen_paths:
            errors.append(f"duplicate path {path!r}")
        seen_ids.add(file_id)
        seen_paths.add(path)

        size = raw.get("size_bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            errors.append(f"file {file_id} has invalid size_bytes")
            continue
        byte_total += size
        checksum_type = raw.get("checksum_type")
        checksum = raw.get("checksum")
        if (
            checksum_type != "MD5"
            or not isinstance(checksum, str)
            or not MD5_RE.fullmatch(checksum.lower())
        ):
            errors.append(f"file {file_id} must have a valid MD5 checksum")
            continue

        url = _source_url(file_id)
        if not _url_is_exact(url, file_id):
            errors.append(f"file {file_id} generated an invalid DaRUS URL")
            continue
        roles = assignments.get(file_id)
        if roles is None:
            missing_role_ids.append(file_id)
            continue
        objects.append(
            {
                "object_id": _object_id(file_id),
                "path": path,
                "size_bytes": size,
                "checksums": {"md5": checksum.lower()},
                "uris": [url],
                "declared_roles": list(roles),
                "media_type": str(raw.get("content_type") or "application/x-hdf5"),
                "metadata": {"darus_file_id": file_id},
            }
        )

    extra_role_ids = sorted(set(assignments) - seen_ids)
    if extra_role_ids:
        errors.append(f"role assignments reference unknown file IDs: {extra_role_ids}")
    declared_count = payload.get("total_files")
    if declared_count != len(rows):
        errors.append(f"total_files mismatch: declared {declared_count!r}, observed {len(rows)}")
    declared_bytes = payload.get("total_bytes")
    if declared_bytes != byte_total:
        errors.append(f"total_bytes mismatch: declared {declared_bytes!r}, observed {byte_total}")

    byte_inventory_valid = not errors
    complete = byte_inventory_valid and not missing_role_ids and len(objects) == len(rows)
    report = {
        "status": "ready" if complete else "blocked",
        "inventory": str(inventory_path),
        "file_count": len(rows),
        "total_bytes": byte_total,
        "byte_inventory_valid": byte_inventory_valid,
        "valid_md5_count": len(rows) if byte_inventory_valid else None,
        "valid_url_count": len(rows) if byte_inventory_valid else None,
        "role_assignment_count": len(assignments),
        "missing_role_assignment_count": len(missing_role_ids),
        "missing_role_file_ids": missing_role_ids,
        "errors": errors,
        "blocker": (
            None
            if complete
            else "A reviewed whole-object train/valid/test role assignment is required; "
            "the byte inventory alone does not establish scientific split authorization."
        ),
    }
    if not complete:
        return report, None

    source = {
        "schema_version": 1,
        "dataset_id": "pdebench",
        "provider": "DaRUS, University of Stuttgart",
        "revision": "darus:doi:10.18419/darus-2986@8.0",
        "native_format": "family-specific HDF5",
        "license": "CC BY 4.0",
        "citation": "PDEBench, NeurIPS Datasets and Benchmarks 2022",
        "metadata_only": False,
        "objects": sorted(objects, key=lambda item: item["object_id"]),
        "metadata": {
            "authority": "DaRUS DOI release 8.0 and checked-in PDEBench file inventory",
            "darus_dataset_version": "8.0",
            "darus_dataset_version_id": 4465,
            "generated_by": "scripts/audit_pdebench_inventory_bridge.py",
        },
    }
    # Validate against the actual control-plane type before allowing publication.
    SourceManifest.from_dict(source)
    return report, source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", default="docs/pdebench_manifest.yaml", type=Path)
    parser.add_argument("--role-assignments", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--output-manifest", type=Path)
    args = parser.parse_args()
    report, source = audit_inventory(args.inventory, role_assignments_path=args.role_assignments)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if args.output_manifest:
        if source is None:
            raise SystemExit("inventory bridge is blocked; no source manifest was written")
        args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.output_manifest.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if source is not None else 2)


if __name__ == "__main__":
    main()
