from __future__ import annotations

from pathlib import Path

import yaml

from scripts.audit_pdebench_inventory_bridge import audit_inventory
from ups.data.manifests import SourceManifest


def _write_inventory(path, files):
    path.write_text(
        yaml.safe_dump(
            {
                "files": files,
                "total_files": len(files),
                "total_bytes": sum(row["size_bytes"] for row in files),
            }
        ),
        encoding="utf-8",
    )


def test_checked_in_inventory_has_exact_bytes_but_cannot_invent_roles():
    report, source = audit_inventory(Path("docs/pdebench_manifest.yaml"))

    assert report["status"] == "blocked"
    assert report["byte_inventory_valid"] is True
    assert report["file_count"] == 336
    assert report["total_bytes"] == 3_902_903_970_421
    assert report["valid_md5_count"] == 336
    assert report["valid_url_count"] == 336
    assert report["missing_role_assignment_count"] == 336
    assert source is None


def test_complete_reviewed_role_sidecar_emits_valid_source_manifest(tmp_path):
    inventory = tmp_path / "inventory.yaml"
    roles = tmp_path / "roles.yaml"
    _write_inventory(
        inventory,
        [
            {
                "file_id": 7,
                "path": "1D/Burgers/Train/example.hdf5",
                "size_bytes": 123,
                "checksum_type": "MD5",
                "checksum": "a" * 32,
                "content_type": "application/x-hdf5",
            }
        ],
    )
    roles.write_text(yaml.safe_dump({"files": {7: ["train"]}}), encoding="utf-8")

    report, source = audit_inventory(inventory, role_assignments_path=roles)

    assert report["status"] == "ready"
    assert source is not None
    parsed = SourceManifest.from_dict(source)
    assert parsed.metadata_only is False
    assert parsed.objects[0].object_id == "darus-datafile-7"
    assert parsed.objects[0].checksums == {"md5": "a" * 32}
    assert parsed.objects[0].uris == (
        "https://darus.uni-stuttgart.de/api/access/datafile/7?format=original",
    )


def test_invalid_totals_and_checksum_fail_closed(tmp_path):
    inventory = tmp_path / "inventory.yaml"
    inventory.write_text(
        yaml.safe_dump(
            {
                "files": [
                    {
                        "file_id": 7,
                        "path": "example.h5",
                        "size_bytes": 123,
                        "checksum_type": "MD5",
                        "checksum": "not-md5",
                    }
                ],
                "total_files": 2,
                "total_bytes": 999,
            }
        ),
        encoding="utf-8",
    )

    report, source = audit_inventory(inventory)

    assert report["status"] == "blocked"
    assert report["byte_inventory_valid"] is False
    assert any("valid MD5" in error for error in report["errors"])
    assert any("total_files mismatch" in error for error in report["errors"])
    assert any("total_bytes mismatch" in error for error in report["errors"])
    assert source is None
