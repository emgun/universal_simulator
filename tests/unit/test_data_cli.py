from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from ups.data.cli import evict_unpinned_cache, main, verify_lock_cache
from ups.data.manifests import load_data_lock


def _write_manifests(tmp_path: Path) -> tuple[Path, Path, Path]:
    source_bytes = b"locked training data"
    source_file = tmp_path / "source.h5"
    source_file.write_bytes(source_bytes)
    source_manifest = {
        "schema_version": 1,
        "dataset_id": "fixture-pde",
        "provider": "fixture",
        "revision": "sha256:0123456789abcdef",
        "native_format": "hdf5",
        "license": "CC-BY-4.0",
        "citation": "Fixture et al.",
        "objects": [
            {
                "object_id": "train-0",
                "path": "burgers1d_train.h5",
                "size_bytes": len(source_bytes),
                "checksums": {"sha256": hashlib.sha256(source_bytes).hexdigest()},
                "uris": [source_file.as_uri()],
                "declared_roles": ["train"],
            }
        ],
    }
    protocol_manifest = {
        "schema_version": 1,
        "protocol_id": "fixture-v1",
        "dataset_id": "fixture-pde",
        "source_revision": "sha256:0123456789abcdef",
        "adapter": "pdebench_hdf5",
        "adapter_revision": "1.0.0",
        "split_authority": "fixture",
        "splits": {"train": ["train-0"]},
        "identity_fields": ["source_file_id", "source_sample_index"],
        "selection": {"algorithm": "identity_hash", "seed": 7},
        "normalization": {"fit_role": "train", "method": "zscore"},
        "test_access": "measurement_contract_required",
    }
    source_path = tmp_path / "source.yaml"
    protocol_path = tmp_path / "protocol.yaml"
    source_path.write_text(yaml.safe_dump(source_manifest))
    protocol_path.write_text(yaml.safe_dump(protocol_manifest))
    return source_path, protocol_path, source_file


def test_cli_resolve_stage_and_verify(tmp_path: Path, capsys):
    source, protocol, source_file = _write_manifests(tmp_path)
    lock_path = tmp_path / "data.lock.json"
    cache = tmp_path / "cache"
    run_dir = tmp_path / "run"
    stage_report = tmp_path / "stage-report.json"

    assert (
        main(
            [
                "resolve",
                "--source",
                str(source),
                "--protocol",
                str(protocol),
                "--output",
                str(lock_path),
                "--roles",
                "train",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "stage",
                "--lock",
                str(lock_path),
                "--cache",
                str(cache),
                "--run-dir",
                str(run_dir),
                "--report",
                str(stage_report),
            ]
        )
        == 0
    )
    assert (run_dir / "burgers1d_train.h5").read_bytes() == source_file.read_bytes()

    lock = load_data_lock(lock_path)
    assert json.loads(stage_report.read_text())["lock_sha256"] == lock.lock_sha256
    report = verify_lock_cache(lock, cache)
    assert report["lock_sha256"] == lock.lock_sha256
    assert report["objects"][0]["id"] == "train-0"
    output = capsys.readouterr().out
    assert "lock_sha256" in output
    assert '"status": "complete"' in output


def test_cache_eviction_is_dry_run_and_respects_lock_pins(tmp_path: Path):
    source, protocol, _ = _write_manifests(tmp_path)
    lock_path = tmp_path / "data.lock.json"
    cache = tmp_path / "cache"
    run_dir = tmp_path / "run"
    main(
        [
            "resolve",
            "--source",
            str(source),
            "--protocol",
            str(protocol),
            "--output",
            str(lock_path),
            "--roles",
            "train",
        ]
    )
    main(["stage", "--lock", str(lock_path), "--cache", str(cache), "--run-dir", str(run_dir)])
    lock = load_data_lock(lock_path)

    pinned = evict_unpinned_cache(cache, [lock], apply=True)
    assert pinned["object_count"] == 0
    assert verify_lock_cache(lock, cache)["status"] == "verified"

    unpinned = evict_unpinned_cache(cache, [], apply=False)
    assert unpinned["object_count"] == 1
    assert Path(unpinned["objects"][0]).exists()
    applied = evict_unpinned_cache(cache, [], apply=True)
    assert applied["object_count"] == 1
    assert not Path(applied["objects"][0]).exists()
