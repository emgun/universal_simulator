#!/usr/bin/env python3
"""Materialize immutable task and universal strat-v1 release controls.

The construction roots contain local ``file://`` fallbacks for development.
Release manifests deliberately remove those fallbacks so every runnable lock is
portable and resolves only content-addressed durable objects.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from ups.data.manifests import (
    ProtocolManifest,
    SourceManifest,
    canonical_sha256,
    resolve_data_lock,
    write_data_lock,
)

TASK_ROOTS = {
    "advection1d": Path("data/pdebench_advection_strat_v1_universal"),
    "burgers1d": Path("data/pdebench_burgers_strat_v1"),
    "darcy2d": Path("data/pdebench_darcy_strat_v1"),
}


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a mapping in {path}")
    return payload


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _remote_only(source: dict[str, Any]) -> dict[str, Any]:
    for item in source.get("objects", []):
        uris = item.get("uris", [])
        item["uris"] = [uri for uri in uris if str(uri).startswith("b2://")]
        if len(item["uris"]) != 1:
            raise ValueError(f"{item.get('object_id')} must have exactly one b2:// release URI")
    return source


def _materialize_task(task: str, root: Path, releases_root: Path) -> dict[str, Any]:
    construction = _load_yaml(root / "manifest.yaml")
    source_payload = _remote_only(_load_yaml(root / "manifest.source.yaml"))
    protocol_payload = _load_yaml(root / "manifest.protocol.yaml")
    release_id = str(source_payload["revision"]).removeprefix("sha256:")
    release_dir = releases_root / task / release_id

    _write_yaml(release_dir / "construction.manifest.yaml", construction)
    _write_yaml(release_dir / "source.manifest.yaml", source_payload)
    _write_yaml(release_dir / "protocol.manifest.yaml", protocol_payload)

    source = SourceManifest.from_dict(source_payload)
    protocol = ProtocolManifest.from_dict(protocol_payload)
    training = resolve_data_lock(source, protocol, requested_roles=("train", "valid"))
    measurement = resolve_data_lock(
        source,
        protocol,
        requested_roles=("test",),
        purpose="measurement",
        measurement_contract_id=f"strat-v1-{task}-publication-verification-20260713",
    )
    write_data_lock(release_dir / "training.lock.json", training)
    write_data_lock(release_dir / "measurement.lock.json", measurement)
    return {
        "task": task,
        "release_id": release_id,
        "release_dir": release_dir,
        "source": source_payload,
        "protocol": protocol_payload,
    }


def _materialize_universal(task_releases: list[dict[str, Any]], releases_root: Path) -> Path:
    identity = {
        item["task"]: {
            "source_revision": item["source"]["revision"],
            "construction_manifest_sha256": item["source"]["metadata"][
                "construction_manifest_sha256"
            ],
        }
        for item in task_releases
    }
    release_id = canonical_sha256(identity)
    release_dir = releases_root / "universal" / release_id
    objects = [obj for item in task_releases for obj in item["source"]["objects"]]
    source_payload = {
        "schema_version": 1,
        "dataset_id": "pdebench-strat-v1-universal",
        "provider": "UPS protocol-gated derivatives of PDEBench",
        "revision": f"sha256:{release_id}",
        "native_format": "HDF5",
        "license": "CC BY 4.0",
        "citation": "PDEBench, NeurIPS Datasets and Benchmarks 2022",
        "objects": objects,
        "metadata": {"task_releases": identity},
    }
    protocol_payload = {
        "schema_version": 1,
        "protocol_id": "pdebench-strat-v1-universal",
        "dataset_id": "pdebench-strat-v1-universal",
        "source_revision": f"sha256:{release_id}",
        "adapter": "pdebench_hdf5",
        "adapter_revision": "1.0.0",
        "split_authority": "constructed_trajectory_disjoint_parameter_stratified",
        "splits": {
            role: sorted(
                object_id
                for item in task_releases
                for object_id in item["protocol"]["splits"][role]
            )
            for role in ("train", "valid", "test")
        },
        "identity_fields": ["task", "source_file_identity", "source_sample_index"],
        "selection": {
            "algorithm": "sha256-protocol-seed-provenance-v1",
            "seed": 0,
            "protocol": "strat-v1",
        },
        "normalization": {"fit_role": "train", "method": "zscore"},
        "test_access": "measurement_contract_required",
        "coverage_dimensions": ["task", "physical_parameter_regime"],
        "metadata": {"task_releases": identity},
    }
    _write_yaml(release_dir / "source.manifest.yaml", source_payload)
    _write_yaml(release_dir / "protocol.manifest.yaml", protocol_payload)

    source = SourceManifest.from_dict(source_payload)
    protocol = ProtocolManifest.from_dict(protocol_payload)
    write_data_lock(
        release_dir / "training.lock.json",
        resolve_data_lock(source, protocol, requested_roles=("train", "valid")),
    )
    write_data_lock(
        release_dir / "measurement.lock.json",
        resolve_data_lock(
            source,
            protocol,
            requested_roles=("test",),
            purpose="measurement",
            measurement_contract_id="strat-v1-universal-publication-verification-20260713",
        ),
    )
    return release_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--releases-root",
        type=Path,
        default=Path("docs/data/releases/strat-v1"),
    )
    parser.add_argument(
        "--catalog-output",
        type=Path,
        default=Path("docs/data/catalog/pdebench.yaml"),
    )
    parser.add_argument(
        "--protocol-output",
        type=Path,
        default=Path("docs/data/protocols/strat_v1.yaml"),
    )
    args = parser.parse_args()
    task_releases = [
        _materialize_task(task, root, args.releases_root) for task, root in TASK_ROOTS.items()
    ]
    universal = _materialize_universal(task_releases, args.releases_root)
    _write_yaml(args.catalog_output, _load_yaml(universal / "source.manifest.yaml"))
    _write_yaml(args.protocol_output, _load_yaml(universal / "protocol.manifest.yaml"))
    print(json.dumps({"universal_release": str(universal)}, sort_keys=True))


if __name__ == "__main__":
    main()
