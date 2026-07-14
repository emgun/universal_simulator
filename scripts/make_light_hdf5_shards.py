#!/usr/bin/env python
from __future__ import annotations

"""Build universally gated PDEBench-style train/validation/test shards."""

import argparse
import hashlib
import json
import urllib.parse
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

LEGACY_VERSION_LABELS = {"smoke-v1", "light-v1", "medium-v1"}
LEGACY_REMOTE_PREFIXES = {"smoke-v1", "light-v1", "medium-v1"}
SELECTION_ALGORITHM = "sha256-protocol-seed-provenance-v1"

try:
    from scripts.protocol_split_gates import evaluate_protocol_splits
except ModuleNotFoundError:  # Direct execution adds scripts/, not the repository root, to sys.path.
    from protocol_split_gates import evaluate_protocol_splits


def _source_paths(root: Path, task: str, split: str) -> list[Path]:
    exact = root / f"{task}_{split}.h5"
    if exact.exists():
        return [exact]
    return sorted(root.glob(f"{task}_{split}_*.h5"))


def _dataset_keys(handle: h5py.File, sample_count: int) -> list[str]:
    keys: list[str] = []
    for key, value in handle.items():
        if isinstance(value, h5py.Dataset) and value.shape and int(value.shape[0]) == sample_count:
            keys.append(key)
    return keys


def _validate_source_complete(path: Path, handle: h5py.File) -> None:
    if "sequential_hydration_complete" not in handle.attrs:
        return
    if not bool(handle.attrs["sequential_hydration_complete"]):
        raise ValueError(
            f"{path} is marked sequential_hydration_complete=False; "
            "refusing to build light shards from partial official hydration"
        )


def _read_indices(paths: Iterable[Path], indices: list[int]) -> dict[str, np.ndarray]:
    if not indices:
        raise ValueError("Requested indexed shard with no indices")

    remaining = sorted((int(index), position) for position, index in enumerate(indices))
    chunks: dict[str, list[np.ndarray | None]] = {}
    attrs: dict[str, dict[str, object]] = {}
    file_attrs: dict[str, object] | None = None
    base = 0

    for path in paths:
        if not remaining:
            break
        with h5py.File(path, "r") as handle:
            _validate_source_complete(path, handle)
            if file_attrs is None:
                file_attrs = dict(handle.attrs.items())
            if "data" not in handle:
                raise KeyError(f"{path} does not contain a 'data' dataset")
            total = int(handle["data"].shape[0])
            local_rows = [
                (index - base, position)
                for index, position in remaining
                if base <= index < base + total
            ]
            if local_rows:
                keys = _dataset_keys(handle, total)
                local_indices = [row for row, _ in local_rows]
                order_positions = [position for _, position in local_rows]
                for key in keys:
                    dataset = handle[key]
                    selected = np.asarray(dataset[local_indices])
                    slots = chunks.setdefault(key, [None] * len(indices))
                    attrs.setdefault(key, dict(dataset.attrs.items()))
                    for source_row, target_position in enumerate(order_positions):
                        slots[target_position] = selected[source_row : source_row + 1]
            remaining = [
                (index, position) for index, position in remaining if index >= base + total
            ]
            base += total

    if remaining:
        missing = ", ".join(str(index) for index, _ in remaining[:5])
        raise ValueError(f"Requested sample indices beyond available source rows: {missing}")

    merged: dict[str, np.ndarray] = {}
    for key, values in chunks.items():
        if any(value is None for value in values):
            raise ValueError(f"Internal indexed read error for dataset {key}")
        merged[key] = np.concatenate([value for value in values if value is not None], axis=0)
    for key, values in attrs.items():
        merged[f"__attrs__:{key}"] = values  # type: ignore[assignment]
    if file_attrs is not None:
        merged["__file_attrs__"] = file_attrs  # type: ignore[assignment]
    return merged


def _read_aligned_dataset(paths: Iterable[Path], key: str) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for path in paths:
        with h5py.File(path, "r") as handle:
            _validate_source_complete(path, handle)
            if "data" not in handle:
                raise KeyError(f"{path} does not contain a 'data' dataset")
            count = int(handle["data"].shape[0])
            if key not in handle or not isinstance(handle[key], h5py.Dataset):
                raise ValueError(f"{path} is missing required dataset: {key}")
            dataset = handle[key]
            if not dataset.shape or int(dataset.shape[0]) != count:
                raise ValueError(f"{path} dataset {key} is not aligned to {count} samples")
            chunks.append(np.asarray(dataset))
    if not chunks:
        raise ValueError(f"No source data available for required dataset: {key}")
    return np.concatenate(chunks, axis=0)


def _write_h5(path: Path, arrays: dict[str, np.ndarray], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    attrs = {
        key.removeprefix("__attrs__:"): value
        for key, value in arrays.items()
        if key.startswith("__attrs__:")
    }
    file_attrs = arrays.get("__file_attrs__", {})
    with h5py.File(path, "w") as handle:
        for attr_key, attr_value in file_attrs.items():  # type: ignore[union-attr]
            handle.attrs[attr_key] = attr_value
        for key, array in arrays.items():
            if key.startswith("__attrs__:") or key == "__file_attrs__":
                continue
            dataset = handle.create_dataset(key, data=array, compression="gzip", compression_opts=3)
            for attr_key, attr_value in attrs.get(key, {}).items():
                dataset.attrs[attr_key] = attr_value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity_value(value: object) -> object:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise ValueError(f"Unsupported provenance identity value: {value!r}")


def _identity_json(values: tuple[object, ...]) -> str:
    return json.dumps(
        [_identity_value(value) for value in values],
        ensure_ascii=True,
        separators=(",", ":"),
    )


def _selection_rank(*, protocol_id: str, seed: int, identity_json: str) -> str:
    payload = json.dumps(
        {
            "algorithm": SELECTION_ALGORITHM,
            "identity": json.loads(identity_json),
            "protocol_id": protocol_id,
            "seed": seed,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _identity_digest(identity_jsons: list[str]) -> str:
    digest = hashlib.sha256()
    for identity_json in identity_jsons:
        encoded = identity_json.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _h5_summary(path: Path) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    sample_count = 0
    with h5py.File(path, "r") as handle:
        for key, value in handle.items():
            if isinstance(value, h5py.Dataset):
                shape = [int(dim) for dim in value.shape]
                datasets[key] = {
                    "shape": shape,
                    "dtype": str(value.dtype),
                }
                if key == "data" and shape:
                    sample_count = shape[0]
    return {
        "sample_count": sample_count,
        "datasets": datasets,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def build_stratified_task_shard_records(
    *,
    root: Path,
    out_root: Path,
    task: str,
    source_split: str,
    train_count: int,
    val_count: int,
    test_count: int,
    overwrite: bool,
    provenance_datasets: list[str],
    regime_dataset: str,
    field_kind: str,
    time_axis: int | None,
    remote_prefix: str | None = None,
    content_addressed_remote: bool = False,
    selection_seed: int = 0,
    selection_protocol: str = "strat-v1",
) -> list[dict[str, Any]]:
    """Build the only supported protocol: deterministic, balanced, and disjoint."""
    counts = {"train": train_count, "val": val_count, "test": test_count}
    if any(count <= 0 for count in counts.values()):
        raise ValueError("The universal protocol requires positive train, val, and test counts")
    paths = _source_paths(root, task, source_split)
    if not paths:
        raise FileNotFoundError(root / f"{task}_{source_split}.h5")
    regime_values = _read_aligned_dataset(paths, regime_dataset)
    if regime_values.ndim != 1:
        raise ValueError(f"{task} identity dataset {regime_dataset} must be one-dimensional")
    provenance_values: dict[str, np.ndarray] = {}
    for key in provenance_datasets:
        values = _read_aligned_dataset(paths, key)
        if values.ndim != 1:
            raise ValueError(f"{task} identity dataset {key} must be one-dimensional")
        provenance_values[key] = values

    regimes: dict[object, list[int]] = {}
    for index, raw_value in enumerate(regime_values):
        value = raw_value.item() if isinstance(raw_value, np.generic) else raw_value
        if isinstance(value, (float, complex)) and not np.isfinite(value):
            raise ValueError(f"{task} regime dataset contains a non-finite value at row {index}")
        regimes.setdefault(value, []).append(index)
    if not regimes:
        raise ValueError(f"{task} source has no regimes")
    regime_count = len(regimes)
    per_split: dict[str, int] = {}
    for split, count in counts.items():
        per_regime, remainder = divmod(count, regime_count)
        if remainder or per_regime <= 0:
            raise ValueError(
                f"{task} {split} count {count} must be a positive multiple of "
                f"the {regime_count} regimes"
            )
        per_split[split] = per_regime
    needed_per_regime = sum(per_split.values())
    for regime, indices in regimes.items():
        if len(indices) < needed_per_regime:
            raise ValueError(
                f"{task} regime {regime!r} has {len(indices)} rows; "
                f"{needed_per_regime} are required"
            )

    identity_jsons = {
        index: _identity_json(tuple(provenance_values[key][index] for key in provenance_datasets))
        for index in range(len(regime_values))
    }
    if len(set(identity_jsons.values())) != len(identity_jsons):
        raise ValueError(f"{task} source contains provenance overlap in composite identities")
    # The terminal provenance component is the trajectory/sample identity shared by
    # parameter-regime files. Ranking by it keeps matched initial conditions in the
    # same split, while the full composite identity remains the uniqueness contract.
    ranking_identity_jsons = {
        index: _identity_json((provenance_values[provenance_datasets[-1]][index],))
        for index in range(len(regime_values))
    }

    selected: dict[str, list[int]] = {split: [] for split in counts}
    for regime in sorted(regimes, key=lambda value: _identity_json((value,))):
        indices = sorted(
            regimes[regime],
            key=lambda index: (
                _selection_rank(
                    protocol_id=selection_protocol,
                    seed=selection_seed,
                    identity_json=ranking_identity_jsons[index],
                ),
                identity_jsons[index],
            ),
        )
        cursor = 0
        for split in counts:
            take = per_split[split]
            selected[split].extend(indices[cursor : cursor + take])
            cursor += take
    selected_arrays = {split: _read_indices(paths, indices) for split, indices in selected.items()}
    selection_digests = {
        split: _identity_digest([identity_jsons[index] for index in indices])
        for split, indices in selected.items()
    }
    for split, arrays in selected_arrays.items():
        file_attrs = dict(arrays.get("__file_attrs__", {}))
        file_attrs.update(
            {
                "selection_algorithm": SELECTION_ALGORITHM,
                "selection_seed": selection_seed,
                "selection_protocol": selection_protocol,
                "selection_ranking_provenance_dataset": provenance_datasets[-1],
                "selected_identity_sha256": selection_digests[split],
            }
        )
        arrays["__file_attrs__"] = file_attrs  # type: ignore[assignment]
    gate = evaluate_protocol_splits(
        selected_arrays,
        provenance_datasets=provenance_datasets,
        regime_dataset=regime_dataset,
        field_kind=field_kind,
        time_axis=time_axis,
    )

    records: list[dict[str, Any]] = []
    output_paths = {split: out_root / f"{task}_{split}.h5" for split in counts}
    if not overwrite:
        existing = [str(path) for path in output_paths.values() if path.exists()]
        if existing:
            raise FileExistsError(
                f"Protocol output already exists; pass --overwrite to replace it: {', '.join(existing)}"
            )
    for split in counts:
        out_path = output_paths[split]
        _write_h5(out_path, selected_arrays[split], overwrite=overwrite)
        summary = _h5_summary(out_path)
        records.append(
            {
                "task": task,
                "split": split,
                "source_split": source_split,
                "preferred_source_split": source_split,
                "derived_from_source_split": split != source_split,
                "source_paths": [str(path) for path in paths],
                "selected_source_indices": selected[split],
                "selection_algorithm": SELECTION_ALGORITHM,
                "selection_seed": selection_seed,
                "selection_protocol": selection_protocol,
                "selection_ranking_provenance_dataset": provenance_datasets[-1],
                "selected_identity_sha256": selection_digests[split],
                "sample_count": summary["sample_count"],
                "output_path": str(out_path),
                "remote_key": (
                    (
                        f"{remote_prefix.rstrip('/')}/immutable/sha256/"
                        f"{summary['sha256']}/{out_path.name}"
                    )
                    if remote_prefix and content_addressed_remote
                    else (
                        f"{remote_prefix.rstrip('/')}/{task}/{out_path.name}"
                        if remote_prefix
                        else None
                    )
                ),
                "bytes": summary["bytes"],
                "sha256": summary["sha256"],
                "datasets": summary["datasets"],
                "protocol_gate": gate,
            }
        )
    return records


def _write_control_manifests(
    *,
    construction_manifest: dict[str, Any],
    records: list[dict[str, Any]],
    provenance_datasets: list[str],
    source_path: Path,
    protocol_path: Path,
    mirror_uri_prefix: str | None,
    content_addressed_mirror: bool,
) -> None:
    """Bridge gated shard construction into the immutable runtime control plane."""

    construction_digest = hashlib.sha256(
        json.dumps(construction_manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    revision = f"sha256:{construction_digest}"
    objects = []
    splits: dict[str, list[str]] = {"train": [], "valid": [], "test": []}
    for record in records:
        split = "valid" if record["split"] == "val" else str(record["split"])
        output = Path(str(record["output_path"])).resolve()
        object_id = f"{record['task']}-{split}"
        uris = [output.as_uri()]
        if mirror_uri_prefix:
            if content_addressed_mirror:
                mirror_suffix = f"immutable/sha256/{record['sha256']}/{output.name}"
            else:
                mirror_suffix = f"{record['task']}/{output.name}"
            uris.insert(
                0,
                f"{mirror_uri_prefix.rstrip('/')}/{mirror_suffix}",
            )
        objects.append(
            {
                "object_id": object_id,
                "path": output.name,
                "size_bytes": int(record["bytes"]),
                "checksums": {"sha256": str(record["sha256"])},
                "uris": uris,
                "declared_roles": [split],
                "media_type": "application/x-hdf5",
                "metadata": {
                    "task": record["task"],
                    "sample_count": record["sample_count"],
                    "selected_identity_sha256": record["selected_identity_sha256"],
                },
            }
        )
        splits[split].append(object_id)

    source = {
        "schema_version": 1,
        "dataset_id": "pdebench",
        "provider": "UPS protocol-gated derivative of PDEBench",
        "revision": revision,
        "native_format": "HDF5",
        "license": "CC BY 4.0",
        "citation": "PDEBench, NeurIPS Datasets and Benchmarks 2022",
        "objects": objects,
        "metadata": {
            "construction_manifest_sha256": construction_digest,
            "construction_protocol": construction_manifest["version"],
        },
    }
    protocol = {
        "schema_version": 1,
        "protocol_id": f"pdebench-{construction_manifest['version']}",
        "dataset_id": "pdebench",
        "source_revision": revision,
        "adapter": "pdebench_hdf5",
        "adapter_revision": "1.0.0",
        "split_authority": "constructed_trajectory_disjoint_parameter_stratified",
        "splits": {role: sorted(ids) for role, ids in splits.items()},
        "identity_fields": provenance_datasets,
        "selection": construction_manifest["selection"],
        "normalization": {"fit_role": "train", "method": "zscore"},
        "test_access": "measurement_contract_required",
        "coverage_dimensions": ["task", "physical_parameter_regime"],
        "metadata": {"construction_manifest_sha256": construction_digest},
    }
    source_path.parent.mkdir(parents=True, exist_ok=True)
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    protocol_path.write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build protocol-gated, provenance-disjoint HDF5 train/val/test shards"
    )
    parser.add_argument(
        "--root",
        default="data/pdebench",
        help="Directory containing source <task>_<split>.h5 files",
    )
    parser.add_argument(
        "--out-root", default="data/pdebench_strat_v1", help="Directory to write gated shards"
    )
    parser.add_argument(
        "--tasks", nargs="+", required=True, help="Task names, e.g. burgers1d advection1d darcy2d"
    )
    parser.add_argument(
        "--source-split", default="train", help="Source split to slice, default train"
    )
    parser.add_argument("--train-count", type=int, default=16)
    parser.add_argument("--val-count", type=int, default=8)
    parser.add_argument("--test-count", type=int, default=8)
    parser.add_argument("--manifest", required=True, help="YAML manifest path for gate evidence")
    parser.add_argument("--source-manifest", help="Output runtime source manifest")
    parser.add_argument("--protocol-manifest", help="Output runtime protocol manifest")
    parser.add_argument(
        "--mirror-uri-prefix",
        help="Optional exact HTTP(S) or b2:// mirror prefix for portable runtime locks",
    )
    parser.add_argument(
        "--content-addressed-mirror",
        action="store_true",
        help=(
            "Address mirror objects as immutable/sha256/<digest>/<logical-filename>; "
            "requires --mirror-uri-prefix"
        ),
    )
    parser.add_argument("--version", default="strat-v1", help="Manifest/data version label")
    parser.add_argument(
        "--selection-seed",
        type=int,
        default=0,
        help="Seed included in stable provenance-identity ranking",
    )
    parser.add_argument(
        "--remote-prefix", help="Optional remote key prefix to record for each output"
    )
    parser.add_argument(
        "--provenance-dataset",
        action="append",
        required=True,
        help="Sample-aligned provenance identity dataset; repeat for composite identities",
    )
    parser.add_argument(
        "--regime-dataset",
        required=True,
        help="Sample-aligned regime dataset",
    )
    parser.add_argument(
        "--field-kind",
        choices=("temporal", "steady"),
        required=True,
        help="Field semantics used by overlap checks",
    )
    parser.add_argument(
        "--time-axis",
        type=int,
        help="Temporal axis in the source HDF5 data dataset; axis 0 is samples",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.version in LEGACY_VERSION_LABELS:
        raise ValueError(
            f"Version label {args.version!r} is reserved for immutable legacy artifacts; "
            "use a strat-v1 label for new construction"
        )
    if args.remote_prefix in LEGACY_REMOTE_PREFIXES:
        raise ValueError(
            f"Remote prefix {args.remote_prefix!r} is reserved for immutable legacy artifacts"
        )
    if args.content_addressed_mirror and not args.mirror_uri_prefix:
        raise ValueError("--content-addressed-mirror requires --mirror-uri-prefix")
    effective_remote_prefix = args.remote_prefix
    if args.content_addressed_mirror:
        mirror = urllib.parse.urlparse(args.mirror_uri_prefix)
        if mirror.scheme == "b2":
            mirror_key_prefix = mirror.path.strip("/")
            if not mirror_key_prefix:
                raise ValueError("content-addressed b2 mirror requires a key prefix")
            if effective_remote_prefix and effective_remote_prefix.rstrip("/") != mirror_key_prefix:
                raise ValueError("--remote-prefix must match the b2 mirror key prefix")
            effective_remote_prefix = mirror_key_prefix

    root = Path(args.root)
    out_root = Path(args.out_root)
    written: list[Path] = []
    records: list[dict[str, Any]] = []
    if args.field_kind == "temporal" and args.time_axis is None:
        raise ValueError("Temporal fields require --time-axis")
    if args.field_kind == "steady" and args.time_axis is not None:
        raise ValueError("Steady fields reject --time-axis")
    for task in args.tasks:
        task_records = build_stratified_task_shard_records(
            root=root,
            out_root=out_root,
            task=str(task),
            source_split=args.source_split,
            train_count=args.train_count,
            val_count=args.val_count,
            test_count=args.test_count,
            overwrite=args.overwrite,
            provenance_datasets=args.provenance_dataset,
            regime_dataset=args.regime_dataset,
            field_kind=args.field_kind,
            time_axis=args.time_axis,
            remote_prefix=effective_remote_prefix,
            content_addressed_remote=args.content_addressed_mirror,
            selection_seed=args.selection_seed,
            selection_protocol=args.version,
        )
        records.extend(task_records)
        written.extend(Path(str(record["output_path"])) for record in task_records)

    for path in written:
        print(path)

    manifest = {
        "version": args.version,
        "source_root": str(root),
        "out_root": str(out_root),
        "remote_prefix": effective_remote_prefix,
        "tasks": [str(task) for task in args.tasks],
        "splits": {
            "train": {
                "samples": args.train_count,
                "preferred_source_split": args.source_split,
            },
            "val": {
                "samples": args.val_count,
                "preferred_source_split": args.source_split,
            },
            "test": {
                "samples": args.test_count,
                "preferred_source_split": args.source_split,
            },
        },
        "protocol_mode": "strat-v1",
        "selection": {
            "algorithm": SELECTION_ALGORITHM,
            "seed": args.selection_seed,
            "protocol": args.version,
        },
        "protocol_gates": {
            str(task): next(
                record["protocol_gate"] for record in records if record["task"] == str(task)
            )
            for task in args.tasks
        },
        "records": records,
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    source_manifest_path = (
        Path(args.source_manifest)
        if args.source_manifest
        else manifest_path.with_name(f"{manifest_path.stem}.source.yaml")
    )
    protocol_manifest_path = (
        Path(args.protocol_manifest)
        if args.protocol_manifest
        else manifest_path.with_name(f"{manifest_path.stem}.protocol.yaml")
    )
    _write_control_manifests(
        construction_manifest=manifest,
        records=records,
        provenance_datasets=list(args.provenance_dataset),
        source_path=source_manifest_path,
        protocol_path=protocol_manifest_path,
        mirror_uri_prefix=args.mirror_uri_prefix,
        content_addressed_mirror=args.content_addressed_mirror,
    )
    print(manifest_path)
    print(source_manifest_path)
    print(protocol_manifest_path)


if __name__ == "__main__":
    main()
