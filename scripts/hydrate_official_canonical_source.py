#!/usr/bin/env python
from __future__ import annotations

"""Build a checksum-bound provenance source from official PDEBench files."""

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml


@dataclass(frozen=True)
class TaskContract:
    manifest_prefix: str
    parameter_name: str
    parameter_pattern: str
    expected_regimes: tuple[float, ...]
    field_kind: str
    time_axis: int | None


TASK_CONTRACTS = {
    "burgers1d": TaskContract(
        manifest_prefix="1D/Burgers/Train/",
        parameter_name="nu",
        parameter_pattern=r"Nu(?P<value>[0-9]+(?:\.[0-9]+)?)",
        expected_regimes=(0.001, 0.002, 0.004, 0.01, 0.02, 0.04, 0.1, 0.2, 0.4, 1.0, 2.0, 4.0),
        field_kind="temporal",
        time_axis=1,
    ),
    "darcy2d": TaskContract(
        manifest_prefix="2D/DarcyFlow/",
        parameter_name="beta",
        parameter_pattern=r"beta(?P<value>[0-9]+(?:\.[0-9]+)?)",
        expected_regimes=(0.01, 0.1, 1.0, 10.0, 100.0),
        field_kind="steady",
        time_axis=None,
    ),
}
SCHEMA_CONTRACT_PATH = Path("docs/protocols/canonical_source_schema.yaml")
SELECTION_ALGORITHM = "sha256-protocol-seed-provenance-v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonicalise_batch(array: np.ndarray, *, raw_layout: str | None) -> np.ndarray:
    """Convert an official batch to state/spatial/channel canonical layout."""

    if raw_layout == "time_x":
        if array.ndim != 3:
            raise ValueError(f"time_x expects [sample,time,x], got {array.shape}")
        return array[..., None]
    if raw_layout == "xy":
        if array.ndim != 3:
            raise ValueError(f"xy expects [sample,x,y], got {array.shape}")
        return array[:, None, :, :, None]
    if raw_layout == "channel_xy":
        if array.ndim != 4:
            raise ValueError(f"channel_xy expects [sample,channel,x,y], got {array.shape}")
        return np.transpose(array, (0, 2, 3, 1))[:, None, ...]
    if raw_layout is not None:
        raise ValueError(f"Unsupported official raw layout: {raw_layout}")
    if array.ndim in (2, 3, 4):
        return array[..., None]
    return array


def _regime_from_path(path: str, contract: TaskContract) -> float:
    match = re.search(contract.parameter_pattern, path, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Cannot parse {contract.parameter_name} from official path: {path}")
    return float(match.group("value"))


def load_official_catalog(manifest: Path, task: str) -> list[dict[str, Any]]:
    contract = TASK_CONTRACTS[task]
    payload = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
    rows = []
    for entry in payload.get("files", []):
        logical_path = str(entry.get("path", ""))
        if not logical_path.startswith(contract.manifest_prefix):
            continue
        if entry.get("checksum_type") != "MD5":
            raise ValueError(f"Official source must use MD5: {logical_path}")
        row = dict(entry)
        row["regime"] = _regime_from_path(logical_path, contract)
        rows.append(row)
    rows.sort(key=lambda row: float(row["regime"]))
    observed = tuple(float(row["regime"]) for row in rows)
    if observed != contract.expected_regimes:
        raise ValueError(
            f"{task} official regime catalog mismatch: observed={observed}, "
            f"expected={contract.expected_regimes}"
        )
    if len({int(row["file_id"]) for row in rows}) != len(rows):
        raise ValueError(f"{task} official catalog contains duplicate file IDs")
    return rows


def _resolve_raw_path(raw_root: Path, logical_path: str) -> Path:
    nested = raw_root / logical_path
    flat = raw_root / Path(logical_path).name
    matches = [path for path in (nested, flat) if path.exists()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one local source for {logical_path}; checked {nested} and {flat}"
        )
    return matches[0]


def load_frozen_schema(task: str, path: Path = SCHEMA_CONTRACT_PATH) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported canonical source schema version in {path}")
    schema = (payload.get("tasks") or {}).get(task)
    if not isinstance(schema, dict):
        raise ValueError(f"Canonical source schema is missing task {task!r}")
    if schema.get("status") != "frozen":
        raise ValueError(
            f"Canonical source schema for {task} is not frozen; inspect an official file first"
        )
    contract = TASK_CONTRACTS[task]
    if schema.get("field_kind") != contract.field_kind:
        raise ValueError(f"Frozen schema field_kind mismatch for {task}")
    if schema.get("parameter_name") != contract.parameter_name:
        raise ValueError(f"Frozen schema parameter_name mismatch for {task}")
    if contract.field_kind == "steady":
        for role in ("input", "target"):
            key_name = f"{role}_dataset_key"
            shape_name = f"expected_{role}_sample_shape"
            if not isinstance(schema.get(key_name), str) or not schema[key_name]:
                raise ValueError(f"Frozen schema {key_name} is invalid for {task}")
            shape = schema.get(shape_name)
            if (
                not isinstance(shape, list)
                or not shape
                or any(not isinstance(dim, int) or dim <= 0 for dim in shape)
            ):
                raise ValueError(f"Frozen schema {shape_name} is invalid for {task}")
        if schema["input_dataset_key"] == schema["target_dataset_key"]:
            raise ValueError(f"Frozen schema must distinguish input and target for {task}")
    else:
        if not isinstance(schema.get("dataset_key"), str) or not schema["dataset_key"]:
            raise ValueError(f"Frozen schema dataset_key is invalid for {task}")
        shape = schema.get("expected_sample_shape")
        if (
            not isinstance(shape, list)
            or not shape
            or any(not isinstance(dim, int) or dim <= 0 for dim in shape)
        ):
            raise ValueError(f"Frozen schema expected_sample_shape is invalid for {task}")
    if not isinstance(schema.get("semantic_role"), str) or not schema["semantic_role"]:
        raise ValueError(f"Frozen schema semantic_role is invalid for {task}")
    return dict(schema)


def _field_fingerprint(sample: np.ndarray, contract: TaskContract) -> str:
    field = np.asarray(sample)
    if contract.field_kind == "temporal":
        assert contract.time_axis is not None
        field = np.take(field, 0, axis=contract.time_axis - 1)
    contiguous = np.ascontiguousarray(field)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _selection_rank(*, protocol_id: str, seed: int, sample_index: int) -> str:
    payload = json.dumps(
        {
            "algorithm": SELECTION_ALGORITHM,
            # Official parameter-regime files share their row/sample identity.
            # Select by that terminal identity so every regime retains the same
            # source rows and the downstream splitter can keep matched samples
            # together. The full (file_id, sample_index) identity remains the
            # uniqueness and evidence contract.
            "identity": [sample_index],
            "protocol_id": protocol_id,
            "seed": seed,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _selected_identity_digest(identities: list[tuple[int, int]]) -> str:
    digest = hashlib.sha256()
    for identity in identities:
        encoded = json.dumps(identity, separators=(",", ":")).encode("ascii")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _read_ranked_rows(
    source: h5py.Dataset,
    *,
    count: int,
    protocol_id: str,
    seed: int,
) -> tuple[np.ndarray, list[int]]:
    ranked_indices = sorted(
        range(int(source.shape[0])),
        key=lambda sample_index: (
            _selection_rank(
                protocol_id=protocol_id,
                seed=seed,
                sample_index=sample_index,
            ),
            sample_index,
        ),
    )[:count]
    ascending = sorted(ranked_indices)
    rows = np.asarray(source[ascending])
    by_index = {sample_index: position for position, sample_index in enumerate(ascending)}
    return rows[[by_index[sample_index] for sample_index in ranked_indices]], ranked_indices


def _read_rows(source: h5py.Dataset, indices: list[int]) -> np.ndarray:
    ascending = sorted(indices)
    rows = np.asarray(source[ascending])
    by_index = {sample_index: position for position, sample_index in enumerate(ascending)}
    return rows[[by_index[sample_index] for sample_index in indices]]


def hydrate_canonical_source(
    *,
    manifest: Path,
    raw_root: Path,
    out_path: Path,
    task: str,
    samples_per_regime: int,
    schema_contract_path: Path = SCHEMA_CONTRACT_PATH,
    overwrite: bool = False,
    selection_seed: int = 0,
    selection_protocol: str = "strat-v1-source-v1",
) -> dict[str, Any]:
    if samples_per_regime <= 0:
        raise ValueError("samples_per_regime must be positive")
    if task not in TASK_CONTRACTS:
        raise ValueError(f"Unsupported canonical task: {task}")
    contract = TASK_CONTRACTS[task]
    schema = load_frozen_schema(task, schema_contract_path)
    if contract.field_kind == "steady":
        input_key = str(schema["input_dataset_key"])
        target_key = str(schema["target_dataset_key"])
        expected_input_shape = tuple(int(dim) for dim in schema["expected_input_sample_shape"])
        expected_target_shape = tuple(int(dim) for dim in schema["expected_target_sample_shape"])
    else:
        input_key = str(schema["dataset_key"])
        target_key = None
        expected_input_shape = tuple(int(dim) for dim in schema["expected_sample_shape"])
        expected_target_shape = None
    catalog = load_official_catalog(manifest, task)
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} exists; pass --overwrite to replace it")

    verified: list[dict[str, Any]] = []
    target_shape: tuple[int, ...] | None = None
    chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    file_ids: list[np.ndarray] = []
    file_indices: list[np.ndarray] = []
    sample_indices: list[np.ndarray] = []
    regimes: list[np.ndarray] = []
    seen_fields: dict[str, tuple[float, int]] = {}
    selected_identities: list[tuple[int, int]] = []
    for source_index, entry in enumerate(catalog):
        logical_path = str(entry["path"])
        source_path = _resolve_raw_path(raw_root, logical_path)
        expected_size = int(entry["size_bytes"])
        if source_path.stat().st_size != expected_size:
            raise ValueError(
                f"Official source size mismatch for {logical_path}: "
                f"{source_path.stat().st_size} != {expected_size}"
            )
        observed_md5 = _md5_file(source_path)
        if observed_md5 != str(entry["checksum"]):
            raise ValueError(f"Official source MD5 mismatch for {logical_path}")
        with h5py.File(source_path, "r") as handle:
            if input_key not in handle or not isinstance(handle[input_key], h5py.Dataset):
                raise ValueError(f"{logical_path} is missing explicit dataset {input_key!r}")
            source = handle[input_key]
            if source.dtype.kind not in "fiu" or source.ndim < 2:
                raise ValueError(
                    f"{logical_path} dataset {input_key!r} is not a numeric field batch"
                )
            if int(source.shape[0]) < samples_per_regime:
                raise ValueError(
                    f"{logical_path} has {source.shape[0]} rows; {samples_per_regime} required"
                )
            if tuple(int(dim) for dim in source.shape[1:]) != expected_input_shape:
                raise ValueError(
                    f"Official source raw sample shape mismatch for {logical_path}: "
                    f"{tuple(source.shape[1:])} != {expected_input_shape}"
                )
            file_id = int(entry["file_id"])
            raw_batch, selected_sample_indices = _read_ranked_rows(
                source,
                count=samples_per_regime,
                protocol_id=selection_protocol,
                seed=selection_seed,
            )
            batch = _canonicalise_batch(
                raw_batch, raw_layout=schema.get("input_raw_layout")
            ).astype(np.float32, copy=False)
            target_batch: np.ndarray | None = None
            if target_key is not None:
                if target_key not in handle or not isinstance(handle[target_key], h5py.Dataset):
                    raise ValueError(
                        f"{logical_path} is missing required target dataset {target_key!r}"
                    )
                target_source = handle[target_key]
                if target_source.dtype.kind not in "fiu" or target_source.ndim < 2:
                    raise ValueError(
                        f"{logical_path} target {target_key!r} is not a numeric field batch"
                    )
                if int(target_source.shape[0]) != int(source.shape[0]):
                    raise ValueError(f"{logical_path} input and target sample counts differ")
                assert expected_target_shape is not None
                if tuple(int(dim) for dim in target_source.shape[1:]) != expected_target_shape:
                    raise ValueError(
                        f"Official target raw sample shape mismatch for {logical_path}: "
                        f"{tuple(target_source.shape[1:])} != {expected_target_shape}"
                    )
                raw_targets = _read_rows(target_source, selected_sample_indices)
                target_batch = _canonicalise_batch(
                    raw_targets, raw_layout=schema.get("target_raw_layout")
                ).astype(np.float32, copy=False)
        if not bool(np.all(np.isfinite(batch))):
            raise ValueError(f"{logical_path} contains non-finite physical fields")
        if target_batch is not None and not bool(np.all(np.isfinite(target_batch))):
            raise ValueError(f"{logical_path} contains non-finite solution targets")
        if target_batch is not None and target_batch.shape != batch.shape:
            raise ValueError(
                f"Canonical Darcy input and target shapes differ: "
                f"{batch.shape} != {target_batch.shape}"
            )
        if target_shape is None:
            target_shape = tuple(int(dim) for dim in batch.shape[1:])
        elif tuple(batch.shape[1:]) != target_shape:
            raise ValueError(
                f"Official source shape mismatch for {logical_path}: "
                f"{tuple(batch.shape[1:])} != {target_shape}"
            )
        regime = float(entry["regime"])
        for sample_index, sample in zip(selected_sample_indices, batch, strict=True):
            fingerprint = _field_fingerprint(sample, contract)
            prior = seen_fields.get(fingerprint)
            if prior is not None and (prior[0] == regime or prior[1] != sample_index):
                raise ValueError(
                    f"Duplicate {contract.field_kind} field in official source collection: "
                    f"regime={regime}, row={sample_index}, prior={prior}"
                )
            seen_fields[fingerprint] = (regime, sample_index)
        count = int(batch.shape[0])
        chunks.append(batch)
        if target_batch is not None:
            target_chunks.append(target_batch)
        file_ids.append(np.full(count, file_id, dtype=np.int64))
        file_indices.append(np.full(count, source_index, dtype=np.int32))
        sample_indices.append(np.asarray(selected_sample_indices, dtype=np.int64))
        regimes.append(np.full(count, regime, dtype=np.float64))
        selected_identities.extend((file_id, index) for index in selected_sample_indices)
        verified.append(
            {
                "source_index": source_index,
                "path": logical_path,
                "local_path": str(source_path),
                "file_id": int(entry["file_id"]),
                "expected_md5": str(entry["checksum"]),
                "observed_md5": observed_md5,
                "size_bytes": expected_size,
                "regime": regime,
                "selected_samples": count,
                "selected_sample_indices": selected_sample_indices,
            }
        )

    selected_identity_sha256 = _selected_identity_digest(selected_identities)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_suffix(out_path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        with h5py.File(temporary, "w") as handle:
            handle.create_dataset("data", data=np.concatenate(chunks), compression="gzip")
            if target_chunks:
                handle.create_dataset(
                    "targets", data=np.concatenate(target_chunks), compression="gzip"
                )
            handle.create_dataset("source_file_id", data=np.concatenate(file_ids))
            handle.create_dataset("source_file_index", data=np.concatenate(file_indices))
            handle.create_dataset("source_sample_index", data=np.concatenate(sample_indices))
            handle.create_dataset(contract.parameter_name, data=np.concatenate(regimes))
            handle.attrs["canonical_source_schema"] = "strat-v1-source-v1"
            handle.attrs["conversion_complete"] = True
            handle.attrs["sequential_hydration_complete"] = True
            handle.attrs["task"] = task
            handle.attrs["field_kind"] = contract.field_kind
            handle.attrs["time_axis"] = -1 if contract.time_axis is None else contract.time_axis
            handle.attrs["raw_input_dataset_key"] = input_key
            if target_key is not None:
                handle.attrs["raw_target_dataset_key"] = target_key
            handle.attrs["mapping_kind"] = (
                "steady_operator" if target_key is not None else "trajectory"
            )
            handle.attrs["source_manifest_sha256"] = _sha256_file(manifest)
            handle.attrs["source_schema_sha256"] = _sha256_file(schema_contract_path)
            handle.attrs["source_semantic_role"] = str(schema["semantic_role"])
            handle.attrs["selection_algorithm"] = SELECTION_ALGORITHM
            handle.attrs["selection_seed"] = selection_seed
            handle.attrs["selection_protocol"] = selection_protocol
            handle.attrs["selection_ranking_provenance_dataset"] = "source_sample_index"
            handle.attrs["selected_identity_sha256"] = selected_identity_sha256
            handle.attrs["source_catalog_json"] = json.dumps(verified, sort_keys=True)
        temporary.replace(out_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    return {
        "status": "complete",
        "task": task,
        "output_path": str(out_path),
        "output_bytes": out_path.stat().st_size,
        "output_sha256": _sha256_file(out_path),
        "samples_per_regime": samples_per_regime,
        "sample_count": samples_per_regime * len(catalog),
        "regime_count": len(catalog),
        "regime_dataset": contract.parameter_name,
        "provenance_datasets": ["source_file_id", "source_sample_index"],
        "selection_algorithm": SELECTION_ALGORITHM,
        "selection_seed": selection_seed,
        "selection_protocol": selection_protocol,
        "selection_ranking_provenance_dataset": "source_sample_index",
        "selected_identity_sha256": selected_identity_sha256,
        "field_kind": contract.field_kind,
        "time_axis": contract.time_axis,
        "input_dataset_key": input_key,
        "target_dataset_key": target_key,
        "expected_raw_input_sample_shape": list(expected_input_shape),
        "expected_raw_target_sample_shape": (
            list(expected_target_shape) if expected_target_shape is not None else None
        ),
        "mapping_kind": "steady_operator" if target_key is not None else "trajectory",
        "semantic_role": str(schema["semantic_role"]),
        "schema_contract_path": str(schema_contract_path),
        "schema_contract_sha256": _sha256_file(schema_contract_path),
        "source_catalog": verified,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="docs/pdebench_manifest.yaml")
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--task", required=True, choices=sorted(TASK_CONTRACTS))
    parser.add_argument("--samples-per-regime", required=True, type=int)
    parser.add_argument("--selection-seed", type=int, default=0)
    parser.add_argument("--selection-protocol", default="strat-v1-source-v1")
    parser.add_argument("--output-json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    out_path = Path(args.out)
    output_json = Path(args.output_json) if args.output_json else None
    if output_json is not None and output_json.resolve() == out_path.resolve():
        raise ValueError("--output-json must differ from --out")
    record = hydrate_canonical_source(
        manifest=Path(args.manifest),
        raw_root=Path(args.raw_root),
        out_path=out_path,
        task=args.task,
        samples_per_regime=args.samples_per_regime,
        overwrite=args.overwrite,
        selection_seed=args.selection_seed,
        selection_protocol=args.selection_protocol,
    )
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_json.with_suffix(output_json.suffix + ".tmp")
        temporary.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(output_json)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
