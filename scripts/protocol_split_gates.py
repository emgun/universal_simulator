from __future__ import annotations

"""Fail-closed integrity gates for provenance-aware protocol splits."""

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _scalar(value: np.generic | object) -> object:
    return value.item() if isinstance(value, np.generic) else value


def _label(value: object) -> str:
    value = _scalar(value)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _validate_finite_identity(value: object, *, split: str, dataset: str, row: int) -> None:
    value = _scalar(value)
    if isinstance(value, (float, complex)) and not bool(np.isfinite(value)):
        raise ValueError(
            f"{split} identity dataset {dataset} contains a non-finite value at row {row}"
        )


def _row_id(arrays: Mapping[str, np.ndarray], keys: Sequence[str], row: int) -> tuple[object, ...]:
    return tuple(_scalar(arrays[key][row]) for key in keys)


def _field_id(data: np.ndarray, row: int, *, field_kind: str, time_axis: int | None) -> str:
    sample = np.asarray(data[row])
    if field_kind == "temporal":
        if time_axis is None:
            raise ValueError("Temporal fields require an explicit HDF5 --time-axis")
        sample_axis = time_axis - 1
        if time_axis <= 0 or sample_axis >= sample.ndim:
            raise ValueError(
                f"Temporal time axis {time_axis} is invalid for data shape {data.shape}; "
                "axis 0 is the sample axis"
            )
        sample = np.take(sample, 0, axis=sample_axis)
    elif field_kind != "steady":
        raise ValueError("Field kind must be 'temporal' or 'steady'")
    contiguous = np.ascontiguousarray(sample)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def evaluate_protocol_splits(
    split_arrays: Mapping[str, Mapping[str, np.ndarray]],
    *,
    provenance_datasets: Sequence[str],
    regime_dataset: str,
    field_kind: str,
    time_axis: int | None,
) -> dict[str, Any]:
    """Validate disjoint, unique, equally covered protocol splits or raise."""
    expected_splits = ("train", "val", "test")
    if tuple(split_arrays) != expected_splits:
        raise ValueError("Gated protocol requires train, val, and test splits in that order")
    if not provenance_datasets:
        raise ValueError("Gated protocol requires at least one provenance dataset")

    provenance_by_split: dict[str, set[tuple[object, ...]]] = {}
    fields_by_split: dict[str, set[str]] = {}
    unique_field_regime_pairs: dict[str, int] = {}
    unique_field_groups: dict[str, int] = {}
    regime_counts: dict[str, dict[str, int]] = {}
    for split in expected_splits:
        arrays = split_arrays[split]
        required = {"data", regime_dataset, *provenance_datasets}
        missing = sorted(required - arrays.keys())
        if missing:
            raise ValueError(f"{split} split is missing required datasets: {', '.join(missing)}")
        count = int(arrays["data"].shape[0])
        data = np.asarray(arrays["data"])
        if not np.issubdtype(data.dtype, np.number):
            raise ValueError(f"{split} physical field data must be numeric")
        if not bool(np.all(np.isfinite(data))):
            raise ValueError(f"{split} physical field data contains non-finite values")
        for key in required:
            if not arrays[key].shape or int(arrays[key].shape[0]) != count:
                raise ValueError(f"{split} dataset {key} is not aligned to {count} samples")
        for key in (regime_dataset, *provenance_datasets):
            if arrays[key].ndim != 1:
                raise ValueError(f"{split} identity dataset {key} must be one-dimensional")
            for row, value in enumerate(arrays[key]):
                _validate_finite_identity(value, split=split, dataset=key, row=row)

        provenance = {_row_id(arrays, provenance_datasets, row) for row in range(count)}
        if len(provenance) != count:
            raise ValueError(f"{split} split contains duplicate provenance identities")
        field_ids = [
            _field_id(arrays["data"], row, field_kind=field_kind, time_axis=time_axis)
            for row in range(count)
        ]
        regime_labels = [_label(value) for value in arrays[regime_dataset]]
        field_regime_pairs = set(zip(field_ids, regime_labels, strict=True))
        if len(field_regime_pairs) != count:
            noun = "initial field" if field_kind == "temporal" else "steady field"
            raise ValueError(f"{split} split repeats an identical {noun} within one regime")

        counts: dict[str, int] = {}
        for label in regime_labels:
            counts[label] = counts.get(label, 0) + 1
        if not counts:
            raise ValueError(f"{split} split has no regime values")
        if len(set(counts.values())) != 1:
            raise ValueError(f"{split} split has unbalanced regime counts: {counts}")
        provenance_by_split[split] = provenance
        fields_by_split[split] = set(field_ids)
        unique_field_regime_pairs[split] = len(field_regime_pairs)
        unique_field_groups[split] = len(fields_by_split[split])
        regime_counts[split] = dict(sorted(counts.items()))

    regime_sets = [set(regime_counts[split]) for split in expected_splits]
    if not all(regimes == regime_sets[0] for regimes in regime_sets[1:]):
        raise ValueError(f"Splits do not have identical regime coverage: {regime_counts}")

    overlap_checks: dict[str, dict[str, int]] = {}
    for index, left in enumerate(expected_splits):
        for right in expected_splits[index + 1 :]:
            provenance_overlap = provenance_by_split[left] & provenance_by_split[right]
            field_overlap = fields_by_split[left] & fields_by_split[right]
            if provenance_overlap:
                raise ValueError(f"{left}/{right} provenance overlap: {len(provenance_overlap)}")
            if field_overlap:
                noun = "initial-field" if field_kind == "temporal" else "steady-field"
                raise ValueError(f"{left}/{right} {noun} overlap: {len(field_overlap)}")
            overlap_checks[f"{left}:{right}"] = {
                "provenance_overlap": 0,
                "field_overlap": 0,
            }

    return {
        "status": "passed",
        "field_kind": field_kind,
        "time_axis": time_axis,
        "provenance_datasets": list(provenance_datasets),
        "regime_dataset": regime_dataset,
        "within_split_provenance_unique": {split: True for split in expected_splits},
        "within_split_field_regime_pairs_unique": {split: True for split in expected_splits},
        "unique_field_regime_pairs": unique_field_regime_pairs,
        "unique_field_groups": unique_field_groups,
        "regime_counts": regime_counts,
        "identical_regime_coverage": True,
        "balanced_regime_coverage": True,
        "cross_split_overlap": overlap_checks,
    }
