#!/usr/bin/env python
from __future__ import annotations

"""Measure the validation-only Darcy ambiguity faced by a beta-blind predictor."""

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ups.data.manifests import RunDataLock, canonical_sha256
from ups.eval.regime_metrics import regime_spread_ratio, weighted_reconstructed_nrmse

EXPECTED_BETAS = (0.01, 0.1, 1.0, 10.0, 100.0)
EPSILON = 1e-8


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_test_path(value: str | Path, label: str) -> None:
    if "test" in Path(str(value)).name.lower():
        raise PermissionError(f"{label} contains a test path")


def _finite_array(value: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.size == 0:
        raise ValueError(f"{label} must not be empty")
    if array.dtype.kind not in "biufc" or not bool(np.isfinite(array).all()):
        raise ValueError(f"{label} must contain only finite numeric values")
    return array


def _provenance_value(value: Any, label: str) -> int | float | str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{label} must be finite")
        return numeric
    if isinstance(value, str) and value:
        return value
    raise ValueError(f"{label} has an unsupported provenance value")


def _beta_index(value: float) -> int:
    matches = [
        index
        for index, expected in enumerate(EXPECTED_BETAS)
        if math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12)
    ]
    if len(matches) != 1:
        raise ValueError(f"unexpected Darcy beta {value!r}")
    return matches[0]


def _global_scale_nrmse(errors: np.ndarray, target_mean_sq: float, *, eps: float) -> float:
    errors = _finite_array(errors, "oracle errors").astype(np.float64, copy=False)
    value = math.sqrt(float(np.square(errors).mean(dtype=np.float64)) / (target_mean_sq + eps))
    if not math.isfinite(value):
        raise ValueError("computed a non-finite NRMSE")
    return value


def _validate_lock(
    *,
    lock_path: Path,
    validation_path: Path,
    expected_lock_sha256: str,
    expected_selection_sha256: str,
    expected_validation_sha256: str,
) -> tuple[RunDataLock, str]:
    _reject_test_path(lock_path, "training lock")
    _reject_test_path(validation_path, "validation shard")
    lock = RunDataLock.from_dict(json.loads(lock_path.read_text(encoding="utf-8")))
    if lock.purpose != "training" or "test" in lock.requested_roles:
        raise PermissionError("diagnostic requires a training lock with no test role")
    if lock.lock_sha256 != expected_lock_sha256:
        raise ValueError("training lock SHA does not match the expected frozen identity")
    selection_sha256 = canonical_sha256(lock.selection)
    if selection_sha256 != expected_selection_sha256:
        raise ValueError("training lock selection does not match the expected frozen identity")
    if any(
        item.role == "test"
        or "test" in item.path.lower()
        or "test" in item.object_id.lower()
        or any("test" in uri.lower() for uri in item.uris)
        for item in lock.objects
    ):
        raise PermissionError("training lock exposes a test object or path")
    records = [
        item
        for item in lock.objects
        if item.role == "valid" and item.object_id == "darcy2d-valid"
    ]
    if len(records) != 1:
        raise ValueError("training lock must contain exactly one darcy2d-valid object")
    record = records[0]
    if Path(record.path).name != validation_path.name:
        raise ValueError("validation shard path does not match the locked Darcy object")
    locked_sha256 = record.checksums.get("sha256")
    if locked_sha256 != expected_validation_sha256:
        raise ValueError("locked validation object SHA does not match the expected identity")
    actual_sha256 = _file_sha256(validation_path)
    if actual_sha256 != expected_validation_sha256:
        raise ValueError("validation shard bytes do not match the locked SHA")
    return lock, selection_sha256


def build_diagnostic(
    *,
    training_lock_path: Path,
    validation_shard_path: Path,
    expected_lock_sha256: str,
    expected_selection_sha256: str,
    expected_validation_sha256: str,
    eps: float = EPSILON,
) -> dict[str, Any]:
    """Build a self-hashed diagnostic without reading train or held-out bytes."""

    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("epsilon must be finite and positive")
    lock, selection_sha256 = _validate_lock(
        lock_path=training_lock_path,
        validation_path=validation_shard_path,
        expected_lock_sha256=expected_lock_sha256,
        expected_selection_sha256=expected_selection_sha256,
        expected_validation_sha256=expected_validation_sha256,
    )

    required = ("data", "targets", "beta", "source_file_id", "source_sample_index")
    with h5py.File(validation_shard_path, "r") as handle:
        missing = [key for key in required if key not in handle]
        if missing:
            raise ValueError(f"validation shard lacks required datasets: {missing}")
        inputs = _finite_array(handle["data"][:], "coefficient inputs")
        targets = _finite_array(handle["targets"][:], "solution targets")
        betas = _finite_array(handle["beta"][:], "beta provenance").reshape(-1)
        source_file_ids = handle["source_file_id"][:].reshape(-1)
        sample_ids = handle["source_sample_index"][:].reshape(-1)

    row_count = len(targets)
    if inputs.shape != targets.shape:
        raise ValueError("Darcy coefficient inputs and solution targets must have identical shapes")
    if any(len(values) != row_count for values in (betas, source_file_ids, sample_ids)):
        raise ValueError("Darcy fields and provenance rows are misaligned")

    grouped: dict[int | float | str, list[int]] = defaultdict(list)
    for row, raw_id in enumerate(sample_ids):
        grouped[_provenance_value(raw_id, "source_sample_index")].append(row)
        _provenance_value(source_file_ids[row], "source_file_id")
    if not grouped:
        raise ValueError("validation shard contains no provenance groups")

    regime_counts = [0] * len(EXPECTED_BETAS)
    group_records: list[dict[str, Any]] = []
    ordered_targets: list[np.ndarray] = []
    ordered_predictions: list[np.ndarray] = []
    for group_id in sorted(grouped, key=lambda value: canonical_sha256(value)):
        rows = grouped[group_id]
        if len(rows) != len(EXPECTED_BETAS):
            raise ValueError("each provenance group must cover all five Darcy betas exactly once")
        by_beta: dict[int, int] = {}
        for row in rows:
            index = _beta_index(float(betas[row]))
            if index in by_beta:
                raise ValueError("provenance group contains duplicate beta coverage")
            by_beta[index] = row
            regime_counts[index] += 1
        if set(by_beta) != set(range(len(EXPECTED_BETAS))):
            raise ValueError("provenance group has incomplete beta coverage")
        ordered_rows = [by_beta[index] for index in range(len(EXPECTED_BETAS))]
        reference = inputs[ordered_rows[0]]
        if any(not np.array_equal(reference, inputs[row]) for row in ordered_rows[1:]):
            raise ValueError("coefficient inputs differ within a shared provenance group")
        group_targets = targets[ordered_rows].astype(np.float64, copy=False)
        oracle = group_targets.mean(axis=0, dtype=np.float64)
        ordered_targets.append(group_targets)
        ordered_predictions.append(np.broadcast_to(oracle, group_targets.shape))
        group_records.append(
            {
                "source_sample_index": group_id,
                "coefficient_input_sha256": hashlib.sha256(
                    np.ascontiguousarray(reference).tobytes()
                ).hexdigest(),
                "source_file_ids_by_beta": [
                    _provenance_value(source_file_ids[row], "source_file_id")
                    for row in ordered_rows
                ],
            }
        )

    if len(set(regime_counts)) != 1 or regime_counts[0] != len(grouped):
        raise ValueError("Darcy beta regimes have unequal provenance-group coverage")
    target_cube = np.stack(ordered_targets)  # group, beta, physical dimensions
    prediction_cube = np.stack(ordered_predictions)
    target_mean_sq = float(np.square(target_cube).mean(dtype=np.float64))
    if not math.isfinite(target_mean_sq) or target_mean_sq < 0:
        raise ValueError("target scale is not finite")
    pooled_nrmse = _global_scale_nrmse(
        prediction_cube - target_cube, target_mean_sq, eps=eps
    )

    regimes = []
    for index, beta in enumerate(EXPECTED_BETAS):
        value = _global_scale_nrmse(
            prediction_cube[:, index] - target_cube[:, index], target_mean_sq, eps=eps
        )
        regimes.append(
            {
                "beta": beta,
                "group_count": regime_counts[index],
                "element_count": int(target_cube[:, index].size),
                "oracle_global_scale_nrmse": value,
                "spread_ratio_to_pooled_nrmse": regime_spread_ratio(value, pooled_nrmse),
            }
        )
    reconstructed = weighted_reconstructed_nrmse(
        [item["oracle_global_scale_nrmse"] for item in regimes],
        [item["element_count"] for item in regimes],
    )
    if not math.isclose(reconstructed, pooled_nrmse, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("per-regime metrics do not reconstruct the pooled oracle NRMSE")

    pairs = []
    for left in range(len(EXPECTED_BETAS)):
        for right in range(left + 1, len(EXPECTED_BETAS)):
            separation = _global_scale_nrmse(
                target_cube[:, left] - target_cube[:, right], target_mean_sq, eps=eps
            )
            pairs.append(
                {
                    "left_beta": EXPECTED_BETAS[left],
                    "right_beta": EXPECTED_BETAS[right],
                    "global_scale_nrmse": separation,
                }
            )
    separation_values = [item["global_scale_nrmse"] for item in pairs]
    payload = {
        "schema_version": 1,
        "artifact_id": "strat-v1-darcy-beta-blind-identifiability-d0",
        "status": "complete_validation_only",
        "access": {
            "split": "valid",
            "read_roles": ["valid"],
            "heldout_reads": 0,
            "held_out_measurements": 0,
        },
        "bindings": {
            "training_lock_path": str(training_lock_path),
            "training_lock_sha256": lock.lock_sha256,
            "selection": dict(lock.selection),
            "selection_sha256": selection_sha256,
            "validation_object_id": "darcy2d-valid",
            "validation_object_path": str(validation_shard_path),
            "validation_object_sha256": expected_validation_sha256,
        },
        "numeric_contract": {
            "epsilon": eps,
            "normalization": "global validation target RMS",
            "oracle": "per-source_sample_index mean target across five betas",
        },
        "coverage": {
            "group_count": len(grouped),
            "betas": list(EXPECTED_BETAS),
            "groups_per_beta": {
                format(beta, ".6g"): count
                for beta, count in zip(EXPECTED_BETAS, regime_counts, strict=True)
            },
            "rows": row_count,
            "equal_complete_group_coverage": True,
        },
        "beta_blind_oracle": {
            "pooled_global_scale_nrmse": pooled_nrmse,
            "reconstructed_global_scale_nrmse": reconstructed,
            "max_corrected_regime_spread": max(
                item["spread_ratio_to_pooled_nrmse"] for item in regimes
            ),
            "regimes": regimes,
        },
        "target_separation": {
            "definition": "pairwise target RMS difference normalized by global validation target RMS",
            "minimum_global_scale_nrmse": min(separation_values),
            "mean_global_scale_nrmse": float(np.mean(separation_values)),
            "maximum_global_scale_nrmse": max(separation_values),
            "pairs": pairs,
        },
        "provenance_groups": group_records,
    }
    payload["artifact_sha256"] = canonical_sha256(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-lock", type=Path, required=True)
    parser.add_argument("--validation-shard", type=Path, required=True)
    parser.add_argument("--expected-lock-sha256", required=True)
    parser.add_argument("--expected-selection-sha256", required=True)
    parser.add_argument("--expected-validation-sha256", required=True)
    parser.add_argument("--epsilon", type=float, default=EPSILON)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_diagnostic(
        training_lock_path=args.training_lock,
        validation_shard_path=args.validation_shard,
        expected_lock_sha256=args.expected_lock_sha256,
        expected_selection_sha256=args.expected_selection_sha256,
        expected_validation_sha256=args.expected_validation_sha256,
        eps=args.epsilon,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "artifact_sha256": payload["artifact_sha256"]}))


if __name__ == "__main__":
    main()
