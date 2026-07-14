#!/usr/bin/env python
from __future__ import annotations

"""Derive strat-v1.1 regime metrics from frozen A4 validation evidence."""

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ups.data.manifests import canonical_sha256
from ups.eval.regime_metrics import regime_spread_ratio, weighted_reconstructed_nrmse

TASKS = ("advection1d", "burgers1d", "darcy2d")
REGIME_KEYS = {"advection1d": "beta", "burgers1d": "nu", "darcy2d": "beta"}
REGIME_COUNTS = {"advection1d": 8, "burgers1d": 12, "darcy2d": 5}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _slug(value: float) -> str:
    label = format(float(value), ".6g")
    slug = label.lower().replace("-", "neg").replace("+", "pos").replace(".", "p")
    slug = re.sub(r"[^a-z0-9]+", "_", slug).strip("_")
    if not slug:
        raise ValueError(f"invalid regime label {value!r}")
    return slug


def _validate_addendum(payload: dict[str, Any]) -> None:
    self_hash = payload.get("self_hash", {})
    expected = self_hash.get("value")
    copy = json.loads(json.dumps(payload))
    copy.get("self_hash", {}).pop("value", None)
    if expected != canonical_sha256(copy):
        raise ValueError("strat-v1.1 addendum self hash does not match")
    access = payload.get("freeze_access", {})
    if access.get("derivation_split") != "valid" or access.get("heldout_reads") != "forbidden":
        raise ValueError("strat-v1.1 derivation must be validation-only and heldout-forbidden")


def _target_scale_stats(path: Path, task: str, *, eps: float) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        regime_values = np.asarray(handle[REGIME_KEYS[task]][:])
        if task == "darcy2d":
            targets = np.asarray(handle["targets"][:], dtype=np.float64)
        else:
            fields = handle["data"]
            if fields.shape[1] < 17:
                raise ValueError(f"{task} validation shard has fewer than 17 temporal frames")
            targets = np.asarray(fields[:, 1:17], dtype=np.float64)
    if len(regime_values) != len(targets):
        raise ValueError(f"{task} regime and target rows are misaligned")
    task_sum_sq = float(np.square(targets).sum(dtype=np.float64))
    task_count = int(targets.size)
    regimes: dict[str, Any] = {}
    for value in sorted(set(float(item) for item in regime_values)):
        mask = np.isclose(regime_values, value, rtol=0.0, atol=1e-12)
        selected = targets[mask]
        slug = _slug(value)
        if slug in regimes:
            raise ValueError(f"{task} regime slug collision: {slug}")
        regimes[slug] = {
            "value": value,
            "target_sum_sq": float(np.square(selected).sum(dtype=np.float64)),
            "element_count": int(selected.size),
        }
    if len(regimes) != REGIME_COUNTS[task]:
        raise ValueError(
            f"{task} has {len(regimes)} validation regimes, expected {REGIME_COUNTS[task]}"
        )
    return {
        "task_target_sum_sq": task_sum_sq,
        "task_element_count": task_count,
        "task_target_mean_sq": task_sum_sq / task_count,
        "epsilon": eps,
        "regimes": regimes,
    }


def _raw_regime_metrics(summary: dict[str, Any], task: str) -> dict[str, float]:
    primary = "decoded_solution_nrmse" if task == "darcy2d" else "decoded_rollout_nrmse"
    pattern = re.compile(rf"^task_{re.escape(task)}_regime_(.+)_{primary}$")
    values = {
        match.group(1): float(value)
        for key, value in summary.get("metrics", {}).items()
        if (match := pattern.match(key))
    }
    if len(values) != REGIME_COUNTS[task] or not all(
        math.isfinite(value) and value >= 0 for value in values.values()
    ):
        raise ValueError(f"{task} summary lacks complete finite raw regime metrics")
    return values


def build_diagnostics(
    *,
    addendum_path: Path,
    scorecard_path: Path,
    validation_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    addendum = yaml.safe_load(addendum_path.read_text(encoding="utf-8"))
    if not isinstance(addendum, dict):
        raise ValueError("addendum must be a mapping")
    _validate_addendum(addendum)
    evidence = addendum["reference_evidence"]
    if _sha256(scorecard_path) != evidence["calibration_scorecard_file_sha256"]:
        raise ValueError("calibration scorecard file hash does not match addendum")
    scorecard = json.loads(scorecard_path.read_text(encoding="utf-8"))
    if scorecard.get("scorecard_sha256") != evidence["calibration_scorecard_sha256"]:
        raise ValueError("calibration scorecard internal hash does not match addendum")
    if scorecard.get("held_out_measurements") != 0:
        raise ValueError("calibration scorecard must contain zero held-out measurements")
    if any("test" in path.name.lower() for path in validation_root.rglob("*.h5")):
        raise PermissionError("validation root contains a test object")

    valid_objects = {
        item["path"]: item
        for item in scorecard["training_lock"]["objects"]
        if item["role"] == "valid"
    }
    eps = float(addendum["numeric_contract"]["epsilon"])
    scale_stats: dict[str, Any] = {}
    for task in TASKS:
        path = validation_root / f"{task}_val.h5"
        record = valid_objects.get(path.name)
        if record is None or not path.is_file():
            raise FileNotFoundError(f"missing locked validation object {path.name}")
        if _sha256(path) != record["sha256"]:
            raise ValueError(f"validation object hash mismatch: {path.name}")
        scale_stats[task] = _target_scale_stats(path, task, eps=eps)

    source_cache: dict[str, dict[str, Any]] = {}
    provisional: list[dict[str, Any]] = []
    for row in scorecard["rows"]:
        summary_rel = row["source_summary"]
        summary_path = repo_root / summary_rel
        if summary_rel not in source_cache:
            if not summary_path.is_file() or _sha256(summary_path) != row["summary_sha256"]:
                raise ValueError(f"frozen source summary hash mismatch: {summary_rel}")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("split") != "val":
                raise ValueError(f"source summary is not validation-only: {summary_rel}")
            if summary.get("held_out_test_policy", {}).get("enabled") or summary.get(
                "extra", {}
            ).get("allow_held_out_test_eval"):
                raise PermissionError(f"held-out access recorded in {summary_rel}")
            source_cache[summary_rel] = summary
        summary = source_cache[summary_rel]
        task = row["task"]
        raw_values = _raw_regime_metrics(summary, task)
        stats = scale_stats[task]
        task_primary = float(row["primary_nrmse"])
        regimes = []
        for slug, raw in sorted(raw_values.items()):
            regime_stats = stats["regimes"].get(slug)
            if regime_stats is None:
                raise ValueError(f"{task} summary regime {slug} is absent from validation shard")
            regime_mean_sq = regime_stats["target_sum_sq"] / regime_stats["element_count"]
            global_value = raw * math.sqrt(
                (regime_mean_sq + eps) / (stats["task_target_mean_sq"] + eps)
            )
            regimes.append(
                {
                    "slug": slug,
                    "value": regime_stats["value"],
                    "slice_normalized_nrmse": raw,
                    "global_scale_nrmse": global_value,
                    "element_count": regime_stats["element_count"],
                    "spread_ratio_to_task_primary": regime_spread_ratio(global_value, task_primary),
                }
            )
        reconstructed = weighted_reconstructed_nrmse(
            [item["global_scale_nrmse"] for item in regimes],
            [item["element_count"] for item in regimes],
        )
        provisional.append(
            {
                "row_id": row["row_id"],
                "model": row["model"],
                "task": task,
                "source_summary": summary_rel,
                "source_summary_sha256": row["summary_sha256"],
                "task_primary_nrmse": task_primary,
                "reconstructed_global_scale_nrmse": reconstructed,
                "reconstruction_delta": reconstructed - task_primary,
                "max_spread_ratio": max(item["spread_ratio_to_task_primary"] for item in regimes),
                "promotion_gate_passed": max(
                    item["spread_ratio_to_task_primary"] for item in regimes
                )
                <= float(addendum["promotion_gate"]["maximum"]),
                "regimes": regimes,
            }
        )

    persistence = {
        (row["task"], item["slug"]): item["global_scale_nrmse"]
        for row in provisional
        if row["model"] == "persistence"
        for item in row["regimes"]
    }
    for row in provisional:
        for item in row["regimes"]:
            denominator = persistence.get((row["task"], item["slug"]))
            if denominator is None or not math.isfinite(denominator) or denominator <= 0:
                raise ValueError("persistence regime reference must be finite and positive")
            item["error_ratio_to_persistence"] = item["global_scale_nrmse"] / denominator

    payload = {
        "schema_version": 1,
        "artifact_id": "strat-v1.1-a4-validation-regime-diagnostics",
        "status": "complete_validation_only_metric_reprojection",
        "addendum_id": addendum["addendum_id"],
        "addendum_sha256": addendum["self_hash"]["value"],
        "base_scorecard_sha256": scorecard["scorecard_sha256"],
        "training_lock_sha256": scorecard["training_lock"]["lock_sha256"],
        "held_out_measurements": 0,
        "derivation": "frozen raw regime NRMSE reprojected using locked validation target scales",
        "target_scale_statistics": scale_stats,
        "rows": provisional,
    }
    payload["artifact_sha256"] = canonical_sha256(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--addendum", type=Path, default=Path("docs/data/protocols/strat_v1_1_metric_addendum.yaml")
    )
    parser.add_argument(
        "--scorecard",
        type=Path,
        default=Path("docs/research/artifacts/strat_v1_a4_validation_scorecard.json"),
    )
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/research/artifacts/strat_v1_1_validation_regime_diagnostics.json"),
    )
    args = parser.parse_args()
    payload = build_diagnostics(
        addendum_path=args.addendum,
        scorecard_path=args.scorecard,
        validation_root=args.validation_root,
        repo_root=args.repo_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "artifact_sha256": payload["artifact_sha256"]}))


if __name__ == "__main__":
    main()
