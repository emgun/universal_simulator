#!/usr/bin/env python
from __future__ import annotations

"""Summarize validation-only Phase 1 rollout-stability recipe sweep results."""

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

MEASUREMENT_TYPE = "p1_rollout_stability_recipe_sweep_results"
DEFAULT_BASELINE_JSON = "docs/research/artifacts/p1_capacity_sweep_medium_v1_val.json"
DEFAULT_CONTRACT_JSON = "docs/research/p1_rollout_stability_recipe_sweep_contract.json"
DEFAULT_OUTPUT_JSON = "docs/research/artifacts/p1_recipe_sweep_medium_v1_val.json"
METRIC = "decoded_rollout_nrmse"
H16_METRIC = "decoded_h16_nrmse"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def _metric(payload: Mapping[str, Any], key: str) -> float:
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError(f"Missing metrics object in {payload.get('run_name', '<unknown>')}")
    value = metrics.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"Missing numeric metric {key}")
    return float(value)


def _summary_paths(output_root: Path) -> list[Path]:
    if not output_root.exists():
        raise FileNotFoundError(f"No summary.json files under {output_root}")
    paths = sorted(output_root.glob("*/summary.json"))
    if not paths:
        raise FileNotFoundError(f"No summary.json files under {output_root}")
    return paths


def _non_empty_metadata(extra: Mapping[str, Any], keys: tuple[str, ...]) -> tuple[str, Any] | None:
    for key in keys:
        value = extra.get(key)
        if value not in ({}, None):
            return key, value
    return None


def _validate_validation_only_summary(
    path: Path,
    summary: Mapping[str, Any],
    *,
    expected_split: str,
) -> None:
    extra = summary.get("extra")
    if not isinstance(extra, Mapping):
        raise ValueError(f"{path} is missing extra metadata")

    split_values = [extra.get("split"), extra.get("decoded_split")]
    concrete_splits = [str(split) for split in split_values if split not in (None, "")]
    if any(split == "test" for split in concrete_splits):
        raise ValueError(f"{path} uses test split")
    if concrete_splits and any(split != expected_split for split in concrete_splits):
        raise ValueError(f"{path} split metadata must be {expected_split}: {concrete_splits}")

    extra_evaluations = summary.get("extra_evaluations") or {}
    if isinstance(extra_evaluations, Mapping) and "test" in extra_evaluations:
        raise ValueError(f"{path} contains test extra evaluation")

    task_roots = _non_empty_metadata(extra, ("task_roots", "decoded_task_roots"))
    if task_roots is not None:
        key, _ = task_roots
        raise ValueError(f"{path} must not use {key}")

    estimator = _non_empty_metadata(
        extra,
        (
            "decoded_context_roll_shift_estimator",
            "decoded_data_conditioned_roll_shift_estimator",
            "decoded_observed_roll_shift_estimator",
            "decoded_prediction_roll_shift_estimator",
            "decoded_decoded_context_roll_shift_estimator",
            "decoded_decoded_data_conditioned_roll_shift_estimator",
            "decoded_decoded_observed_roll_shift_estimator",
            "decoded_decoded_prediction_roll_shift_estimator",
        ),
    )
    if estimator is not None:
        key, _ = estimator
        raise ValueError(f"{path} enables {key}")


def _validated_contract_split(protocol: Mapping[str, Any]) -> str:
    if protocol.get("estimators") != "none":
        raise ValueError("contract must disable estimators")
    expected_split = str(protocol.get("eval_split", ""))
    if expected_split == "test":
        raise ValueError("contract eval_split must not be test")
    if not expected_split:
        raise ValueError("contract eval_split must be set")
    return expected_split


def _baseline_run(baseline: Mapping[str, Any], run_name: str) -> dict[str, float]:
    runs = baseline.get("runs")
    if not isinstance(runs, Mapping) or run_name not in runs:
        raise KeyError(f"Baseline artifact is missing run {run_name}")
    payload = runs[run_name]
    if not isinstance(payload, Mapping):
        raise TypeError(f"Baseline run {run_name} must be an object")
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        raise TypeError(f"Baseline run {run_name} must contain metrics")
    return {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float))
        and (
            key
            in {
                METRIC,
                H16_METRIC,
                "decoded_step1_nrmse",
                "task_advection1d_decoded_rollout_nrmse",
                "task_burgers1d_decoded_rollout_nrmse",
                "task_darcy2d_decoded_rollout_nrmse",
            }
        )
    }


def _run_record(path: Path, summary: Mapping[str, Any]) -> dict[str, Any]:
    metrics = summary.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError(f"{path} is missing metrics")
    run_name = str(summary.get("run_name") or path.parent.name)
    selected_metrics = {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float))
        and (
            key.startswith("task_")
            or key.startswith("family_")
            or key
            in {
                METRIC,
                H16_METRIC,
                "decoded_h4_nrmse",
                "decoded_step1_nrmse",
                "decoded_rollout_spectral_energy_error",
            }
        )
    }
    return {
        "summary_json": str(path),
        "duration_sec": float(summary.get("duration_sec", 0.0) or 0.0),
        "metrics": selected_metrics,
        "run_name": run_name,
    }


def summarize_recipe_sweep(
    *,
    output_root: Path,
    baseline_json: Path,
    contract_json: Path,
    artifact: str | None = None,
) -> dict[str, Any]:
    baseline = load_json(baseline_json)
    contract = load_json(contract_json)
    if contract.get("measurement_type") != "p1_rollout_stability_recipe_sweep_contract":
        raise ValueError(
            "contract measurement_type must be p1_rollout_stability_recipe_sweep_contract"
        )
    if contract.get("held_out_test_data_read") is not False:
        raise ValueError("contract must be validation-only")
    protocol = contract.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("contract protocol must be an object")
    expected_split = _validated_contract_split(protocol)

    gate = contract.get("gate")
    if not isinstance(gate, Mapping):
        raise ValueError("contract gate must be an object")
    persistence_name = str(gate.get("reference_run", "persistence_medium_v1_val"))
    tier_name = str(gate.get("diagnostic_reference_run", "ups_medium_capacity_tier_b"))
    persistence = _baseline_run(baseline, persistence_name)
    tier = _baseline_run(baseline, tier_name)

    runs: dict[str, Any] = {}
    for path in _summary_paths(output_root):
        summary = load_json(path)
        _validate_validation_only_summary(path, summary, expected_split=expected_split)
        record = _run_record(path, summary)
        runs[record["run_name"]] = {
            key: value for key, value in record.items() if key != "run_name"
        }

    best_run, best_record = min(
        runs.items(), key=lambda item: float(item[1]["metrics"].get(METRIC, float("inf")))
    )
    best_metric = float(best_record["metrics"][METRIC])
    persistence_metric = float(persistence[METRIC])
    tier_metric = float(tier[METRIC])
    best_h16 = float(best_record["metrics"].get(H16_METRIC, float("nan")))
    tier_h16 = float(tier.get(H16_METRIC, float("nan")))

    return {
        "measurement_type": MEASUREMENT_TYPE,
        "artifact": artifact,
        "contract_json": str(contract_json),
        "baseline_json": str(baseline_json),
        "held_out_test_data_read": False,
        "test_ledger_writes": [],
        "protocol": dict(protocol),
        "recipes": list(contract.get("recipes", [])),
        "baselines": {
            persistence_name: persistence,
            tier_name: tier,
        },
        "runs": runs,
        "decision": {
            "best_run": best_run,
            "best_metric_name": METRIC,
            "best_metric_value": best_metric,
            "improves_persistence": best_metric < persistence_metric,
            "absolute_delta_vs_persistence": best_metric - persistence_metric,
            "improvement_fraction_vs_persistence": (persistence_metric - best_metric)
            / persistence_metric,
            "improves_tier_b_capacity": best_metric < tier_metric,
            "absolute_delta_vs_tier_b_capacity": best_metric - tier_metric,
            "improvement_fraction_vs_tier_b_capacity": (tier_metric - best_metric) / tier_metric,
            "h16_absolute_delta_vs_tier_b_capacity": best_h16 - tier_h16,
            "held_out_test_allowed_by_this_artifact": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--baseline-json", type=Path, default=Path(DEFAULT_BASELINE_JSON))
    parser.add_argument("--contract-json", type=Path, default=Path(DEFAULT_CONTRACT_JSON))
    parser.add_argument("--artifact")
    parser.add_argument("--output-json", type=Path, default=Path(DEFAULT_OUTPUT_JSON))
    args = parser.parse_args(argv)

    record = summarize_recipe_sweep(
        output_root=args.output_root,
        baseline_json=args.baseline_json,
        contract_json=args.contract_json,
        artifact=args.artifact,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
