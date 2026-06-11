#!/usr/bin/env python
from __future__ import annotations

"""Summarize validation-only Phase 1 data-budget sweep results."""

import argparse
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from scripts.summarize_recipe_sweep import (
    H16_METRIC,
    METRIC,
    _baseline_run,
    _run_record,
    _summary_paths,
    _validate_validation_only_summary,
    _validated_contract_split,
    load_json,
)

MEASUREMENT_TYPE = "p1_data_budget_sweep_results"
DEFAULT_BASELINE_JSON = "docs/research/artifacts/p1_capacity_sweep_medium_v1_val.json"
DEFAULT_CONTRACT_JSON = "docs/research/p1_data_budget_sweep_contract.json"
DEFAULT_OUTPUT_JSON = "docs/research/artifacts/p1_data_budget_sweep_medium_v1_val.json"


def _expected_budgets(protocol: Mapping[str, Any]) -> list[int]:
    values = protocol.get("train_sample_budgets")
    if not isinstance(values, list) or not values:
        raise ValueError("contract protocol must define non-empty train_sample_budgets")
    budgets: list[int] = []
    for value in values:
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"invalid train_sample_budget: {value!r}")
        budgets.append(value)
    return budgets


def _run_name_prefix(contract: Mapping[str, Any]) -> str:
    runner = contract.get("runner")
    if not isinstance(runner, Mapping):
        raise ValueError("contract runner must be an object")
    prefix = runner.get("run_name_prefix")
    if not isinstance(prefix, str) or not prefix:
        raise ValueError("contract runner.run_name_prefix must be set")
    return prefix


def _budget_from_run_name(run_name: str, prefix: str) -> int:
    match = re.fullmatch(rf"{re.escape(prefix)}_n([0-9]+)", run_name)
    if match is None:
        raise ValueError(f"run_name {run_name!r} does not match {prefix}_n<samples>")
    return int(match.group(1))


def summarize_data_budget_sweep(
    *,
    output_root: Path,
    baseline_json: Path,
    contract_json: Path,
    artifact: str | None = None,
) -> dict[str, Any]:
    baseline = load_json(baseline_json)
    contract = load_json(contract_json)
    if contract.get("measurement_type") != "p1_data_budget_sweep_contract":
        raise ValueError("contract measurement_type must be p1_data_budget_sweep_contract")
    if contract.get("held_out_test_data_read") is not False:
        raise ValueError("contract must be validation-only")
    protocol = contract.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("contract protocol must be an object")
    expected_split = _validated_contract_split(protocol)
    expected_budgets = _expected_budgets(protocol)
    run_name_prefix = _run_name_prefix(contract)

    gate = contract.get("gate")
    if not isinstance(gate, Mapping):
        raise ValueError("contract gate must be an object")
    persistence_name = str(gate.get("reference_run", "persistence_medium_v1_val"))
    tier_name = str(gate.get("diagnostic_reference_run", "ups_medium_capacity_tier_b"))
    persistence = _baseline_run(baseline, persistence_name)
    tier = _baseline_run(baseline, tier_name)

    runs: dict[str, Any] = {}
    by_budget: dict[int, str] = {}
    for path in _summary_paths(output_root):
        summary = load_json(path)
        _validate_validation_only_summary(path, summary, expected_split=expected_split)
        record = _run_record(path, summary)
        run_name = record["run_name"]
        budget = _budget_from_run_name(run_name, run_name_prefix)
        if budget not in expected_budgets:
            raise ValueError(f"{path} has unexpected train-sample budget {budget}")
        if budget in by_budget:
            raise ValueError(f"duplicate summary for train-sample budget {budget}")
        by_budget[budget] = run_name
        runs[run_name] = {key: value for key, value in record.items() if key != "run_name"} | {
            "train_samples": budget
        }

    if not runs:
        raise FileNotFoundError(f"No data-budget run summaries under {output_root}")

    best_run, best_record = min(
        runs.items(), key=lambda item: float(item[1]["metrics"].get(METRIC, float("inf")))
    )
    best_metric = float(best_record["metrics"][METRIC])
    best_h16 = float(best_record["metrics"].get(H16_METRIC, float("nan")))
    persistence_metric = float(persistence[METRIC])
    tier_metric = float(tier[METRIC])
    tier_h16 = float(tier.get(H16_METRIC, float("nan")))
    missing_budgets = [budget for budget in expected_budgets if budget not in by_budget]

    curve = []
    for budget in sorted(by_budget):
        run_name = by_budget[budget]
        metrics = runs[run_name]["metrics"]
        metric = float(metrics[METRIC])
        curve.append(
            {
                "train_samples": budget,
                "run_name": run_name,
                "decoded_rollout_nrmse": metric,
                "decoded_h16_nrmse": float(metrics.get(H16_METRIC, float("nan"))),
                "absolute_delta_vs_tier_b_capacity": metric - tier_metric,
                "absolute_delta_vs_persistence": metric - persistence_metric,
            }
        )

    return {
        "measurement_type": MEASUREMENT_TYPE,
        "artifact": artifact,
        "contract_json": str(contract_json),
        "baseline_json": str(baseline_json),
        "held_out_test_data_read": False,
        "test_ledger_writes": [],
        "protocol": dict(protocol),
        "baselines": {
            persistence_name: persistence,
            tier_name: tier,
        },
        "runs": runs,
        "budget_curve": curve,
        "missing_train_sample_budgets": missing_budgets,
        "complete_budget_curve": not missing_budgets,
        "decision": {
            "best_run": best_run,
            "best_train_samples": int(best_record["train_samples"]),
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

    record = summarize_data_budget_sweep(
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
