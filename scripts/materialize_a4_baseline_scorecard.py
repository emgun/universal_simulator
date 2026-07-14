#!/usr/bin/env python
from __future__ import annotations

"""Bind completed A4 validation summaries into one deterministic scorecard."""

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from ups.data.manifests import canonical_sha256

TASKS = ("advection1d", "burgers1d", "darcy2d")
COMPLETE_MODELS = ("persistence", "fno", "uno", "unet")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_plan(plan: dict[str, Any]) -> None:
    expected = plan.get("plan_sha256")
    payload = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if expected != canonical_sha256(payload):
        raise ValueError("A4 plan SHA-256 does not match its canonical payload")
    if plan.get("mode") != "validation_only" or plan.get("test_access") != "forbidden":
        raise ValueError("A4 scorecard accepts validation-only, test-forbidden plans")


def _metric_coverage(metrics: dict[str, Any], task: str) -> dict[str, int]:
    regime_prefix = f"task_{task}_regime_"
    primary = "decoded_solution_nrmse" if task == "darcy2d" else "decoded_rollout_nrmse"
    regime_count = sum(
        key.startswith(regime_prefix)
        and key.endswith(f"_{primary}")
        and "_global_scale_" not in key
        for key in metrics
    )
    global_scale_regime_count = sum(
        key.startswith(regime_prefix)
        and key.endswith(f"_{primary.replace('_nrmse', '_global_scale_nrmse')}")
        for key in metrics
    )
    horizon_count = sum(
        f"task_{task}_decoded_h{horizon}_nrmse" in metrics for horizon in range(1, 17)
    )
    expected_regimes = {"advection1d": 8, "burgers1d": 12, "darcy2d": 5}[task]
    if regime_count != expected_regimes:
        raise ValueError(
            f"{task} summary has {regime_count} regime metrics, expected {expected_regimes}"
        )
    if global_scale_regime_count not in {0, expected_regimes}:
        raise ValueError(
            f"{task} summary has {global_scale_regime_count} global-scale regime metrics, "
            f"expected 0 or {expected_regimes}"
        )
    expected_horizons = 0 if task == "darcy2d" else 16
    if horizon_count != expected_horizons:
        raise ValueError(
            f"{task} summary has {horizon_count} horizon metrics, expected {expected_horizons}"
        )
    return {
        "regime_metric_count": regime_count,
        "global_scale_regime_metric_count": global_scale_regime_count,
        "horizon_metric_count": horizon_count,
    }


def build_scorecard(plan: dict[str, Any], *, repo_root: Path) -> dict[str, Any]:
    _validate_plan(plan)
    runs = {run["run_id"]: run for run in plan["runs"]}
    run_by_summary = {run["expected_summary"]: run for run in runs.values()}
    rows: list[dict[str, Any]] = []
    values_by_model: dict[str, dict[str, float]] = defaultdict(dict)

    for row_plan in plan["scorecard_plan"]["rows"]:
        summary_rel = row_plan["source_summary"]
        summary_path = repo_root / summary_rel
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        source_run = run_by_summary.get(summary_rel)
        if source_run is None:
            raise ValueError(f"Scorecard summary is not produced by the frozen plan: {summary_rel}")
        if summary.get("status", "complete") != "complete" or summary.get("split") != "val":
            raise ValueError(f"Incomplete or non-validation summary: {summary_rel}")
        held_out = summary.get("held_out_test_policy", {})
        if held_out.get("enabled") or summary.get("extra", {}).get("allow_held_out_test_eval"):
            raise ValueError(f"Held-out access detected in validation summary: {summary_rel}")
        value = summary.get("metrics", {}).get(row_plan["metric_key"])
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(
                f"Missing finite primary metric {row_plan['metric_key']}: {summary_rel}"
            )
        task = row_plan["task"]
        model = row_plan["model"]
        coverage = _metric_coverage(summary["metrics"], task)
        values_by_model[model][task] = float(value)
        rows.append(
            {
                **row_plan,
                "primary_nrmse": float(value),
                "summary_sha256": _sha256(summary_path),
                "run_id": source_run["run_id"],
                "runner_sha256": source_run["model_identity"]["runner_sha256"],
                "embedded_data_provenance": summary.get("data_provenance") is not None,
                **coverage,
            }
        )

    model_rollup: dict[str, Any] = {}
    for model, task_values in sorted(values_by_model.items()):
        applicable_to_all = set(task_values) == set(TASKS)
        model_rollup[model] = {
            "tasks": task_values,
            "applicable_to_all_tasks": applicable_to_all,
            "macro_primary_nrmse": (
                sum(task_values.values()) / len(TASKS) if applicable_to_all else None
            ),
            "partial_macro_primary_nrmse": sum(task_values.values()) / len(task_values),
        }

    task_wall: dict[str, Any] = {}
    for task in TASKS:
        candidates = {
            model: values[task] for model, values in values_by_model.items() if task in values
        }
        best_model = min(candidates, key=candidates.get)
        task_wall[task] = {"model": best_model, "primary_nrmse": candidates[best_model]}

    complete = {
        model: model_rollup[model]["macro_primary_nrmse"]
        for model in COMPLETE_MODELS
        if model_rollup.get(model, {}).get("applicable_to_all_tasks")
    }
    best_complete = min(complete, key=complete.get)
    payload = {
        "schema_version": 1,
        "scorecard_id": "a4-strat-v1-baseline-validation-v1",
        "status": "complete_validation_only",
        "plan_sha256": plan["plan_sha256"],
        "training_lock": plan["training_lock"],
        "config": plan["config"],
        "metric_contract": plan["metric_contract"],
        "rows": rows,
        "model_rollup": model_rollup,
        "task_wall": task_wall,
        "overall_wall": {
            "model": best_complete,
            "macro_primary_nrmse": complete[best_complete],
            "eligible_models": complete,
        },
        "cno_scope": plan["scorecard_plan"]["cno_exclusion"],
        "evidence_binding": (
            "frozen plan identities plus content-hashed complete summaries; "
            "historic external summaries enforced the lock at runtime but did not embed it"
        ),
        "held_out_measurements": 0,
    }
    return {**payload, "scorecard_sha256": canonical_sha256(payload)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="reports/a4_strat_v1_baseline_plan.json")
    parser.add_argument(
        "--output",
        default="docs/research/artifacts/strat_v1_a4_validation_scorecard.json",
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    plan = json.loads((repo_root / args.plan).read_text(encoding="utf-8"))
    scorecard = build_scorecard(plan, repo_root=repo_root)
    output = repo_root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(scorecard, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"output": str(output), "scorecard_sha256": scorecard["scorecard_sha256"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
