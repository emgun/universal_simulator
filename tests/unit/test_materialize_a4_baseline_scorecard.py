from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.materialize_a4_baseline_scorecard import build_scorecard
from ups.data.manifests import canonical_sha256


def _metrics(task: str, value: float) -> dict[str, float]:
    primary = "decoded_solution_nrmse" if task == "darcy2d" else "decoded_rollout_nrmse"
    metrics = {f"task_{task}_{primary}": value}
    if task != "darcy2d":
        metrics.update({f"task_{task}_decoded_h{horizon}_nrmse": value for horizon in range(1, 17)})
    regime_count = {"advection1d": 8, "burgers1d": 12, "darcy2d": 5}[task]
    metrics.update(
        {f"task_{task}_regime_{index}_{primary}": value for index in range(regime_count)}
    )
    return metrics


def _add_global_scale_regime_metrics(metrics: dict[str, float], task: str, value: float) -> None:
    primary = "decoded_solution_nrmse" if task == "darcy2d" else "decoded_rollout_nrmse"
    global_primary = primary.replace("_nrmse", "_global_scale_nrmse")
    regime_count = {"advection1d": 8, "burgers1d": 12, "darcy2d": 5}[task]
    metrics.update(
        {f"task_{task}_regime_{index}_{global_primary}": value for index in range(regime_count)}
    )


def _plan_and_summaries(tmp_path: Path) -> dict:
    tasks = ("advection1d", "burgers1d", "darcy2d")
    values = {"persistence": 0.7, "fno": 0.5, "uno": 0.4, "unet": 0.8, "cno": 0.6}
    runs = []
    rows = []
    persistence_metrics = {}
    for task in tasks:
        persistence_metrics.update(_metrics(task, values["persistence"]))
    persistence_path = Path("reports/persistence/summary.json")
    (tmp_path / persistence_path).parent.mkdir(parents=True)
    (tmp_path / persistence_path).write_text(
        json.dumps({"status": "complete", "split": "val", "metrics": persistence_metrics})
    )
    runs.append(
        {
            "run_id": "persistence",
            "expected_summary": str(persistence_path),
            "model_identity": {"runner_sha256": "a" * 64},
        }
    )
    for task in tasks:
        primary = "decoded_solution_nrmse" if task == "darcy2d" else "decoded_rollout_nrmse"
        rows.append(
            {
                "row_id": f"{task}-persistence",
                "task": task,
                "model": "persistence",
                "source_summary": str(persistence_path),
                "metric_key": f"task_{task}_{primary}",
            }
        )
        models = ("fno", "uno", "unet") + (("cno",) if task != "darcy2d" else ())
        for model in models:
            path = Path(f"reports/{task}-{model}/summary.json")
            (tmp_path / path).parent.mkdir(parents=True)
            (tmp_path / path).write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "split": "val",
                        "metrics": _metrics(task, values[model]),
                        "held_out_test_policy": {"enabled": False},
                        "extra": {"allow_held_out_test_eval": False},
                    }
                )
            )
            runs.append(
                {
                    "run_id": f"{task}-{model}",
                    "expected_summary": str(path),
                    "model_identity": {"runner_sha256": model[0] * 64},
                }
            )
            rows.append(
                {
                    "row_id": f"{task}-{model}",
                    "task": task,
                    "model": model,
                    "source_summary": str(path),
                    "metric_key": f"task_{task}_{primary}",
                }
            )
    payload = {
        "mode": "validation_only",
        "test_access": "forbidden",
        "training_lock": {"lock_sha256": "b" * 64},
        "config": {"sha256": "c" * 64},
        "metric_contract": {"primary": "decoded_rollout_nrmse"},
        "runs": runs,
        "scorecard_plan": {
            "rows": rows,
            "cno_exclusion": "CNO1d-only",
        },
    }
    return {**payload, "plan_sha256": canonical_sha256(payload)}


def test_build_scorecard_binds_summaries_and_excludes_partial_cno(tmp_path):
    scorecard = build_scorecard(_plan_and_summaries(tmp_path), repo_root=tmp_path)

    assert scorecard["status"] == "complete_validation_only"
    assert scorecard["overall_wall"] == {
        "model": "uno",
        "macro_primary_nrmse": pytest.approx(0.4),
        "eligible_models": {
            "persistence": pytest.approx(0.7),
            "fno": pytest.approx(0.5),
            "uno": pytest.approx(0.4),
            "unet": pytest.approx(0.8),
        },
    }
    assert scorecard["model_rollup"]["cno"]["applicable_to_all_tasks"] is False
    assert scorecard["held_out_measurements"] == 0


def test_build_scorecard_rejects_tampered_plan(tmp_path):
    plan = _plan_and_summaries(tmp_path)
    plan["mode"] = "tampered"

    with pytest.raises(ValueError, match="plan SHA-256"):
        build_scorecard(plan, repo_root=tmp_path)


def test_build_scorecard_counts_raw_and_global_scale_regime_metrics_separately(tmp_path):
    plan = _plan_and_summaries(tmp_path)
    for run in plan["runs"]:
        path = tmp_path / run["expected_summary"]
        summary = json.loads(path.read_text(encoding="utf-8"))
        tasks = {
            row["task"]
            for row in plan["scorecard_plan"]["rows"]
            if row["source_summary"] == run["expected_summary"]
        }
        for task in tasks:
            _add_global_scale_regime_metrics(summary["metrics"], task, 0.25)
        path.write_text(json.dumps(summary), encoding="utf-8")

    scorecard = build_scorecard(plan, repo_root=tmp_path)

    assert all(
        row["global_scale_regime_metric_count"]
        == {"advection1d": 8, "burgers1d": 12, "darcy2d": 5}[row["task"]]
        for row in scorecard["rows"]
    )
