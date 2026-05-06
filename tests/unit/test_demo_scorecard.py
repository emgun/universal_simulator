from __future__ import annotations

import json

from scripts.build_demo_report import _summary_paths as report_summary_paths
from scripts.collect_light_results import _summary_paths as collect_summary_paths
from ups.eval.demo_scorecard import (
    collect_scorecard,
    load_cost_index,
    render_scorecard_html,
    write_scorecard_json,
    write_scorecard_tsv,
)
from ups.eval.demo_plots import write_scorecard_plots


def _write_summary(path, *, run_name: str, decoded_rollout_nrmse: float, step1: float = 0.5):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "run_name": run_name,
                "stages": ["operator", "decoder"],
                "config": "resolved_train.yaml",
                "eval_config": "resolved_eval.yaml",
                "duration_sec": 1.25,
                "metrics": {
                    "decoded_rollout_nrmse": decoded_rollout_nrmse,
                    "decoded_step1_nrmse": step1,
                    "task_burgers1d_decoded_rollout_nrmse": decoded_rollout_nrmse,
                },
                "tracking": {
                    "wandb": {
                        "requested": True,
                        "enabled": True,
                        "project": "universal-simulator",
                        "entity": "physics-team",
                        "group": "light-v1",
                        "run_name": run_name,
                        "run_count": 1,
                        "runs": [
                            {
                                "id": f"{run_name}-id",
                                "url": f"https://wandb.ai/physics-team/universal-simulator/runs/{run_name}-id",
                            }
                        ],
                    }
                },
                "extra": {},
            }
        ),
        encoding="utf-8",
    )


def test_collect_scorecard_rows_and_promotion_rules(tmp_path):
    first = tmp_path / "run_a" / "summary.json"
    second = tmp_path / "run_b" / "summary.json"
    _write_summary(first, run_name="run_a", decoded_rollout_nrmse=0.8)
    _write_summary(second, run_name="run_b", decoded_rollout_nrmse=1.1)

    scorecard = collect_scorecard(
        [second, first],
        data_manifest="docs/demo_data_manifest.yaml",
        commit="abc123",
        promotion_rules=["decoded_rollout_nrmse<=0.9"],
    )

    assert [row["run_name"] for row in scorecard.rows] == ["run_a", "run_b"]
    assert "metric:decoded_rollout_nrmse" in scorecard.metric_keys
    assert scorecard.rows[0]["promotion_passed"] is True
    assert scorecard.rows[1]["promotion_passed"] is False
    assert scorecard.rows[0]["data_manifest"] == "docs/demo_data_manifest.yaml"
    assert scorecard.rows[0]["commit"] == "abc123"
    assert scorecard.rows[0]["tracking_wandb_run_ids"] == "run_a-id"
    assert scorecard.rows[0]["tracking_wandb_urls"].endswith("/run_a-id")


def test_write_scorecard_outputs(tmp_path):
    summary = tmp_path / "run_a" / "summary.json"
    _write_summary(summary, run_name="run_a", decoded_rollout_nrmse=0.8)
    scorecard = collect_scorecard([summary])

    tsv_path = tmp_path / "metrics.tsv"
    json_path = tmp_path / "scorecard.json"
    write_scorecard_tsv(scorecard, tsv_path)
    write_scorecard_json(scorecard, json_path)
    html = render_scorecard_html(scorecard, title="Demo <Scorecard>")

    assert "metric:decoded_rollout_nrmse" in tsv_path.read_text(encoding="utf-8")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["rows"][0]["run_name"] == "run_a"
    assert "Demo &lt;Scorecard&gt;" in html
    assert "run_a" in html


def test_write_scorecard_plots_and_embed_html(tmp_path):
    summary = tmp_path / "run_a" / "summary.json"
    _write_summary(summary, run_name="run_a", decoded_rollout_nrmse=0.8)
    scorecard = collect_scorecard([summary])

    plots = write_scorecard_plots(scorecard, tmp_path / "report")
    html = render_scorecard_html(scorecard, plots=plots)

    assert "decoded_rollout_nrmse" in plots
    assert (tmp_path / "report" / plots["decoded_rollout_nrmse"]).exists()
    assert "Metric Plots" in html
    assert "plots/decoded_rollout_nrmse.png" in html


def test_collect_scorecard_includes_optional_cost_json(tmp_path):
    summary = tmp_path / "run_a" / "summary.json"
    cost_json = tmp_path / "run_a" / "cost.json"
    _write_summary(summary, run_name="run_a", decoded_rollout_nrmse=0.8)
    cost_json.write_text(
        json.dumps(
            {
                "run_name": "run_a",
                "provider": "vast",
                "instance_type": "rtx4090-spot",
                "gpu_type": "RTX 4090",
                "gpu_count": 1,
                "wall_clock_hours": 0.5,
                "hourly_usd": 0.35,
            }
        ),
        encoding="utf-8",
    )

    scorecard = collect_scorecard([summary], cost_paths=[cost_json])
    row = scorecard.rows[0]

    assert row["cost_provider"] == "vast"
    assert row["cost_instance_type"] == "rtx4090-spot"
    assert row["cost_gpu_hours"] == 0.5
    assert row["cost_estimated_usd"] == 0.175

    tsv_path = tmp_path / "metrics.tsv"
    write_scorecard_tsv(scorecard, tsv_path)
    assert "cost_estimated_usd" in tsv_path.read_text(encoding="utf-8")


def test_load_cost_index_supports_aggregate_runs_file(tmp_path):
    cost_json = tmp_path / "costs.json"
    cost_json.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "run_name": "run_a",
                        "summary_json": "/tmp/run_a/summary.json",
                        "provider": "runpod",
                        "estimated_usd": 1.25,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    cost_index = load_cost_index([cost_json])

    assert cost_index["run_a"]["provider"] == "runpod"
    assert cost_index["/tmp/run_a/summary.json"]["estimated_usd"] == 1.25


def test_collect_scorecard_adds_baseline_delta_fields(tmp_path):
    baseline = tmp_path / "persistence" / "summary.json"
    candidate = tmp_path / "ups_candidate" / "summary.json"
    _write_summary(baseline, run_name="persistence", decoded_rollout_nrmse=1.0)
    _write_summary(candidate, run_name="ups_candidate", decoded_rollout_nrmse=0.75)

    scorecard = collect_scorecard(
        [candidate, baseline],
        baseline_run_name="persistence",
        baseline_metric_name="decoded_rollout_nrmse",
        baseline_min_improvement=0.2,
    )
    rows = {row["run_name"]: row for row in scorecard.rows}

    assert rows["ups_candidate"]["baseline_metric_value"] == 1.0
    assert rows["ups_candidate"]["baseline_metric_delta"] == -0.25
    assert rows["ups_candidate"]["baseline_metric_ratio"] == 0.75
    assert rows["ups_candidate"]["baseline_improvement_fraction"] == 0.25
    assert rows["ups_candidate"]["baseline_improvement_passed"] is True
    assert rows["persistence"]["baseline_improvement_passed"] is None


def test_report_clis_accept_absolute_glob_patterns(tmp_path):
    first = tmp_path / "run_a" / "summary.json"
    second = tmp_path / "run_b" / "summary.json"
    _write_summary(first, run_name="run_a", decoded_rollout_nrmse=0.8)
    _write_summary(second, run_name="run_b", decoded_rollout_nrmse=0.9)
    pattern = str(tmp_path / "*" / "summary.json")

    assert report_summary_paths([], [pattern]) == [first.resolve(), second.resolve()]
    assert collect_summary_paths([], [pattern]) == [first.resolve(), second.resolve()]
