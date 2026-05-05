from __future__ import annotations

import json

from ups.eval.demo_scorecard import (
    collect_scorecard,
    render_scorecard_html,
    write_scorecard_json,
    write_scorecard_tsv,
)


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

