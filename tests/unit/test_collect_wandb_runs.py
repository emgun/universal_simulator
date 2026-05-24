from __future__ import annotations

import json

from scripts.collect_wandb_runs import run_to_record, write_json, write_tsv


class FakeRun:
    id = "abc123"
    name = "ups_light_v1_task_signature_only"
    state = "finished"
    url = "https://wandb.ai/team/universal-simulator/runs/abc123"
    project = "universal-simulator"
    entity = "team"
    group = "light-v1"
    job_type = "light-experiment"
    tags = ["light-experiment", "residual"]
    created_at = "2026-05-06T00:00:00"
    updated_at = "2026-05-06T00:10:00"
    summary = {
        "decoded_rollout_nrmse": 0.75,
        "operator/loss": 0.1,
        "_runtime": 12,
        "nested": {"skip": True},
    }
    config = {
        "logging": {
            "wandb": {
                "run_name": "ups_light_v1_task_signature_only",
            }
        }
    }


def test_run_to_record_flattens_wandb_summary_metrics():
    record = run_to_record(FakeRun(), metric_prefixes=["decoded_"])

    assert record["wandb_id"] == "abc123"
    assert record["wandb_group"] == "light-v1"
    assert record["config_run_name"] == "ups_light_v1_task_signature_only"
    assert record["metric:decoded_rollout_nrmse"] == 0.75
    assert "metric:operator/loss" not in record
    assert "metric:_runtime" not in record
    assert "metric:nested" not in record


def test_write_wandb_registry_outputs_json_and_tsv(tmp_path):
    row = run_to_record(FakeRun(), metric_prefixes=["decoded_"])
    json_path = tmp_path / "runs.json"
    tsv_path = tmp_path / "runs.tsv"

    write_json([row], json_path)
    write_tsv([row], tsv_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["runs"][0]["wandb_id"] == "abc123"
    tsv = tsv_path.read_text(encoding="utf-8")
    assert "metric:decoded_rollout_nrmse" in tsv
    assert "abc123" in tsv
