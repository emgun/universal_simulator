from __future__ import annotations

import json

from scripts.extract_experiment_metrics import append_row, extract_row


def test_extract_row_computes_baseline_ratio_and_task_metrics(tmp_path):
    summary_path = tmp_path / "summary.json"
    baseline_path = tmp_path / "baseline.json"
    summary_path.write_text(
        json.dumps(
            {
                "run_name": "candidate",
                "duration_sec": 12.5,
                "metrics": {
                    "decoded_rollout_nrmse": 0.45,
                    "decoded_h4_nrmse": 0.4,
                    "decoded_rollout_spectral_energy_error": 0.2,
                    "task_advection1d_decoded_rollout_nrmse": 0.7,
                    "task_burgers1d_decoded_rollout_nrmse": 0.2,
                    "task_darcy2d_decoded_rollout_nrmse": 0.3,
                    "family_transport_decoded_rollout_nrmse": 0.7,
                },
                "tracking": {
                    "wandb": {
                        "runs": [
                            {"url": "https://wandb.ai/example/one"},
                            {"url": "https://wandb.ai/example/two"},
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    baseline_path.write_text(
        json.dumps({"metrics": {"decoded_rollout_nrmse": 0.5}}), encoding="utf-8"
    )

    row = extract_row(
        summary_path,
        baseline_summary_path=baseline_path,
        status="keep",
        description="learned gate",
    )

    assert row["run_name"] == "candidate"
    assert row["primary_metric"] == "decoded_rollout_nrmse"
    assert row["primary_metric_value"] == 0.45
    assert row["baseline_metric_value"] == 0.5
    assert round(row["baseline_ratio"], 6) == 0.9
    assert round(row["baseline_improvement_fraction"], 6) == 0.1
    assert row["advection_nrmse"] == 0.7
    assert row["burgers_nrmse"] == 0.2
    assert row["darcy_nrmse"] == 0.3
    assert row["transport_nrmse"] == 0.7
    assert row["h4_nrmse"] == 0.4
    assert row["spectral_error"] == 0.2
    assert row["duration_sec"] == 12.5
    assert row["wandb_urls"] == "https://wandb.ai/example/one,https://wandb.ai/example/two"
    assert row["status"] == "keep"
    assert row["description"] == "learned gate"


def test_append_row_writes_header_once(tmp_path):
    output_path = tmp_path / "results.tsv"
    row = extract_row(tmp_path / "summary.json") if False else {"run_name": "candidate"}

    append_row(output_path, row)
    append_row(output_path, row)

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("timestamp\tbranch\tcommit\trun_name")
    assert len(lines) == 3
