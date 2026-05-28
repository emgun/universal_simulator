from __future__ import annotations

import json
import sys

import h5py
import torch
import yaml

from scripts import run_persistence_baseline as baseline_script
from ups.eval.persistence_baselines import evaluate_persistence_decoded


def test_evaluate_persistence_decoded_constant_sequence_is_zero(tmp_path):
    with h5py.File(tmp_path / "burgers1d_test.h5", "w") as handle:
        handle.create_dataset("data", data=torch.ones(2, 4, 8).numpy())

    cfg = {
        "data": {
            "task": "burgers1d",
            "split": "test",
            "root": str(tmp_path),
            "max_samples": 1,
        }
    }

    report = evaluate_persistence_decoded(cfg, rollout_steps=2)

    assert report.metrics["decoded_rollout_nrmse"] == 0.0
    assert report.metrics["decoded_step1_nrmse"] == 0.0
    assert report.extra["baseline"] == "persistence"
    assert report.extra["samples"] == 2


def test_run_persistence_baseline_writes_light_summary(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    output_root = tmp_path / "runs"
    data_root.mkdir()
    with h5py.File(data_root / "burgers1d_test.h5", "w") as handle:
        data = torch.stack(
            [
                torch.arange(4 * 8, dtype=torch.float32).view(4, 8),
                torch.arange(4 * 8, dtype=torch.float32).view(4, 8) + 1.0,
            ]
        )
        handle.create_dataset("data", data=data.numpy())

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "task": "burgers1d",
                    "split": "test",
                    "root": str(data_root),
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_persistence_baseline",
            "--config",
            str(cfg_path),
            "--name",
            "persistence_burgers",
            "--output-root",
            str(output_root),
            "--max-samples",
            "1",
            "--rollout-steps",
            "2",
            "--promotion-rule",
            "decoded_rollout_nrmse<=2.0",
        ],
    )

    baseline_script.main()

    summary_path = output_root / "persistence_burgers" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["run_name"] == "persistence_burgers"
    assert summary["stages"] == ["persistence"]
    assert "decoded_rollout_nrmse" in summary["metrics"]
    assert summary["extra"]["promotion_passed"] is True
    assert (output_root / "results.tsv").exists()


def test_append_results_row_preserves_existing_tracking_columns(tmp_path):
    results_path = tmp_path / "runs" / "results.tsv"
    results_path.parent.mkdir()
    results_path.write_text(
        "\t".join(
            [
                "run_name",
                "timestamp",
                "stages",
                "decoded",
                "train_split",
                "eval_split",
                "transfer_tasks",
                "promotion_passed",
                "main_metric_name",
                "main_metric_value",
                "summary_json",
                "wandb_run_ids",
                "wandb_urls",
            ]
        )
        + "\n"
        + "\t".join(
            [
                "candidate",
                "123",
                "train,eval",
                "True",
                "train",
                "test",
                "",
                "True",
                "decoded_rollout_nrmse",
                "0.3",
                "candidate/summary.json",
                "run-1",
                "https://wandb.invalid/run-1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    baseline_script._append_results_row(
        results_path,
        {
            "run_name": "persistence",
            "timestamp": 456,
            "stages": "persistence",
            "decoded": True,
            "train_split": "",
            "eval_split": "test",
            "transfer_tasks": "",
            "promotion_passed": "",
            "main_metric_name": "decoded_rollout_nrmse",
            "main_metric_value": 0.6,
            "summary_json": "persistence/summary.json",
        },
    )

    lines = results_path.read_text(encoding="utf-8").strip().splitlines()
    header = lines[0].split("\t")
    assert "wandb_run_ids" in header
    assert "wandb_urls" in header
    assert len(lines) == 3

    candidate = dict(zip(header, lines[1].split("\t", maxsplit=len(header) - 1)))
    persistence = dict(zip(header, lines[2].split("\t", maxsplit=len(header) - 1)))
    assert candidate["wandb_run_ids"] == "run-1"
    assert candidate["wandb_urls"] == "https://wandb.invalid/run-1"
    assert persistence["run_name"] == "persistence"
