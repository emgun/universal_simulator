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

