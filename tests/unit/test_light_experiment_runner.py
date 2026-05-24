from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py

from scripts import run_light_experiment as runner_script
from scripts import train as train_script


def test_train_load_config_supports_include():
    cfg = train_script.load_config("configs/train_burgers_light_operator.yaml")
    assert cfg["latent"]["dim"] == 16
    assert cfg["training"]["auto_conditioning"] is True
    assert cfg["stages"]["operator"]["epochs"] == 1


def test_run_light_experiment_bootstraps_and_records_results(tmp_path, monkeypatch):
    output_root = tmp_path / "light_runs"
    args = [
        "run_light_experiment",
        "--config",
        "configs/train_burgers_light_operator.yaml",
        "--eval-config",
        "configs/eval_burgers_light_proxy.yaml",
        "--name",
        "smoke_operator",
        "--output-root",
        str(output_root),
        "--bootstrap-synthetic",
        "--device",
        "cpu",
    ]
    monkeypatch.setattr(sys, "argv", args)

    runner_script.main()

    run_dir = output_root / "smoke_operator"
    summary_path = run_dir / "summary.json"
    results_tsv = output_root / "results.tsv"
    assert summary_path.exists()
    assert results_tsv.exists()
    assert (run_dir / "resolved_train.yaml").exists()
    assert (run_dir / "resolved_eval.yaml").exists()
    assert (run_dir / "checkpoints" / "operator.pt").exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "metrics" in summary
    assert summary["extra"]["promotion_passed"] is True
    assert summary["run_name"] == "smoke_operator"
    assert summary["stages"] == ["operator"]
    assert summary["tracking"]["wandb"]["requested"] is False
    assert summary["tracking"]["wandb"]["enabled"] is False

    lines = results_tsv.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert "smoke_operator" in lines[1]
    assert "wandb_run_ids" in lines[0]


def test_run_light_experiment_applies_eval_overrides_without_eval_config(tmp_path, monkeypatch):
    output_root = tmp_path / "light_runs"
    args = [
        "run_light_experiment",
        "--config",
        "configs/train_burgers_light_operator.yaml",
        "--name",
        "smoke_operator_val",
        "--output-root",
        str(output_root),
        "--bootstrap-synthetic",
        "--device",
        "cpu",
        "--eval-override",
        "data.split=val",
        "--extra-eval-split",
        "test",
        "--synthetic-samples",
        "3",
        "--synthetic-steps",
        "5",
    ]
    monkeypatch.setattr(sys, "argv", args)

    runner_script.main()

    run_dir = output_root / "smoke_operator_val"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    assert "split: val" in (run_dir / "resolved_eval.yaml").read_text(encoding="utf-8")
    assert (run_dir / "summary_test.json").exists()
    assert summary["extra_evaluations"]["test"]["summary"].endswith("summary_test.json")
    assert (run_dir / "synthetic_pdebench" / "burgers1d_val.h5").exists()
    assert (run_dir / "synthetic_pdebench" / "burgers1d_test.h5").exists()
    with h5py.File(run_dir / "synthetic_pdebench" / "burgers1d_val.h5", "r") as handle:
        assert handle["data"].shape == (3, 5, 8)
    with h5py.File(run_dir / "synthetic_pdebench" / "burgers1d_test.h5", "r") as handle:
        assert handle["data"].shape == (3, 5, 8)


def test_run_light_experiment_can_reuse_checkpoints_for_eval_only(tmp_path, monkeypatch):
    output_root = tmp_path / "light_runs"
    train_args = [
        "run_light_experiment",
        "--config",
        "configs/train_burgers_light_operator.yaml",
        "--eval-config",
        "configs/eval_burgers_light_proxy.yaml",
        "--name",
        "trained_operator",
        "--output-root",
        str(output_root),
        "--bootstrap-synthetic",
        "--device",
        "cpu",
    ]
    monkeypatch.setattr(sys, "argv", train_args)
    runner_script.main()

    eval_args = [
        "run_light_experiment",
        "--config",
        "configs/train_burgers_light_operator.yaml",
        "--eval-config",
        "configs/eval_burgers_light_proxy.yaml",
        "--name",
        "eval_only",
        "--output-root",
        str(output_root),
        "--checkpoint-source",
        str(output_root / "trained_operator"),
        "--skip-training",
        "--bootstrap-synthetic",
        "--device",
        "cpu",
    ]
    monkeypatch.setattr(sys, "argv", eval_args)
    runner_script.main()

    run_dir = output_root / "eval_only"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["skip_training"] is True
    assert summary["stages"] == []
    assert summary["checkpoint_source"].endswith("trained_operator")
    assert (run_dir / "checkpoints" / "operator.pt").exists()
    assert "metrics" in summary


def test_run_light_experiment_logs_benchmark_summary_to_wandb(tmp_path, monkeypatch):
    output_root = tmp_path / "light_runs"
    logged_payloads = []

    class FakeSession:
        metadata = {"id": "summary123", "url": "https://wandb.ai/entity/project/runs/summary123"}

        def log(self, payload):
            logged_payloads.append(payload)

        def finish(self):
            pass

    def fake_init_monitoring_session(cfg, *, component, file_path=None):
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with (Path(file_path).parent / "wandb_runs.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"component": component, "id": "summary123", "url": FakeSession.metadata["url"]}
                )
                + "\n"
            )
        return FakeSession()

    monkeypatch.setattr(runner_script, "init_monitoring_session", fake_init_monitoring_session)
    monkeypatch.setattr(train_script, "init_monitoring_session", fake_init_monitoring_session)
    args = [
        "run_light_experiment",
        "--config",
        "configs/train_burgers_light_operator.yaml",
        "--eval-config",
        "configs/eval_burgers_light_proxy.yaml",
        "--name",
        "wandb_summary",
        "--output-root",
        str(output_root),
        "--bootstrap-synthetic",
        "--device",
        "cpu",
        "--allow-wandb",
        "--wandb-project",
        "universal-simulator",
    ]
    monkeypatch.setattr(sys, "argv", args)

    runner_script.main()

    summary = json.loads(
        (output_root / "wandb_summary" / "summary.json").read_text(encoding="utf-8")
    )
    assert logged_payloads
    assert "summary/mse" in logged_payloads[-1]
    assert summary["tracking"]["wandb"]["run_count"] >= 1
    assert "summary123" in [run["id"] for run in summary["tracking"]["wandb"]["runs"]]


def test_bootstrap_synthetic_2d_scalar_keeps_channel_dim(tmp_path):
    root = tmp_path / "synthetic"
    runner_script.bootstrap_synthetic_pdebench(
        root,
        tasks=("darcy2d",),
        splits=("train",),
    )

    with h5py.File(root / "darcy2d_train.h5", "r") as handle:
        data = handle["data"][:]

    assert data.shape == (2, 4, 1, 4, 4)
