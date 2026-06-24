from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py

from scripts import run_light_experiment as runner_script
from scripts import train as train_script
from scripts.validate_model_side_transport_head_summary import validate_summary
from ups.eval.reports import MetricReport


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
    ledger_path = tmp_path / "test_ledger.json"
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
        "--promotion-rule",
        "mse>=0.0",
        "--held-out-test-ledger-json",
        str(ledger_path),
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
    test_summary = json.loads((run_dir / "summary_test.json").read_text(encoding="utf-8"))
    assert test_summary["duration_sec"] >= 0.0
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert len(ledger["measurements"]) == 1
    assert ledger["measurements"][0]["test_split"] == "test"
    assert ledger["measurements"][0]["validation_metric_name"] == "mse"
    assert ledger["measurements"][0]["test_metric_name"] == "mse"
    assert (run_dir / "synthetic_pdebench" / "burgers1d_val.h5").exists()
    assert (run_dir / "synthetic_pdebench" / "burgers1d_test.h5").exists()
    with h5py.File(run_dir / "synthetic_pdebench" / "burgers1d_val.h5", "r") as handle:
        assert handle["data"].shape == (3, 5, 8)
    with h5py.File(run_dir / "synthetic_pdebench" / "burgers1d_test.h5", "r") as handle:
        assert handle["data"].shape == (3, 5, 8)


def test_held_out_measurement_key_ignores_cosmetic_run_name():
    args = argparse.Namespace(
        checkpoint_source="checkpoints/source",
        config="configs/train.yaml",
        decoded=True,
        decoded_rollout_steps=16,
        eval_config=None,
        eval_override=["evaluation.foo=1"],
        extra_eval_split=["test"],
        name="first_name",
        override=["data.max_samples=32"],
        promotion_rule=["decoded_rollout_nrmse<=0.35"],
        skip_training=True,
        stage=None,
    )
    split_cfg = {"data": {"split": "test", "max_samples": 32}}

    first_key = runner_script._held_out_measurement_key(
        args=args,
        split_name="test",
        split_cfg=split_cfg,
    )
    args.name = "second_name"
    second_key = runner_script._held_out_measurement_key(
        args=args,
        split_name="test",
        split_cfg=split_cfg,
    )

    assert first_key == second_key


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


def test_run_light_experiment_preserves_model_side_transport_head_summary_extra(
    tmp_path, monkeypatch
):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    for name in ("operator.pt", "encoder.pt", "decoder.pt"):
        (checkpoint_dir / name).write_text(name, encoding="utf-8")

    def fake_load_state_dict_compat(_model, _path, *, prefix_to_strip=""):
        return None

    def fake_evaluate_latent_operator(_cfg, _operator, **_kwargs):
        return MetricReport(metrics={"mse": 0.0}, extra={}), {"latent": True}

    def fake_evaluate_decoded_operator(_cfg, _encoder, _operator, _decoder, **_kwargs):
        return MetricReport(
            metrics={
                "decoded_rollout_nrmse": 0.11,
                "task_advection1d_decoded_rollout_nrmse": 0.001,
                "task_advection1d_decoded_h16_nrmse": 0.001,
                "task_burgers1d_decoded_rollout_nrmse": 0.14,
                "task_darcy2d_decoded_rollout_nrmse": 0.18,
            },
            extra={
                "model_side_transport_head": {
                    "enabled": True,
                    "tasks": ["advection1d"],
                    "required_params": ["beta"],
                    "mode": "periodic_roll",
                    "apply_at": "decoded_rollout",
                },
                "model_side_transport_head_metrics": {
                    "applied_count": 4,
                    "skipped_count": 0,
                    "beta_missing_count": 0,
                },
            },
        )

    monkeypatch.setattr(runner_script.evaluate_script, "make_operator", lambda _cfg: object())
    monkeypatch.setattr(runner_script.evaluate_script, "make_encoder", lambda _cfg: object())
    monkeypatch.setattr(runner_script.evaluate_script, "make_decoder", lambda _cfg: object())
    monkeypatch.setattr(
        runner_script.evaluate_script, "_load_state_dict_compat", fake_load_state_dict_compat
    )
    monkeypatch.setattr(runner_script, "evaluate_latent_operator", fake_evaluate_latent_operator)
    monkeypatch.setattr(runner_script, "evaluate_decoded_operator", fake_evaluate_decoded_operator)

    summary = runner_script._evaluate_once(
        {"data": {"split": "val", "task": ["advection1d", "burgers1d", "darcy2d"]}},
        checkpoint_dir=checkpoint_dir,
        operator_checkpoint_names=("operator.pt",),
        decoded=True,
        device="cpu",
        decoded_rollout_steps=16,
        transfer_tasks=(),
        transfer_split=None,
        cli_promotion_rules=(),
    )

    extra = summary["extra"]
    assert extra["model_side_transport_head"]["enabled"] is True
    assert extra["model_side_transport_head_metrics"]["beta_missing_count"] == 0
    assert extra["decoded_model_side_transport_head"]["enabled"] is True
    assert extra["decoded_model_side_transport_head_metrics"]["applied_count"] == 4
    assert validate_summary(summary) == []


def test_run_light_experiment_evaluates_checkpoint_from_last_operator_stage(tmp_path, monkeypatch):
    output_root = tmp_path / "light_runs"
    source_dir = tmp_path / "source" / "checkpoints"
    source_dir.mkdir(parents=True)
    for name in ("operator_joint.pt", "operator_decoded.pt", "operator.pt"):
        (source_dir / name).write_text(f"source {name}", encoding="utf-8")

    loaded_checkpoints = []

    def fake_operator_decoded(cfg):
        checkpoint_dir = Path(cfg["checkpoint"]["dir"])
        (checkpoint_dir / "operator_decoded.pt").write_text("new decoded", encoding="utf-8")
        (checkpoint_dir / "operator.pt").write_text("new decoded", encoding="utf-8")

    def fake_load_state_dict_compat(_model, path, *, prefix_to_strip=""):
        loaded_checkpoints.append(path)

    def fake_evaluate_latent_operator(_cfg, _operator, **_kwargs):
        return MetricReport(metrics={"mse": 0.0}, extra={}), {"loaded": loaded_checkpoints[-1]}

    monkeypatch.setitem(runner_script.STAGE_FUNCTIONS, "operator_decoded", fake_operator_decoded)
    monkeypatch.setattr(runner_script.evaluate_script, "make_operator", lambda _cfg: object())
    monkeypatch.setattr(
        runner_script.evaluate_script, "_load_state_dict_compat", fake_load_state_dict_compat
    )
    monkeypatch.setattr(runner_script, "evaluate_latent_operator", fake_evaluate_latent_operator)

    args = [
        "run_light_experiment",
        "--config",
        "configs/train_burgers_light_operator.yaml",
        "--name",
        "decoded_after_joint",
        "--output-root",
        str(output_root),
        "--checkpoint-source",
        str(source_dir.parent),
        "--stage",
        "operator_decoded",
    ]
    monkeypatch.setattr(sys, "argv", args)

    runner_script.main()

    summary = json.loads(
        (output_root / "decoded_after_joint" / "summary.json").read_text(encoding="utf-8")
    )
    assert Path(summary["checkpoints"]["operator"]).name == "operator_decoded.pt"


def test_run_light_experiment_skip_training_can_prefer_requested_stage_checkpoint(
    tmp_path, monkeypatch
):
    output_root = tmp_path / "light_runs"
    source_dir = tmp_path / "source" / "checkpoints"
    source_dir.mkdir(parents=True)
    for name in ("operator_joint.pt", "operator_decoded.pt", "operator.pt"):
        (source_dir / name).write_text(f"source {name}", encoding="utf-8")

    loaded_checkpoints = []

    def fake_load_state_dict_compat(_model, path, *, prefix_to_strip=""):
        loaded_checkpoints.append(path)

    def fake_evaluate_latent_operator(_cfg, _operator, **_kwargs):
        return MetricReport(metrics={"mse": 0.0}, extra={}), {"loaded": loaded_checkpoints[-1]}

    monkeypatch.setattr(runner_script.evaluate_script, "make_operator", lambda _cfg: object())
    monkeypatch.setattr(
        runner_script.evaluate_script, "_load_state_dict_compat", fake_load_state_dict_compat
    )
    monkeypatch.setattr(runner_script, "evaluate_latent_operator", fake_evaluate_latent_operator)

    args = [
        "run_light_experiment",
        "--config",
        "configs/train_burgers_light_operator.yaml",
        "--name",
        "decoded_skip_training",
        "--output-root",
        str(output_root),
        "--checkpoint-source",
        str(source_dir.parent),
        "--skip-training",
        "--stage",
        "operator_decoded",
    ]
    monkeypatch.setattr(sys, "argv", args)

    runner_script.main()

    summary = json.loads(
        (output_root / "decoded_skip_training" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["stages"] == []
    assert summary["checkpoint_preference_stages"] == ["operator_decoded"]
    assert Path(summary["checkpoints"]["operator"]).name == "operator_decoded.pt"


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
