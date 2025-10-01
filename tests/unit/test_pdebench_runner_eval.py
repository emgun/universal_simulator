from __future__ import annotations

import sys

import h5py
import torch
import yaml

from scripts import evaluate as evaluate_script
from scripts import benchmark as benchmark_script
from scripts import train as train_script
from scripts import train_baselines as train_baselines_script
from ups.eval.pdebench_runner import evaluate_latent_operator


def _write_minimal_hdf5(tmp_path) -> None:
    data = torch.randn(2, 3, 4, dtype=torch.float32)
    file_path = tmp_path / "burgers1d_train.h5"
    with h5py.File(file_path, "w") as handle:
        handle.create_dataset("data", data=data.numpy())


def test_evaluate_latent_operator_runs(tmp_path):
    _write_minimal_hdf5(tmp_path)
    cfg = {
        "training": {"batch_size": 2, "dt": 0.1},
        "latent": {"dim": 8, "tokens": 4},
        "data": {
            "task": "burgers1d",
            "split": "train",
            "root": str(tmp_path),
            "patch_size": 1,
        },
    }

    operator = train_script.make_operator(cfg)
    report = evaluate_latent_operator(cfg, operator)

    assert "mse" in report.metrics
    assert report.metrics["mse"] >= 0.0


def test_evaluate_cli_main(tmp_path, monkeypatch, capsys):
    _write_minimal_hdf5(tmp_path)
    cfg = {
        "training": {"batch_size": 2, "dt": 0.1},
        "latent": {"dim": 8, "tokens": 4},
        "data": {
            "task": "burgers1d",
            "split": "train",
            "root": str(tmp_path),
            "patch_size": 1,
        },
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    operator = evaluate_script.make_operator(cfg)
    operator_path = tmp_path / "operator.pt"
    torch.save(operator.state_dict(), operator_path)

    output_prefix = tmp_path / "eval_run"
    args = [
        "evaluate",
        "--config",
        str(cfg_path),
        "--operator",
        str(operator_path),
        "--output-prefix",
        str(output_prefix),
        "--log-path",
        str(tmp_path / "eval_log.jsonl"),
        "--print-json",
    ]
    monkeypatch.setattr(sys, "argv", args)

    evaluate_script.main()
    output = capsys.readouterr().out
    assert "\"metrics\"" in output
    assert output_prefix.with_suffix(".json").exists()
    assert output_prefix.with_suffix(".csv").exists()
    assert output_prefix.with_suffix(".html").exists()
    assert output_prefix.with_suffix(".config.yaml").exists()
    assert (output_prefix.parent / f"{output_prefix.name}_metrics.png").exists()
    assert (output_prefix.parent / f"{output_prefix.name}_mse_hist.png").exists()
    assert (output_prefix.parent / f"{output_prefix.name}_mae_hist.png").exists()
    assert (tmp_path / "eval_log.jsonl").exists()


def test_benchmark_cli(tmp_path, monkeypatch, capsys):
    _write_minimal_hdf5(tmp_path)
    cfg = {
        "latent": {"dim": 8, "tokens": 4},
        "training": {"batch_size": 2, "dt": 0.1},
        "data": {
            "task": "burgers1d",
            "split": "train",
            "root": str(tmp_path),
            "patch_size": 1,
        },
        "checkpoint": {"dir": str(tmp_path / "ckpts")},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    operator = train_script.make_operator(cfg)
    operator_path = tmp_path / "ckpts" / "operator.pt"
    operator_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(operator.state_dict(), operator_path)

    log_path = tmp_path / "benchmark_log.jsonl"
    args = [
        "benchmark",
        "--config",
        str(cfg_path),
        "--operator",
        str(operator_path),
        "--baseline",
        "identity",
        "--output",
        str(tmp_path / "benchmark.json"),
        "--log-path",
        str(log_path),
    ]
    monkeypatch.setattr(sys, "argv", args)

    benchmark_script.main()
    captured = capsys.readouterr().out
    assert "Benchmark results" in captured
    out_path = tmp_path / "benchmark.json"
    assert out_path.exists()
    assert log_path.exists()


def test_train_baseline_cli(tmp_path, monkeypatch, capsys):
    _write_minimal_hdf5(tmp_path)
    cfg = {
        "latent": {"dim": 8, "tokens": 4},
        "training": {"batch_size": 2, "dt": 0.1},
        "baseline": {"epochs": 1, "log_path": str(tmp_path / "baseline_log.jsonl")},
        "data": {
            "task": "burgers1d",
            "split": "train",
            "root": str(tmp_path),
            "patch_size": 1,
        },
        "checkpoint": {"dir": str(tmp_path / "ckpts")},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    log_path = tmp_path / "baseline_log.jsonl"
    args = [
        "train_baselines",
        "--config",
        str(cfg_path),
        "--baseline",
        "identity",
        "--seed",
        "0",
    ]
    monkeypatch.setattr(sys, "argv", args)

    train_baselines_script.main()
    out = capsys.readouterr().out
    assert "Saved baseline checkpoint" in out
    ckpt = tmp_path / "ckpts" / "baseline_identity.pt"
    assert ckpt.exists()
    assert log_path.exists()
