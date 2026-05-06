from __future__ import annotations

import sys

import h5py
import torch
import yaml

from scripts import evaluate as evaluate_script
from scripts import benchmark as benchmark_script
from scripts import train as train_script
from scripts import train_baselines as train_baselines_script
from ups.core.latent_state import LatentState
from ups.eval.persistence_baselines import evaluate_persistence_decoded
from ups.eval import pdebench_runner
from ups.eval.pdebench_runner import evaluate_decoded_operator, evaluate_latent_operator


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


class _DummyEncoder(torch.nn.Module):
    def forward(self, fields, coords, *, meta=None, params=None, bc=None, geom=None):
        return fields["u"]


class _IdentityOperator(torch.nn.Module):
    def forward(self, state: LatentState, dt):
        return LatentState(z=state.z, t=dt if state.t is None else state.t + dt, cond=state.cond)


class _AddOperator(torch.nn.Module):
    def __init__(self, delta: float) -> None:
        super().__init__()
        self.delta = delta

    def forward(self, state: LatentState, dt):
        return LatentState(z=state.z + self.delta, t=dt if state.t is None else state.t + dt, cond=state.cond)


class _DummyDecoder(torch.nn.Module):
    def forward(self, points, latent_tokens, *, conditioning=None):
        return {"u": latent_tokens}


def test_evaluate_decoded_operator_runs_on_constant_sequence(tmp_path):
    data = torch.ones(1, 3, 4, dtype=torch.float32)
    file_path = tmp_path / "burgers1d_train.h5"
    with h5py.File(file_path, "w") as handle:
        handle.create_dataset("data", data=data.numpy())

    cfg = {
        "training": {"batch_size": 1, "dt": 0.1},
        "data": {
            "task": "burgers1d",
            "split": "train",
            "root": str(tmp_path),
            "patch_size": 1,
            "field_name": "u",
        },
    }

    report = evaluate_decoded_operator(
        cfg,
        _DummyEncoder(),
        _IdentityOperator(),
        _DummyDecoder(),
    )

    assert "decoded_mse" in report.metrics
    assert report.metrics["decoded_mse"] == 0.0
    assert report.metrics["decoded_mae"] == 0.0
    assert report.metrics["decoded_rollout_nrmse"] == 0.0
    assert report.metrics["decoded_step1_nrmse"] == 0.0


def test_flatten_field_step_handles_channel_first_scalar_2d():
    field_step = torch.randn(1, 4, 4)
    grid_shape = (4, 4)

    flattened_train = train_script._flatten_field_step(field_step, grid_shape)
    flattened_eval = pdebench_runner._flatten_field_step(field_step, grid_shape)

    assert flattened_train.shape == (1, 16, 1)
    assert flattened_eval.shape == (1, 16, 1)


def test_evaluate_decoded_operator_reports_horizon_metrics_when_available(tmp_path):
    data = torch.ones(1, 17, 4, dtype=torch.float32)
    file_path = tmp_path / "burgers1d_train.h5"
    with h5py.File(file_path, "w") as handle:
        handle.create_dataset("data", data=data.numpy())

    cfg = {
        "training": {"batch_size": 1, "dt": 0.1},
        "data": {
            "task": "burgers1d",
            "split": "train",
            "root": str(tmp_path),
            "patch_size": 1,
            "field_name": "u",
        },
    }

    report = evaluate_decoded_operator(
        cfg,
        _DummyEncoder(),
        _IdentityOperator(),
        _DummyDecoder(),
        rollout_steps=16,
    )

    assert report.metrics["decoded_h4_nrmse"] == 0.0
    assert report.metrics["decoded_h16_nrmse"] == 0.0


def test_evaluate_decoded_operator_can_blend_against_persistence_residual(tmp_path):
    data = torch.tensor([[[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]], dtype=torch.float32)
    file_path = tmp_path / "burgers1d_train.h5"
    with h5py.File(file_path, "w") as handle:
        handle.create_dataset("data", data=data.numpy())

    cfg = {
        "training": {"batch_size": 1, "dt": 0.1},
        "evaluation": {"decoded_persistence_residual_alpha": 0.5},
        "data": {
            "task": "burgers1d",
            "split": "train",
            "root": str(tmp_path),
            "patch_size": 1,
            "field_name": "u",
        },
    }

    persistence_cfg = {**cfg, "evaluation": {"decoded_persistence_residual_alpha": 0.0}}
    persistence = evaluate_persistence_decoded(persistence_cfg, rollout_steps=1)
    report = evaluate_decoded_operator(
        persistence_cfg,
        _DummyEncoder(),
        _AddOperator(delta=2.0),
        _DummyDecoder(),
        rollout_steps=1,
    )

    assert report.metrics["decoded_rollout_nrmse"] == persistence.metrics["decoded_rollout_nrmse"]
    assert report.extra["decoded_persistence_residual_alpha"] == 0.0


def test_evaluate_decoded_operator_can_apply_task_specific_residual_alpha(tmp_path):
    for task_name in ("burgers1d", "advection1d"):
        data = torch.tensor([[[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]], dtype=torch.float32)
        file_path = tmp_path / f"{task_name}_train.h5"
        with h5py.File(file_path, "w") as handle:
            handle.create_dataset("data", data=data.numpy())

    cfg = {
        "training": {"batch_size": 1, "dt": 0.1},
        "evaluation": {
            "decoded_persistence_residual_alpha": 0.0,
            "decoded_persistence_residual_alpha_by_task": {"advection1d": 1.0},
        },
        "data": {
            "task": ["burgers1d", "advection1d"],
            "split": "train",
            "root": str(tmp_path),
            "patch_size": 1,
            "field_name": "u",
        },
    }

    report = evaluate_decoded_operator(
        cfg,
        _DummyEncoder(),
        _AddOperator(delta=1.0),
        _DummyDecoder(),
        rollout_steps=1,
    )

    assert report.metrics["task_burgers1d_decoded_rollout_nrmse"] > report.metrics["task_advection1d_decoded_rollout_nrmse"]
    assert report.metrics["task_advection1d_decoded_rollout_nrmse"] > 0.0
    assert report.extra["decoded_persistence_residual_alpha_by_task"] == {"advection1d": 1.0}


def test_evaluate_decoded_operator_reports_multitask_metrics(tmp_path):
    for task_name in ("burgers1d", "advection1d"):
        data = torch.ones(1, 3, 4, dtype=torch.float32)
        file_path = tmp_path / f"{task_name}_train.h5"
        with h5py.File(file_path, "w") as handle:
            handle.create_dataset("data", data=data.numpy())

    cfg = {
        "training": {"batch_size": 1, "dt": 0.1, "auto_conditioning": True},
        "data": {
            "task": ["burgers1d", "advection1d"],
            "split": "train",
            "root": str(tmp_path),
            "patch_size": 1,
            "field_name": "u",
        },
    }

    report = evaluate_decoded_operator(
        cfg,
        _DummyEncoder(),
        _IdentityOperator(),
        _DummyDecoder(),
    )

    assert report.metrics["task_burgers1d_decoded_rollout_nrmse"] == 0.0
    assert report.metrics["task_advection1d_decoded_rollout_nrmse"] == 0.0
    assert report.metrics["family_conservation_decoded_rollout_nrmse"] == 0.0
    assert report.metrics["family_transport_decoded_rollout_nrmse"] == 0.0


def test_evaluate_decoded_operator_reports_heterogeneous_multitask_metrics(tmp_path):
    burgers = torch.ones(1, 3, 4, dtype=torch.float32)
    advection = torch.ones(1, 3, 4, dtype=torch.float32)
    darcy = torch.ones(1, 3, 1, 4, 4, dtype=torch.float32)

    with h5py.File(tmp_path / "burgers1d_train.h5", "w") as handle:
        handle.create_dataset("data", data=burgers.numpy())
    with h5py.File(tmp_path / "advection1d_train.h5", "w") as handle:
        handle.create_dataset("data", data=advection.numpy())
    with h5py.File(tmp_path / "darcy2d_train.h5", "w") as handle:
        handle.create_dataset("data", data=darcy.numpy())

    cfg = {
        "training": {"batch_size": 1, "dt": 0.1, "auto_conditioning": True},
        "data": {
            "task": ["burgers1d", "advection1d", "darcy2d"],
            "split": "train",
            "root": str(tmp_path),
            "patch_size": 1,
            "field_name": "u",
        },
    }

    report = evaluate_decoded_operator(
        cfg,
        _DummyEncoder(),
        _IdentityOperator(),
        _DummyDecoder(),
    )

    assert report.metrics["decoded_rollout_nrmse"] == 0.0
    assert report.metrics["task_burgers1d_decoded_rollout_nrmse"] == 0.0
    assert report.metrics["task_advection1d_decoded_rollout_nrmse"] == 0.0
    assert report.metrics["task_darcy2d_decoded_rollout_nrmse"] == 0.0
    assert report.metrics["family_elliptic_decoded_rollout_nrmse"] == 0.0


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


def test_evaluate_cli_main_with_decoded_metrics(tmp_path, monkeypatch, capsys):
    _write_minimal_hdf5(tmp_path)
    cfg = {
        "training": {"batch_size": 2, "dt": 0.1},
        "latent": {"dim": 8, "tokens": 4},
        "decoder": {"hidden_dim": 16, "mlp_hidden_dim": 16, "num_heads": 4, "num_layers": 1},
        "data": {
            "task": "burgers1d",
            "split": "train",
            "root": str(tmp_path),
            "patch_size": 1,
        },
    }
    cfg_path = tmp_path / "cfg_decoded.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    operator = evaluate_script.make_operator(cfg)
    encoder = evaluate_script.make_encoder(cfg)
    decoder = evaluate_script.make_decoder(cfg)
    dataset = evaluate_script.PDEBenchDataset(
        evaluate_script.PDEBenchConfig(task="burgers1d", split="train", root=str(tmp_path))
    )
    grid_shape = evaluate_script.infer_grid_shape(dataset.fields[0])
    coords = train_script.make_grid_coords(grid_shape, torch.device("cpu"))
    field_step = dataset[0]["fields"][0]
    flattened = train_script._flatten_field_step(field_step, grid_shape)
    with torch.no_grad():
        encoder({"u": flattened}, coords, meta={"grid_shape": grid_shape})
    operator_path = tmp_path / "operator.pt"
    encoder_path = tmp_path / "encoder.pt"
    decoder_path = tmp_path / "decoder.pt"
    torch.save(operator.state_dict(), operator_path)
    torch.save(encoder.state_dict(), encoder_path)
    torch.save(decoder.state_dict(), decoder_path)

    output_prefix = tmp_path / "eval_decoded"
    args = [
        "evaluate",
        "--config",
        str(cfg_path),
        "--operator",
        str(operator_path),
        "--encoder",
        str(encoder_path),
        "--decoder",
        str(decoder_path),
        "--decoded",
        "--decoded-rollout-steps",
        "1",
        "--output-prefix",
        str(output_prefix),
        "--log-path",
        str(tmp_path / "eval_decoded_log.jsonl"),
        "--print-json",
    ]
    monkeypatch.setattr(sys, "argv", args)

    evaluate_script.main()
    output = capsys.readouterr().out
    assert "\"decoded_mse\"" in output
    assert "\"decoded_rollout_nrmse\"" in output
    assert "\"decoded_h4_nrmse\"" not in output


def test_evaluate_cli_main_with_transfer_tasks(tmp_path, monkeypatch, capsys):
    for task_name in ("burgers1d", "advection1d"):
        data = torch.ones(1, 3, 4, dtype=torch.float32)
        file_path = tmp_path / f"{task_name}_train.h5"
        with h5py.File(file_path, "w") as handle:
            handle.create_dataset("data", data=data.numpy())

    cfg = {
        "training": {"batch_size": 1, "dt": 0.1, "auto_conditioning": True},
        "latent": {"dim": 8, "tokens": 4},
        "decoder": {"hidden_dim": 16, "mlp_hidden_dim": 16, "num_heads": 4, "num_layers": 1},
        "data": {
            "task": "burgers1d",
            "split": "train",
            "root": str(tmp_path),
            "patch_size": 1,
        },
    }
    cfg_path = tmp_path / "cfg_transfer.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    operator = evaluate_script.make_operator(cfg)
    encoder = evaluate_script.make_encoder(cfg)
    decoder = evaluate_script.make_decoder(cfg)
    dataset = evaluate_script.PDEBenchDataset(
        evaluate_script.PDEBenchConfig(task="burgers1d", split="train", root=str(tmp_path))
    )
    grid_shape = evaluate_script.infer_grid_shape(dataset.fields[0])
    coords = train_script.make_grid_coords(grid_shape, torch.device("cpu"))
    field_step = dataset[0]["fields"][0]
    flattened = train_script._flatten_field_step(field_step, grid_shape)
    with torch.no_grad():
        encoder({"u": flattened}, coords, meta={"grid_shape": grid_shape})
    operator_path = tmp_path / "operator_transfer.pt"
    encoder_path = tmp_path / "encoder_transfer.pt"
    decoder_path = tmp_path / "decoder_transfer.pt"
    torch.save(operator.state_dict(), operator_path)
    torch.save(encoder.state_dict(), encoder_path)
    torch.save(decoder.state_dict(), decoder_path)

    output_prefix = tmp_path / "eval_transfer"
    args = [
        "evaluate",
        "--config",
        str(cfg_path),
        "--operator",
        str(operator_path),
        "--encoder",
        str(encoder_path),
        "--decoder",
        str(decoder_path),
        "--decoded",
        "--decoded-rollout-steps",
        "1",
        "--transfer-tasks",
        "advection1d",
        "--output-prefix",
        str(output_prefix),
        "--log-path",
        str(tmp_path / "eval_transfer_log.jsonl"),
        "--print-json",
    ]
    monkeypatch.setattr(sys, "argv", args)

    evaluate_script.main()
    output = capsys.readouterr().out
    assert "\"transfer_mse\"" in output
    assert "\"transfer_decoded_rollout_nrmse\"" in output
    assert "\"transfer_task_advection1d_decoded_rollout_nrmse\"" in output


def test_evaluate_cli_main_with_promotion_rules(tmp_path, monkeypatch, capsys):
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
        "evaluation": {"promotion": {"rules": ["mse>=0.0"]}},
    }
    cfg_path = tmp_path / "cfg_promotion.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    operator = evaluate_script.make_operator(cfg)
    operator_path = tmp_path / "operator_promotion.pt"
    torch.save(operator.state_dict(), operator_path)

    args = [
        "evaluate",
        "--config",
        str(cfg_path),
        "--operator",
        str(operator_path),
        "--promotion-rule",
        "rmse>=0.0",
        "--print-json",
    ]
    monkeypatch.setattr(sys, "argv", args)

    evaluate_script.main()
    output = capsys.readouterr().out
    assert "\"promotion_passed\": true" in output


def test_evaluate_cli_main_with_wildcard_family_promotion_rule(tmp_path, monkeypatch, capsys):
    for task_name in ("burgers1d", "advection1d"):
        data = torch.ones(1, 3, 4, dtype=torch.float32)
        file_path = tmp_path / f"{task_name}_train.h5"
        with h5py.File(file_path, "w") as handle:
            handle.create_dataset("data", data=data.numpy())

    cfg = {
        "training": {"batch_size": 1, "dt": 0.1, "auto_conditioning": True},
        "latent": {"dim": 8, "tokens": 4},
        "decoder": {"hidden_dim": 16, "mlp_hidden_dim": 16, "num_heads": 4, "num_layers": 1},
        "data": {
            "task": ["burgers1d", "advection1d"],
            "split": "train",
            "root": str(tmp_path),
            "patch_size": 1,
        },
    }
    cfg_path = tmp_path / "cfg_family_promotion.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    operator = evaluate_script.make_operator(cfg)
    operator_path = tmp_path / "operator_family_promotion.pt"
    torch.save(operator.state_dict(), operator_path)

    encoder = _DummyEncoder()
    decoder = _DummyDecoder()
    monkeypatch.setattr(evaluate_script, "make_encoder", lambda _cfg: encoder)
    monkeypatch.setattr(evaluate_script, "make_decoder", lambda _cfg: decoder)
    real_load_state_dict = evaluate_script._load_state_dict_compat
    monkeypatch.setattr(
        evaluate_script,
        "_load_state_dict_compat",
        lambda model, *args, **kwargs: real_load_state_dict(model, *args, **kwargs)
        if isinstance(model, evaluate_script.LatentOperator)
        else None,
    )

    output_prefix = tmp_path / "eval_family_promotion"
    args = [
        "evaluate",
        "--config",
        str(cfg_path),
        "--operator",
        str(operator_path),
        "--encoder",
        str(operator_path),
        "--decoder",
        str(operator_path),
        "--decoded",
        "--decoded-rollout-steps",
        "1",
        "--promotion-rule",
        "max:family_*_decoded_rollout_nrmse>=0.0",
        "--output-prefix",
        str(output_prefix),
        "--print-json",
    ]
    monkeypatch.setattr(sys, "argv", args)

    evaluate_script.main()
    output = capsys.readouterr().out
    assert "\"promotion_passed\": true" in output


def test_evaluate_cli_main_fails_on_promotion_failure(tmp_path, monkeypatch):
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
    cfg_path = tmp_path / "cfg_promotion_fail.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    operator = evaluate_script.make_operator(cfg)
    operator_path = tmp_path / "operator_promotion_fail.pt"
    torch.save(operator.state_dict(), operator_path)

    args = [
        "evaluate",
        "--config",
        str(cfg_path),
        "--operator",
        str(operator_path),
        "--promotion-rule",
        "mse<0.0",
        "--fail-on-promotion",
    ]
    monkeypatch.setattr(sys, "argv", args)

    try:
        evaluate_script.main()
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("Expected promotion failure to exit with code 2")


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
