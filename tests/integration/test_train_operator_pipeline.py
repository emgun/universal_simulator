from __future__ import annotations

import h5py
import torch

from scripts import train as train_script


def _write_minimal_pdebench(tmp_path) -> None:
    data = torch.randn(2, 3, 4, dtype=torch.float32)
    file_path = tmp_path / "burgers1d_train.h5"
    with h5py.File(file_path, "w") as handle:
        handle.create_dataset("data", data=data.numpy())


def test_train_operator_runs_with_pdebench_loader(tmp_path):
    _write_minimal_pdebench(tmp_path)
    cfg = {
        "training": {"batch_size": 2, "dt": 0.1},
        "latent": {"dim": 8, "tokens": 4},
        "stages": {"operator": {"epochs": 1}},
        "optimizer": {"lr": 1e-3},
        "data": {
            "task": "burgers1d",
            "split": "train",
            "root": str(tmp_path),
            "patch_size": 1,
        },
        "checkpoint": {"dir": str(tmp_path / "ckpts")},
    }

    train_script.train_operator(cfg)
    ckpt_path = tmp_path / "ckpts" / "operator.pt"
    assert ckpt_path.exists()
    assert (tmp_path / "ckpts" / "encoder.pt").exists()


def test_train_operator_runs_with_semigroup_loss(tmp_path):
    _write_minimal_pdebench(tmp_path)
    cfg = {
        "training": {"batch_size": 2, "dt": 0.1, "lambda_semigroup": 0.1},
        "latent": {"dim": 8, "tokens": 4},
        "stages": {"operator": {"epochs": 1}},
        "optimizer": {"lr": 1e-3},
        "data": {
            "task": "burgers1d",
            "split": "train",
            "root": str(tmp_path),
            "patch_size": 1,
        },
        "checkpoint": {"dir": str(tmp_path / "ckpts_semigroup")},
    }

    train_script.train_operator(cfg)
    ckpt_path = tmp_path / "ckpts_semigroup" / "operator.pt"
    assert ckpt_path.exists()


def test_train_operator_runs_with_multitask_auto_conditioning(tmp_path):
    for task_name in ("burgers1d", "advection1d"):
        data = torch.randn(2, 3, 4, dtype=torch.float32)
        file_path = tmp_path / f"{task_name}_train.h5"
        with h5py.File(file_path, "w") as handle:
            handle.create_dataset("data", data=data.numpy())

    cfg = {
        "training": {"batch_size": 2, "dt": 0.1, "auto_conditioning": True},
        "latent": {"dim": 8, "tokens": 4},
        "stages": {"operator": {"epochs": 1}},
        "optimizer": {"lr": 1e-3},
        "data": {
            "task": ["burgers1d", "advection1d"],
            "split": "train",
            "root": str(tmp_path),
            "patch_size": 1,
        },
        "checkpoint": {"dir": str(tmp_path / "ckpts_multitask_conditioned")},
    }

    train_script.train_operator(cfg)
    ckpt_path = tmp_path / "ckpts_multitask_conditioned" / "operator.pt"
    assert ckpt_path.exists()
    assert (tmp_path / "ckpts_multitask_conditioned" / "encoder.pt").exists()
    operator = train_script.make_operator(cfg)
    assert operator.conditioner is not None


def test_train_diffusion_runs_with_pdebench_loader(tmp_path):
    _write_minimal_pdebench(tmp_path)
    cfg = {
        "training": {"batch_size": 2, "dt": 0.1},
        "latent": {"dim": 8, "tokens": 4},
        "stages": {"diff_residual": {"epochs": 1}},
        "optimizer": {"lr": 1e-3},
        "data": {
            "task": "burgers1d",
            "split": "train",
            "root": str(tmp_path),
            "patch_size": 1,
        },
        "checkpoint": {"dir": str(tmp_path / "ckpts")},
    }

    train_script.train_operator(cfg)
    train_script.train_diffusion(cfg)
    ckpt_path = tmp_path / "ckpts" / "diffusion_residual.pt"
    assert ckpt_path.exists()


def test_train_decoder_runs_with_saved_encoder(tmp_path):
    _write_minimal_pdebench(tmp_path)
    cfg = {
        "training": {"batch_size": 2, "dt": 0.1},
        "latent": {"dim": 8, "tokens": 4},
        "decoder": {"hidden_dim": 16, "mlp_hidden_dim": 16, "num_heads": 4, "num_layers": 1},
        "stages": {"operator": {"epochs": 1}, "decoder": {"epochs": 1}},
        "optimizer": {"lr": 1e-3},
        "data": {
            "task": "burgers1d",
            "split": "train",
            "root": str(tmp_path),
            "patch_size": 1,
        },
        "checkpoint": {"dir": str(tmp_path / "ckpts_decoder")},
    }

    train_script.train_operator(cfg)
    train_script.train_decoder(cfg)
    ckpt_path = tmp_path / "ckpts_decoder" / "decoder.pt"
    assert ckpt_path.exists()


def test_train_operator_decoded_runs_with_saved_codec(tmp_path):
    _write_minimal_pdebench(tmp_path)
    cfg = {
        "training": {"batch_size": 2, "dt": 0.1},
        "latent": {"dim": 8, "tokens": 4},
        "decoder": {"hidden_dim": 16, "mlp_hidden_dim": 16, "num_heads": 4, "num_layers": 1},
        "stages": {
            "operator": {"epochs": 1},
            "decoder": {"epochs": 1},
            "operator_decoded": {"epochs": 1, "rollout_steps": 2, "lambda_rollout": 0.5},
        },
        "optimizer": {"lr": 1e-3},
        "data": {
            "task": "burgers1d",
            "split": "train",
            "root": str(tmp_path),
            "patch_size": 1,
        },
        "checkpoint": {"dir": str(tmp_path / "ckpts_operator_decoded")},
    }

    train_script.train_operator(cfg)
    train_script.train_decoder(cfg)
    train_script.train_operator_decoded(cfg)
    checkpoint_dir = tmp_path / "ckpts_operator_decoded"
    assert (checkpoint_dir / "operator.pt").exists()
    assert (checkpoint_dir / "operator_decoded.pt").exists()


def test_train_joint_codec_operator_runs_with_saved_codec(tmp_path):
    _write_minimal_pdebench(tmp_path)
    cfg = {
        "training": {"batch_size": 2, "dt": 0.1},
        "latent": {"dim": 8, "tokens": 4},
        "decoder": {"hidden_dim": 16, "mlp_hidden_dim": 16, "num_heads": 4, "num_layers": 1},
        "stages": {
            "operator": {"epochs": 1},
            "decoder": {"epochs": 1},
            "joint_codec_operator": {"epochs": 1, "rollout_steps": 2, "lambda_rollout": 0.5, "lambda_reconstruction": 0.25},
        },
        "optimizer": {"lr": 1e-3},
        "data": {
            "task": "burgers1d",
            "split": "train",
            "root": str(tmp_path),
            "patch_size": 1,
        },
        "checkpoint": {"dir": str(tmp_path / "ckpts_joint_codec")},
    }

    train_script.train_operator(cfg)
    train_script.train_decoder(cfg)
    train_script.train_joint_codec_operator(cfg)
    checkpoint_dir = tmp_path / "ckpts_joint_codec"
    assert (checkpoint_dir / "operator.pt").exists()
    assert (checkpoint_dir / "encoder.pt").exists()
    assert (checkpoint_dir / "decoder.pt").exists()
    assert (checkpoint_dir / "operator_joint.pt").exists()
    assert (checkpoint_dir / "encoder_joint.pt").exists()
    assert (checkpoint_dir / "decoder_joint.pt").exists()


def test_train_joint_codec_operator_runs_with_multitask_variable_grid_batch(tmp_path):
    task_shapes = {"burgers1d": 4, "advection1d": 6}
    for task_name, width in task_shapes.items():
        data = torch.randn(2, 3, width, dtype=torch.float32)
        file_path = tmp_path / f"{task_name}_train.h5"
        with h5py.File(file_path, "w") as handle:
            handle.create_dataset("data", data=data.numpy())

    cfg = {
        "training": {"batch_size": 2, "dt": 0.1, "auto_conditioning": True},
        "latent": {"dim": 8, "tokens": 4},
        "decoder": {"hidden_dim": 16, "mlp_hidden_dim": 16, "num_heads": 4, "num_layers": 1},
        "stages": {
            "operator": {"epochs": 1},
            "decoder": {"epochs": 1},
            "joint_codec_operator": {"epochs": 1, "rollout_steps": 2, "lambda_rollout": 0.5, "lambda_reconstruction": 0.25},
        },
        "optimizer": {"lr": 1e-3},
        "data": {
            "task": ["burgers1d", "advection1d"],
            "split": "train",
            "root": str(tmp_path),
            "patch_size": 1,
        },
        "checkpoint": {"dir": str(tmp_path / "ckpts_joint_codec_multitask")},
    }

    train_script.train_operator(cfg)
    train_script.train_decoder(cfg)
    train_script.train_joint_codec_operator(cfg)
    checkpoint_dir = tmp_path / "ckpts_joint_codec_multitask"
    assert (checkpoint_dir / "operator_joint.pt").exists()
    assert (checkpoint_dir / "encoder_joint.pt").exists()
    assert (checkpoint_dir / "decoder_joint.pt").exists()
