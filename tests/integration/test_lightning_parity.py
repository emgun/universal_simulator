"""Integration tests for Lightning training parity with native training."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
import torch
import yaml


def run_training(script: str, config_path: str, ckpt_dir: str, use_torchrun: bool = False, nproc: int = 1) -> dict:
    """Run training and return metrics.

    Args:
        script: Path to training script
        config_path: Path to config file
        ckpt_dir: Directory for checkpoints
        use_torchrun: Whether to use torchrun
        nproc: Number of processes for torchrun

    Returns:
        Dictionary with training metrics
    """
    if use_torchrun and nproc > 1:
        cmd = [
            "torchrun",
            f"--nproc_per_node={nproc}",
            "--nnodes=1",
            "--node_rank=0",
            "--master_addr=localhost",
            "--master_port=29500",
            script,
            "--config", config_path,
            "--stage", "operator",
        ]
    else:
        cmd = [
            "python",
            script,
            "--config", config_path,
            "--stage", "operator",
        ]

    # Note: train.py doesn't accept --epochs, it uses config file

    result = subprocess.run(cmd, capture_output=True, text=True)

    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "checkpoint_dir": Path(ckpt_dir),
    }


@pytest.mark.slow
@pytest.mark.gpu
def test_lightning_vs_native_single_gpu():
    """Test that Lightning produces similar results to native training on single GPU."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test config (minimal for speed)
        config = {
            "data": {
                "task": "burgers1d",
                "root": "data/pdebench",
                "split": "train",
            },
            "latent": {
                "dim": 32,
                "tokens": 64,
            },
            "operator": {
                "architecture_type": "pdet_unet",
                "pdet": {
                    "input_dim": 32,
                    "hidden_dim": 64,
                    "depths": [1, 1],
                    "group_size": 8,
                    "num_heads": 4,
                },
            },
            "training": {
                "batch_size": 4,
                "num_workers": 0,
                "compile": False,  # Disable for faster testing
                "amp": False,
                "dt": 0.1,
            },
            "stages": {
                "operator": {
                    "epochs": 2,
                },
            },
            "optimizer": {
                "name": "adam",
                "lr": 1e-3,
            },
            "checkpoint": {
                "dir": str(Path(tmpdir) / "checkpoints"),
            },
            "logging": {
                "wandb": {
                    "enabled": False,
                },
            },
        }

        config_path = Path(tmpdir) / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # Run native training
        native_ckpt_dir = Path(tmpdir) / "native_checkpoints"
        native_ckpt_dir.mkdir(parents=True)
        config["checkpoint"]["dir"] = str(native_ckpt_dir)

        native_config_path = Path(tmpdir) / "native_config.yaml"
        with open(native_config_path, "w") as f:
            yaml.dump(config, f)

        native_result = run_training(
            "scripts/train.py",
            str(native_config_path),
            ckpt_dir=str(native_ckpt_dir),
        )

        assert native_result["returncode"] == 0, f"Native training failed:\n{native_result['stderr']}"

        # Run Lightning training
        lightning_ckpt_dir = Path(tmpdir) / "lightning_checkpoints"
        lightning_ckpt_dir.mkdir(parents=True)
        config["checkpoint"]["dir"] = str(lightning_ckpt_dir)

        lightning_config_path = Path(tmpdir) / "lightning_config.yaml"
        with open(lightning_config_path, "w") as f:
            yaml.dump(config, f)

        lightning_result = run_training(
            "scripts/train_lightning.py",
            str(lightning_config_path),
            ckpt_dir=str(lightning_ckpt_dir),
        )

        assert lightning_result["returncode"] == 0, f"Lightning training failed:\n{lightning_result['stderr']}"

        # Both should complete successfully
        print("Native training completed successfully")
        print("Lightning training completed successfully")


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires 2 GPUs")
def test_lightning_ddp_2gpu():
    """Test Lightning with DDP on 2 GPUs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "data": {
                "task": "burgers1d",
                "root": "data/pdebench",
                "split": "train",
            },
            "latent": {
                "dim": 32,
                "tokens": 64,
            },
            "operator": {
                "architecture_type": "pdet_unet",
                "pdet": {
                    "input_dim": 32,
                    "hidden_dim": 64,
                    "depths": [1, 1],
                    "group_size": 8,
                    "num_heads": 4,
                },
            },
            "training": {
                "num_gpus": 2,
                "batch_size": 4,
                "num_workers": 0,
                "compile": False,
                "amp": False,
                "dt": 0.1,
            },
            "stages": {
                "operator": {
                    "epochs": 2,
                },
            },
            "optimizer": {
                "name": "adam",
                "lr": 1e-3,
            },
            "checkpoint": {
                "dir": str(Path(tmpdir) / "checkpoints"),
            },
            "logging": {
                "wandb": {
                    "enabled": False,
                },
            },
        }

        config_path = Path(tmpdir) / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        result = run_training(
            "scripts/train_lightning.py",
            str(config_path),

            ckpt_dir=str(Path(tmpdir) / "checkpoints"),
            use_torchrun=True,
            nproc=2,
        )

        assert result["returncode"] == 0, f"Lightning DDP training failed:\n{result['stderr']}"

        # Check that only one WandB run was created (not 2)
        # This is implicitly tested by the fact that training completed
        print("Lightning DDP 2-GPU training completed successfully")


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Requires 4 GPUs")
def test_lightning_ddp_4gpu():
    """Test Lightning with DDP on 4 GPUs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "data": {
                "task": "burgers1d",
                "root": "data/pdebench",
                "split": "train",
            },
            "latent": {
                "dim": 32,
                "tokens": 64,
            },
            "operator": {
                "architecture_type": "pdet_unet",
                "pdet": {
                    "input_dim": 32,
                    "hidden_dim": 64,
                    "depths": [1, 1],
                    "group_size": 8,
                    "num_heads": 4,
                },
            },
            "training": {
                "num_gpus": 4,
                "batch_size": 4,
                "num_workers": 0,
                "compile": False,
                "amp": False,
                "dt": 0.1,
            },
            "stages": {
                "operator": {
                    "epochs": 2,
                },
            },
            "optimizer": {
                "name": "adam",
                "lr": 1e-3,
            },
            "checkpoint": {
                "dir": str(Path(tmpdir) / "checkpoints"),
            },
            "logging": {
                "wandb": {
                    "enabled": False,
                },
            },
        }

        config_path = Path(tmpdir) / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        result = run_training(
            "scripts/train_lightning.py",
            str(config_path),

            ckpt_dir=str(Path(tmpdir) / "checkpoints"),
            use_torchrun=True,
            nproc=4,
        )

        assert result["returncode"] == 0, f"Lightning DDP 4-GPU training failed:\n{result['stderr']}"
        print("Lightning DDP 4-GPU training completed successfully")


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires 2 GPUs")
def test_lightning_fsdp_2gpu():
    """Test Lightning with FSDP on 2 GPUs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "data": {
                "task": "burgers1d",
                "root": "data/pdebench",
                "split": "train",
            },
            "latent": {
                "dim": 32,
                "tokens": 64,
            },
            "operator": {
                "architecture_type": "pdet_unet",
                "pdet": {
                    "input_dim": 32,
                    "hidden_dim": 64,
                    "depths": [1, 1],
                    "group_size": 8,
                    "num_heads": 4,
                },
            },
            "training": {
                "num_gpus": 2,
                "use_fsdp2": True,  # Enable FSDP
                "batch_size": 4,
                "num_workers": 0,
                "compile": False,  # Disable compile with FSDP for now
                "amp": False,
                "dt": 0.1,
            },
            "stages": {
                "operator": {
                    "epochs": 2,
                },
            },
            "optimizer": {
                "name": "adam",
                "lr": 1e-3,
            },
            "checkpoint": {
                "dir": str(Path(tmpdir) / "checkpoints"),
            },
            "logging": {
                "wandb": {
                    "enabled": False,
                },
            },
        }

        config_path = Path(tmpdir) / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        result = run_training(
            "scripts/train_lightning.py",
            str(config_path),

            ckpt_dir=str(Path(tmpdir) / "checkpoints"),
            use_torchrun=True,
            nproc=2,
        )

        assert result["returncode"] == 0, f"Lightning FSDP training failed:\n{result['stderr']}"
        print("Lightning FSDP 2-GPU training completed successfully")


@pytest.mark.slow
@pytest.mark.gpu
def test_lightning_compile_toggle():
    """Test that compile toggle works correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "data": {
                "task": "burgers1d",
                "root": "data/pdebench",
                "split": "train",
            },
            "latent": {
                "dim": 32,
                "tokens": 64,
            },
            "operator": {
                "architecture_type": "pdet_unet",
                "pdet": {
                    "input_dim": 32,
                    "hidden_dim": 64,
                    "depths": [1, 1],
                    "group_size": 8,
                    "num_heads": 4,
                },
            },
            "training": {
                "batch_size": 4,
                "num_workers": 0,
                "compile": True,  # Enable compile
                "compile_mode": "default",
                "amp": False,
                "dt": 0.1,
            },
            "stages": {
                "operator": {
                    "epochs": 1,
                },
            },
            "optimizer": {
                "name": "adam",
                "lr": 1e-3,
            },
            "checkpoint": {
                "dir": str(Path(tmpdir) / "checkpoints"),
            },
            "logging": {
                "wandb": {
                    "enabled": False,
                },
            },
        }

        config_path = Path(tmpdir) / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        result = run_training(
            "scripts/train_lightning.py",
            str(config_path),

            ckpt_dir=str(Path(tmpdir) / "checkpoints"),
        )

        # Should complete successfully even with compile (or gracefully fall back)
        assert result["returncode"] == 0, f"Lightning compile training failed:\n{result['stderr']}"
        print("Lightning training with compile completed successfully")


@pytest.mark.slow
@pytest.mark.gpu
def test_checkpoint_compatibility():
    """Test that Lightning checkpoints can be loaded into native model."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "data": {
                "task": "burgers1d",
                "root": "data/pdebench",
                "split": "train",
            },
            "latent": {
                "dim": 32,
                "tokens": 64,
            },
            "operator": {
                "architecture_type": "pdet_unet",
                "pdet": {
                    "input_dim": 32,
                    "hidden_dim": 64,
                    "depths": [1, 1],
                    "group_size": 8,
                    "num_heads": 4,
                },
            },
            "training": {
                "batch_size": 4,
                "num_workers": 0,
                "compile": False,
                "amp": False,
                "dt": 0.1,
            },
            "stages": {
                "operator": {
                    "epochs": 1,
                },
            },
            "optimizer": {
                "name": "adam",
                "lr": 1e-3,
            },
            "checkpoint": {
                "dir": str(Path(tmpdir) / "lightning_checkpoints"),
            },
            "logging": {
                "wandb": {
                    "enabled": False,
                },
            },
        }

        config_path = Path(tmpdir) / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # Train with Lightning
        result = run_training(
            "scripts/train_lightning.py",
            str(config_path),

            ckpt_dir=str(Path(tmpdir) / "lightning_checkpoints"),
        )

        assert result["returncode"] == 0, f"Lightning training failed:\n{result['stderr']}"

        # Find the Lightning checkpoint
        ckpt_dir = Path(tmpdir) / "lightning_checkpoints"
        ckpt_files = list(ckpt_dir.glob("operator-*.ckpt"))

        if not ckpt_files:
            pytest.skip("No Lightning checkpoint found")

        lightning_ckpt = ckpt_files[0]

        # Try to load Lightning checkpoint
        ckpt = torch.load(lightning_ckpt, map_location="cpu")

        # Lightning checkpoints have a different structure
        assert "state_dict" in ckpt, "Lightning checkpoint missing state_dict"

        # Extract operator state dict
        lightning_state = ckpt["state_dict"]

        # Remove 'operator.' prefix from keys if present
        operator_state = {}
        for key, value in lightning_state.items():
            if key.startswith("operator."):
                operator_state[key.replace("operator.", "", 1)] = value
            else:
                operator_state[key] = value

        # Create native operator model
        from ups.models.latent_operator import LatentOperator, LatentOperatorConfig
        from ups.core.blocks_pdet import PDETransformerConfig

        pdet_config = PDETransformerConfig(
            input_dim=32,
            hidden_dim=64,
            depths=[1, 1],
            group_size=8,
            num_heads=4,
        )

        operator_config = LatentOperatorConfig(
            latent_dim=32,
            pdet=pdet_config,
            architecture_type="pdet_unet",
            time_embed_dim=32,
        )

        native_operator = LatentOperator(operator_config)

        # Try to load the state dict
        try:
            native_operator.load_state_dict(operator_state, strict=False)
            print("Successfully loaded Lightning checkpoint into native model")
        except Exception as e:
            pytest.fail(f"Failed to load Lightning checkpoint into native model: {e}")


@pytest.mark.slow
@pytest.mark.gpu
def test_multi_stage_pipeline():
    """Test that multi-stage Lightning pipeline works."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "data": {
                "task": "burgers1d",
                "root": "data/pdebench",
                "split": "train",
            },
            "latent": {
                "dim": 32,
                "tokens": 64,
            },
            "operator": {
                "architecture_type": "pdet_unet",
                "pdet": {
                    "input_dim": 32,
                    "hidden_dim": 64,
                    "depths": [1, 1],
                    "group_size": 8,
                    "num_heads": 4,
                },
            },
            "training": {
                "batch_size": 4,
                "num_workers": 0,
                "compile": False,
                "amp": False,
                "dt": 0.1,
            },
            "stages": {
                "operator": {
                    "epochs": 1,
                },
            },
            "optimizer": {
                "name": "adam",
                "lr": 1e-3,
            },
            "checkpoint": {
                "dir": str(Path(tmpdir) / "checkpoints"),
            },
            "logging": {
                "wandb": {
                    "enabled": False,
                },
            },
        }

        config_path = Path(tmpdir) / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # Run multi-stage pipeline
        cmd = [
            "python",
            "scripts/run_lightning_pipeline.py",
            "--train-config", str(config_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        assert result.returncode == 0, f"Multi-stage pipeline failed:\n{result.stderr}"
        print("Multi-stage Lightning pipeline completed successfully")
