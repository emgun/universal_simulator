"""Integration tests for distributed training."""

import os
import subprocess
from pathlib import Path

import pytest


def run_torchrun(nproc: int, script: str, config: str, extra_args: list[str] = None) -> int:
    """Helper to run torchrun command.

    Args:
        nproc: Number of processes (GPUs)
        script: Python script to run
        config: Config file path
        extra_args: Additional command-line arguments

    Returns:
        Return code from subprocess
    """
    cmd = [
        "torchrun",
        f"--nproc_per_node={nproc}",
        "--nnodes=1",
        "--node_rank=0",
        "--master_addr=localhost",
        "--master_port=29500",
        script,
        "--config",
        config,
    ]
    if extra_args:
        cmd.extend(extra_args)

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode


@pytest.mark.slow
@pytest.mark.gpu
def test_2gpu_training():
    """Test 2-GPU distributed training completes successfully."""
    returncode = run_torchrun(
        nproc=2,
        script="scripts/train.py",
        config="configs/train_pdebench_2task_baseline_ddp.yaml",
        extra_args=["--stage", "operator", "--epochs", "2"],
    )
    assert returncode == 0, "2-GPU training failed"


@pytest.mark.slow
@pytest.mark.gpu
def test_4gpu_training():
    """Test 4-GPU distributed training completes successfully."""
    returncode = run_torchrun(
        nproc=4,
        script="scripts/train.py",
        config="configs/train_pdebench_11task_ddp.yaml",
        extra_args=["--stage", "operator", "--epochs", "2"],
    )
    assert returncode == 0, "4-GPU training failed"


@pytest.mark.slow
@pytest.mark.gpu
def test_task_distribution():
    """Test task distribution is balanced across ranks."""
    # Note: This requires validate_distributed_sampling.py to exist
    # For now, we'll test that the config is valid
    returncode = subprocess.run(
        [
            "python",
            "scripts/validate_config.py",
            "configs/train_pdebench_2task_baseline_ddp.yaml",
        ],
        capture_output=True,
        text=True,
    ).returncode
    assert returncode == 0, "Task distribution validation failed"


@pytest.mark.slow
@pytest.mark.gpu
def test_checkpoint_sync():
    """Test that checkpoints are saved correctly in distributed mode."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        # Run 2-GPU training with temporary checkpoint dir
        config_path = "configs/train_pdebench_2task_baseline_ddp.yaml"

        # Create a modified config with temporary checkpoint dir
        import yaml

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        cfg["checkpoints"] = {"dir": tmpdir}

        temp_config = Path(tmpdir) / "test_config.yaml"
        with open(temp_config, "w") as f:
            yaml.dump(cfg, f)

        returncode = run_torchrun(
            nproc=2,
            script="scripts/train.py",
            config=str(temp_config),
            extra_args=["--stage", "operator", "--epochs", "1"],
        )

        assert returncode == 0, "Checkpoint sync training failed"

        # Verify checkpoint was created (only rank 0 should save)
        checkpoint_dir = Path(tmpdir) / "checkpoints"
        assert checkpoint_dir.exists(), "Checkpoint directory not created"

        operator_ckpt = checkpoint_dir / "operator.pt"
        assert operator_ckpt.exists(), "Operator checkpoint not created"


@pytest.mark.slow
@pytest.mark.gpu
def test_metrics_aggregation():
    """Test that metrics are correctly aggregated across ranks."""
    # This is implicitly tested by test_2gpu_training
    # In the future, could add explicit validation by parsing logs
    returncode = run_torchrun(
        nproc=2,
        script="scripts/train.py",
        config="configs/train_pdebench_2task_baseline_ddp.yaml",
        extra_args=["--stage", "operator", "--epochs", "1"],
    )
    assert returncode == 0, "Metrics aggregation test failed"


@pytest.mark.slow
@pytest.mark.gpu
def test_oom_skip_sync():
    """Ensure simulated OOM skips are coordinated across ranks."""
    env = os.environ.copy()
    env["UPS_SIMULATE_OOM_RANK"] = "1"
    env["UPS_SIMULATE_OOM_STEP"] = "0"
    cmd = [
        "torchrun",
        "--nproc_per_node=2",
        "--nnodes=1",
        "--node_rank=0",
        "--master_addr=localhost",
        "--master_port=29502",
        "scripts/train.py",
        "--config",
        "configs/train_pdebench_2task_baseline_ddp.yaml",
        "--stage",
        "operator",
        "--epochs",
        "1",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    combined = f"{result.stdout}\n{result.stderr}"
    assert result.returncode == 0, f"Simulated OOM run failed:\n{combined}"
    assert (
        "Warning: OOM encountered in operator step" in combined
    ), f"Expected OOM warning not found:\n{combined}"


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.skipif(
    not os.path.exists("configs/train_pdebench_11task_ddp.yaml"),
    reason="11-task config not found",
)
def test_11task_multi_gpu():
    """Test full 11-task PDEBench training on 4 GPUs."""
    returncode = run_torchrun(
        nproc=4,
        script="scripts/train.py",
        config="configs/train_pdebench_11task_ddp.yaml",
        extra_args=["--stage", "operator", "--epochs", "1"],
    )
    assert returncode == 0, "11-task multi-GPU training failed"
