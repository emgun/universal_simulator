"""Unit tests for WandBContext module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from ups.utils.wandb_context import (
    WandBContext,
    create_wandb_context,
    load_wandb_context,
    load_wandb_context_from_env,
    load_wandb_context_from_file,
    save_wandb_context,
)


@pytest.fixture
def mock_wandb():
    """Mock wandb module for testing."""
    with patch("ups.utils.wandb_context.wandb") as mock_wandb:
        mock_wandb.Table = MagicMock(return_value=MagicMock())
        mock_wandb.Image = MagicMock(return_value=MagicMock())
        mock_wandb.AlertLevel = MagicMock()
        mock_wandb.AlertLevel.INFO = "INFO"
        mock_wandb.AlertLevel.WARN = "WARN"
        mock_wandb.AlertLevel.ERROR = "ERROR"
        yield mock_wandb


@pytest.fixture
def mock_wandb_run():
    """Mock WandB run object."""
    run = MagicMock()
    run.id = "test-run-id"
    run.project = "test-project"
    run.entity = "test-entity"
    run.url = "https://wandb.ai/test-entity/test-project/runs/test-run-id"
    run.tags = ["tag1", "tag2"]
    run.config = MagicMock()
    run.summary = {}
    return run


def test_wandb_context_disabled():
    """Test WandBContext in disabled mode."""
    ctx = WandBContext(run=None, run_id="test-id", enabled=False)

    # All methods should be no-ops
    ctx.log_training_metric("operator", "loss", 0.5, step=10)
    ctx.log_eval_summary({"nrmse": 0.1}, prefix="eval")
    ctx.log_table("test_table", ["col1"], [["val1"]])
    ctx.log_image("test_img", Path("test.png"))
    ctx.save_file(Path("test.txt"))
    ctx.update_config({"key": "value"})
    ctx.add_tags(["new_tag"])
    ctx.finish()

    # No errors should be raised
    assert not ctx.enabled


def test_log_training_metric(mock_wandb_run):
    """Test logging training metrics as time series."""
    ctx = WandBContext(run=mock_wandb_run, run_id="test-id", enabled=True)

    ctx.log_training_metric("operator", "loss", 0.5, step=10)

    mock_wandb_run.log.assert_called_once_with(
        {"training/operator/loss": 0.5}, step=10
    )


def test_log_eval_summary(mock_wandb_run):
    """Test logging evaluation metrics as summary (not time series)."""
    ctx = WandBContext(run=mock_wandb_run, run_id="test-id", enabled=True)

    metrics = {"nrmse": 0.09, "mse": 0.001}
    ctx.log_eval_summary(metrics, prefix="eval")

    assert mock_wandb_run.summary["eval/nrmse"] == 0.09
    assert mock_wandb_run.summary["eval/mse"] == 0.001


def test_log_eval_summary_filters_non_numeric(mock_wandb_run):
    """Test that non-numeric values are filtered out."""
    ctx = WandBContext(run=mock_wandb_run, run_id="test-id", enabled=True)

    metrics = {"nrmse": 0.09, "invalid": "text", "also_invalid": None}
    ctx.log_eval_summary(metrics, prefix="eval")

    assert "eval/nrmse" in mock_wandb_run.summary
    assert "eval/invalid" not in mock_wandb_run.summary
    assert "eval/also_invalid" not in mock_wandb_run.summary


def test_log_table(mock_wandb, mock_wandb_run):
    """Test logging tables for multi-value comparisons."""
    ctx = WandBContext(run=mock_wandb_run, run_id="test-id", enabled=True)

    columns = ["Metric", "Baseline", "TTC"]
    data = [["NRMSE", 0.78, 0.09], ["MSE", 0.99, 0.01]]

    ctx.log_table("eval/comparison", columns, data)

    mock_wandb.Table.assert_called_once_with(columns=columns, data=data)
    mock_wandb_run.log.assert_called_once()


def test_log_image(mock_wandb, mock_wandb_run, tmp_path):
    """Test logging images."""
    ctx = WandBContext(run=mock_wandb_run, run_id="test-id", enabled=True)

    img_path = tmp_path / "test.png"
    img_path.touch()

    ctx.log_image("eval/mse_histogram", img_path)

    mock_wandb.Image.assert_called_once_with(str(img_path))
    mock_wandb_run.log.assert_called_once()


def test_save_file(mock_wandb, mock_wandb_run, tmp_path):
    """Test saving files to WandB."""
    ctx = WandBContext(run=mock_wandb_run, run_id="test-id", enabled=True)

    file_path = tmp_path / "checkpoints" / "operator.pt"
    file_path.parent.mkdir(parents=True)
    file_path.touch()

    ctx.save_file(file_path)

    mock_wandb.save.assert_called_once()


def test_update_config(mock_wandb_run):
    """Test updating run config."""
    ctx = WandBContext(run=mock_wandb_run, run_id="test-id", enabled=True)

    updates = {"eval_samples": 3072, "eval_tau": 0.5}
    ctx.update_config(updates)

    mock_wandb_run.config.update.assert_called_once_with(
        updates, allow_val_change=True
    )


def test_add_tags(mock_wandb_run):
    """Test adding tags to run."""
    mock_wandb_run.tags = ("tag1", "tag2")
    ctx = WandBContext(run=mock_wandb_run, run_id="test-id", enabled=True)

    ctx.add_tags(["tag3", "tag1"])  # tag1 already exists

    # Should have unique tags
    assert set(mock_wandb_run.tags) == {"tag1", "tag2", "tag3"}


def test_finish(mock_wandb_run):
    """Test finishing WandB run."""
    ctx = WandBContext(run=mock_wandb_run, run_id="test-id", enabled=True)

    ctx.finish()

    mock_wandb_run.finish.assert_called_once()


def test_create_wandb_context_disabled():
    """Test creating WandB context in disabled mode."""
    config = {"data": {"task": "burgers"}, "latent": {"dim": 32, "tokens": 64}}

    with patch("ups.utils.wandb_context.WANDB_AVAILABLE", False):
        ctx = create_wandb_context(config, run_id="test-id", mode="online")

    assert ctx is not None
    assert not ctx.enabled
    assert ctx.run is None


def test_create_wandb_context_mode_disabled():
    """Test creating WandB context with mode='disabled'."""
    config = {"data": {"task": "burgers"}, "latent": {"dim": 32, "tokens": 64}}

    ctx = create_wandb_context(config, run_id="test-id", mode="disabled")

    assert ctx is not None
    assert not ctx.enabled
    assert ctx.run is None


def test_create_wandb_context_logging_disabled():
    """Test creating WandB context when logging.wandb.enabled=false."""
    config = {
        "data": {"task": "burgers"},
        "latent": {"dim": 32, "tokens": 64},
        "logging": {"wandb": {"enabled": False}},
    }

    ctx = create_wandb_context(config, run_id="test-id", mode="online")

    assert ctx is not None
    assert not ctx.enabled
    assert ctx.run is None


@patch("ups.utils.wandb_context.WANDB_AVAILABLE", True)
@patch("ups.utils.wandb_context.wandb")
def test_create_wandb_context_success(mock_wandb, mock_wandb_run):
    """Test successfully creating WandB context."""
    mock_wandb.init.return_value = mock_wandb_run

    config = {
        "data": {"task": "burgers"},
        "latent": {"dim": 32, "tokens": 64},
        "operator": {"pdet": {"hidden_dim": 256, "num_heads": 4, "depths": [2, 2]}},
        "training": {"batch_size": 8, "time_stride": 5},
        "stages": {
            "operator": {"epochs": 50, "optimizer": {"lr": 1e-4}},
            "diff_residual": {"epochs": 10, "optimizer": {"lr": 1e-5}},
        },
        "logging": {
            "wandb": {
                "enabled": True,
                "project": "test-project",
                "entity": "test-entity",
                "tags": ["test-tag"],
            }
        },
    }

    ctx = create_wandb_context(config, run_id="test-id", mode="online")

    assert ctx is not None
    assert ctx.enabled
    assert ctx.run == mock_wandb_run
    assert ctx.run_id == "test-id"

    # Verify wandb.init was called with correct params
    mock_wandb.init.assert_called_once()
    call_kwargs = mock_wandb.init.call_args[1]
    assert call_kwargs["project"] == "test-project"
    assert call_kwargs["entity"] == "test-entity"
    assert call_kwargs["id"] == "test-id"
    assert call_kwargs["mode"] == "online"
    assert call_kwargs["reinit"] is False


def test_save_wandb_context(mock_wandb_run, tmp_path):
    """Test saving WandB context to file."""
    ctx = WandBContext(run=mock_wandb_run, run_id="test-id", enabled=True)

    context_file = tmp_path / "wandb_context.json"
    save_wandb_context(ctx, context_file)

    assert context_file.exists()

    data = json.loads(context_file.read_text(encoding="utf-8"))
    assert data["run_id"] == "test-id"
    assert data["project"] == "test-project"
    assert data["entity"] == "test-entity"
    assert data["url"] == mock_wandb_run.url


def test_save_wandb_context_disabled(tmp_path):
    """Test saving disabled WandB context (should be no-op)."""
    ctx = WandBContext(run=None, run_id="test-id", enabled=False)

    context_file = tmp_path / "wandb_context.json"
    save_wandb_context(ctx, context_file)

    # File should not be created
    assert not context_file.exists()


@patch("ups.utils.wandb_context.WANDB_AVAILABLE", True)
@patch("ups.utils.wandb_context.wandb")
def test_load_wandb_context(mock_wandb, mock_wandb_run):
    """Test loading WandB context by run ID."""
    mock_wandb.init.return_value = mock_wandb_run

    ctx = load_wandb_context(
        run_id="test-id", project="test-project", entity="test-entity"
    )

    assert ctx is not None
    assert ctx.enabled
    assert ctx.run_id == "test-id"

    # Verify wandb.init was called with resume
    mock_wandb.init.assert_called_once()
    call_kwargs = mock_wandb.init.call_args[1]
    assert call_kwargs["id"] == "test-id"
    assert call_kwargs["project"] == "test-project"
    assert call_kwargs["entity"] == "test-entity"
    assert call_kwargs["resume"] == "allow"


@patch("ups.utils.wandb_context.WANDB_AVAILABLE", False)
def test_load_wandb_context_unavailable():
    """Test loading WandB context when WandB unavailable."""
    ctx = load_wandb_context(
        run_id="test-id", project="test-project", entity="test-entity"
    )

    assert ctx is not None
    assert not ctx.enabled


@patch("ups.utils.wandb_context.WANDB_AVAILABLE", True)
@patch("ups.utils.wandb_context.wandb")
def test_load_wandb_context_from_file(mock_wandb, mock_wandb_run, tmp_path):
    """Test loading WandB context from file."""
    mock_wandb.init.return_value = mock_wandb_run

    context_file = tmp_path / "wandb_context.json"
    context_data = {
        "run_id": "test-id",
        "project": "test-project",
        "entity": "test-entity",
        "url": "https://wandb.ai/test-entity/test-project/runs/test-id",
    }
    context_file.write_text(json.dumps(context_data), encoding="utf-8")

    ctx = load_wandb_context_from_file(context_file)

    assert ctx is not None
    assert ctx.enabled
    assert ctx.run_id == "test-id"


def test_load_wandb_context_from_file_not_exists(tmp_path):
    """Test loading WandB context from non-existent file."""
    context_file = tmp_path / "nonexistent.json"

    ctx = load_wandb_context_from_file(context_file)

    assert ctx is None


@patch("ups.utils.wandb_context.WANDB_AVAILABLE", True)
@patch("ups.utils.wandb_context.wandb")
def test_load_wandb_context_from_env(mock_wandb, mock_wandb_run, tmp_path, monkeypatch):
    """Test loading WandB context from environment variable."""
    mock_wandb.init.return_value = mock_wandb_run

    context_file = tmp_path / "wandb_context.json"
    context_data = {
        "run_id": "test-id",
        "project": "test-project",
        "entity": "test-entity",
        "url": "https://wandb.ai/test-entity/test-project/runs/test-id",
    }
    context_file.write_text(json.dumps(context_data), encoding="utf-8")

    monkeypatch.setenv("WANDB_CONTEXT_FILE", str(context_file))

    ctx = load_wandb_context_from_env()

    assert ctx is not None
    assert ctx.enabled
    assert ctx.run_id == "test-id"


def test_load_wandb_context_from_env_not_set(monkeypatch):
    """Test loading WandB context when env var not set."""
    monkeypatch.delenv("WANDB_CONTEXT_FILE", raising=False)

    ctx = load_wandb_context_from_env()

    assert ctx is None


def test_error_handling_graceful():
    """Test that errors in WandB calls are handled gracefully."""
    mock_run = MagicMock()
    mock_run.log.side_effect = Exception("WandB API error")
    mock_run.summary = {}

    ctx = WandBContext(run=mock_run, run_id="test-id", enabled=True)

    # Should not raise exceptions
    ctx.log_training_metric("operator", "loss", 0.5, step=10)
    ctx.log_eval_summary({"nrmse": 0.1}, prefix="eval")
