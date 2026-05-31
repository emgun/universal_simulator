from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch import nn

from scripts.run_external_poseidon_scot_finetune import (
    SCALAR_ADAPTER_MODE,
    configure_trainable_poseidon_parameters,
    train_poseidon_scot_adapter,
    validate_poseidon_finetune_summary,
)

ROOT = Path(__file__).resolve().parents[2]


class TinyOutput:
    def __init__(self, output: torch.Tensor) -> None:
        self.output = output


class TinyPoseidon(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embeddings = nn.Module()
        self.embeddings.patch_embeddings = nn.Module()
        self.embeddings.patch_embeddings.projection = nn.Conv2d(1, 2, kernel_size=1)
        self.encoder = nn.Conv2d(2, 2, kernel_size=1)
        self.patch_recovery = nn.Module()
        self.patch_recovery.projection = nn.Conv2d(2, 1, kernel_size=1)
        self.patch_recovery.mixup = nn.Conv2d(1, 1, kernel_size=1)

    def forward(self, *, pixel_values: torch.Tensor, time: torch.Tensor) -> TinyOutput:
        assert time.shape == (pixel_values.shape[0],)
        hidden = self.embeddings.patch_embeddings.projection(pixel_values)
        hidden = self.encoder(hidden)
        output = self.patch_recovery.projection(hidden)
        output = self.patch_recovery.mixup(output)
        return TinyOutput(output)


class NonFinitePoseidon(TinyPoseidon):
    def forward(self, *, pixel_values: torch.Tensor, time: torch.Tensor) -> TinyOutput:
        output = super().forward(pixel_values=pixel_values, time=time).output
        return TinyOutput(output * torch.tensor(float("inf"), dtype=output.dtype))


def _write_h5(path: Path, shape: tuple[int, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.linspace(0.0, 1.0, num=int(np.prod(shape)), dtype=np.float32).reshape(shape)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=values)


def test_configure_trainable_poseidon_parameters_keeps_backbone_frozen():
    model = TinyPoseidon()

    info = configure_trainable_poseidon_parameters(model, adapter_mode=SCALAR_ADAPTER_MODE)

    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert "embeddings.patch_embeddings.projection.weight" in trainable_names
    assert "patch_recovery.projection.weight" in trainable_names
    assert "patch_recovery.mixup.weight" in trainable_names
    assert "encoder.weight" not in trainable_names
    assert info["trainable_parameter_count"] > 0
    assert info["frozen_parameter_count"] > 0


def test_train_poseidon_scot_adapter_uses_train_split_pairs(tmp_path):
    data_root = tmp_path / "data"
    _write_h5(data_root / "advection1d_train.h5", (2, 4, 16, 1))
    model = TinyPoseidon()
    configure_trainable_poseidon_parameters(model)

    info = train_poseidon_scot_adapter(
        {"data": {"root": str(data_root)}},
        model,
        tasks=["advection1d"],
        split="train",
        data_root=str(data_root),
        max_train_samples=1,
        rollout_steps=2,
        image_size=8,
        time_value=1.0,
        epochs=1,
        learning_rate=0.01,
        weight_decay=0.0,
        batch_size=2,
        seed=7,
        device="cpu",
    )

    assert info["train_split"] == "train"
    assert info["train_pairs"] == 2
    assert info["training_records"][0]["pairs_collected"] == 2
    assert np.isfinite(info["best_train_mse"])


def test_train_poseidon_scot_adapter_rejects_nonfinite_training_loss(tmp_path):
    data_root = tmp_path / "data"
    _write_h5(data_root / "advection1d_train.h5", (2, 4, 16, 1))
    model = NonFinitePoseidon()
    configure_trainable_poseidon_parameters(model)

    with pytest.raises(RuntimeError, match="Non-finite Poseidon finetune prediction"):
        train_poseidon_scot_adapter(
            {"data": {"root": str(data_root)}},
            model,
            tasks=["advection1d"],
            split="train",
            data_root=str(data_root),
            max_train_samples=1,
            rollout_steps=1,
            image_size=8,
            time_value=1.0,
            epochs=1,
            learning_rate=0.01,
            weight_decay=0.0,
            batch_size=1,
            seed=7,
            device="cpu",
        )


def test_poseidon_finetune_summary_requires_checkpoint_hash():
    summary = {
        "schema_version": 1,
        "status": "validation_finetune_measurement_complete",
        "measurement_type": "poseidon_scot_finetune_validation_measurement",
        "train_split": "train",
        "split": "val",
        "held_out_test_used": False,
        "claim_comparable": False,
        "published_numbers_directly_comparable": False,
        "metrics": {"decoded_rollout_nrmse": 1.0},
        "details": {
            "pretrained_checkpoint": {"sha256": ""},
            "adapter_mode": SCALAR_ADAPTER_MODE,
            "trainable_parameters": {"trainable_parameter_count": 1},
        },
    }

    errors = validate_poseidon_finetune_summary(summary)

    assert "details.pretrained_checkpoint.sha256 is required" in errors


def test_poseidon_finetune_summary_rejects_validation_marked_as_held_out():
    summary = {
        "schema_version": 1,
        "status": "validation_finetune_measurement_complete",
        "measurement_type": "poseidon_scot_finetune_validation_measurement",
        "train_split": "train",
        "split": "val",
        "held_out_test_used": True,
        "claim_comparable": False,
        "published_numbers_directly_comparable": False,
        "metrics": {"decoded_rollout_nrmse": 1.0},
        "details": {
            "pretrained_checkpoint": {"sha256": "abc"},
            "adapter_mode": SCALAR_ADAPTER_MODE,
            "trainable_parameters": {"trainable_parameter_count": 1},
        },
    }

    errors = validate_poseidon_finetune_summary(summary)

    assert "validation summaries must mark held_out_test_used false" in errors


def test_poseidon_finetune_cli_blocks_test_split_before_loading_data(tmp_path):
    output_root = tmp_path / "out"

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_external_poseidon_scot_finetune.py",
            "--config",
            "configs/train_multitask_heterogeneous_light_best.yaml",
            "--name",
            "blocked_poseidon_finetune_test",
            "--output-root",
            str(output_root),
            "--eval-split",
            "test",
            "--data-root",
            str(tmp_path / "missing-data"),
            "--tasks",
            "advection1d",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode != 0
    assert "--allow-held-out-test-eval" in proc.stderr
    assert not output_root.exists()
