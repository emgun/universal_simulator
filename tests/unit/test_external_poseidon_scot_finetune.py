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
    CHANNEL_LIFT_ADAPTER_MODE,
    SCALAR_ADAPTER_MODE,
    ChannelLiftScOT,
    collect_poseidon_training_sequences,
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


class TinyMultiChannelPoseidon(nn.Module):
    """Fake 4-channel ScOT backbone with the same call signature."""

    def __init__(self, channels: int = 4) -> None:
        super().__init__()
        self.channels = channels
        self.encoder = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, *, pixel_values: torch.Tensor, time: torch.Tensor) -> TinyOutput:
        assert pixel_values.shape[1] == self.channels
        assert time.shape == (pixel_values.shape[0],)
        return TinyOutput(self.encoder(pixel_values))


def test_channel_lift_replicate_init_matches_pretrained_channel_mean():
    backbone = TinyMultiChannelPoseidon()
    wrapper = ChannelLiftScOT(backbone, backbone_channels=4)
    pixels = torch.randn(3, 1, 8, 8)
    time = torch.ones(3)

    wrapped = wrapper(pixel_values=pixels, time=time)
    direct = backbone(pixel_values=pixels.expand(-1, 4, -1, -1), time=time).output.mean(
        dim=1, keepdim=True
    )

    assert torch.allclose(wrapped, direct, atol=1e-6)


def test_channel_lift_mode_trains_only_lift_and_readout():
    wrapper = ChannelLiftScOT(TinyMultiChannelPoseidon(), backbone_channels=4)

    info = configure_trainable_poseidon_parameters(wrapper, adapter_mode=CHANNEL_LIFT_ADAPTER_MODE)

    trainable_names = {
        name for name, parameter in wrapper.named_parameters() if parameter.requires_grad
    }
    assert trainable_names == {
        "lift.weight",
        "lift.bias",
        "readout.weight",
        "readout.bias",
    }
    # 1x1 lift (4 weights + 4 biases) + 1x1 readout (4 weights + 1 bias).
    assert info["trainable_parameter_count"] == 13
    assert info["adapter_mode"] == CHANNEL_LIFT_ADAPTER_MODE


def test_channel_lift_backward_leaves_backbone_without_grads():
    wrapper = ChannelLiftScOT(TinyMultiChannelPoseidon(), backbone_channels=4)
    configure_trainable_poseidon_parameters(wrapper, adapter_mode=CHANNEL_LIFT_ADAPTER_MODE)

    out = wrapper(pixel_values=torch.randn(2, 1, 8, 8), time=torch.ones(2))
    out.sum().backward()

    for name, parameter in wrapper.named_parameters():
        if name.startswith(("lift.", "readout.")):
            assert parameter.grad is not None, name
        else:
            assert parameter.grad is None, name


def test_collect_poseidon_training_sequences_windows(tmp_path):
    data_root = tmp_path / "data"
    _write_h5(data_root / "advection1d_train.h5", (2, 5, 16, 1))

    sequences = collect_poseidon_training_sequences(
        {"data": {"root": str(data_root)}},
        tasks=["advection1d"],
        split="train",
        data_root=str(data_root),
        max_train_samples=2,
        sequence_steps=2,
        image_size=8,
    )

    assert sequences is not None
    # 5 steps -> windows of 3 frames with stride 2 -> 2 windows per sample.
    assert sequences.shape == (4, 3, 1, 8, 8)
    assert (
        collect_poseidon_training_sequences(
            {"data": {"root": str(data_root)}},
            tasks=["advection1d"],
            split="train",
            data_root=str(data_root),
            max_train_samples=2,
            sequence_steps=1,
            image_size=8,
        )
        is None
    )


def test_train_with_rollout_loss_records_sequences(tmp_path):
    data_root = tmp_path / "data"
    _write_h5(data_root / "advection1d_train.h5", (2, 5, 16, 1))
    wrapper = ChannelLiftScOT(TinyMultiChannelPoseidon(), backbone_channels=4)
    configure_trainable_poseidon_parameters(wrapper, adapter_mode=CHANNEL_LIFT_ADAPTER_MODE)

    info = train_poseidon_scot_adapter(
        {"data": {"root": str(data_root)}},
        wrapper,
        tasks=["advection1d"],
        split="train",
        data_root=str(data_root),
        max_train_samples=2,
        rollout_steps=2,
        image_size=8,
        time_value=1.0,
        epochs=1,
        learning_rate=0.01,
        weight_decay=0.0,
        batch_size=2,
        rollout_loss_steps=2,
        rollout_loss_weight=1.0,
        seed=7,
        device="cpu",
    )

    assert info["rollout_loss_steps"] == 2
    assert info["rollout_sequences"] == 4
    assert np.isfinite(info["best_train_mse"])


def test_channel_lift_summary_requires_intact_embedding_recovery():
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
            "pretrained_checkpoint": {"sha256": "abc"},
            "adapter_mode": CHANNEL_LIFT_ADAPTER_MODE,
            "trainable_parameters": {"trainable_parameter_count": 13},
            "model": {"embedding_recovery_replaced": True},
        },
    }

    errors = validate_poseidon_finetune_summary(summary)
    assert any("embedding_recovery_replaced" in error for error in errors)

    summary["details"]["model"]["embedding_recovery_replaced"] = False
    assert validate_poseidon_finetune_summary(summary) == []
