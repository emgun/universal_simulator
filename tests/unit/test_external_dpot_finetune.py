from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch import nn

from scripts.run_external_dpot_finetune import (
    CHANNEL_LIFT_ADAPTER_MODE,
    ChannelLiftDPOT,
    append_prediction_to_history,
    build_repeat_current_history,
    configure_trainable_dpot_parameters,
    train_dpot_adapter,
    validate_dpot_finetune_summary,
)

ROOT = Path(__file__).resolve().parents[2]


class TinyDPOT(nn.Module):
    """Fake DPOT backbone with the real tensor contract."""

    def __init__(self, channels: int = 4) -> None:
        super().__init__()
        self.channels = int(channels)
        self.mixer = nn.Conv2d(self.channels, self.channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert x.dim() == 5
        assert x.shape[-1] == self.channels
        last = x[:, :, :, -1, :].permute(0, 3, 1, 2).contiguous()
        mixed = self.mixer(last)
        output = mixed.permute(0, 2, 3, 1).unsqueeze(3).contiguous()
        cls = torch.zeros(x.shape[0], 1, dtype=x.dtype, device=x.device)
        return output, cls


class NonFiniteDPOT(TinyDPOT):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output, cls = super().forward(x)
        return output * torch.tensor(float("inf"), dtype=output.dtype), cls


def _write_h5(path: Path, shape: tuple[int, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.linspace(0.0, 1.0, num=int(np.prod(shape)), dtype=np.float32).reshape(shape)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=values)


def test_channel_lift_dpot_replicate_init_matches_backbone_channel_mean():
    backbone = TinyDPOT()
    wrapper = ChannelLiftDPOT(backbone, history_steps=10, backbone_channels=4)
    pixels = torch.randn(3, 1, 8, 8)
    history = build_repeat_current_history(pixels, history_steps=10)

    wrapped = wrapper(history)
    dpot_input = history.expand(-1, -1, 4, -1, -1).permute(0, 3, 4, 1, 2).contiguous()
    direct = backbone(dpot_input)[0][:, :, :, -1, :].permute(0, 3, 1, 2).mean(
        dim=1,
        keepdim=True,
    )

    assert torch.allclose(wrapped, direct, atol=1e-6)


def test_configure_trainable_dpot_parameters_keeps_backbone_frozen():
    wrapper = ChannelLiftDPOT(TinyDPOT(), history_steps=10, backbone_channels=4)

    info = configure_trainable_dpot_parameters(
        wrapper,
        adapter_mode=CHANNEL_LIFT_ADAPTER_MODE,
    )

    trainable_names = {
        name for name, parameter in wrapper.named_parameters() if parameter.requires_grad
    }
    assert trainable_names == {
        "lift.weight",
        "lift.bias",
        "readout.weight",
        "readout.bias",
    }
    assert info["trainable_parameter_count"] == 13
    assert info["adapter_mode"] == CHANNEL_LIFT_ADAPTER_MODE
    assert "backbone.mixer.weight" in info["frozen_parameter_names_sample"]


def test_channel_lift_backward_leaves_backbone_without_grads():
    wrapper = ChannelLiftDPOT(TinyDPOT(), history_steps=10, backbone_channels=4)
    configure_trainable_dpot_parameters(wrapper)
    history = build_repeat_current_history(torch.randn(2, 1, 8, 8), history_steps=10)

    wrapper(history).sum().backward()

    assert wrapper.lift.weight.grad is not None
    assert wrapper.readout.weight.grad is not None
    assert wrapper.backbone.mixer.weight.grad is None


def test_repeat_current_history_is_deterministic():
    pixels = torch.arange(18, dtype=torch.float32).reshape(1, 1, 3, 6)

    history = build_repeat_current_history(pixels, history_steps=4)

    assert history.shape == (1, 4, 1, 3, 6)
    for step in range(4):
        assert torch.equal(history[:, step], pixels)


def test_append_prediction_to_history_shifts_and_appends():
    history = torch.arange(1 * 4 * 1 * 2 * 2, dtype=torch.float32).reshape(1, 4, 1, 2, 2)
    prediction = torch.full((1, 1, 2, 2), 99.0)

    updated = append_prediction_to_history(history, prediction)

    assert torch.equal(updated[:, 0], history[:, 1])
    assert torch.equal(updated[:, 1], history[:, 2])
    assert torch.equal(updated[:, 2], history[:, 3])
    assert torch.equal(updated[:, 3], prediction)


def test_train_dpot_adapter_uses_train_split_pairs(tmp_path):
    data_root = tmp_path / "data"
    _write_h5(data_root / "advection1d_train.h5", (2, 4, 16, 1))
    wrapper = ChannelLiftDPOT(TinyDPOT(), history_steps=3, backbone_channels=4)
    configure_trainable_dpot_parameters(wrapper)

    info = train_dpot_adapter(
        {"data": {"root": str(data_root)}},
        wrapper,
        tasks=["advection1d"],
        split="train",
        data_root=str(data_root),
        max_train_samples=1,
        rollout_steps=2,
        image_size=8,
        history_steps=3,
        epochs=1,
        learning_rate=0.01,
        weight_decay=0.0,
        batch_size=2,
        grad_clip_norm=1.0,
        seed=7,
        device="cpu",
    )

    assert info["train_split"] == "train"
    assert info["train_pairs"] == 2
    assert info["training_records"][0]["pairs_collected"] == 2
    assert np.isfinite(info["best_train_mse"])


def test_train_dpot_adapter_rejects_nonfinite_training_loss(tmp_path):
    data_root = tmp_path / "data"
    _write_h5(data_root / "advection1d_train.h5", (2, 4, 16, 1))
    wrapper = ChannelLiftDPOT(NonFiniteDPOT(), history_steps=3, backbone_channels=4)
    configure_trainable_dpot_parameters(wrapper)

    with pytest.raises(RuntimeError, match="Non-finite DPOT finetune prediction"):
        train_dpot_adapter(
            {"data": {"root": str(data_root)}},
            wrapper,
            tasks=["advection1d"],
            split="train",
            data_root=str(data_root),
            max_train_samples=1,
            rollout_steps=1,
            image_size=8,
            history_steps=3,
            epochs=1,
            learning_rate=0.01,
            weight_decay=0.0,
            batch_size=1,
            grad_clip_norm=1.0,
            seed=7,
            device="cpu",
        )


def test_dpot_finetune_summary_requires_checkpoint_hash():
    summary = {
        "schema_version": 1,
        "status": "validation_finetune_measurement_complete",
        "measurement_type": "dpot_finetune_validation_measurement",
        "train_split": "train",
        "split": "val",
        "held_out_test_used": False,
        "held_out_test_data_read": False,
        "claim_comparable": False,
        "published_numbers_directly_comparable": False,
        "metrics": {"decoded_rollout_nrmse": 1.0},
        "details": {
            "dpot_source": {"commit": "abc"},
            "pretrained_checkpoint": {"sha256": ""},
            "adapter_mode": CHANNEL_LIFT_ADAPTER_MODE,
            "history_steps": 10,
            "history_init": "repeat_current",
            "trainable_parameters": {"trainable_parameter_count": 13},
        },
    }

    errors = validate_dpot_finetune_summary(summary)

    assert "details.pretrained_checkpoint.sha256 is required" in errors


def test_dpot_finetune_summary_rejects_validation_marked_as_held_out():
    summary = {
        "schema_version": 1,
        "status": "validation_finetune_measurement_complete",
        "measurement_type": "dpot_finetune_validation_measurement",
        "train_split": "train",
        "split": "val",
        "held_out_test_used": True,
        "held_out_test_data_read": True,
        "claim_comparable": False,
        "published_numbers_directly_comparable": False,
        "metrics": {"decoded_rollout_nrmse": 1.0},
        "details": {
            "dpot_source": {"commit": "abc"},
            "pretrained_checkpoint": {"sha256": "abc"},
            "adapter_mode": CHANNEL_LIFT_ADAPTER_MODE,
            "history_steps": 10,
            "history_init": "repeat_current",
            "trainable_parameters": {"trainable_parameter_count": 13},
        },
    }

    errors = validate_dpot_finetune_summary(summary)

    assert "validation summaries must mark held_out_test_used false" in errors
    assert "validation summaries must mark held_out_test_data_read false" in errors


def test_dpot_finetune_cli_blocks_test_split_before_loading_repo(tmp_path):
    output_root = tmp_path / "out"

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_external_dpot_finetune.py",
            "--config",
            "configs/train_multitask_heterogeneous_light_best.yaml",
            "--name",
            "blocked_dpot_test",
            "--output-root",
            str(output_root),
            "--eval-split",
            "test",
            "--dpot-repo",
            str(tmp_path / "missing-dpot"),
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
    assert "does not allow split=test" in proc.stderr
    assert not output_root.exists()
