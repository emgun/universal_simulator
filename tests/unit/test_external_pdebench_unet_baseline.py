from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from scripts.run_external_pdebench_unet_baseline import (
    PDEBenchUNet1d,
    PDEBenchUNet2d,
    _external_test_measurement_key,
    build_parser,
    build_pdebench_unet_model,
    run_baseline,
    train_unet_group_model,
)

ROOT = Path(__file__).resolve().parents[2]


class TinyUNet1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, init_features: int) -> None:
        super().__init__()
        self.init_features = init_features
        self.net = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        nn.init.zeros_(self.net.weight)
        nn.init.zeros_(self.net.bias)

    def forward(self, current: torch.Tensor) -> torch.Tensor:
        return self.net(current)


class TinyUNet2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, init_features: int) -> None:
        super().__init__()
        self.init_features = init_features
        self.net = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        nn.init.zeros_(self.net.weight)
        nn.init.zeros_(self.net.bias)

    def forward(self, current: torch.Tensor) -> torch.Tensor:
        return self.net(current)


class TinyMetadataUNet1d(TinyUNet1d):
    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        state["_metadata"] = {"init_features": self.init_features}
        return state


def test_official_pdebench_unet_1d_preserves_width():
    model = PDEBenchUNet1d(in_channels=1, out_channels=1, init_features=2)
    model.eval()

    with torch.no_grad():
        pred = model(torch.randn(2, 1, 32))

    assert pred.shape == (2, 1, 32)


def test_official_pdebench_unet_2d_preserves_grid():
    model = PDEBenchUNet2d(in_channels=1, out_channels=1, init_features=2)
    model.eval()

    with torch.no_grad():
        pred = model(torch.randn(2, 1, 16, 16))

    assert pred.shape == (2, 1, 16, 16)


def test_build_pdebench_unet_model_adapts_1d_grid_to_repo_grid_shape():
    model = build_pdebench_unet_model(
        channels=1,
        grid_shape=(1, 32),
        init_features=4,
        residual=False,
        unet1d_cls=TinyUNet1d,
        unet2d_cls=TinyUNet2d,
    )

    pred = model(torch.randn(2, 1, 1, 32))

    assert pred.shape == (2, 1, 1, 32)
    assert model.model.init_features == 4


def test_build_pdebench_unet_model_rejects_non_poolable_grid():
    with pytest.raises(ValueError, match="divisible by 16"):
        build_pdebench_unet_model(
            channels=1,
            grid_shape=(1, 30),
            init_features=4,
            residual=False,
            unet1d_cls=TinyUNet1d,
            unet2d_cls=TinyUNet2d,
        )


def test_train_unet_group_model_can_learn_simple_residual_with_fake_unet():
    generator = torch.Generator().manual_seed(13)
    currents = torch.randn(16, 1, 1, 32, generator=generator)
    targets = currents + 0.25

    model, fit = train_unet_group_model(
        currents,
        targets,
        init_features=4,
        residual=True,
        epochs=80,
        learning_rate=0.05,
        batch_size=4,
        seed=7,
        unet1d_cls=TinyUNet1d,
        unet2d_cls=TinyUNet2d,
    )

    with torch.no_grad():
        mse = torch.mean((model(currents) - targets) ** 2).item()
    assert fit["implementation"] == "pdebench.models.unet.unet.UNet1d/UNet2d"
    assert fit["source_commit"] == "4ff3e3a4aa1561721b5571fa3a048a0a463e0568"
    assert mse < 0.01


def test_train_unet_group_model_ignores_state_metadata():
    generator = torch.Generator().manual_seed(23)
    currents = torch.randn(16, 1, 1, 32, generator=generator)
    targets = currents + 0.25

    model, fit = train_unet_group_model(
        currents,
        targets,
        init_features=4,
        residual=True,
        epochs=80,
        learning_rate=0.05,
        batch_size=4,
        seed=9,
        unet1d_cls=TinyMetadataUNet1d,
        unet2d_cls=TinyUNet2d,
    )

    with torch.no_grad():
        mse = torch.mean((model(currents) - targets) ** 2).item()
    assert fit["implementation"] == "pdebench.models.unet.unet.UNet1d/UNet2d"
    assert mse < 0.01


def test_external_pdebench_unet_dry_run_writes_contract_summary(tmp_path):
    args = build_parser().parse_args(
        [
            "--dry-run",
            "--config",
            "configs/train_multitask_heterogeneous_light_best.yaml",
            "--name",
            "external_pdebench_unet_dry_run",
            "--output-root",
            str(tmp_path),
            "--data-root",
            "data/pdebench",
            "--eval-split",
            "test",
            "--tasks",
            "advection1d",
            "burgers1d",
            "darcy2d",
        ]
    )

    summary_path = run_baseline(args)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["status"] == "dry_run"
    assert summary["metrics"] == {}
    assert summary["extra"]["baseline"] == "external_pdebench_unet"
    assert summary["extra"]["source_commit"] == "4ff3e3a4aa1561721b5571fa3a048a0a463e0568"
    assert summary["extra"]["split"] == "test"
    assert summary["extra"]["epochs"] == 3
    assert summary["extra"]["batch_size"] == 8
    assert summary["extra"]["train_stride"] == 4
    assert "advection1d" in summary["extra"]["command"]
    assert "--dry-run" in summary["extra"]["command"]
    assert summary["details"]["contract"]["published_numbers_directly_comparable"] is False


def test_live_test_split_requires_explicit_held_out_flag_before_data(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_external_pdebench_unet_baseline.py",
            "--config",
            "configs/train_multitask_heterogeneous_light_best.yaml",
            "--name",
            "blocked_live_test",
            "--output-root",
            str(tmp_path),
            "--data-root",
            "data/pdebench",
            "--eval-split",
            "test",
            "--max-train-samples",
            "1",
            "--max-eval-samples",
            "1",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode != 0
    assert "--allow-held-out-test-eval" in proc.stderr


def test_live_test_split_blocks_repeat_external_ledger_before_measurement(tmp_path):
    ledger_path = tmp_path / "external-test-ledger.json"
    args = build_parser().parse_args(
        [
            "--config",
            "configs/train_multitask_heterogeneous_light_best.yaml",
            "--name",
            "repeat_blocked",
            "--output-root",
            str(tmp_path / "out"),
            "--data-root",
            "data/pdebench",
            "--eval-split",
            "test",
            "--tasks",
            "advection1d",
            "burgers1d",
            "darcy2d",
            "--max-train-samples",
            "32",
            "--max-eval-samples",
            "32",
            "--held-out-ledger-json",
            str(ledger_path),
            "--allow-held-out-test-eval",
        ]
    )
    measurement_key = _external_test_measurement_key(
        args=args,
        tasks=["advection1d", "burgers1d", "darcy2d"],
    )
    ledger_path.write_text(
        json.dumps({"measurements": [{"measurement_key": measurement_key}]}),
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError, match="held-out external PDEBench U-Net test measurement already recorded"
    ):
        run_baseline(args)


def test_allow_repeat_test_is_explicit_in_command_record(tmp_path):
    args = build_parser().parse_args(
        [
            "--dry-run",
            "--config",
            "configs/train_multitask_heterogeneous_light_best.yaml",
            "--name",
            "external_pdebench_unet_repeat_debug",
            "--output-root",
            str(tmp_path),
            "--eval-split",
            "test",
            "--allow-repeat-test",
        ]
    )

    summary_path = run_baseline(args)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert "--allow-repeat-test" in summary["extra"]["command"]
