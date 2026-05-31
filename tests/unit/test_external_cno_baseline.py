from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from scripts.run_external_cno_baseline import (
    CNO1d,
    _external_test_measurement_key,
    build_cno1d_model,
    build_parser,
    run_baseline,
    train_cno_group_model,
)

ROOT = Path(__file__).resolve().parents[2]


class TinyCNO1d(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        size: int,
        n_layers: int,
        *,
        n_res: int,
        n_res_neck: int,
        channel_multiplier: int,
        lift_latent_dim: int,
        use_bn: bool,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.n_res = n_res
        self.n_res_neck = n_res_neck
        self.channel_multiplier = channel_multiplier
        self.lift_latent_dim = lift_latent_dim
        self.use_bn = use_bn
        self.net = nn.Conv1d(in_dim, out_dim, kernel_size=1)
        nn.init.zeros_(self.net.weight)
        nn.init.zeros_(self.net.bias)

    def forward(self, current: torch.Tensor) -> torch.Tensor:
        return self.net(current)


def test_official_cno1d_preserves_width():
    model = CNO1d(
        in_dim=1,
        out_dim=1,
        size=32,
        n_layers=2,
        n_res=1,
        n_res_neck=1,
        channel_multiplier=4,
        lift_latent_dim=4,
        use_bn=False,
    )
    model.eval()

    with torch.no_grad():
        pred = model(torch.randn(2, 1, 32))

    assert pred.shape == (2, 1, 32)


def test_build_cno1d_model_adapts_repo_grid_shape():
    model = build_cno1d_model(
        channels=1,
        grid_shape=(1, 32),
        n_layers=2,
        n_res=1,
        n_res_neck=1,
        channel_multiplier=4,
        lift_latent_dim=8,
        use_bn=True,
        residual=False,
        cno_cls=TinyCNO1d,
    )

    pred = model(torch.randn(2, 1, 1, 32))

    assert pred.shape == (2, 1, 1, 32)
    assert model.cno.lift_latent_dim == 8


def test_build_cno1d_model_rejects_non_height_one_grid():
    with pytest.raises(ValueError, match="requires height=1"):
        build_cno1d_model(
            channels=1,
            grid_shape=(16, 16),
            n_layers=2,
            n_res=1,
            n_res_neck=1,
            channel_multiplier=4,
            lift_latent_dim=8,
            use_bn=True,
            residual=False,
            cno_cls=TinyCNO1d,
        )


def test_build_cno1d_model_rejects_non_poolable_width():
    with pytest.raises(ValueError, match="divisible by 2\\*\\*n_layers"):
        build_cno1d_model(
            channels=1,
            grid_shape=(1, 30),
            n_layers=3,
            n_res=1,
            n_res_neck=1,
            channel_multiplier=4,
            lift_latent_dim=8,
            use_bn=True,
            residual=False,
            cno_cls=TinyCNO1d,
        )


def test_train_cno_group_model_can_learn_simple_residual_with_fake_cno():
    generator = torch.Generator().manual_seed(13)
    currents = torch.randn(16, 1, 1, 32, generator=generator)
    targets = currents + 0.25

    model, fit = train_cno_group_model(
        currents,
        targets,
        n_layers=2,
        n_res=1,
        n_res_neck=1,
        channel_multiplier=4,
        lift_latent_dim=8,
        residual=True,
        epochs=80,
        learning_rate=0.05,
        batch_size=4,
        seed=7,
        cno_cls=TinyCNO1d,
    )

    with torch.no_grad():
        mse = torch.mean((model(currents) - targets) ** 2).item()
    assert fit["implementation"] == "camlab-ethz.ConvolutionalNeuralOperator.CNO1d_simplified.CNO1d"
    assert fit["source_commit"] == "6e765198aa02b56352e0a3437104b9d9e337176e"
    assert mse < 0.01


def test_external_cno_dry_run_writes_contract_summary(tmp_path):
    args = build_parser().parse_args(
        [
            "--dry-run",
            "--config",
            "configs/train_multitask_heterogeneous_light_best.yaml",
            "--name",
            "external_cno_dry_run",
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
    assert summary["extra"]["baseline"] == "external_cno1d"
    assert summary["extra"]["source_commit"] == "6e765198aa02b56352e0a3437104b9d9e337176e"
    assert summary["extra"]["split"] == "test"
    assert summary["extra"]["epochs"] == 3
    assert summary["extra"]["batch_size"] == 8
    assert summary["extra"]["train_stride"] == 4
    assert "advection1d" in summary["extra"]["command"]
    assert "--dry-run" in summary["extra"]["command"]
    assert summary["details"]["contract"]["published_numbers_directly_comparable"] is False
    assert (
        summary["details"]["contract"]["current_adapter_scope"] == "CNO1d height=1 light-v1 grids"
    )


def test_live_test_split_requires_explicit_held_out_flag_before_data(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_external_cno_baseline.py",
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
        RuntimeError, match="held-out external CNO1d test measurement already recorded"
    ):
        run_baseline(args)
