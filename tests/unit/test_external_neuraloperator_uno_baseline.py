from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from scripts.run_external_neuraloperator_uno_baseline import (
    _external_test_measurement_key,
    build_neuraloperator_uno_model,
    build_parser,
    run_baseline,
    train_uno_group_model,
    uno_modes_for_grid,
    uno_scalings_for_grid,
)

ROOT = Path(__file__).resolve().parents[2]


class TinyUNO(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        lifting_channels: int,
        projection_channels: int,
        n_layers: int,
        uno_out_channels: list[int],
        uno_n_modes: list[list[int]],
        uno_scalings: list[list[float]],
        positional_embedding: str,
        channel_mlp_skip: str,
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.lifting_channels = lifting_channels
        self.projection_channels = projection_channels
        self.n_layers = n_layers
        self.uno_out_channels = uno_out_channels
        self.uno_n_modes = uno_n_modes
        self.uno_scalings = uno_scalings
        self.positional_embedding = positional_embedding
        self.channel_mlp_skip = channel_mlp_skip
        if len(uno_n_modes[0]) == 1:
            self.net = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.net = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        nn.init.zeros_(self.net.weight)
        nn.init.zeros_(self.net.bias)

    def forward(self, current: torch.Tensor) -> torch.Tensor:
        return self.net(current)


class TinyMetadataUNO(TinyUNO):
    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        state["_metadata"] = {"uno_n_modes": self.uno_n_modes}
        return state


def test_uno_modes_for_grid_uses_1d_for_flat_tasks():
    assert uno_modes_for_grid((1, 64), 16) == [16]
    assert uno_modes_for_grid((16, 32), 16) == [8, 16]


def test_uno_scalings_keep_end_to_end_resolution():
    assert uno_scalings_for_grid((1, 64), 4, identity_scaling=False) == [
        [1.0],
        [0.5],
        [1.0],
        [2.0],
    ]
    assert uno_scalings_for_grid((16, 32), 2, identity_scaling=False) == [
        [0.5, 0.5],
        [2.0, 2.0],
    ]
    assert uno_scalings_for_grid((16, 32), 3, identity_scaling=True) == [
        [1.0, 1.0],
        [1.0, 1.0],
        [1.0, 1.0],
    ]


def test_build_neuraloperator_uno_model_adapts_1d_grid_to_repo_grid_shape():
    model = build_neuraloperator_uno_model(
        channels=1,
        grid_shape=(1, 8),
        hidden_channels=4,
        fourier_modes=4,
        n_layers=2,
        lifting_channels=8,
        projection_channels=8,
        channel_mlp_skip="linear",
        identity_scaling=False,
        residual=False,
        uno_cls=TinyUNO,
    )

    pred = model(torch.randn(2, 1, 1, 8))

    assert pred.shape == (2, 1, 1, 8)
    assert model.uno.uno_n_modes == [[4], [4]]
    assert model.uno.uno_scalings == [[0.5], [2.0]]
    assert model.uno.channel_mlp_skip == "linear"


def test_train_uno_group_model_can_learn_simple_residual_with_fake_uno():
    generator = torch.Generator().manual_seed(13)
    currents = torch.randn(16, 1, 1, 8, generator=generator)
    targets = currents + 0.25

    model, fit = train_uno_group_model(
        currents,
        targets,
        hidden_channels=4,
        fourier_modes=4,
        n_layers=2,
        lifting_channels=8,
        projection_channels=8,
        residual=True,
        epochs=80,
        learning_rate=0.05,
        batch_size=4,
        seed=7,
        uno_cls=TinyUNO,
    )

    with torch.no_grad():
        mse = torch.mean((model(currents) - targets) ** 2).item()
    assert fit["implementation"] == "neuralop.models.UNO"
    assert fit["uno_n_modes"] == [[4], [4]]
    assert mse < 0.01


def test_train_uno_group_model_ignores_neuraloperator_state_metadata():
    generator = torch.Generator().manual_seed(23)
    currents = torch.randn(16, 1, 1, 8, generator=generator)
    targets = currents + 0.25

    model, fit = train_uno_group_model(
        currents,
        targets,
        hidden_channels=4,
        fourier_modes=4,
        n_layers=2,
        lifting_channels=8,
        projection_channels=8,
        residual=True,
        epochs=80,
        learning_rate=0.05,
        batch_size=4,
        seed=9,
        uno_cls=TinyMetadataUNO,
    )

    with torch.no_grad():
        mse = torch.mean((model(currents) - targets) ** 2).item()
    assert fit["implementation"] == "neuralop.models.UNO"
    assert mse < 0.01


def test_external_neuraloperator_uno_dry_run_writes_contract_summary(tmp_path):
    args = build_parser().parse_args(
        [
            "--dry-run",
            "--config",
            "configs/train_multitask_heterogeneous_light_best.yaml",
            "--name",
            "external_uno_dry_run",
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
    assert summary["extra"]["baseline"] == "external_neuraloperator_uno"
    assert summary["extra"]["implementation"] == "neuralop.models.UNO"
    assert summary["extra"]["split"] == "test"
    assert summary["extra"]["epochs"] == 3
    assert summary["extra"]["batch_size"] == 8
    assert summary["extra"]["train_stride"] == 4
    assert "advection1d" in summary["extra"]["command"]
    assert "--dry-run" in summary["extra"]["command"]
    assert summary["details"]["contract"]["published_numbers_directly_comparable"] is False


def test_live_test_split_requires_explicit_held_out_flag_before_import_or_data(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_external_neuraloperator_uno_baseline.py",
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
        RuntimeError, match="held-out external UNO test measurement already recorded"
    ):
        run_baseline(args)


def test_allow_repeat_test_is_explicit_in_command_record(tmp_path):
    args = build_parser().parse_args(
        [
            "--dry-run",
            "--config",
            "configs/train_multitask_heterogeneous_light_best.yaml",
            "--name",
            "external_uno_repeat_debug",
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
