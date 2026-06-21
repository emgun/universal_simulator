from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch
from torch import nn

from scripts.run_physicsnemo_compatibility_smoke import (
    PHYSICSNEMO_FNO_IMPORT,
    build_parser,
    build_physicsnemo_fno_model,
    run_compatibility_smoke,
    run_live_recipe_adapter,
    train_physicsnemo_fno_group_model,
    validate_physicsnemo_smoke_summary,
)

ROOT = Path(__file__).resolve().parents[2]


class TinyPhysicsNeMoFNO(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        decoder_layers: int,
        decoder_layer_size: int,
        dimension: int,
        latent_channels: int,
        num_fno_layers: int,
        num_fno_modes: int,
        padding: int,
    ) -> None:
        super().__init__()
        self.decoder_layers = int(decoder_layers)
        self.decoder_layer_size = int(decoder_layer_size)
        self.dimension = int(dimension)
        self.latent_channels = int(latent_channels)
        self.num_fno_layers = int(num_fno_layers)
        self.num_fno_modes = int(num_fno_modes)
        self.padding = int(padding)
        if self.dimension == 1:
            self.net = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.net = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        nn.init.zeros_(self.net.weight)
        nn.init.zeros_(self.net.bias)

    def forward(self, current: torch.Tensor) -> torch.Tensor:
        return self.net(current)


def test_physicsnemo_compatibility_smoke_writes_non_metric_manifest(tmp_path):
    evidence_json = tmp_path / "physicsnemo_smoke.json"
    args = build_parser().parse_args(
        [
            "--name",
            "physicsnemo_smoke_test",
            "--output-root",
            str(tmp_path / "out"),
            "--evidence-json",
            str(evidence_json),
            "--tasks",
            "advection1d",
            "burgers1d",
            "darcy2d",
            "--train-split",
            "train",
            "--eval-split",
            "val",
        ]
    )

    summary_path = run_compatibility_smoke(args)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    evidence = json.loads(evidence_json.read_text(encoding="utf-8"))

    assert summary["status"] == "compatibility_smoke_ready"
    assert summary["measurement_type"] == "physicsnemo_compatibility_smoke"
    assert summary["held_out_test_used"] is False
    assert summary["held_out_test_data_read"] is False
    assert summary["claim_comparable"] is False
    assert summary["published_numbers_directly_comparable"] is False
    assert "decoded_rollout_nrmse" not in summary["metrics"]
    assert summary["metrics"] == {}
    assert summary["inspected_splits"] == ["train", "val"]
    assert summary["details"]["package"]["pip_name"] == "nvidia-physicsnemo"
    assert summary["details"]["package"]["import_name"] == "physicsnemo"
    assert summary["details"]["recipe_contract"]["tasks"] == [
        "advection1d",
        "burgers1d",
        "darcy2d",
    ]
    assert summary["details"]["recipe_contract"]["live_metric_allowed"] is False
    assert summary["details"]["recipe_contract"]["next_gate"].startswith(
        "Run a live PhysicsNeMo recipe adapter"
    )
    assert evidence == summary
    assert validate_physicsnemo_smoke_summary(summary) == []


def test_physicsnemo_compatibility_smoke_blocks_test_split_before_output(tmp_path):
    output_root = tmp_path / "out"
    evidence_json = tmp_path / "physicsnemo_smoke.json"

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_physicsnemo_compatibility_smoke.py",
            "--name",
            "blocked_physicsnemo_smoke",
            "--output-root",
            str(output_root),
            "--evidence-json",
            str(evidence_json),
            "--eval-split",
            "test",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode != 0
    assert "must not inspect split=test" in proc.stderr
    assert not output_root.exists()
    assert not evidence_json.exists()


def test_physicsnemo_smoke_validator_rejects_metric_overclaim():
    summary = {
        "schema_version": 1,
        "status": "compatibility_smoke_ready",
        "measurement_type": "physicsnemo_compatibility_smoke",
        "run_name": "physicsnemo_smoke_test",
        "split": "val",
        "inspected_splits": ["train", "val"],
        "metrics": {"decoded_rollout_nrmse": 0.1},
        "claim_comparable": False,
        "published_numbers_directly_comparable": False,
        "held_out_test_used": False,
        "held_out_test_data_read": False,
        "details": {
            "package": {"pip_name": "nvidia-physicsnemo", "import_name": "physicsnemo"},
            "recipe_contract": {
                "tasks": ["advection1d"],
                "live_metric_allowed": False,
                "next_gate": "Run a live PhysicsNeMo recipe adapter on train/val.",
            },
        },
    }

    assert "compatibility smoke must not report decoded_rollout_nrmse" in (
        validate_physicsnemo_smoke_summary(summary)
    )


def test_build_physicsnemo_fno_model_adapts_1d_grid_to_repo_grid_shape():
    model = build_physicsnemo_fno_model(
        channels=1,
        grid_shape=(1, 8),
        latent_channels=4,
        fourier_modes=4,
        num_fno_layers=2,
        decoder_layers=1,
        decoder_layer_size=8,
        padding=2,
        residual=False,
        fno_cls=TinyPhysicsNeMoFNO,
    )

    pred = model(torch.randn(2, 1, 1, 8))

    assert pred.shape == (2, 1, 1, 8)
    assert model.fno.dimension == 1
    assert model.fno.num_fno_modes == 4
    assert model.fno.num_fno_layers == 2


def test_train_physicsnemo_fno_group_model_can_learn_simple_residual_with_fake_fno():
    generator = torch.Generator().manual_seed(31)
    currents = torch.randn(16, 1, 1, 8, generator=generator)
    targets = currents + 0.25

    model, fit = train_physicsnemo_fno_group_model(
        currents,
        targets,
        latent_channels=4,
        fourier_modes=4,
        num_fno_layers=1,
        decoder_layers=1,
        decoder_layer_size=8,
        padding=2,
        residual=True,
        epochs=80,
        learning_rate=0.05,
        batch_size=4,
        seed=11,
        fno_cls=TinyPhysicsNeMoFNO,
    )

    with torch.no_grad():
        mse = torch.mean((model(currents) - targets) ** 2).item()
    assert fit["implementation"] == PHYSICSNEMO_FNO_IMPORT
    assert fit["num_fno_modes"] == 4
    assert fit["dimension"] == 1
    assert mse < 0.01


def test_physicsnemo_live_recipe_dry_run_writes_validation_contract(tmp_path):
    args = build_parser().parse_args(
        [
            "--live-recipe",
            "--dry-run",
            "--config",
            "configs/train_multitask_heterogeneous_light_best.yaml",
            "--name",
            "physicsnemo_live_recipe_dry_run",
            "--output-root",
            str(tmp_path),
            "--data-root",
            "data/pdebench",
            "--eval-split",
            "val",
            "--tasks",
            "advection1d",
            "burgers1d",
            "darcy2d",
        ]
    )

    summary_path = run_live_recipe_adapter(args)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["status"] == "dry_run"
    assert summary["measurement_type"] == "physicsnemo_recipe_validation_adapter"
    assert summary["metrics"] == {}
    assert summary["claim_comparable"] is False
    assert summary["published_numbers_directly_comparable"] is False
    assert summary["held_out_test_used"] is False
    assert summary["held_out_test_data_read"] is False
    assert summary["details"]["contract"]["requires_optional_dependency"] == "nvidia-physicsnemo"
    assert summary["details"]["contract"]["live_test_allowed"] is False
    assert summary["extra"]["implementation"] == PHYSICSNEMO_FNO_IMPORT
    assert "--live-recipe" in summary["extra"]["command"]
    assert "--dry-run" in summary["extra"]["command"]


def test_physicsnemo_live_recipe_blocks_test_split_before_import_or_output(tmp_path):
    output_root = tmp_path / "out"
    evidence_json = tmp_path / "physicsnemo_smoke.json"

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_physicsnemo_compatibility_smoke.py",
            "--live-recipe",
            "--name",
            "blocked_physicsnemo_live_recipe",
            "--output-root",
            str(output_root),
            "--evidence-json",
            str(evidence_json),
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
    assert "must not inspect split=test" in proc.stderr
    assert not output_root.exists()
    assert not evidence_json.exists()
