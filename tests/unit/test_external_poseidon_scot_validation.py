from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch import nn

from scripts.run_external_poseidon_scot_validation import (
    evaluate_poseidon_scot_validation,
    poseidon_checkpoint_handle,
    validate_poseidon_scot_summary,
)

ROOT = Path(__file__).resolve().parents[2]


class IdentityScOT(nn.Module):
    def forward(self, *, pixel_values: torch.Tensor, time: torch.Tensor):
        assert time.shape == (pixel_values.shape[0],)
        return type("Output", (), {"output": pixel_values})()


def _write_h5(path: Path, shape: tuple[int, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.linspace(0.0, 1.0, num=int(np.prod(shape)), dtype=np.float32).reshape(shape)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=values)


def test_poseidon_checkpoint_handle_validates_model_size():
    assert poseidon_checkpoint_handle("t") == "camlab-ethz/Poseidon-T"
    with pytest.raises(ValueError, match="one of T, B, or L"):
        poseidon_checkpoint_handle("XL")


def test_evaluate_poseidon_scot_validation_reports_decoded_rollout_metrics(tmp_path):
    data_root = tmp_path / "data"
    _write_h5(data_root / "advection1d_val.h5", (2, 4, 16, 1))

    metrics, records = evaluate_poseidon_scot_validation(
        {"data": {"root": str(data_root)}},
        IdentityScOT(),
        tasks=["advection1d"],
        split="val",
        data_root=str(data_root),
        max_eval_samples=1,
        rollout_steps=2,
        image_size=8,
        time_value=1.0,
        device="cpu",
    )

    assert "decoded_rollout_nrmse" in metrics
    assert "task_advection1d_decoded_rollout_nrmse" in metrics
    assert records[0]["teacher_forced_steps"] is True
    assert records[0]["pairs_evaluated"] == 2


def test_poseidon_scot_summary_requires_checkpoint_hash():
    summary = {
        "schema_version": 1,
        "status": "validation_model_measurement_complete",
        "measurement_type": "poseidon_scot_validation_measurement",
        "split": "val",
        "held_out_test_used": False,
        "claim_comparable": False,
        "published_numbers_directly_comparable": False,
        "metrics": {"decoded_rollout_nrmse": 1.0},
        "details": {
            "pretrained_checkpoint": {"sha256": ""},
            "model": {"embedding_recovery_replaced": True},
        },
    }

    errors = validate_poseidon_scot_summary(summary)

    assert "details.pretrained_checkpoint.sha256 is required" in errors


def test_poseidon_scot_cli_blocks_test_split_before_checkpoint_download(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_external_poseidon_scot_validation.py",
            "--config",
            "configs/train_multitask_heterogeneous_light_best.yaml",
            "--name",
            "blocked_poseidon_test",
            "--output-root",
            str(tmp_path / "out"),
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
    assert not (tmp_path / "out").exists()
