from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

from scripts.run_external_poseidon_transfer_adapter import (
    build_parser,
    light_step_to_poseidon_pixels,
    poseidon_pixels_to_repo_flat,
    run_adapter_manifest,
    validate_poseidon_adapter_summary,
)

ROOT = Path(__file__).resolve().parents[2]


def _write_h5(path: Path, shape: tuple[int, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.linspace(0.0, 1.0, num=int(np.prod(shape)), dtype=np.float32).reshape(shape)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=values)


def _write_fake_poseidon_repo(path: Path) -> None:
    (path / "scOT").mkdir(parents=True)
    (path / "scOT" / "problems").mkdir(parents=True)
    (path / "README.md").write_text(
        'ScOT.from_pretrained("camlab-ethz/Poseidon-T")\n',
        encoding="utf-8",
    )
    (path / "pyproject.toml").write_text("[project]\nname='scOT'\n", encoding="utf-8")
    (path / "scOT" / "model.py").write_text(
        "class ScOT: pass\nclass ScOTConfig: pass\n", encoding="utf-8"
    )
    (path / "scOT" / "train.py").write_text("ignore_mismatched_sizes=True\n", encoding="utf-8")
    (path / "scOT" / "inference.py").write_text("python -m scOT.inference\n", encoding="utf-8")
    (path / "scOT" / "problems" / "base.py").write_text(
        "class BaseDataset: pass\n", encoding="utf-8"
    )


def test_light_step_to_poseidon_pixels_roundtrips_to_repo_flat_shape():
    field_step = torch.linspace(0.0, 1.0, steps=16).reshape(16, 1)

    pixels = light_step_to_poseidon_pixels(field_step, (1, 16), image_size=8)
    flat = poseidon_pixels_to_repo_flat(pixels, (1, 16))

    assert pixels.shape == (1, 1, 8, 8)
    assert flat.shape == (1, 16, 1)
    assert torch.isfinite(flat).all()


def test_poseidon_adapter_manifest_writes_train_val_only_evidence(tmp_path):
    data_root = tmp_path / "data"
    for split in ("train", "val"):
        _write_h5(data_root / f"advection1d_{split}.h5", (2, 4, 16, 1))
    poseidon_repo = tmp_path / "poseidon"
    _write_fake_poseidon_repo(poseidon_repo)

    args = build_parser().parse_args(
        [
            "--config",
            "configs/train_multitask_heterogeneous_light_best.yaml",
            "--name",
            "poseidon_adapter_test",
            "--output-root",
            str(tmp_path / "out"),
            "--data-root",
            str(data_root),
            "--tasks",
            "advection1d",
            "--max-samples",
            "1",
            "--max-steps",
            "2",
            "--image-size",
            "8",
            "--poseidon-repo",
            str(poseidon_repo),
        ]
    )

    summary_path = run_adapter_manifest(args)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["status"] == "validation_adapter_manifest_complete"
    assert summary["measurement_type"] == "poseidon_validation_adapter_manifest"
    assert summary["held_out_test_used"] is False
    assert summary["held_out_test_data_read"] is False
    assert summary["inspected_splits"] == ["train", "val"]
    assert summary["claim_comparable"] is False
    assert "adapter_roundtrip_nrmse" in summary["metrics"]
    assert "decoded_rollout_nrmse" not in summary["metrics"]
    assert summary["details"]["pretrained_checkpoint"]["handle"] == "camlab-ethz/Poseidon-T"
    assert summary["details"]["pretrained_checkpoint"]["sha256_status"] == "pending"
    assert summary["details"]["adapter_records"][0]["poseidon_pixel_shape"] == [1, 1, 8, 8]
    assert validate_poseidon_adapter_summary(summary) == []


def test_poseidon_adapter_cli_blocks_test_split_before_data(tmp_path):
    output_root = tmp_path / "out"

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_external_poseidon_transfer_adapter.py",
            "--config",
            "configs/train_multitask_heterogeneous_light_best.yaml",
            "--name",
            "blocked_poseidon_adapter_test",
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
    assert "must not inspect split=test" in proc.stderr
    assert not output_root.exists()
