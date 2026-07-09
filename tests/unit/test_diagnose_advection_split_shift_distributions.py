from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts.diagnose_advection_split_shift_distributions import (
    estimate_shift_per_step,
    main,
    nearest_beta,
)


def _rolled_trajectory(width: int, steps: int, shift_per_step: float) -> np.ndarray:
    x = np.linspace(0.0, 2.0 * np.pi, width, endpoint=False)
    base = np.sin(x) + 0.4 * np.sin(3.0 * x + 0.7)
    frames = []
    for t in range(steps):
        total = shift_per_step * t
        k = int(np.floor(total))
        frac = total - k
        rolled = np.roll(base, k)
        frames.append((1.0 - frac) * rolled + frac * np.roll(base, k + 1))
    return np.stack(frames, axis=0)


@pytest.mark.parametrize("true_shift", [1.0, 10.25, 41.0, -7.5])
def test_estimate_shift_per_step_recovers_known_shift(true_shift):
    fields = _rolled_trajectory(width=256, steps=24, shift_per_step=true_shift)

    estimate = estimate_shift_per_step(fields, transitions=16)

    assert abs(estimate - true_shift) < 0.3


def test_nearest_beta_maps_shift_to_official_grid():
    assert nearest_beta(1.02, 10.24) == 0.1
    assert nearest_beta(41.0, 10.24) == 4.0
    assert nearest_beta(71.7, 10.24) == 7.0


def test_cli_refuses_test_split_without_flag(tmp_path):
    with pytest.raises(SystemExit, match="--include-test"):
        main(["--root", str(tmp_path), "--splits", "train", "test"])


def test_cli_writes_honest_diagnostic_record(tmp_path):
    fields = _rolled_trajectory(width=128, steps=12, shift_per_step=4.0)
    with h5py.File(tmp_path / "advection1d_val.h5", "w") as handle:
        handle.create_dataset("data", data=fields[None, :, :, None].astype(np.float32))
    output = tmp_path / "out.json"

    assert (
        main(
            [
                "--root",
                str(tmp_path),
                "--splits",
                "val",
                "--transitions",
                "8",
                "--output-json",
                str(output),
            ]
        )
        == 0
    )

    record = json.loads(Path(output).read_text())
    assert record["candidate_scored"] is False
    assert record["held_out_test_data_read"] is False
    assert record["test_ledger_writes"] == []
    assert record["splits"]["val"]["samples_analyzed"] == 1
    assert abs(record["splits"]["val"]["shift_per_step_pixels"]["mean"] - 4.0) < 0.3
