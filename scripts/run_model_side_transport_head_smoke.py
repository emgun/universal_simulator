#!/usr/bin/env python
from __future__ import annotations

"""Run a no-provider synthetic smoke for the model-side beta transport head."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from ups.core.latent_state import LatentState
from ups.eval.pdebench_runner import evaluate_decoded_operator


class _IdentityEncoder(torch.nn.Module):
    def forward(self, fields, coords, *, meta=None, params=None, bc=None, geom=None):
        return fields["u"]


class _IdentityOperator(torch.nn.Module):
    def forward(self, state: LatentState, dt):
        return LatentState(z=state.z, t=dt if state.t is None else state.t + dt, cond=state.cond)


class _IdentityDecoder(torch.nn.Module):
    def forward(self, points, latent_tokens, *, conditioning=None):
        return {"u": latent_tokens}


def _write_synthetic_hdf5(root: Path, *, steps: int, width: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    advection = torch.zeros(1, steps + 1, width, dtype=torch.float32)
    for step in range(steps + 1):
        advection[0, step, step % width] = 1.0
    with h5py.File(root / "advection1d_val.h5", "w") as handle:
        handle.create_dataset("data", data=advection.numpy())
        handle.create_dataset("source_file_index", data=torch.tensor([0]).numpy())
        handle.attrs["source_paths"] = ["1D/Advection/Train/1D_Advection_Sols_beta0.1.hdf5"]

    burgers = torch.full((1, steps + 1, width), 2.0, dtype=torch.float32)
    with h5py.File(root / "burgers1d_val.h5", "w") as handle:
        handle.create_dataset("data", data=burgers.numpy())

    darcy = torch.full((1, steps + 1, width), 3.0, dtype=torch.float32)
    with h5py.File(root / "darcy2d_val.h5", "w") as handle:
        handle.create_dataset("data", data=darcy.numpy())


def run_smoke(*, output_dir: Path, rollout_steps: int = 16, width: int = 8) -> dict[str, Any]:
    data_root = output_dir / "synthetic_data"
    _write_synthetic_hdf5(data_root, steps=rollout_steps, width=width)
    cfg: dict[str, Any] = {
        "training": {"batch_size": 1, "dt": 0.1},
        "model_side_transport_head": {
            "enabled": True,
            "tasks": ["advection1d"],
            "required_params": ["beta"],
            "features": ["param:beta", "bias"],
            "init": {"param:beta": 10.0, "bias": 0.0},
            "mode": "periodic_roll",
            "apply_at": "decoded_rollout",
            "missing_param_policy": "skip",
        },
        "evaluation": {
            "decoded_persistence_residual_alpha": 0.0,
            "report_all_horizon_metrics": True,
        },
        "data": {
            "task": ["advection1d", "burgers1d", "darcy2d"],
            "split": "val",
            "root": str(data_root),
            "patch_size": 1,
            "field_name": "u",
            "param_keys": ["beta"],
        },
    }
    report = evaluate_decoded_operator(
        cfg,
        _IdentityEncoder(),
        _IdentityOperator(),
        _IdentityDecoder(),
        rollout_steps=rollout_steps,
    )
    summary = {
        "run_name": "model_side_transport_head_smoke_val_light_v1",
        "measurement_type": "synthetic_model_side_transport_head_smoke",
        "held_out_test_used": False,
        "held_out_test_data_read": False,
        "command_args": [
            "python",
            "scripts/run_model_side_transport_head_smoke.py",
            "--output-dir",
            str(output_dir),
        ],
        "data": {
            "root": str(data_root),
            "split": "val",
            "tasks": ["advection1d", "burgers1d", "darcy2d"],
            "synthetic": True,
        },
        "metrics": report.metrics,
        "extra": report.extra,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/research/sota_loop/model_side_transport_head_smoke_val_light_v1"),
    )
    parser.add_argument("--rollout-steps", type=int, default=16)
    parser.add_argument("--width", type=int, default=8)
    args = parser.parse_args()

    summary = run_smoke(
        output_dir=args.output_dir,
        rollout_steps=int(args.rollout_steps),
        width=int(args.width),
    )
    print(
        json.dumps(
            {
                "summary_json": str(args.output_dir / "summary.json"),
                "metrics": summary["metrics"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
