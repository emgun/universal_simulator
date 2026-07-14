#!/usr/bin/env python
from __future__ import annotations

"""Train/evaluate an external official CNO1d architecture baseline on light-v1.

The CNO1d classes in this file are adapted from:
https://github.com/camlab-ethz/ConvolutionalNeuralOperator/blob/6e765198aa02b56352e0a3437104b9d9e337176e/CNO1d_simplified/CNO1d.py

The upstream ConvolutionalNeuralOperator source includes the following notice:

MIT License

Copyright (c) 2024 Computational and Applied Mathematics Laboratory @ ETH Zurich

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import argparse
import csv
import hashlib
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts import run_external_neuraloperator_fno_baseline as fno_runner

CNO_IMPLEMENTATION = "camlab-ethz.ConvolutionalNeuralOperator.CNO1d_simplified.CNO1d"
CNO_SOURCE_URL = "https://github.com/camlab-ethz/ConvolutionalNeuralOperator"
CNO_SOURCE_COMMIT = "6e765198aa02b56352e0a3437104b9d9e337176e"
CNO_SOURCE_PATH = "CNO1d_simplified/CNO1d.py"


class CNOLeakyReLU(nn.Module):
    """Official CNO activation: antialiased upsample, LeakyReLU, downsample."""

    def __init__(self, in_size: int, out_size: int) -> None:
        super().__init__()
        self.in_size = int(in_size)
        self.out_size = int(out_size)
        self.act = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x.unsqueeze(2),
            size=(1, 2 * self.in_size),
            mode="bicubic",
            antialias=True,
        )
        x = self.act(x)
        x = F.interpolate(
            x,
            size=(1, self.out_size),
            mode="bicubic",
            antialias=True,
        )
        return x[:, :, 0]


class CNOBlock1d(nn.Module):
    """Official CNO 1D Conv-BN-activation block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        in_size: int,
        out_size: int,
        *,
        use_bn: bool = True,
    ) -> None:
        super().__init__()
        self.convolution = nn.Conv1d(
            in_channels=int(in_channels),
            out_channels=int(out_channels),
            kernel_size=3,
            padding=1,
        )
        self.batch_norm = nn.BatchNorm1d(int(out_channels)) if use_bn else nn.Identity()
        self.act = CNOLeakyReLU(in_size=int(in_size), out_size=int(out_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convolution(x)
        x = self.batch_norm(x)
        return self.act(x)


class CNOLiftProjectBlock1d(nn.Module):
    """Official CNO lift/project block with exposed latent dimension."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        size: int,
        *,
        latent_dim: int = 64,
    ) -> None:
        super().__init__()
        self.inter_cno_block = CNOBlock1d(
            in_channels=int(in_channels),
            out_channels=int(latent_dim),
            in_size=int(size),
            out_size=int(size),
            use_bn=False,
        )
        self.convolution = nn.Conv1d(
            in_channels=int(latent_dim),
            out_channels=int(out_channels),
            kernel_size=3,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inter_cno_block(x)
        return self.convolution(x)


class CNOResidualBlock1d(nn.Module):
    """Official CNO residual block."""

    def __init__(self, channels: int, size: int, *, use_bn: bool = True) -> None:
        super().__init__()
        self.convolution1 = nn.Conv1d(
            in_channels=int(channels),
            out_channels=int(channels),
            kernel_size=3,
            padding=1,
        )
        self.convolution2 = nn.Conv1d(
            in_channels=int(channels),
            out_channels=int(channels),
            kernel_size=3,
            padding=1,
        )
        self.batch_norm1 = nn.BatchNorm1d(int(channels)) if use_bn else nn.Identity()
        self.batch_norm2 = nn.BatchNorm1d(int(channels)) if use_bn else nn.Identity()
        self.act = CNOLeakyReLU(in_size=int(size), out_size=int(size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.convolution1(x)
        out = self.batch_norm1(out)
        out = self.act(out)
        out = self.convolution2(out)
        out = self.batch_norm2(out)
        return x + out


class CNOResNet1d(nn.Module):
    """Official stack of CNO residual blocks."""

    def __init__(self, channels: int, size: int, num_blocks: int, *, use_bn: bool = True) -> None:
        super().__init__()
        self.res_nets = nn.Sequential(
            *[
                CNOResidualBlock1d(channels=int(channels), size=int(size), use_bn=use_bn)
                for _ in range(int(num_blocks))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res_nets(x)


class CNO1d(nn.Module):
    """Simplified official CNO1d adapted for the repo baseline harness."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        size: int,
        n_layers: int,
        *,
        n_res: int = 4,
        n_res_neck: int = 4,
        channel_multiplier: int = 16,
        lift_latent_dim: int = 64,
        use_bn: bool = True,
    ) -> None:
        super().__init__()
        self.n_layers = int(n_layers)
        self.lift_dim = int(channel_multiplier) // 2
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.channel_multiplier = int(channel_multiplier)

        self.encoder_features = [self.lift_dim]
        for index in range(self.n_layers):
            self.encoder_features.append(2**index * self.channel_multiplier)

        self.decoder_features_in = list(reversed(self.encoder_features[1:]))
        self.decoder_features_out = list(reversed(self.encoder_features[:-1]))
        for index in range(1, self.n_layers):
            self.decoder_features_in[index] = 2 * self.decoder_features_in[index]

        self.encoder_sizes = []
        self.decoder_sizes = []
        for index in range(self.n_layers + 1):
            self.encoder_sizes.append(int(size) // 2**index)
            self.decoder_sizes.append(int(size) // 2 ** (self.n_layers - index))

        self.lift = CNOLiftProjectBlock1d(
            in_channels=self.in_dim,
            out_channels=self.encoder_features[0],
            size=int(size),
            latent_dim=int(lift_latent_dim),
        )
        self.project = CNOLiftProjectBlock1d(
            in_channels=self.encoder_features[0] + self.decoder_features_out[-1],
            out_channels=self.out_dim,
            size=int(size),
            latent_dim=int(lift_latent_dim),
        )

        self.encoder = nn.ModuleList(
            [
                CNOBlock1d(
                    in_channels=self.encoder_features[index],
                    out_channels=self.encoder_features[index + 1],
                    in_size=self.encoder_sizes[index],
                    out_size=self.encoder_sizes[index + 1],
                    use_bn=use_bn,
                )
                for index in range(self.n_layers)
            ]
        )
        self.ed_expansion = nn.ModuleList(
            [
                CNOBlock1d(
                    in_channels=self.encoder_features[index],
                    out_channels=self.encoder_features[index],
                    in_size=self.encoder_sizes[index],
                    out_size=self.decoder_sizes[self.n_layers - index],
                    use_bn=use_bn,
                )
                for index in range(self.n_layers + 1)
            ]
        )
        self.decoder = nn.ModuleList(
            [
                CNOBlock1d(
                    in_channels=self.decoder_features_in[index],
                    out_channels=self.decoder_features_out[index],
                    in_size=self.decoder_sizes[index],
                    out_size=self.decoder_sizes[index + 1],
                    use_bn=use_bn,
                )
                for index in range(self.n_layers)
            ]
        )
        self.res_nets = nn.Sequential(
            *[
                CNOResNet1d(
                    channels=self.encoder_features[index],
                    size=self.encoder_sizes[index],
                    num_blocks=int(n_res),
                    use_bn=use_bn,
                )
                for index in range(self.n_layers)
            ]
        )
        self.res_net_neck = CNOResNet1d(
            channels=self.encoder_features[self.n_layers],
            size=self.encoder_sizes[self.n_layers],
            num_blocks=int(n_res_neck),
            use_bn=use_bn,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(x)
        skip = []
        for index in range(self.n_layers):
            skip.append(self.res_nets[index](x))
            x = self.encoder[index](x)

        x = self.res_net_neck(x)

        for index in range(self.n_layers):
            if index == 0:
                x = self.ed_expansion[self.n_layers - index](x)
            else:
                x = torch.cat((x, self.ed_expansion[self.n_layers - index](skip[-index])), 1)
            x = self.decoder[index](x)

        x = torch.cat((x, self.ed_expansion[0](skip[0])), 1)
        return self.project(x)


class CNO1dGridAdapter(nn.Module):
    """Adapt CNO1d tensors to the repo's standard (B,C,H,W) grid."""

    def __init__(
        self,
        cno: nn.Module,
        *,
        grid_shape: tuple[int, int],
        residual: bool = False,
    ) -> None:
        super().__init__()
        self.cno = cno
        self.grid_shape = (int(grid_shape[0]), int(grid_shape[1]))
        self.residual = bool(residual)

    def forward(self, current: torch.Tensor) -> torch.Tensor:
        pred = self.cno(current.squeeze(-2)).unsqueeze(-2)
        if self.residual:
            pred = current + pred
        return pred


def _check_cno_grid_shape(grid_shape: tuple[int, int], n_layers: int) -> None:
    height, width = int(grid_shape[0]), int(grid_shape[1])
    if height != 1:
        raise ValueError(
            "The current CNO adapter uses official CNO1d and requires height=1 grids, "
            f"got {(height, width)}"
        )
    divisor = 2 ** int(n_layers)
    if width < divisor or width % divisor != 0:
        raise ValueError(
            f"CNO1d requires width divisible by 2**n_layers, got width={width}, n_layers={n_layers}"
        )


def build_cno1d_model(
    *,
    channels: int,
    grid_shape: tuple[int, int],
    n_layers: int,
    n_res: int,
    n_res_neck: int,
    channel_multiplier: int,
    lift_latent_dim: int,
    use_bn: bool,
    residual: bool,
    cno_cls: type[nn.Module] = CNO1d,
) -> nn.Module:
    _check_cno_grid_shape(grid_shape, n_layers)
    _, width = int(grid_shape[0]), int(grid_shape[1])
    cno = cno_cls(
        in_dim=int(channels),
        out_dim=int(channels),
        size=width,
        n_layers=int(n_layers),
        n_res=int(n_res),
        n_res_neck=int(n_res_neck),
        channel_multiplier=int(channel_multiplier),
        lift_latent_dim=int(lift_latent_dim),
        use_bn=bool(use_bn),
    )
    return CNO1dGridAdapter(cno, grid_shape=grid_shape, residual=residual)


def train_cno_group_model(
    currents: torch.Tensor,
    targets: torch.Tensor,
    *,
    n_layers: int = 3,
    n_res: int = 1,
    n_res_neck: int = 1,
    channel_multiplier: int = 8,
    lift_latent_dim: int = 64,
    use_bn: bool = True,
    residual: bool = False,
    epochs: int = 5,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 8,
    seed: int = 0,
    device: str | torch.device = "cpu",
    cno_cls: type[nn.Module] = CNO1d,
) -> tuple[nn.Module, dict[str, Any]]:
    if currents.shape != targets.shape:
        raise ValueError("currents and targets must have the same shape")
    if currents.dim() != 4:
        raise ValueError(f"Expected training tensors shaped (N,C,H,W), got {tuple(currents.shape)}")

    torch.manual_seed(int(seed))
    device = torch.device(device)
    grid_shape = (int(currents.shape[2]), int(currents.shape[3]))
    model = build_cno1d_model(
        channels=int(currents.shape[1]),
        grid_shape=grid_shape,
        n_layers=n_layers,
        n_res=n_res,
        n_res_neck=n_res_neck,
        channel_multiplier=channel_multiplier,
        lift_latent_dim=lift_latent_dim,
        use_bn=use_bn,
        residual=residual,
        cno_cls=cno_cls,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    generator = torch.Generator().manual_seed(int(seed))
    best_loss = float("inf")
    best_state = fno_runner._clone_tensor_state_dict(model)
    for _ in range(int(epochs)):
        order = torch.randperm(int(currents.shape[0]), generator=generator)
        total_loss = 0.0
        batches = 0
        for start in range(0, int(currents.shape[0]), max(int(batch_size), 1)):
            index = order[start : start + max(int(batch_size), 1)]
            current = currents.index_select(0, index).to(device)
            target = targets.index_select(0, index).to(device)
            pred = model(current)
            loss = torch.mean((pred - target) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu().item())
            batches += 1
        mean_loss = total_loss / max(batches, 1)
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_state = fno_runner._clone_tensor_state_dict(model)
    model.load_state_dict(best_state)
    model.to("cpu")
    return model, {
        "model": "external_cno1d_baseline",
        "implementation": CNO_IMPLEMENTATION,
        "source_url": CNO_SOURCE_URL,
        "source_commit": CNO_SOURCE_COMMIT,
        "source_path": CNO_SOURCE_PATH,
        "train_frames": int(currents.shape[0]),
        "channels": int(currents.shape[1]),
        "height": int(currents.shape[2]),
        "width": int(currents.shape[3]),
        "n_layers": int(n_layers),
        "n_res": int(n_res),
        "n_res_neck": int(n_res_neck),
        "channel_multiplier": int(channel_multiplier),
        "lift_latent_dim": int(lift_latent_dim),
        "use_bn": bool(use_bn),
        "residual": bool(residual),
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "batch_size": int(batch_size),
        "train_mse": best_loss,
    }


def train_cno_groups(
    grouped_pairs: dict[tuple[str, int, int, int], tuple[torch.Tensor, torch.Tensor]],
    *,
    n_layers: int,
    n_res: int,
    n_res_neck: int,
    channel_multiplier: int,
    lift_latent_dim: int,
    use_bn: bool,
    residual: bool,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    seed: int,
    device: str | torch.device,
    cno_cls: type[nn.Module] = CNO1d,
) -> tuple[dict[tuple[str, int, int, int], nn.Module], dict[str, Any]]:
    models: dict[tuple[str, int, int, int], nn.Module] = {}
    fit: dict[str, Any] = {"groups": {}}
    for offset, (key, (currents, targets)) in enumerate(sorted(grouped_pairs.items())):
        model, group_fit = train_cno_group_model(
            currents,
            targets,
            n_layers=n_layers,
            n_res=n_res,
            n_res_neck=n_res_neck,
            channel_multiplier=channel_multiplier,
            lift_latent_dim=lift_latent_dim,
            use_bn=use_bn,
            residual=residual,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            seed=seed + offset,
            device=device,
            cno_cls=cno_cls,
        )
        models[key] = model
        fit["groups"][str(key)] = group_fit
    fit["group_count"] = len(models)
    fit["train_frames"] = sum(int(pair[0].shape[0]) for pair in grouped_pairs.values())
    return models, fit


def _external_test_measurement_key(
    *,
    args: argparse.Namespace,
    tasks: Sequence[str],
) -> str:
    payload = {
        "adapter": "external_cno1d_baseline",
        "batch_size": args.batch_size,
        "channel_multiplier": args.channel_multiplier,
        "config": args.config,
        "data_root": args.data_root,
        "data_sources": fno_runner._external_data_sources(args, tasks),
        "device": args.device,
        "eval_split": args.eval_split,
        "epochs": args.epochs,
        "implementation": CNO_IMPLEMENTATION,
        "learning_rate": args.learning_rate,
        "lift_latent_dim": args.lift_latent_dim,
        "max_eval_samples": args.max_eval_samples,
        "max_pairs_per_task": args.max_pairs_per_task,
        "max_train_samples": args.max_train_samples,
        "metric": args.metric,
        "n_layers": args.n_layers,
        "n_res": args.n_res,
        "n_res_neck": args.n_res_neck,
        "residual": bool(args.residual),
        "strict_contract": bool(getattr(args, "strict_contract", False)),
        "data_lock_path": getattr(args, "data_lock", None),
        "data_lock_sha256": getattr(args, "expected_data_lock_sha256", None),
        "rollout_steps": args.rollout_steps,
        "seed": args.seed,
        "source_commit": CNO_SOURCE_COMMIT,
        "tasks": list(tasks),
        "train_split": args.train_split,
        "train_stride": args.train_stride,
        "use_bn": not bool(args.no_batch_norm),
        "weight_decay": args.weight_decay,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _guard_external_test_measurement(
    *,
    args: argparse.Namespace,
    tasks: Sequence[str],
) -> dict[str, Any]:
    measurement_key = _external_test_measurement_key(args=args, tasks=tasks)
    ledger = fno_runner._load_test_ledger(args.held_out_ledger_json)
    existing_keys = {
        str(entry.get("measurement_key"))
        for entry in ledger.get("measurements", [])
        if isinstance(entry, dict)
    }
    already_recorded = measurement_key in existing_keys
    if already_recorded and not args.allow_repeat_test:
        raise RuntimeError(
            "held-out external CNO1d test measurement already recorded; "
            "set --allow-repeat-test only for explicit debugging repeats"
        )
    return {
        "enabled": True,
        "allow_repeat_test": bool(args.allow_repeat_test),
        "already_recorded": already_recorded,
        "ledger_path": args.held_out_ledger_json,
        "measurement_key": measurement_key,
        "recorded": False,
    }


def _record_external_test_measurement(
    *,
    args: argparse.Namespace,
    tasks: Sequence[str],
    policy: dict[str, Any],
    metrics: dict[str, float],
    summary_path: Path,
) -> bool:
    if not policy.get("enabled") or policy.get("allow_repeat_test"):
        return False
    ledger = fno_runner._load_test_ledger(args.held_out_ledger_json)
    ledger.setdefault("measurements", []).append(
        {
            "adapter": "external_cno1d_baseline",
            "measurement_key": policy["measurement_key"],
            "metric": args.metric,
            "run_name": args.name,
            "summary": str(summary_path),
            "test_metric_value": float(metrics[args.metric]),
            "test_split": args.eval_split,
            "tasks": list(tasks),
        }
    )
    fno_runner._write_test_ledger(args.held_out_ledger_json, ledger)
    return True


def _command_record(args: argparse.Namespace) -> list[str]:
    command = [
        "python",
        "scripts/run_external_cno_baseline.py",
        "--config",
        args.config,
        "--name",
        args.name,
        "--output-root",
        args.output_root,
        "--train-split",
        args.train_split,
        "--eval-split",
        args.eval_split,
        "--max-train-samples",
        str(args.max_train_samples),
        "--max-eval-samples",
        str(args.max_eval_samples),
        "--max-pairs-per-task",
        str(args.max_pairs_per_task),
        "--rollout-steps",
        str(args.rollout_steps),
        "--train-stride",
        str(args.train_stride),
        "--n-layers",
        str(args.n_layers),
        "--n-res",
        str(args.n_res),
        "--n-res-neck",
        str(args.n_res_neck),
        "--channel-multiplier",
        str(args.channel_multiplier),
        "--lift-latent-dim",
        str(args.lift_latent_dim),
        "--epochs",
        str(args.epochs),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--batch-size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--metric",
        args.metric,
        "--held-out-ledger-json",
        args.held_out_ledger_json,
    ]
    if args.data_root:
        command.extend(["--data-root", args.data_root])
    tasks = list(args.tasks or args.task)
    if tasks:
        command.append("--tasks")
        command.extend(str(task) for task in tasks)
    if args.no_batch_norm:
        command.append("--no-batch-norm")
    if args.residual:
        command.append("--residual")
    if args.dry_run:
        command.append("--dry-run")
    if args.allow_held_out_test_eval:
        command.append("--allow-held-out-test-eval")
    if args.allow_repeat_test:
        command.append("--allow-repeat-test")
    return command


def _summary_common(args: argparse.Namespace, *, tasks: Sequence[str]) -> dict[str, Any]:
    return {
        "data_provenance": fno_runner.training_lock_provenance(args),
        "extra": {
            "baseline": "external_cno1d",
            "implementation": CNO_IMPLEMENTATION,
            "source_url": CNO_SOURCE_URL,
            "source_commit": CNO_SOURCE_COMMIT,
            "source_path": CNO_SOURCE_PATH,
            "task": tasks[0] if len(tasks) == 1 else list(tasks),
            "train_split": args.train_split,
            "split": args.eval_split,
            "max_train_samples": args.max_train_samples,
            "max_eval_samples": args.max_eval_samples,
            "max_pairs_per_task": args.max_pairs_per_task,
            "rollout_steps": args.rollout_steps,
            "train_stride": args.train_stride,
            "metric": args.metric,
            "n_layers": args.n_layers,
            "n_res": args.n_res,
            "n_res_neck": args.n_res_neck,
            "channel_multiplier": args.channel_multiplier,
            "lift_latent_dim": args.lift_latent_dim,
            "use_bn": not bool(args.no_batch_norm),
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "device": args.device,
            "residual": bool(args.residual),
            "allow_held_out_test_eval": bool(args.allow_held_out_test_eval),
            "held_out_ledger_reference": args.held_out_ledger_json,
            "command": _command_record(args),
        },
        "checkpoints": {},
        "run_name": args.name,
        "split": args.eval_split,
        "stages": ["external_cno1d_baseline"],
        "config": args.config,
        "eval_config": args.config,
    }


def write_dry_run_summary(args: argparse.Namespace, *, tasks: Sequence[str]) -> Path:
    output_root = Path(args.output_root)
    run_dir = output_root / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    summary = {
        "status": "dry_run",
        "metrics": {},
        **_summary_common(args, tasks=tasks),
        "details": {
            "contract": {
                "published_numbers_directly_comparable": False,
                "official_source_commit": CNO_SOURCE_COMMIT,
                "official_source_path": CNO_SOURCE_PATH,
                "live_test_requires_explicit_flag": True,
                "current_adapter_scope": "CNO1d height=1 light-v1 grids",
            }
        },
        "duration_sec": 0.0,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "dry_run", "summary": str(summary_path)}, indent=2))
    return summary_path


def _append_external_results_row(
    output_root: Path,
    *,
    args: argparse.Namespace,
    metrics: dict[str, float],
    summary_path: Path,
    finished: float,
) -> None:
    main_metric_name, main_metric_value = fno_runner._main_metric(metrics)
    fno_runner._append_results_row(
        output_root / "results.tsv",
        {
            "run_name": args.name,
            "timestamp": int(finished),
            "stages": "external_cno1d_baseline",
            "decoded": True,
            "train_split": args.train_split,
            "eval_split": args.eval_split,
            "transfer_tasks": "",
            "promotion_passed": "",
            "main_metric_name": main_metric_name,
            "main_metric_value": main_metric_value,
            "summary_json": str(summary_path),
        },
    )


def _write_group_manifest(path: Path, fit: dict[str, Any]) -> None:
    fieldnames = [
        "group",
        "model",
        "implementation",
        "source_commit",
        "train_frames",
        "channels",
        "height",
        "width",
        "n_layers",
        "n_res",
        "n_res_neck",
        "channel_multiplier",
        "lift_latent_dim",
        "use_bn",
        "residual",
        "train_mse",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for group_name, group_fit in sorted(fit.get("groups", {}).items()):
            row = {field: group_fit.get(field, "") for field in fieldnames}
            row["group"] = group_name
            writer.writerow(row)


def run_baseline(args: argparse.Namespace) -> Path:
    cfg = fno_runner._load_cfg(args.config)
    fno_runner.bind_training_lock(cfg, args)
    tasks = fno_runner._as_task_names(cfg, args.tasks or args.task)
    if args.dry_run:
        return write_dry_run_summary(args, tasks=tasks)
    if args.eval_split == "test" and not args.allow_held_out_test_eval:
        raise RuntimeError(
            "Live external CNO1d evaluation on split=test requires --allow-held-out-test-eval. "
            "Use --eval-split val while debugging adapter behavior."
        )
    held_out_test_policy = {"enabled": False, "recorded": False}
    if args.eval_split == "test":
        held_out_test_policy = _guard_external_test_measurement(args=args, tasks=tasks)

    started = time.time()
    groups = fno_runner.collect_training_pairs(
        cfg,
        tasks=tasks,
        split=args.train_split,
        data_root=args.data_root,
        max_samples=args.max_train_samples,
        max_pairs_per_task=args.max_pairs_per_task,
        rollout_steps=args.rollout_steps,
        stride=args.train_stride,
    )
    if not groups:
        raise RuntimeError("No training pairs collected for external CNO1d baseline")
    models, fit = train_cno_groups(
        groups,
        n_layers=args.n_layers,
        n_res=args.n_res,
        n_res_neck=args.n_res_neck,
        channel_multiplier=args.channel_multiplier,
        lift_latent_dim=args.lift_latent_dim,
        use_bn=not args.no_batch_norm,
        residual=args.residual,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )
    metrics = fno_runner.evaluate_external_fno_baseline(
        cfg,
        models,
        tasks=tasks,
        split=args.eval_split,
        data_root=args.data_root,
        max_samples=args.max_eval_samples,
        rollout_steps=args.rollout_steps,
        device=args.device,
        strict_contract=bool(getattr(args, "strict_contract", False)),
    )
    finished = time.time()
    output_root = Path(args.output_root)
    run_dir = output_root / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    group_manifest = run_dir / "cno_groups.tsv"
    _write_group_manifest(group_manifest, fit)
    summary_path = run_dir / "summary.json"
    held_out_test_policy["recorded"] = _record_external_test_measurement(
        args=args,
        tasks=tasks,
        policy=held_out_test_policy,
        metrics=metrics,
        summary_path=summary_path,
    )
    summary = {
        "status": "complete",
        "metrics": metrics,
        **_summary_common(args, tasks=tasks),
        "details": {
            "fit": fit,
            "group_manifest": str(group_manifest),
            "held_out_test_policy": held_out_test_policy,
        },
        "held_out_test_policy": held_out_test_policy,
        "duration_sec": finished - started,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _append_external_results_row(
        output_root,
        args=args,
        metrics=metrics,
        summary_path=summary_path,
        finished=finished,
    )
    main_metric_name, main_metric_value = fno_runner._main_metric(metrics)
    print(
        json.dumps(
            {"summary": str(summary_path), "main_metric": {main_metric_name: main_metric_value}},
            indent=2,
        )
    )
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train/evaluate an external official CNO1d architecture baseline"
    )
    parser.add_argument("--config", default="configs/train_multitask_heterogeneous_light_best.yaml")
    parser.add_argument("--name", default="external_cno1d_light_val")
    parser.add_argument("--output-root", default="reports/research/sota_loop/external_baselines")
    parser.add_argument("--data-root")
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--tasks", nargs="+", default=[])
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="val")
    parser.add_argument("--max-train-samples", type=int, default=2)
    parser.add_argument("--max-eval-samples", type=int, default=2)
    parser.add_argument("--max-pairs-per-task", type=int, default=64)
    parser.add_argument("--rollout-steps", type=int, default=16)
    parser.add_argument("--train-stride", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-res", type=int, default=1)
    parser.add_argument("--n-res-neck", type=int, default=1)
    parser.add_argument("--channel-multiplier", type=int, default=8)
    parser.add_argument("--lift-latent-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--metric", default="decoded_rollout_nrmse")
    parser.add_argument(
        "--held-out-ledger-json",
        default="reports/research/sota_loop/external_baselines/test_ledger.json",
    )
    parser.add_argument("--no-batch-norm", action="store_true")
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--strict-contract", action="store_true")
    parser.add_argument("--data-lock")
    parser.add_argument("--expected-data-lock-sha256")
    parser.add_argument("--allow-held-out-test-eval", action="store_true")
    parser.add_argument("--allow-repeat-test", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_baseline(args)


if __name__ == "__main__":
    main()
