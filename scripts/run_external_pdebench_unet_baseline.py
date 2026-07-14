#!/usr/bin/env python
from __future__ import annotations

"""Train/evaluate an external PDEBench U-Net architecture baseline on light-v1.

The U-Net classes in this file are adapted from:
https://github.com/pdebench/PDEBench/blob/4ff3e3a4aa1561721b5571fa3a048a0a463e0568/pdebench/models/unet/unet.py

The upstream PDEBench U-Net source includes the following notice:

MIT License

Copyright (c) 2019 mateuszbuda

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
"""

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts import run_external_neuraloperator_fno_baseline as fno_runner

PDEBENCH_UNET_IMPLEMENTATION = "pdebench.models.unet.unet.UNet1d/UNet2d"
PDEBENCH_SOURCE_URL = "https://github.com/pdebench/PDEBench"
PDEBENCH_SOURCE_COMMIT = "4ff3e3a4aa1561721b5571fa3a048a0a463e0568"
PDEBENCH_UNET_SOURCE_PATH = "pdebench/models/unet/unet.py"


class PDEBenchUNet1d(nn.Module):
    """PDEBench U-Net 1D architecture adapted from the official PDEBench source."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1, init_features: int = 32):
        super().__init__()
        features = int(init_features)
        self.encoder1 = self._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder2 = self._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder3 = self._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder4 = self._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.bottleneck = self._block(features * 8, features * 16, name="bottleneck")
        self.upconv4 = nn.ConvTranspose1d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = self._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose1d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = self._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose1d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose1d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self._block(features * 2, features, name="dec1")
        self.conv = nn.Conv1d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        bottleneck = self.bottleneck(self.pool4(enc4))
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.conv(dec1)

    @staticmethod
    def _block(in_channels: int, features: int, name: str) -> nn.Sequential:
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm1d(num_features=features)),
                    (name + "tanh1", nn.Tanh()),
                    (
                        name + "conv2",
                        nn.Conv1d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm1d(num_features=features)),
                    (name + "tanh2", nn.Tanh()),
                ]
            )
        )


class PDEBenchUNet2d(nn.Module):
    """PDEBench U-Net 2D architecture adapted from the official PDEBench source."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1, init_features: int = 32):
        super().__init__()
        features = int(init_features)
        self.encoder1 = self._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = self._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = self._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = self._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = self._block(features * 8, features * 16, name="bottleneck")
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = self._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = self._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self._block(features * 2, features, name="dec1")
        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        bottleneck = self.bottleneck(self.pool4(enc4))
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.conv(dec1)

    @staticmethod
    def _block(in_channels: int, features: int, name: str) -> nn.Sequential:
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "tanh1", nn.Tanh()),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "tanh2", nn.Tanh()),
                ]
            )
        )


class PDEBenchUNetGridAdapter(nn.Module):
    """Adapt official PDEBench U-Net tensors to the repo's standard (B,C,H,W) grid."""

    def __init__(
        self,
        model: nn.Module,
        *,
        grid_shape: tuple[int, int],
        residual: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.grid_shape = (int(grid_shape[0]), int(grid_shape[1]))
        self.residual = bool(residual)

    def forward(self, current: torch.Tensor) -> torch.Tensor:
        height, _ = self.grid_shape
        if height == 1:
            pred = self.model(current.squeeze(-2)).unsqueeze(-2)
        else:
            pred = self.model(current)
        if self.residual:
            pred = current + pred
        return pred


def _check_unet_grid_shape(grid_shape: tuple[int, int]) -> None:
    height, width = int(grid_shape[0]), int(grid_shape[1])
    if height == 1:
        if width < 16 or width % 16 != 0:
            raise ValueError(f"PDEBench UNet1d requires width divisible by 16, got {width}")
    elif height < 16 or width < 16 or height % 16 != 0 or width % 16 != 0:
        raise ValueError(
            f"PDEBench UNet2d requires height and width divisible by 16, got {(height, width)}"
        )


def build_pdebench_unet_model(
    *,
    channels: int,
    grid_shape: tuple[int, int],
    init_features: int,
    residual: bool,
    unet1d_cls: type[nn.Module] = PDEBenchUNet1d,
    unet2d_cls: type[nn.Module] = PDEBenchUNet2d,
) -> nn.Module:
    _check_unet_grid_shape(grid_shape)
    height, _ = int(grid_shape[0]), int(grid_shape[1])
    if height == 1:
        model = unet1d_cls(
            in_channels=int(channels),
            out_channels=int(channels),
            init_features=int(init_features),
        )
    else:
        model = unet2d_cls(
            in_channels=int(channels),
            out_channels=int(channels),
            init_features=int(init_features),
        )
    return PDEBenchUNetGridAdapter(model, grid_shape=grid_shape, residual=residual)


def train_unet_group_model(
    currents: torch.Tensor,
    targets: torch.Tensor,
    *,
    init_features: int = 32,
    residual: bool = False,
    epochs: int = 5,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 8,
    seed: int = 0,
    device: str | torch.device = "cpu",
    unet1d_cls: type[nn.Module] = PDEBenchUNet1d,
    unet2d_cls: type[nn.Module] = PDEBenchUNet2d,
) -> tuple[nn.Module, dict[str, Any]]:
    if currents.shape != targets.shape:
        raise ValueError("currents and targets must have the same shape")
    if currents.dim() != 4:
        raise ValueError(f"Expected training tensors shaped (N,C,H,W), got {tuple(currents.shape)}")

    torch.manual_seed(int(seed))
    device = torch.device(device)
    grid_shape = (int(currents.shape[2]), int(currents.shape[3]))
    model = build_pdebench_unet_model(
        channels=int(currents.shape[1]),
        grid_shape=grid_shape,
        init_features=init_features,
        residual=residual,
        unet1d_cls=unet1d_cls,
        unet2d_cls=unet2d_cls,
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
        "model": "external_pdebench_unet_baseline",
        "implementation": PDEBENCH_UNET_IMPLEMENTATION,
        "source_url": PDEBENCH_SOURCE_URL,
        "source_commit": PDEBENCH_SOURCE_COMMIT,
        "source_path": PDEBENCH_UNET_SOURCE_PATH,
        "train_frames": int(currents.shape[0]),
        "channels": int(currents.shape[1]),
        "height": int(currents.shape[2]),
        "width": int(currents.shape[3]),
        "init_features": int(init_features),
        "residual": bool(residual),
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "batch_size": int(batch_size),
        "train_mse": best_loss,
    }


def train_unet_groups(
    grouped_pairs: dict[tuple[str, int, int, int], tuple[torch.Tensor, torch.Tensor]],
    *,
    init_features: int,
    residual: bool,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    seed: int,
    device: str | torch.device,
    unet1d_cls: type[nn.Module] = PDEBenchUNet1d,
    unet2d_cls: type[nn.Module] = PDEBenchUNet2d,
) -> tuple[dict[tuple[str, int, int, int], nn.Module], dict[str, Any]]:
    models: dict[tuple[str, int, int, int], nn.Module] = {}
    fit: dict[str, Any] = {"groups": {}}
    for offset, (key, (currents, targets)) in enumerate(sorted(grouped_pairs.items())):
        model, group_fit = train_unet_group_model(
            currents,
            targets,
            init_features=init_features,
            residual=residual,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            seed=seed + offset,
            device=device,
            unet1d_cls=unet1d_cls,
            unet2d_cls=unet2d_cls,
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
        "adapter": "external_pdebench_unet_baseline",
        "batch_size": args.batch_size,
        "config": args.config,
        "data_root": args.data_root,
        "data_sources": fno_runner._external_data_sources(args, tasks),
        "device": args.device,
        "eval_split": args.eval_split,
        "epochs": args.epochs,
        "implementation": PDEBENCH_UNET_IMPLEMENTATION,
        "init_features": args.init_features,
        "learning_rate": args.learning_rate,
        "max_eval_samples": args.max_eval_samples,
        "max_pairs_per_task": args.max_pairs_per_task,
        "max_train_samples": args.max_train_samples,
        "metric": args.metric,
        "residual": bool(args.residual),
        "strict_contract": bool(getattr(args, "strict_contract", False)),
        "data_lock_path": getattr(args, "data_lock", None),
        "data_lock_sha256": getattr(args, "expected_data_lock_sha256", None),
        "rollout_steps": args.rollout_steps,
        "seed": args.seed,
        "source_commit": PDEBENCH_SOURCE_COMMIT,
        "tasks": list(tasks),
        "train_split": args.train_split,
        "train_stride": args.train_stride,
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
            "held-out external PDEBench U-Net test measurement already recorded; "
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
            "adapter": "external_pdebench_unet_baseline",
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
        "scripts/run_external_pdebench_unet_baseline.py",
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
        "--init-features",
        str(args.init_features),
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
            "baseline": "external_pdebench_unet",
            "implementation": PDEBENCH_UNET_IMPLEMENTATION,
            "source_url": PDEBENCH_SOURCE_URL,
            "source_commit": PDEBENCH_SOURCE_COMMIT,
            "source_path": PDEBENCH_UNET_SOURCE_PATH,
            "task": tasks[0] if len(tasks) == 1 else list(tasks),
            "train_split": args.train_split,
            "split": args.eval_split,
            "max_train_samples": args.max_train_samples,
            "max_eval_samples": args.max_eval_samples,
            "max_pairs_per_task": args.max_pairs_per_task,
            "rollout_steps": args.rollout_steps,
            "train_stride": args.train_stride,
            "metric": args.metric,
            "init_features": args.init_features,
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
        "stages": ["external_pdebench_unet_baseline"],
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
                "official_source_commit": PDEBENCH_SOURCE_COMMIT,
                "live_test_requires_explicit_flag": True,
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
            "stages": "external_pdebench_unet_baseline",
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
        "init_features",
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
            "Live external PDEBench U-Net evaluation on split=test requires "
            "--allow-held-out-test-eval. Use --eval-split val while debugging adapter behavior."
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
        raise RuntimeError("No training pairs collected for external PDEBench U-Net baseline")
    models, fit = train_unet_groups(
        groups,
        init_features=args.init_features,
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
    group_manifest = run_dir / "pdebench_unet_groups.tsv"
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
        description="Train/evaluate an external PDEBench U-Net architecture baseline"
    )
    parser.add_argument("--config", default="configs/train_multitask_heterogeneous_light_best.yaml")
    parser.add_argument("--name", default="external_pdebench_unet_light_val")
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
    parser.add_argument("--init-features", type=int, default=32)
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
