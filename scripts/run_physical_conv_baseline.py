#!/usr/bin/env python
from __future__ import annotations

"""Train and evaluate a physical-space convolutional baseline on PDEBench shards."""

import argparse
import csv
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ups.data.latent_pairs import infer_grid_shape
from ups.data.pdebench import PDEBenchConfig, PDEBenchDataset, get_pdebench_spec
from ups.eval.pdebench_runner import _aggregate_chunk_metrics, _flatten_field_step
from ups.utils.config_loader import load_config_with_includes


class PhysicalConvBaseline(nn.Module):
    """Small residual convolutional one-step predictor in physical space."""

    def __init__(self, *, channels: int, hidden_channels: int = 32) -> None:
        super().__init__()
        self.channels = int(channels)
        self.hidden_channels = int(hidden_channels)
        self.network = nn.Sequential(
            nn.Conv2d(self.channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.hidden_channels, self.channels, kernel_size=1),
        )
        final = self.network[-1]
        if isinstance(final, nn.Conv2d):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(self, current: torch.Tensor) -> torch.Tensor:
        return current + self.network(current)


class PhysicalResidualUNetBaseline(nn.Module):
    """Tiny residual U-Net-style baseline with a width-downsampled skip path."""

    def __init__(self, *, channels: int, hidden_channels: int = 32) -> None:
        super().__init__()
        self.channels = int(channels)
        self.hidden_channels = int(hidden_channels)
        self.input_proj = nn.Sequential(
            nn.Conv2d(self.channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.down = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(
                self.hidden_channels * 2,
                self.hidden_channels * 2,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
        )
        self.up = nn.Sequential(
            nn.Conv2d(
                self.hidden_channels * 3,
                self.hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv2d(self.hidden_channels, self.channels, kernel_size=1),
        )
        final = self.up[-1]
        if isinstance(final, nn.Conv2d):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(self, current: torch.Tensor) -> torch.Tensor:
        skip = self.input_proj(current)
        pooled = F.avg_pool2d(skip, kernel_size=(1, 2), stride=(1, 2), ceil_mode=True)
        low = self.down(pooled)
        upsampled = F.interpolate(low, size=skip.shape[-2:], mode="nearest")
        residual = self.up(torch.cat((skip, upsampled), dim=1))
        return current + residual


def build_physical_model(
    architecture: str,
    *,
    channels: int,
    hidden_channels: int,
) -> nn.Module:
    name = architecture.lower()
    if name == "conv":
        return PhysicalConvBaseline(channels=channels, hidden_channels=hidden_channels)
    if name == "unet":
        return PhysicalResidualUNetBaseline(channels=channels, hidden_channels=hidden_channels)
    raise ValueError(f"Unknown physical baseline architecture: {architecture}")


def field_step_to_grid(field_step: torch.Tensor, grid_shape: tuple[int, int]) -> torch.Tensor:
    """Convert a PDEBench field step to (B,C,H,W)."""

    flat = _flatten_field_step(field_step.float(), grid_shape)
    height, width = grid_shape
    batch, nodes, channels = flat.shape
    if nodes != height * width:
        raise ValueError(f"Flattened field has {nodes} nodes, expected {height * width}")
    return flat.transpose(1, 2).reshape(batch, channels, height, width).contiguous()


def grid_to_flat(grid: torch.Tensor) -> torch.Tensor:
    """Convert (B,C,H,W) to flattened (B,N,C)."""

    if grid.dim() != 4:
        raise ValueError(f"Expected grid shaped (B,C,H,W), got {tuple(grid.shape)}")
    batch, channels, height, width = grid.shape
    return grid.reshape(batch, channels, height * width).transpose(1, 2).contiguous()


def group_key(task: str, grid_shape: tuple[int, int], channels: int) -> tuple[str, int, int, int]:
    return (str(task), int(grid_shape[0]), int(grid_shape[1]), int(channels))


def _as_task_names(cfg: dict[str, Any], tasks: Sequence[str] | None) -> list[str]:
    if tasks:
        return [str(task) for task in tasks]
    task_cfg = cfg.get("data", {}).get("task")
    if isinstance(task_cfg, str):
        return [task_cfg]
    if isinstance(task_cfg, (list, tuple)) and task_cfg:
        return [str(task) for task in task_cfg]
    raise ValueError("data.task must be set or --task must be provided")


def _load_cfg(path: str) -> dict[str, Any]:
    try:
        return load_config_with_includes(path)
    except FileNotFoundError:
        with Path(path).open(encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}


def _dataset(
    cfg: dict[str, Any],
    *,
    task: str,
    split: str,
    data_root: str | None,
    max_samples: int | None,
) -> PDEBenchDataset:
    data_cfg = cfg.get("data", {})
    return PDEBenchDataset(
        PDEBenchConfig(
            task=task,
            split=split,
            root=data_root or data_cfg.get("root"),
            param_keys=tuple(data_cfg.get("param_keys", ())),
            bc_keys=tuple(data_cfg.get("bc_keys", ())),
            max_samples=max_samples,
        )
    )


def collect_training_pairs(
    cfg: dict[str, Any],
    *,
    tasks: Sequence[str],
    split: str,
    data_root: str | None,
    max_samples: int | None,
    max_pairs_per_task: int,
    rollout_steps: int,
    stride: int,
) -> dict[tuple[str, int, int, int], tuple[torch.Tensor, torch.Tensor]]:
    """Collect capped one-step training pairs grouped by (H,W,C)."""

    groups: dict[tuple[str, int, int, int], list[tuple[torch.Tensor, torch.Tensor]]] = {}
    for task in tasks:
        dataset = _dataset(
            cfg,
            task=task,
            split=split,
            data_root=data_root,
            max_samples=max_samples,
        )
        task_pairs = 0
        for sample_idx in range(len(dataset)):
            fields = dataset[sample_idx]["fields"].float()
            grid_shape = infer_grid_shape(fields)
            max_steps = min(int(fields.shape[0]) - 1, int(rollout_steps))
            for step in range(0, max_steps, max(int(stride), 1)):
                current = field_step_to_grid(fields[step], grid_shape)
                target = field_step_to_grid(fields[step + 1], grid_shape)
                key = group_key(task, grid_shape, int(current.shape[1]))
                groups.setdefault(key, []).append((current.squeeze(0), target.squeeze(0)))
                task_pairs += 1
                if task_pairs >= int(max_pairs_per_task):
                    break
            if task_pairs >= int(max_pairs_per_task):
                break

    stacked: dict[tuple[str, int, int, int], tuple[torch.Tensor, torch.Tensor]] = {}
    for key, pairs in groups.items():
        currents, targets = zip(*pairs)
        stacked[key] = (torch.stack(tuple(currents), dim=0), torch.stack(tuple(targets), dim=0))
    return stacked


def train_group_model(
    currents: torch.Tensor,
    targets: torch.Tensor,
    *,
    hidden_channels: int = 32,
    architecture: str = "conv",
    epochs: int = 5,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 8,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    if currents.shape != targets.shape:
        raise ValueError("currents and targets must have the same shape")
    if currents.dim() != 4:
        raise ValueError(f"Expected training tensors shaped (N,C,H,W), got {tuple(currents.shape)}")

    torch.manual_seed(int(seed))
    device = torch.device(device)
    model = build_physical_model(
        architecture,
        channels=int(currents.shape[1]),
        hidden_channels=int(hidden_channels),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    generator = torch.Generator().manual_seed(int(seed))
    best_loss = float("inf")
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
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
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
    model.load_state_dict(best_state)
    model.to("cpu")
    return model, {
        "model": f"physical_{architecture.lower()}_residual_baseline",
        "architecture": architecture.lower(),
        "train_frames": int(currents.shape[0]),
        "channels": int(currents.shape[1]),
        "height": int(currents.shape[2]),
        "width": int(currents.shape[3]),
        "hidden_channels": int(hidden_channels),
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "batch_size": int(batch_size),
        "train_mse": best_loss,
    }


def train_groups(
    grouped_pairs: dict[tuple[str, int, int, int], tuple[torch.Tensor, torch.Tensor]],
    *,
    hidden_channels: int,
    architecture: str,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    seed: int,
    device: str | torch.device,
) -> tuple[dict[tuple[str, int, int, int], nn.Module], dict[str, Any]]:
    models: dict[tuple[str, int, int, int], nn.Module] = {}
    fit: dict[str, Any] = {"groups": {}}
    for offset, (key, (currents, targets)) in enumerate(sorted(grouped_pairs.items())):
        model, group_fit = train_group_model(
            currents,
            targets,
            hidden_channels=hidden_channels,
            architecture=architecture,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            seed=seed + offset,
            device=device,
        )
        models[key] = model
        fit["groups"][str(key)] = group_fit
    fit["group_count"] = len(models)
    fit["train_frames"] = sum(int(pair[0].shape[0]) for pair in grouped_pairs.values())
    return models, fit


def _add_rollout_metrics(
    metrics: dict[str, float],
    *,
    prefix: str,
    pred_chunks: list[torch.Tensor],
    target_chunks: list[torch.Tensor],
) -> None:
    if not pred_chunks:
        return
    stats = _aggregate_chunk_metrics(pred_chunks, target_chunks)
    metrics[f"{prefix}decoded_rollout_nrmse"] = stats["nrmse"]
    metrics[f"{prefix}decoded_rollout_rrmse"] = stats["rrmse"]
    metrics[f"{prefix}decoded_rollout_mse"] = stats["mse"]
    metrics[f"{prefix}decoded_rollout_mae"] = stats["mae"]


def evaluate_physical_conv_baseline(
    cfg: dict[str, Any],
    models: dict[tuple[str, int, int, int], nn.Module],
    *,
    tasks: Sequence[str],
    split: str,
    data_root: str | None,
    max_samples: int | None,
    rollout_steps: int,
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    device = torch.device(device)
    total_pred: list[torch.Tensor] = []
    total_target: list[torch.Tensor] = []
    per_task_pred: dict[str, list[torch.Tensor]] = {}
    per_task_target: dict[str, list[torch.Tensor]] = {}
    per_family_pred: dict[str, list[torch.Tensor]] = {}
    per_family_target: dict[str, list[torch.Tensor]] = {}

    for task in tasks:
        dataset = _dataset(
            cfg,
            task=task,
            split=split,
            data_root=data_root,
            max_samples=max_samples,
        )
        family = get_pdebench_spec(task).family
        for sample_idx in range(len(dataset)):
            fields = dataset[sample_idx]["fields"].float()
            grid_shape = infer_grid_shape(fields)
            max_steps = min(int(fields.shape[0]) - 1, int(rollout_steps))
            for step in range(max_steps):
                current_grid = field_step_to_grid(fields[step], grid_shape)
                key = group_key(task, grid_shape, int(current_grid.shape[1]))
                if key not in models:
                    raise ValueError(f"No trained physical conv baseline for group {key}")
                model = models[key].to(device).eval()
                with torch.no_grad():
                    pred_grid = model(current_grid.to(device)).cpu()
                pred = grid_to_flat(pred_grid)
                target = _flatten_field_step(fields[step + 1].float(), grid_shape).cpu()
                total_pred.append(pred)
                total_target.append(target)
                per_task_pred.setdefault(task, []).append(pred)
                per_task_target.setdefault(task, []).append(target)
                per_family_pred.setdefault(family, []).append(pred)
                per_family_target.setdefault(family, []).append(target)

    if not total_pred:
        raise RuntimeError("Physical conv baseline received no eval pairs")

    stats = _aggregate_chunk_metrics(total_pred, total_target)
    metrics = {
        "decoded_mse": stats["mse"],
        "decoded_mae": stats["mae"],
        "decoded_nrmse": stats["nrmse"],
        "decoded_rrmse": stats["rrmse"],
        "decoded_spectral_energy_error": stats["spectral_energy_error"],
        "decoded_rollout_mse": stats["mse"],
        "decoded_rollout_mae": stats["mae"],
        "decoded_rollout_nrmse": stats["nrmse"],
        "decoded_rollout_rrmse": stats["rrmse"],
        "decoded_rollout_spectral_energy_error": stats["spectral_energy_error"],
        "mse": stats["mse"],
        "mae": stats["mae"],
        "rmse": stats["mse"] ** 0.5,
    }
    for task, pred_chunks in per_task_pred.items():
        _add_rollout_metrics(
            metrics,
            prefix=f"task_{task}_",
            pred_chunks=pred_chunks,
            target_chunks=per_task_target[task],
        )
    for family, pred_chunks in per_family_pred.items():
        _add_rollout_metrics(
            metrics,
            prefix=f"family_{family}_",
            pred_chunks=pred_chunks,
            target_chunks=per_family_target[family],
        )
    return metrics


def _main_metric(metrics: dict[str, float]) -> tuple[str, float]:
    for key in ("decoded_rollout_nrmse", "decoded_step1_nrmse", "mse"):
        if key in metrics:
            return key, float(metrics[key])
    first = next(iter(metrics))
    return first, float(metrics[first])


def _append_results_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_name",
        "timestamp",
        "stages",
        "decoded",
        "train_split",
        "eval_split",
        "transfer_tasks",
        "promotion_passed",
        "main_metric_name",
        "main_metric_value",
        "summary_json",
    ]
    row_map: dict[str, dict[str, Any]] = {}
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for name in reader.fieldnames or []:
                if name not in fieldnames:
                    fieldnames.append(name)
            for existing in reader:
                run_name = existing.get("run_name")
                if run_name:
                    row_map[run_name] = dict(existing)
    row_map[str(row["run_name"])] = row
    for name in row:
        if name not in fieldnames:
            fieldnames.append(name)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for name in sorted(row_map):
            payload = row_map[name]
            writer.writerow({field: payload.get(field, "") for field in fieldnames})


def run_baseline(args: argparse.Namespace) -> Path:
    cfg = _load_cfg(args.config)
    tasks = _as_task_names(cfg, args.task)
    started = time.time()
    groups = collect_training_pairs(
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
        raise RuntimeError("No training pairs collected for physical conv baseline")
    models, fit = train_groups(
        groups,
        hidden_channels=args.hidden_channels,
        architecture=args.architecture,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )
    metrics = evaluate_physical_conv_baseline(
        cfg,
        models,
        tasks=tasks,
        split=args.eval_split,
        data_root=args.data_root,
        max_samples=args.max_eval_samples,
        rollout_steps=args.rollout_steps,
        device=args.device,
    )
    finished = time.time()
    output_root = Path(args.output_root)
    run_dir = output_root / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    summary = {
        "metrics": metrics,
        "extra": {
            "baseline": f"physical_{args.architecture}",
            "architecture": args.architecture,
            "task": tasks[0] if len(tasks) == 1 else list(tasks),
            "train_split": args.train_split,
            "split": args.eval_split,
            "group_count": fit["group_count"],
            "train_frames": fit["train_frames"],
        },
        "details": {"fit": fit},
        "checkpoints": {},
        "run_name": args.name,
        "split": args.eval_split,
        "stages": ["physical_conv_baseline"],
        "config": args.config,
        "eval_config": args.config,
        "duration_sec": finished - started,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    main_metric_name, main_metric_value = _main_metric(metrics)
    _append_results_row(
        output_root / "results.tsv",
        {
            "run_name": args.name,
            "timestamp": int(finished),
            "stages": "physical_conv_baseline",
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
    print(
        json.dumps(
            {"summary": str(summary_path), "main_metric": {main_metric_name: main_metric_value}},
            indent=2,
        )
    )
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate a physical ConvNet baseline")
    parser.add_argument("--config", default="configs/train_multitask_heterogeneous_light_best.yaml")
    parser.add_argument("--name", default="physical_conv_light_val")
    parser.add_argument("--output-root", default="reports/physical_baselines")
    parser.add_argument("--data-root")
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="val")
    parser.add_argument("--max-train-samples", type=int, default=2)
    parser.add_argument("--max-eval-samples", type=int, default=2)
    parser.add_argument("--max-pairs-per-task", type=int, default=64)
    parser.add_argument("--rollout-steps", type=int, default=16)
    parser.add_argument("--train-stride", type=int, default=4)
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument("--architecture", choices=["conv", "unet"], default="conv")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run_baseline(args)


if __name__ == "__main__":
    main()
