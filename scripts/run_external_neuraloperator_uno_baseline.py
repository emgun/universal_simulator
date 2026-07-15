#!/usr/bin/env python
from __future__ import annotations

"""Train/evaluate an optional NeuralOperator UNO baseline on the light-v1 protocol."""

import argparse
import contextlib
import csv
import hashlib
import importlib.metadata as metadata
import io
import json
import math
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts import run_external_neuraloperator_fno_baseline as fno_runner

NEURALOP_IMPORT = "neuralop.models.UNO"
NEURALOP_SOURCE_URL = "https://github.com/neuraloperator/neuraloperator"


class MissingNeuralOperatorError(RuntimeError):
    """Raised when a live external UNO run is requested without neuralop installed."""


class UNOGridAdapter(nn.Module):
    """Adapt NeuralOperator UNO's ND tensors to the repo's standard (B,C,H,W) grid."""

    def __init__(
        self, uno: nn.Module, *, grid_shape: tuple[int, int], residual: bool = False
    ) -> None:
        super().__init__()
        self.uno = uno
        self.grid_shape = (int(grid_shape[0]), int(grid_shape[1]))
        self.residual = bool(residual)

    def forward(self, current: torch.Tensor) -> torch.Tensor:
        height, _ = self.grid_shape
        if height == 1:
            pred = self.uno(current.squeeze(-2)).unsqueeze(-2)
        else:
            pred = self.uno(current)
        if self.residual:
            pred = current + pred
        return pred


def neuraloperator_import_status() -> dict[str, Any]:
    try:
        from neuralop.models import UNO  # noqa: F401
    except Exception as exc:
        return {
            "available": False,
            "import": NEURALOP_IMPORT,
            "source_url": NEURALOP_SOURCE_URL,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    version = "unknown"
    for package_name in ("neuraloperator", "neuralop"):
        try:
            version = metadata.version(package_name)
            break
        except metadata.PackageNotFoundError:
            continue
    return {
        "available": True,
        "import": NEURALOP_IMPORT,
        "source_url": NEURALOP_SOURCE_URL,
        "version": version,
    }


def load_neuraloperator_uno_class() -> type[nn.Module]:
    try:
        from neuralop.models import UNO
    except Exception as exc:
        raise MissingNeuralOperatorError(
            "neuralop.models.UNO is required for a live external UNO run. "
            "Install the optional NeuralOperator package or run with --dry-run."
        ) from exc
    return UNO


def uno_modes_for_grid(grid_shape: tuple[int, int], requested_modes: int) -> list[int]:
    height, width = int(grid_shape[0]), int(grid_shape[1])
    modes = max(int(requested_modes), 1)
    if height == 1:
        return [min(modes, max(1, width // 2))]
    return [
        min(modes, max(1, height // 2)),
        min(modes, max(1, width // 2)),
    ]


def uno_scalings_for_grid(
    grid_shape: tuple[int, int],
    n_layers: int,
    *,
    identity_scaling: bool,
) -> list[list[float]]:
    height, _ = int(grid_shape[0]), int(grid_shape[1])
    dims = 1 if height == 1 else 2
    layer_count = max(int(n_layers), 1)
    unit = [1.0] * dims
    if identity_scaling or layer_count == 1:
        return [unit[:] for _ in range(layer_count)]
    if layer_count == 2:
        return [[0.5] * dims, [2.0] * dims]
    return [unit[:], [0.5] * dims, *[unit[:] for _ in range(layer_count - 3)], [2.0] * dims]


def build_neuraloperator_uno_model(
    *,
    channels: int | None = None,
    in_channels: int | None = None,
    out_channels: int | None = None,
    grid_shape: tuple[int, int],
    hidden_channels: int,
    fourier_modes: int,
    n_layers: int,
    lifting_channels: int,
    projection_channels: int,
    channel_mlp_skip: str,
    identity_scaling: bool,
    residual: bool,
    uno_cls: type[nn.Module] | None = None,
) -> nn.Module:
    if channels is not None:
        if in_channels is not None or out_channels is not None:
            raise ValueError("channels cannot be combined with explicit input/output channels")
        in_channels = out_channels = int(channels)
    if in_channels is None or out_channels is None:
        raise ValueError("channels or both in_channels and out_channels are required")
    if residual and int(in_channels) != int(out_channels):
        raise ValueError("residual UNO requires equal input and output channels")
    uno_cls = uno_cls or load_neuraloperator_uno_class()
    modes = uno_modes_for_grid(grid_shape, fourier_modes)
    scalings = uno_scalings_for_grid(
        grid_shape,
        n_layers,
        identity_scaling=identity_scaling,
    )
    kwargs = {
        "in_channels": int(in_channels),
        "out_channels": int(out_channels),
        "hidden_channels": int(hidden_channels),
        "lifting_channels": int(lifting_channels),
        "projection_channels": int(projection_channels),
        "n_layers": int(n_layers),
        "uno_out_channels": [int(hidden_channels)] * int(n_layers),
        "uno_n_modes": [modes[:] for _ in range(int(n_layers))],
        "uno_scalings": scalings,
        "positional_embedding": "grid",
        "channel_mlp_skip": str(channel_mlp_skip),
    }
    with contextlib.redirect_stdout(io.StringIO()):
        uno = uno_cls(**kwargs)
    return UNOGridAdapter(uno, grid_shape=grid_shape, residual=residual)


def train_uno_group_model(
    currents: torch.Tensor,
    targets: torch.Tensor,
    *,
    hidden_channels: int = 32,
    fourier_modes: int = 16,
    n_layers: int = 4,
    lifting_channels: int = 64,
    projection_channels: int = 64,
    channel_mlp_skip: str = "linear",
    identity_scaling: bool = False,
    residual: bool = False,
    epochs: int = 5,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 8,
    seed: int = 0,
    device: str | torch.device = "cpu",
    uno_cls: type[nn.Module] | None = None,
    checkpoint_epochs: Sequence[int] = (),
    checkpoint_states: dict[int, dict[str, torch.Tensor]] | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    if currents.shape != targets.shape:
        raise ValueError("currents and targets must have the same shape")
    if currents.dim() != 4:
        raise ValueError(f"Expected training tensors shaped (N,C,H,W), got {tuple(currents.shape)}")

    torch.manual_seed(int(seed))
    device = torch.device(device)
    grid_shape = (int(currents.shape[2]), int(currents.shape[3]))
    model = build_neuraloperator_uno_model(
        channels=int(currents.shape[1]),
        grid_shape=grid_shape,
        hidden_channels=hidden_channels,
        fourier_modes=fourier_modes,
        n_layers=n_layers,
        lifting_channels=lifting_channels,
        projection_channels=projection_channels,
        channel_mlp_skip=channel_mlp_skip,
        identity_scaling=identity_scaling,
        residual=residual,
        uno_cls=uno_cls,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    generator = torch.Generator().manual_seed(int(seed))
    best_loss = float("inf")
    best_state = fno_runner._clone_tensor_state_dict(model)
    epoch_train_mse: list[float] = []
    best_epoch = 0
    checkpoint_epoch_set = {int(epoch) for epoch in checkpoint_epochs}
    for epoch in range(1, int(epochs) + 1):
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
        epoch_train_mse.append(mean_loss)
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_state = fno_runner._clone_tensor_state_dict(model)
            best_epoch = epoch
        if epoch in checkpoint_epoch_set and checkpoint_states is not None:
            checkpoint_states[epoch] = fno_runner._clone_tensor_state_dict(model)
    model.load_state_dict(best_state)
    model.to("cpu")
    return model, {
        "model": "external_neuraloperator_uno_baseline",
        "implementation": NEURALOP_IMPORT,
        "train_frames": int(currents.shape[0]),
        "channels": int(currents.shape[1]),
        "height": int(currents.shape[2]),
        "width": int(currents.shape[3]),
        "hidden_channels": int(hidden_channels),
        "fourier_modes": int(fourier_modes),
        "uno_n_modes": [uno_modes_for_grid(grid_shape, fourier_modes)] * int(n_layers),
        "uno_scalings": uno_scalings_for_grid(
            grid_shape,
            n_layers,
            identity_scaling=identity_scaling,
        ),
        "n_layers": int(n_layers),
        "lifting_channels": int(lifting_channels),
        "projection_channels": int(projection_channels),
        "channel_mlp_skip": str(channel_mlp_skip),
        "identity_scaling": bool(identity_scaling),
        "residual": bool(residual),
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "batch_size": int(batch_size),
        "train_mse": best_loss,
        "best_epoch": best_epoch,
        "epoch_train_mse": epoch_train_mse,
        "optimizer_steps": int(math.ceil(currents.shape[0] / max(int(batch_size), 1)))
        * int(epochs),
        "examples_seen": int(currents.shape[0]) * int(epochs),
    }


def train_uno_groups(
    grouped_pairs: dict[tuple[str, int, int, int], tuple[torch.Tensor, torch.Tensor]],
    *,
    hidden_channels: int,
    fourier_modes: int,
    n_layers: int,
    lifting_channels: int,
    projection_channels: int,
    channel_mlp_skip: str,
    identity_scaling: bool,
    residual: bool,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    seed: int,
    device: str | torch.device,
    uno_cls: type[nn.Module] | None = None,
) -> tuple[dict[tuple[str, int, int, int], nn.Module], dict[str, Any]]:
    models: dict[tuple[str, int, int, int], nn.Module] = {}
    fit: dict[str, Any] = {"groups": {}}
    for offset, (key, (currents, targets)) in enumerate(sorted(grouped_pairs.items())):
        model, group_fit = train_uno_group_model(
            currents,
            targets,
            hidden_channels=hidden_channels,
            fourier_modes=fourier_modes,
            n_layers=n_layers,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            channel_mlp_skip=channel_mlp_skip,
            identity_scaling=identity_scaling,
            residual=residual,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            seed=seed + offset,
            device=device,
            uno_cls=uno_cls,
        )
        models[key] = model
        fit["groups"][str(key)] = group_fit
    fit["group_count"] = len(models)
    fit["train_frames"] = sum(int(pair[0].shape[0]) for pair in grouped_pairs.values())
    fit["optimizer_steps"] = sum(int(group["optimizer_steps"]) for group in fit["groups"].values())
    fit["examples_seen"] = sum(int(group["examples_seen"]) for group in fit["groups"].values())
    return models, fit


def train_uno_groups_with_rungs(
    grouped_pairs: dict[tuple[str, int, int, int], tuple[torch.Tensor, torch.Tensor]],
    *,
    validation_rungs: Sequence[int],
    hidden_channels: int,
    fourier_modes: int,
    n_layers: int,
    lifting_channels: int,
    projection_channels: int,
    channel_mlp_skip: str,
    identity_scaling: bool,
    residual: bool,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    seed: int,
    device: str | torch.device,
    uno_cls: type[nn.Module] | None = None,
) -> tuple[
    dict[tuple[str, int, int, int], nn.Module],
    dict[str, Any],
    dict[int, dict[tuple[str, int, int, int], nn.Module]],
]:
    """Train each specialist once and retain a common model ensemble at every rung."""

    rungs = tuple(int(epoch) for epoch in validation_rungs)
    models: dict[tuple[str, int, int, int], nn.Module] = {}
    rung_models: dict[int, dict[tuple[str, int, int, int], nn.Module]] = {
        epoch: {} for epoch in rungs
    }
    fit: dict[str, Any] = {"groups": {}}
    for offset, (key, (currents, targets)) in enumerate(sorted(grouped_pairs.items())):
        snapshots: dict[int, dict[str, torch.Tensor]] = {}
        model, group_fit = train_uno_group_model(
            currents,
            targets,
            hidden_channels=hidden_channels,
            fourier_modes=fourier_modes,
            n_layers=n_layers,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            channel_mlp_skip=channel_mlp_skip,
            identity_scaling=identity_scaling,
            residual=residual,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            seed=seed + offset,
            device=device,
            uno_cls=uno_cls,
            checkpoint_epochs=rungs,
            checkpoint_states=snapshots,
        )
        missing = sorted(set(rungs).difference(snapshots))
        if missing:
            raise RuntimeError(f"UNO training did not produce requested rungs: {missing}")
        models[key] = model
        for epoch in rungs:
            rung_models[epoch][key] = fno_runner._clone_module_at_state(model, snapshots[epoch])
        fit["groups"][str(key)] = group_fit
    fit["group_count"] = len(models)
    fit["train_frames"] = sum(int(pair[0].shape[0]) for pair in grouped_pairs.values())
    fit["optimizer_steps"] = sum(int(group["optimizer_steps"]) for group in fit["groups"].values())
    fit["examples_seen"] = sum(int(group["examples_seen"]) for group in fit["groups"].values())
    return models, fit, rung_models


def _external_test_measurement_key(
    *,
    args: argparse.Namespace,
    tasks: Sequence[str],
) -> str:
    payload = {
        "adapter": "external_neuraloperator_uno_baseline",
        "batch_size": args.batch_size,
        "channel_mlp_skip": args.channel_mlp_skip,
        "config": args.config,
        "data_root": args.data_root,
        "data_sources": fno_runner._external_data_sources(args, tasks),
        "device": args.device,
        "eval_split": args.eval_split,
        "epochs": args.epochs,
        "fourier_modes": args.fourier_modes,
        "hidden_channels": args.hidden_channels,
        "identity_scaling": bool(args.identity_scaling),
        "implementation": NEURALOP_IMPORT,
        "learning_rate": args.learning_rate,
        "lifting_channels": args.lifting_channels,
        "max_eval_samples": args.max_eval_samples,
        "max_pairs_per_task": args.max_pairs_per_task,
        "max_train_samples": args.max_train_samples,
        "metric": args.metric,
        "n_layers": args.n_layers,
        "projection_channels": args.projection_channels,
        "residual": bool(args.residual),
        "strict_contract": bool(getattr(args, "strict_contract", False)),
        "data_lock_path": getattr(args, "data_lock", None),
        "data_lock_sha256": getattr(args, "expected_data_lock_sha256", None),
        "rollout_steps": args.rollout_steps,
        "seed": args.seed,
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
            "held-out external UNO test measurement already recorded; "
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
            "adapter": "external_neuraloperator_uno_baseline",
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
        "scripts/run_external_neuraloperator_uno_baseline.py",
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
        "--hidden-channels",
        str(args.hidden_channels),
        "--fourier-modes",
        str(args.fourier_modes),
        "--n-layers",
        str(args.n_layers),
        "--lifting-channels",
        str(args.lifting_channels),
        "--projection-channels",
        str(args.projection_channels),
        "--channel-mlp-skip",
        args.channel_mlp_skip,
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
    if getattr(args, "strict_contract", False):
        command.append("--strict-contract")
    if getattr(args, "data_lock", None):
        command.extend(["--data-lock", args.data_lock])
    if getattr(args, "expected_data_lock_sha256", None):
        command.extend(["--expected-data-lock-sha256", args.expected_data_lock_sha256])
    if getattr(args, "validation_rungs", None):
        command.append("--validation-rungs")
        command.extend(str(epoch) for epoch in args.validation_rungs)
    if getattr(args, "refuse_overwrite", False):
        command.append("--refuse-overwrite")
    tasks = list(args.tasks or args.task)
    if tasks:
        command.append("--tasks")
        command.extend(str(task) for task in tasks)
    if args.identity_scaling:
        command.append("--identity-scaling")
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
            "baseline": "external_neuraloperator_uno",
            "implementation": NEURALOP_IMPORT,
            "source_url": NEURALOP_SOURCE_URL,
            "neuraloperator": neuraloperator_import_status(),
            "task": tasks[0] if len(tasks) == 1 else list(tasks),
            "train_split": args.train_split,
            "split": args.eval_split,
            "max_train_samples": args.max_train_samples,
            "max_eval_samples": args.max_eval_samples,
            "max_pairs_per_task": args.max_pairs_per_task,
            "rollout_steps": args.rollout_steps,
            "train_stride": args.train_stride,
            "metric": args.metric,
            "hidden_channels": args.hidden_channels,
            "fourier_modes": args.fourier_modes,
            "n_layers": args.n_layers,
            "lifting_channels": args.lifting_channels,
            "projection_channels": args.projection_channels,
            "channel_mlp_skip": args.channel_mlp_skip,
            "identity_scaling": bool(args.identity_scaling),
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "device": args.device,
            "residual": bool(args.residual),
            "validation_rungs": list(getattr(args, "validation_rungs", None) or ()),
            "refuse_overwrite": bool(getattr(args, "refuse_overwrite", False)),
            "allow_held_out_test_eval": bool(args.allow_held_out_test_eval),
            "held_out_ledger_reference": args.held_out_ledger_json,
            "command": _command_record(args),
        },
        "checkpoints": {},
        "run_name": args.name,
        "split": args.eval_split,
        "stages": ["external_neuraloperator_uno_baseline"],
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
                "requires_optional_dependency": "neuraloperator",
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
            "stages": "external_neuraloperator_uno_baseline",
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
        "train_frames",
        "channels",
        "height",
        "width",
        "hidden_channels",
        "fourier_modes",
        "uno_n_modes",
        "uno_scalings",
        "n_layers",
        "lifting_channels",
        "projection_channels",
        "channel_mlp_skip",
        "identity_scaling",
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
            for key in ("uno_n_modes", "uno_scalings"):
                if isinstance(row.get(key), list):
                    row[key] = json.dumps(row[key])
            writer.writerow(row)


def run_baseline(args: argparse.Namespace) -> Path:
    cfg = fno_runner._load_cfg(args.config)
    fno_runner.bind_training_lock(cfg, args)
    tasks = fno_runner._as_task_names(cfg, args.tasks or args.task)
    validation_rungs = fno_runner._validated_rungs(args)
    output_root = Path(args.output_root)
    run_dir = output_root / args.name
    if getattr(args, "refuse_overwrite", False) and run_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run directory: {run_dir}")
    if args.dry_run:
        return write_dry_run_summary(args, tasks=tasks)
    if args.eval_split == "test" and not args.allow_held_out_test_eval:
        raise RuntimeError(
            "Live external UNO evaluation on split=test requires --allow-held-out-test-eval. "
            "Use --eval-split val while debugging adapter behavior."
        )
    held_out_test_policy = {"enabled": False, "recorded": False}
    if args.eval_split == "test":
        held_out_test_policy = _guard_external_test_measurement(args=args, tasks=tasks)

    load_neuraloperator_uno_class()
    fno_runner._begin_compute_tracking(args.device)
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
        raise RuntimeError("No training pairs collected for external UNO baseline")
    training_kwargs = {
        "hidden_channels": args.hidden_channels,
        "fourier_modes": args.fourier_modes,
        "n_layers": args.n_layers,
        "lifting_channels": args.lifting_channels,
        "projection_channels": args.projection_channels,
        "channel_mlp_skip": args.channel_mlp_skip,
        "identity_scaling": args.identity_scaling,
        "residual": args.residual,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": args.device,
    }
    validation_history: list[dict[str, Any]] = []
    rung_models: dict[int, dict[tuple[str, int, int, int], nn.Module]] = {}
    if validation_rungs:
        models, fit, rung_models = train_uno_groups_with_rungs(
            groups, validation_rungs=validation_rungs, **training_kwargs
        )
        for epoch in validation_rungs:
            evaluation_started = time.time()
            rung_metrics = fno_runner.evaluate_external_fno_baseline(
                cfg,
                rung_models[epoch],
                tasks=tasks,
                split=args.eval_split,
                data_root=args.data_root,
                max_samples=args.max_eval_samples,
                rollout_steps=args.rollout_steps,
                device=args.device,
                strict_contract=bool(getattr(args, "strict_contract", False)),
            )
            if args.metric not in rung_metrics:
                raise KeyError(f"selection metric {args.metric!r} is absent at epoch {epoch}")
            metric_value = float(rung_metrics[args.metric])
            validation_history.append(
                {
                    "epoch": epoch,
                    "metric_name": args.metric,
                    "metric_value": metric_value,
                    "metrics": rung_metrics,
                    "duration_sec": time.time() - evaluation_started,
                }
            )
        finite_history = [
            record for record in validation_history if math.isfinite(record["metric_value"])
        ]
        if not finite_history:
            raise RuntimeError("no validation rung produced a finite selection metric")
        selected = min(finite_history, key=lambda record: (record["metric_value"], record["epoch"]))
        selected_epoch = int(selected["epoch"])
        selected_models = rung_models[selected_epoch]
        metrics = selected["metrics"]
    else:
        models, fit = train_uno_groups(groups, **training_kwargs)
        selected_models = models
        selected_epoch = None
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
    run_dir.mkdir(parents=True, exist_ok=True)
    group_manifest = run_dir / "uno_groups.tsv"
    _write_group_manifest(group_manifest, fit)
    if validation_rungs:
        rung_checkpoints = {
            str(epoch): fno_runner.write_group_checkpoint(
                run_dir / f"models_epoch_{epoch}.pt",
                rung_models[epoch],
                model_family="uno",
                fit=fit,
                epoch=epoch,
            )
            for epoch in validation_rungs
        }
        checkpoints = {
            "rungs": rung_checkpoints,
            "selected": dict(rung_checkpoints[str(selected_epoch)]),
        }
    else:
        checkpoint = fno_runner.write_group_checkpoint(
            run_dir / "models.pt", selected_models, model_family="uno", fit=fit
        )
        checkpoints = {"models": checkpoint}
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
            "validation_history": validation_history,
        },
        "held_out_test_policy": held_out_test_policy,
        "duration_sec": finished - started,
        "compute": fno_runner._compute_evidence(
            selected_models,
            fit,
            duration_sec=finished - started,
            device=args.device,
        ),
    }
    summary["checkpoints"] = checkpoints
    if validation_rungs:
        summary["recipe_adequacy"] = {
            "validation_rungs": list(validation_rungs),
            "selection_metric": args.metric,
            "selected_epoch": selected_epoch,
            "selected_metric_value": float(metrics[args.metric]),
            "selection_rule": "minimum_finite_validation_metric_earliest_tie",
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
        description="Train/evaluate an optional external NeuralOperator UNO baseline"
    )
    parser.add_argument("--config", default="configs/train_multitask_heterogeneous_light_best.yaml")
    parser.add_argument("--name", default="external_neuraloperator_uno_light_val")
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
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument("--fourier-modes", type=int, default=16)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--lifting-channels", type=int, default=32)
    parser.add_argument("--projection-channels", type=int, default=32)
    parser.add_argument("--channel-mlp-skip", default="linear")
    parser.add_argument("--identity-scaling", action="store_true")
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
    parser.add_argument("--validation-rungs", nargs="+", type=int)
    parser.add_argument("--refuse-overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_baseline(args)


if __name__ == "__main__":
    main()
