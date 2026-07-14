#!/usr/bin/env python
from __future__ import annotations

"""Train/evaluate an optional NeuralOperator FNO baseline on the light-v1 protocol."""

import argparse
import copy
import csv
import hashlib
import importlib.metadata as metadata
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

from scripts.run_physical_conv_baseline import (
    _add_rollout_metrics,
    _append_results_row,
    _as_task_names,
    _dataset,
    _load_cfg,
    _main_metric,
    collect_training_pairs,
    field_step_to_grid,
    grid_to_flat,
    group_key,
)
from ups.data.latent_pairs import infer_grid_shape
from ups.data.manifests import canonical_sha256, load_data_lock
from ups.data.pdebench import get_pdebench_spec
from ups.eval.pdebench_runner import _aggregate_chunk_metrics, _flatten_field_step
from ups.eval.persistence_baselines import _regime_label, _regime_slug, _regime_value
from ups.eval.regime_metrics import aligned_element_count, global_scale_regime_nrmse

NEURALOP_IMPORT = "neuralop.models.FNO"
NEURALOP_SOURCE_URL = "https://github.com/neuraloperator/neuraloperator"


class MissingNeuralOperatorError(RuntimeError):
    """Raised when a live external FNO run is requested without neuralop installed."""


def bind_training_lock(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    """Bind external runners to one verified train+valid lock in strict mode."""

    lock_path = getattr(args, "data_lock", None)
    strict = bool(getattr(args, "strict_contract", False))
    if not lock_path:
        if strict:
            raise ValueError("strict external baseline runs require --data-lock")
        return
    data_cfg = cfg.setdefault("data", {})
    configured_expected = data_cfg.get("data_lock_sha256")
    cli_expected = getattr(args, "expected_data_lock_sha256", None)
    if configured_expected and cli_expected and configured_expected != cli_expected:
        raise ValueError("configured data lock identity disagrees with --expected-data-lock-sha256")
    expected = cli_expected or configured_expected
    if strict and not expected:
        raise ValueError("strict external baseline runs require an expected data lock identity")
    lock = load_data_lock(lock_path)
    if lock.purpose != "training" or set(lock.requested_roles) != {"train", "valid"}:
        raise ValueError("external baseline data lock must contain exactly train and valid roles")
    if any(item.role == "test" for item in lock.objects):
        raise ValueError("external baseline training lock must not expose test objects")
    if expected and expected != lock.lock_sha256:
        raise ValueError("external baseline data lock identity does not match expected SHA-256")
    data_cfg["data_lock_path"] = str(Path(lock_path).resolve())
    data_cfg["data_lock_sha256"] = lock.lock_sha256
    data_cfg["selection_sha256"] = canonical_sha256(lock.selection)


def training_lock_provenance(args: argparse.Namespace) -> dict[str, Any] | None:
    """Return the immutable data identity serialized into external summaries."""

    lock_path = getattr(args, "data_lock", None)
    if not lock_path:
        return None
    lock = load_data_lock(lock_path)
    return {
        "path": str(lock_path),
        "lock_sha256": lock.lock_sha256,
        "purpose": lock.purpose,
        "requested_roles": list(lock.requested_roles),
        "source_revision": lock.source_revision,
        "source_manifest_sha256": lock.source_manifest_sha256,
        "protocol_manifest_sha256": lock.protocol_manifest_sha256,
        "selection_sha256": canonical_sha256(lock.selection),
        "normalization": dict(lock.normalization),
        "objects": [
            {
                "object_id": item.object_id,
                "role": item.role,
                "sha256": item.checksums["sha256"],
            }
            for item in lock.objects
        ],
    }


class FNOGridAdapter(nn.Module):
    """Adapt NeuralOperator FNO's ND tensors to the repo's standard (B,C,H,W) grid."""

    def __init__(
        self, fno: nn.Module, *, grid_shape: tuple[int, int], residual: bool = False
    ) -> None:
        super().__init__()
        self.fno = fno
        self.grid_shape = (int(grid_shape[0]), int(grid_shape[1]))
        self.residual = bool(residual)

    def forward(self, current: torch.Tensor) -> torch.Tensor:
        height, _ = self.grid_shape
        if height == 1:
            pred = self.fno(current.squeeze(-2)).unsqueeze(-2)
        else:
            pred = self.fno(current)
        if self.residual:
            pred = current + pred
        return pred


def neuraloperator_import_status() -> dict[str, Any]:
    try:
        from neuralop.models import FNO  # noqa: F401
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


def load_neuraloperator_fno_class() -> type[nn.Module]:
    try:
        from neuralop.models import FNO
    except Exception as exc:
        raise MissingNeuralOperatorError(
            "neuralop.models.FNO is required for a live external FNO run. "
            "Install the optional NeuralOperator package or run with --dry-run."
        ) from exc
    return FNO


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_source_record(data_root: str | None, task: str, split: str) -> dict[str, Any]:
    if not data_root:
        return {"task": task, "split": split, "path": "", "exists": False}
    path = Path(data_root) / f"{task}_{split}.h5"
    if not path.exists():
        return {"task": task, "split": split, "path": str(path), "exists": False}
    return {
        "task": task,
        "split": split,
        "path": str(path),
        "exists": True,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _external_data_sources(args: argparse.Namespace, tasks: Sequence[str]) -> dict[str, Any]:
    splits = [args.train_split, args.eval_split]
    return {
        task: {
            split: _split_source_record(args.data_root, task, split)
            for split in dict.fromkeys(splits)
        }
        for task in tasks
    }


def _load_test_ledger(path: str | None) -> dict[str, Any]:
    if not path:
        return {"measurements": []}
    ledger_path = Path(path)
    if not ledger_path.exists():
        return {"measurements": []}
    return json.loads(ledger_path.read_text(encoding="utf-8"))


def _write_test_ledger(path: str | None, ledger: dict[str, Any]) -> None:
    if not path:
        return
    ledger_path = Path(path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True), encoding="utf-8")


def _external_test_measurement_key(
    *,
    args: argparse.Namespace,
    tasks: Sequence[str],
) -> str:
    payload = {
        "adapter": "external_neuraloperator_fno_baseline",
        "batch_size": args.batch_size,
        "config": args.config,
        "data_root": args.data_root,
        "data_sources": _external_data_sources(args, tasks),
        "device": args.device,
        "eval_split": args.eval_split,
        "epochs": args.epochs,
        "fourier_modes": args.fourier_modes,
        "hidden_channels": args.hidden_channels,
        "implementation": NEURALOP_IMPORT,
        "learning_rate": args.learning_rate,
        "max_eval_samples": args.max_eval_samples,
        "max_pairs_per_task": args.max_pairs_per_task,
        "max_train_samples": args.max_train_samples,
        "metric": args.metric,
        "n_layers": args.n_layers,
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
    ledger = _load_test_ledger(args.held_out_ledger_json)
    existing_keys = {
        str(entry.get("measurement_key"))
        for entry in ledger.get("measurements", [])
        if isinstance(entry, dict)
    }
    already_recorded = measurement_key in existing_keys
    if already_recorded and not args.allow_repeat_test:
        raise RuntimeError(
            "held-out external FNO test measurement already recorded; "
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
    ledger = _load_test_ledger(args.held_out_ledger_json)
    ledger.setdefault("measurements", []).append(
        {
            "measurement_key": policy["measurement_key"],
            "metric": args.metric,
            "run_name": args.name,
            "summary": str(summary_path),
            "test_metric_value": float(metrics[args.metric]),
            "test_split": args.eval_split,
            "tasks": list(tasks),
        }
    )
    _write_test_ledger(args.held_out_ledger_json, ledger)
    return True


def fno_modes_for_grid(grid_shape: tuple[int, int], requested_modes: int) -> tuple[int, ...]:
    height, width = int(grid_shape[0]), int(grid_shape[1])
    modes = max(int(requested_modes), 1)
    if height == 1:
        return (min(modes, max(1, width // 2)),)
    return (
        min(modes, max(1, height // 2)),
        min(modes, max(1, width // 2)),
    )


def build_neuraloperator_fno_model(
    *,
    channels: int,
    grid_shape: tuple[int, int],
    hidden_channels: int,
    fourier_modes: int,
    n_layers: int,
    residual: bool,
    fno_cls: type[nn.Module] | None = None,
) -> nn.Module:
    fno_cls = fno_cls or load_neuraloperator_fno_class()
    fno = fno_cls(
        n_modes=fno_modes_for_grid(grid_shape, fourier_modes),
        in_channels=int(channels),
        out_channels=int(channels),
        hidden_channels=int(hidden_channels),
        n_layers=int(n_layers),
    )
    return FNOGridAdapter(fno, grid_shape=grid_shape, residual=residual)


def _clone_tensor_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
        if isinstance(value, torch.Tensor)
    }


def _clone_module_at_state(module: nn.Module, state: dict[str, torch.Tensor]) -> nn.Module:
    clone = copy.deepcopy(module).to("cpu")
    clone.load_state_dict(state)
    return clone


def train_fno_group_model(
    currents: torch.Tensor,
    targets: torch.Tensor,
    *,
    hidden_channels: int = 32,
    fourier_modes: int = 16,
    n_layers: int = 4,
    residual: bool = False,
    epochs: int = 5,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 8,
    seed: int = 0,
    device: str | torch.device = "cpu",
    fno_cls: type[nn.Module] | None = None,
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
    model = build_neuraloperator_fno_model(
        channels=int(currents.shape[1]),
        grid_shape=grid_shape,
        hidden_channels=hidden_channels,
        fourier_modes=fourier_modes,
        n_layers=n_layers,
        residual=residual,
        fno_cls=fno_cls,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    generator = torch.Generator().manual_seed(int(seed))
    best_loss = float("inf")
    best_state = _clone_tensor_state_dict(model)
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
            best_state = _clone_tensor_state_dict(model)
            best_epoch = epoch
        if epoch in checkpoint_epoch_set and checkpoint_states is not None:
            checkpoint_states[epoch] = _clone_tensor_state_dict(model)
    model.load_state_dict(best_state)
    model.to("cpu")
    return model, {
        "model": "external_neuraloperator_fno_baseline",
        "implementation": NEURALOP_IMPORT,
        "train_frames": int(currents.shape[0]),
        "channels": int(currents.shape[1]),
        "height": int(currents.shape[2]),
        "width": int(currents.shape[3]),
        "hidden_channels": int(hidden_channels),
        "fourier_modes": int(fourier_modes),
        "n_modes": list(fno_modes_for_grid(grid_shape, fourier_modes)),
        "n_layers": int(n_layers),
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


def train_fno_groups(
    grouped_pairs: dict[tuple[str, int, int, int], tuple[torch.Tensor, torch.Tensor]],
    *,
    hidden_channels: int,
    fourier_modes: int,
    n_layers: int,
    residual: bool,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    seed: int,
    device: str | torch.device,
    fno_cls: type[nn.Module] | None = None,
) -> tuple[dict[tuple[str, int, int, int], nn.Module], dict[str, Any]]:
    models: dict[tuple[str, int, int, int], nn.Module] = {}
    fit: dict[str, Any] = {"groups": {}}
    for offset, (key, (currents, targets)) in enumerate(sorted(grouped_pairs.items())):
        model, group_fit = train_fno_group_model(
            currents,
            targets,
            hidden_channels=hidden_channels,
            fourier_modes=fourier_modes,
            n_layers=n_layers,
            residual=residual,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            seed=seed + offset,
            device=device,
            fno_cls=fno_cls,
        )
        models[key] = model
        fit["groups"][str(key)] = group_fit
    fit["group_count"] = len(models)
    fit["train_frames"] = sum(int(pair[0].shape[0]) for pair in grouped_pairs.values())
    fit["optimizer_steps"] = sum(int(group["optimizer_steps"]) for group in fit["groups"].values())
    fit["examples_seen"] = sum(int(group["examples_seen"]) for group in fit["groups"].values())
    return models, fit


def train_fno_groups_with_rungs(
    grouped_pairs: dict[tuple[str, int, int, int], tuple[torch.Tensor, torch.Tensor]],
    *,
    validation_rungs: Sequence[int],
    hidden_channels: int,
    fourier_modes: int,
    n_layers: int,
    residual: bool,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    seed: int,
    device: str | torch.device,
    fno_cls: type[nn.Module] | None = None,
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
        model, group_fit = train_fno_group_model(
            currents,
            targets,
            hidden_channels=hidden_channels,
            fourier_modes=fourier_modes,
            n_layers=n_layers,
            residual=residual,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            seed=seed + offset,
            device=device,
            fno_cls=fno_cls,
            checkpoint_epochs=rungs,
            checkpoint_states=snapshots,
        )
        missing = sorted(set(rungs).difference(snapshots))
        if missing:
            raise RuntimeError(f"FNO training did not produce requested rungs: {missing}")
        models[key] = model
        for epoch in rungs:
            rung_models[epoch][key] = _clone_module_at_state(model, snapshots[epoch])
        fit["groups"][str(key)] = group_fit
    fit["group_count"] = len(models)
    fit["train_frames"] = sum(int(pair[0].shape[0]) for pair in grouped_pairs.values())
    fit["optimizer_steps"] = sum(int(group["optimizer_steps"]) for group in fit["groups"].values())
    fit["examples_seen"] = sum(int(group["examples_seen"]) for group in fit["groups"].values())
    return models, fit, rung_models


def write_group_checkpoint(
    path: Path,
    models: dict[tuple[str, int, int, int], nn.Module],
    *,
    model_family: str,
    fit: dict[str, Any],
    epoch: int | None = None,
) -> dict[str, Any]:
    """Persist the exact selected group states for recipe-adequacy evidence."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "model_family": model_family,
        "epoch": epoch,
        "groups": {
            str(key): _clone_tensor_state_dict(model) for key, model in sorted(models.items())
        },
        "fit": fit,
    }
    torch.save(payload, path)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "epoch": epoch,
    }


def evaluate_external_fno_baseline(
    cfg: dict[str, Any],
    models: dict[tuple[str, int, int, int], nn.Module],
    *,
    tasks: Sequence[str],
    split: str,
    data_root: str | None,
    max_samples: int | None,
    rollout_steps: int,
    device: str | torch.device = "cpu",
    strict_contract: bool = False,
) -> dict[str, float]:
    """Evaluate shared external models under temporal and steady semantics.

    Temporal predictions are genuinely autoregressive: horizon ``h`` consumes
    the prediction from ``h-1``.  Steady operators map the coefficient field to
    the explicit solution target once.  The strict strat-v1 mode additionally
    requires 16 temporal horizons and scalar regime metadata.
    """
    device = torch.device(device)
    total_pred: list[torch.Tensor] = []
    total_target: list[torch.Tensor] = []
    per_task_pred: dict[str, list[torch.Tensor]] = {}
    per_task_target: dict[str, list[torch.Tensor]] = {}
    per_family_pred: dict[str, list[torch.Tensor]] = {}
    per_family_target: dict[str, list[torch.Tensor]] = {}
    task_horizon_pred: dict[str, dict[int, list[torch.Tensor]]] = {}
    task_horizon_target: dict[str, dict[int, list[torch.Tensor]]] = {}
    task_regime_pred: dict[str, dict[str, list[torch.Tensor]]] = {}
    task_regime_target: dict[str, dict[str, list[torch.Tensor]]] = {}

    for task in tasks:
        dataset = _dataset(
            cfg,
            task=task,
            split=split,
            data_root=data_root,
            max_samples=max_samples,
        )
        family = get_pdebench_spec(task).family
        spec = get_pdebench_spec(task)
        for sample_idx in range(len(dataset)):
            sample = dataset[sample_idx]
            fields = sample["fields"].float()
            grid_shape = infer_grid_shape(fields)
            regime = _regime_label(
                _regime_value(sample, task=task, strict_contract=strict_contract)
            )
            if spec.mapping_kind == "steady_operator":
                targets = sample["targets"].float()
                if fields.shape[0] != 1 or targets.shape[0] != 1:
                    raise ValueError("Steady external baseline requires one input and one target")
                current_grid = field_step_to_grid(fields[0], grid_shape)
                key = group_key(task, grid_shape, int(current_grid.shape[1]))
                if key not in models:
                    raise ValueError(f"No trained external baseline for steady group {key}")
                with torch.no_grad():
                    pred_grid = models[key].to(device).eval()(current_grid.to(device)).cpu()
                predictions = [grid_to_flat(pred_grid)]
                targets_for_sample = [_flatten_field_step(targets[0], grid_shape).cpu()]
            else:
                available_steps = int(fields.shape[0]) - 1
                steps = int(rollout_steps)
                if strict_contract and steps != 16:
                    raise ValueError("strat-v1 external baselines require 16 rollout steps")
                if available_steps < steps:
                    raise ValueError(
                        f"{task} sample {sample_idx} has {available_steps} horizons; {steps} required"
                    )
                current_grid = field_step_to_grid(fields[0], grid_shape)
                key = group_key(task, grid_shape, int(current_grid.shape[1]))
                if key not in models:
                    raise ValueError(f"No trained external baseline for group {key}")
                model = models[key].to(device).eval()
                predictions = []
                targets_for_sample = []
                for horizon in range(1, steps + 1):
                    with torch.no_grad():
                        current_grid = model(current_grid.to(device)).cpu()
                    prediction = grid_to_flat(current_grid)
                    target = _flatten_field_step(fields[horizon].float(), grid_shape).cpu()
                    predictions.append(prediction)
                    targets_for_sample.append(target)
                    task_horizon_pred.setdefault(task, {}).setdefault(horizon, []).append(
                        prediction
                    )
                    task_horizon_target.setdefault(task, {}).setdefault(horizon, []).append(target)

            for pred, target in zip(predictions, targets_for_sample, strict=True):
                total_pred.append(pred)
                total_target.append(target)
                per_task_pred.setdefault(task, []).append(pred)
                per_task_target.setdefault(task, []).append(target)
                per_family_pred.setdefault(family, []).append(pred)
                per_family_target.setdefault(family, []).append(target)
                task_regime_pred.setdefault(task, {}).setdefault(regime, []).append(pred)
                task_regime_target.setdefault(task, {}).setdefault(regime, []).append(target)

    if not total_pred:
        raise RuntimeError("External FNO baseline received no eval pairs")

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
    primary_values = []
    temporal_horizon_values: dict[int, list[float]] = {}
    for task, pred_chunks in per_task_pred.items():
        spec = get_pdebench_spec(task)
        _add_rollout_metrics(
            metrics,
            prefix=f"task_{task}_",
            pred_chunks=pred_chunks,
            target_chunks=per_task_target[task],
        )
        task_stats = _aggregate_chunk_metrics(pred_chunks, per_task_target[task])
        primary_name = (
            "decoded_solution_nrmse"
            if spec.mapping_kind == "steady_operator"
            else "decoded_rollout_nrmse"
        )
        metrics[f"task_{task}_{primary_name}"] = task_stats["nrmse"]
        primary_values.append(task_stats["nrmse"])
        for horizon in sorted(task_horizon_pred.get(task, {})):
            horizon_stats = _aggregate_chunk_metrics(
                task_horizon_pred[task][horizon], task_horizon_target[task][horizon]
            )
            metrics[f"task_{task}_decoded_h{horizon}_nrmse"] = horizon_stats["nrmse"]
            temporal_horizon_values.setdefault(horizon, []).append(horizon_stats["nrmse"])
        suffix = primary_name
        seen_slugs = set()
        for regime in sorted(task_regime_pred[task], key=lambda value: (value == "unknown", value)):
            slug = _regime_slug(regime)
            if slug in seen_slugs:
                raise ValueError(f"{task} regime labels collide after metric slugging")
            seen_slugs.add(slug)
            regime_stats = _aggregate_chunk_metrics(
                task_regime_pred[task][regime], task_regime_target[task][regime]
            )
            metrics[f"task_{task}_regime_{slug}_{suffix}"] = regime_stats["nrmse"]
            global_scale_key = suffix.replace("_nrmse", "_global_scale_nrmse")
            metrics[f"task_{task}_regime_{slug}_{global_scale_key}"] = global_scale_regime_nrmse(
                task_regime_pred[task][regime],
                task_regime_target[task][regime],
                per_task_target[task],
            )
            element_count_key = suffix.replace("_nrmse", "_element_count")
            metrics[f"task_{task}_regime_{slug}_{element_count_key}"] = aligned_element_count(
                task_regime_pred[task][regime], task_regime_target[task][regime]
            )
    for family, pred_chunks in per_family_pred.items():
        _add_rollout_metrics(
            metrics,
            prefix=f"family_{family}_",
            pred_chunks=pred_chunks,
            target_chunks=per_family_target[family],
        )
    metrics["micro_decoded_rollout_nrmse"] = stats["nrmse"]
    metrics["macro_primary_nrmse"] = sum(primary_values) / len(primary_values)
    metrics["decoded_rollout_nrmse"] = metrics["macro_primary_nrmse"]
    for horizon, values in sorted(temporal_horizon_values.items()):
        metrics[f"temporal_macro_decoded_h{horizon}_nrmse"] = sum(values) / len(values)
    return metrics


def _validated_rungs(args: argparse.Namespace) -> tuple[int, ...]:
    rungs = tuple(int(epoch) for epoch in (getattr(args, "validation_rungs", None) or ()))
    if not rungs:
        return ()
    if any(epoch <= 0 for epoch in rungs):
        raise ValueError("validation rungs must be positive epochs")
    if tuple(sorted(set(rungs))) != rungs:
        raise ValueError("validation rungs must be unique and strictly increasing")
    if int(args.epochs) != rungs[-1]:
        raise ValueError("--epochs must equal the final validation rung")
    if args.eval_split == "test":
        raise ValueError("validation rungs forbid held-out test evaluation")
    return rungs


def _begin_compute_tracking(device: str | torch.device) -> None:
    resolved = torch.device(device)
    if resolved.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(resolved)


def _compute_evidence(
    models: dict[tuple[str, int, int, int], nn.Module],
    fit: dict[str, Any],
    *,
    duration_sec: float,
    device: str | torch.device,
) -> dict[str, Any]:
    parameters = [parameter for model in models.values() for parameter in model.parameters()]
    result: dict[str, Any] = {
        "total_parameter_count": sum(int(parameter.numel()) for parameter in parameters),
        "trainable_parameter_count": sum(
            int(parameter.numel()) for parameter in parameters if parameter.requires_grad
        ),
        "optimizer_steps": int(fit.get("optimizer_steps", 0)),
        "examples_seen": int(fit.get("examples_seen", 0)),
        "duration_sec": float(duration_sec),
        "device": str(device),
    }
    resolved = torch.device(device)
    if resolved.type == "cuda" and torch.cuda.is_available():
        result.update(
            {
                "cuda_device_name": torch.cuda.get_device_name(resolved),
                "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(resolved)),
            }
        )
    return result


def _command_record(args: argparse.Namespace) -> list[str]:
    command = [
        "python",
        "scripts/run_external_neuraloperator_fno_baseline.py",
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
        "data_provenance": training_lock_provenance(args),
        "extra": {
            "baseline": "external_neuraloperator_fno",
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
        "stages": ["external_neuraloperator_fno_baseline"],
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
    main_metric_name, main_metric_value = _main_metric(metrics)
    _append_results_row(
        output_root / "results.tsv",
        {
            "run_name": args.name,
            "timestamp": int(finished),
            "stages": "external_neuraloperator_fno_baseline",
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
        "n_modes",
        "n_layers",
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
            if isinstance(row.get("n_modes"), list):
                row["n_modes"] = json.dumps(row["n_modes"])
            writer.writerow(row)


def run_baseline(args: argparse.Namespace) -> Path:
    cfg = _load_cfg(args.config)
    bind_training_lock(cfg, args)
    tasks = _as_task_names(cfg, args.tasks or args.task)
    validation_rungs = _validated_rungs(args)
    output_root = Path(args.output_root)
    run_dir = output_root / args.name
    if getattr(args, "refuse_overwrite", False) and run_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run directory: {run_dir}")
    if args.dry_run:
        return write_dry_run_summary(args, tasks=tasks)
    if args.eval_split == "test" and not args.allow_held_out_test_eval:
        raise RuntimeError(
            "Live external FNO evaluation on split=test requires --allow-held-out-test-eval. "
            "Use --eval-split val while debugging adapter behavior."
        )
    held_out_test_policy = {"enabled": False, "recorded": False}
    if args.eval_split == "test":
        held_out_test_policy = _guard_external_test_measurement(args=args, tasks=tasks)

    load_neuraloperator_fno_class()
    _begin_compute_tracking(args.device)
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
        raise RuntimeError("No training pairs collected for external FNO baseline")
    training_kwargs = {
        "hidden_channels": args.hidden_channels,
        "fourier_modes": args.fourier_modes,
        "n_layers": args.n_layers,
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
        models, fit, rung_models = train_fno_groups_with_rungs(
            groups, validation_rungs=validation_rungs, **training_kwargs
        )
        for epoch in validation_rungs:
            evaluation_started = time.time()
            rung_metrics = evaluate_external_fno_baseline(
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
        models, fit = train_fno_groups(groups, **training_kwargs)
        selected_models = models
        selected_epoch = None
        metrics = evaluate_external_fno_baseline(
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
    group_manifest = run_dir / "fno_groups.tsv"
    _write_group_manifest(group_manifest, fit)
    if validation_rungs:
        rung_checkpoints = {
            str(epoch): write_group_checkpoint(
                run_dir / f"models_epoch_{epoch}.pt",
                rung_models[epoch],
                model_family="fno",
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
        checkpoint = write_group_checkpoint(
            run_dir / "models.pt", selected_models, model_family="fno", fit=fit
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
        "compute": _compute_evidence(
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
    main_metric_name, main_metric_value = _main_metric(metrics)
    print(
        json.dumps(
            {"summary": str(summary_path), "main_metric": {main_metric_name: main_metric_value}},
            indent=2,
        )
    )
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train/evaluate an optional external NeuralOperator FNO baseline"
    )
    parser.add_argument("--config", default="configs/train_multitask_heterogeneous_light_best.yaml")
    parser.add_argument("--name", default="external_neuraloperator_fno_light_val")
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
