#!/usr/bin/env python
from __future__ import annotations

"""Write a dry PhysicsNeMo recipe-compatibility smoke manifest.

This is not a PhysicsNeMo performance measurement. The default path avoids a
mandatory PhysicsNeMo install and records the recipe contract that must be
satisfied before any live validation metric or held-out test comparison.
"""

import argparse
import csv
import hashlib
import importlib
import importlib.metadata as metadata
import importlib.util
import json
import platform
import sys
import time
from collections.abc import Mapping
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
from ups.data.pdebench import get_pdebench_spec
from ups.eval.pdebench_runner import _aggregate_chunk_metrics, _flatten_field_step

PHYSICSNEMO_PACKAGE = "nvidia-physicsnemo"
PHYSICSNEMO_IMPORT = "physicsnemo"
PHYSICSNEMO_FNO_IMPORT = "physicsnemo.models.fno.fno.FNO"
PHYSICSNEMO_SOURCE_URL = "https://github.com/NVIDIA/physicsnemo"
PHYSICSNEMO_DOCS_URL = "https://docs.nvidia.com/physicsnemo/latest/"
PHYSICSNEMO_RECIPE_DOCS_URL = (
    "https://docs.nvidia.com/physicsnemo/latest/user-guide/simple_training_example.html"
)
PHYSICSNEMO_EXAMPLES_URL = "https://docs.nvidia.com/physicsnemo/latest/examples_catalog.html"
PHYSICSNEMO_INSTALL_URL = (
    "https://docs.nvidia.com/physicsnemo/latest/getting-started/installation.html"
)
DEFAULT_TASKS = ("advection1d", "burgers1d", "darcy2d")
SMOKE_MEASUREMENT_TYPE = "physicsnemo_compatibility_smoke"
SMOKE_STATUS = "compatibility_smoke_ready"
LIVE_RECIPE_MEASUREMENT_TYPE = "physicsnemo_recipe_validation_adapter"
LIVE_RECIPE_STATUS = "validation_recipe_adapter_complete"


class MissingPhysicsNeMoError(RuntimeError):
    """Raised when a live PhysicsNeMo recipe run is requested without PhysicsNeMo."""


def _selected_tasks(args: argparse.Namespace, cfg: dict[str, Any] | None = None) -> list[str]:
    tasks = list(getattr(args, "tasks", []) or getattr(args, "task", []))
    if tasks:
        return [str(task) for task in tasks]
    if cfg is not None:
        return _as_task_names(cfg, [])
    return list(DEFAULT_TASKS)


def _command_record(args: argparse.Namespace) -> list[str]:
    tasks = _selected_tasks(args)
    command = [
        "python",
        "scripts/run_physicsnemo_compatibility_smoke.py",
        "--name",
        args.name,
        "--output-root",
        args.output_root,
        "--evidence-json",
        args.evidence_json,
        "--train-split",
        args.train_split,
        "--eval-split",
        args.eval_split,
        "--tasks",
        *tasks,
    ]
    if args.live_import:
        command.append("--live-import")
    if args.require_live_import:
        command.append("--require-live-import")
    return command


def _package_probe(*, live_import: bool, require_live_import: bool) -> dict[str, Any]:
    base: dict[str, Any] = {
        "pip_name": PHYSICSNEMO_PACKAGE,
        "import_name": PHYSICSNEMO_IMPORT,
        "declared_python_requires": ">=3.11,<=3.14",
        "install_url": PHYSICSNEMO_INSTALL_URL,
        "source_url": PHYSICSNEMO_SOURCE_URL,
        "docs_url": PHYSICSNEMO_DOCS_URL,
        "live_import_requested": bool(live_import),
        "live_import_required": bool(require_live_import),
    }
    if not live_import:
        base["live_import_status"] = "not_requested"
        base["module_spec_checked"] = False
        return base
    spec = importlib.util.find_spec(PHYSICSNEMO_IMPORT)
    base["module_spec_checked"] = True
    base["module_spec_available"] = spec is not None
    try:
        module = importlib.import_module(PHYSICSNEMO_IMPORT)
    except Exception as exc:
        if require_live_import:
            raise RuntimeError(
                f"{PHYSICSNEMO_IMPORT} import is required but failed: {type(exc).__name__}: {exc}"
            ) from exc
        base.update(
            {
                "live_import_status": "failed_optional",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        return base
    base.update(
        {
            "live_import_status": "available",
            "version": str(getattr(module, "__version__", "")),
        }
    )
    return base


def _recipe_contract(args: argparse.Namespace) -> dict[str, Any]:
    tasks = list(dict.fromkeys(_selected_tasks(args)))
    inspected_splits = list(dict.fromkeys([str(args.train_split), str(args.eval_split)]))
    return {
        "scope": "dry ecosystem compatibility recipe manifest",
        "tasks": tasks,
        "inspected_splits": inspected_splits,
        "data_interface": {
            "source": "repo light-v1 PDEBench-shaped HDF5 shards",
            "expected_shard_pattern": "{task}_{split}.h5",
            "candidate_adapter": (
                "repo PDEBench tensors to a PhysicsNeMo data-driven recipe/datapipe"
            ),
            "held_out_test_data_read": False,
        },
        "candidate_recipe": {
            "first_target": (
                "PhysicsNeMo data-driven neural-operator recipe adapted to light-v1 "
                "train/validation tensors"
            ),
            "examples_catalog_url": PHYSICSNEMO_EXAMPLES_URL,
            "why_first": (
                "Recipe compatibility proves framework interop before reporting a "
                "framework metric."
            ),
        },
        "live_metric_allowed": False,
        "model_training_performed": False,
        "held_out_test_policy": "No test split or held-out ledger write in this smoke gate.",
        "next_gate": (
            "Run a live PhysicsNeMo recipe adapter with `python "
            "scripts/run_physicsnemo_compatibility_smoke.py --live-recipe "
            "--eval-split val` in a Python 3.11+ or PhysicsNeMo container "
            "environment, record validation-only provenance, then decide whether "
            "any held-out test budget is justified."
        ),
    }


def build_physicsnemo_smoke_summary(args: argparse.Namespace) -> dict[str, Any]:
    inspected_splits = list(dict.fromkeys([str(args.train_split), str(args.eval_split)]))
    if "test" in inspected_splits:
        raise RuntimeError("PhysicsNeMo compatibility smoke must not inspect split=test")
    package = _package_probe(
        live_import=bool(args.live_import),
        require_live_import=bool(args.require_live_import),
    )
    output_root = Path(args.output_root)
    summary_path = output_root / args.name / "summary.json"
    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": SMOKE_STATUS,
        "measurement_type": SMOKE_MEASUREMENT_TYPE,
        "run_name": args.name,
        "summary_json": str(summary_path),
        "evidence_json": str(args.evidence_json),
        "split": args.eval_split,
        "inspected_splits": inspected_splits,
        "metrics": {},
        "claim_comparable": False,
        "published_numbers_directly_comparable": False,
        "held_out_test_used": False,
        "held_out_test_data_read": False,
        "stages": ["external_physicsnemo_compatibility_smoke"],
        "source_refs": ["physicsnemo_official_repo", "physicsnemo_docs"],
        "extra": {
            "baseline": "external_physicsnemo_compatibility",
            "implementation": PHYSICSNEMO_IMPORT,
            "source_url": PHYSICSNEMO_SOURCE_URL,
            "docs_url": PHYSICSNEMO_DOCS_URL,
            "examples_catalog_url": PHYSICSNEMO_EXAMPLES_URL,
            "command": _command_record(args),
        },
        "details": {
            "package": package,
            "recipe_contract": _recipe_contract(args),
            "claim_boundary": (
                "Compatibility smoke only; no model training, no PhysicsNeMo metric, "
                "and no held-out test access."
            ),
        },
    }
    errors = validate_physicsnemo_smoke_summary(summary)
    if errors:
        summary["status"] = "invalid"
        summary["validation_errors"] = errors
    return summary


def validate_physicsnemo_smoke_summary(summary: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if summary.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if summary.get("status") not in {SMOKE_STATUS, "invalid"}:
        errors.append(f"status must be one of {[SMOKE_STATUS, 'invalid']}")
    if summary.get("measurement_type") != SMOKE_MEASUREMENT_TYPE:
        errors.append(f"measurement_type must be {SMOKE_MEASUREMENT_TYPE}")
    if summary.get("claim_comparable") is not False:
        errors.append("claim_comparable must be false")
    if summary.get("published_numbers_directly_comparable") is not False:
        errors.append("published_numbers_directly_comparable must be false")
    if summary.get("held_out_test_used") is not False:
        errors.append("held_out_test_used must be false")
    if summary.get("held_out_test_data_read") is not False:
        errors.append("held_out_test_data_read must be false")

    inspected_splits = summary.get("inspected_splits")
    if not isinstance(inspected_splits, list) or not inspected_splits:
        errors.append("inspected_splits must be a non-empty list")
        inspected_splits = []
    if "test" in inspected_splits:
        errors.append("compatibility smoke must not inspect split=test")

    metrics = summary.get("metrics")
    if not isinstance(metrics, Mapping):
        errors.append("metrics must be an object")
        metrics = {}
    if metrics:
        errors.append("compatibility smoke must not report metrics")
    if "decoded_rollout_nrmse" in metrics:
        errors.append("compatibility smoke must not report decoded_rollout_nrmse")

    details = summary.get("details")
    if not isinstance(details, Mapping):
        errors.append("details must be an object")
        details = {}
    package = details.get("package")
    if not isinstance(package, Mapping):
        errors.append("details.package is required")
        package = {}
    if package.get("pip_name") != PHYSICSNEMO_PACKAGE:
        errors.append(f"details.package.pip_name must be {PHYSICSNEMO_PACKAGE}")
    if package.get("import_name") != PHYSICSNEMO_IMPORT:
        errors.append(f"details.package.import_name must be {PHYSICSNEMO_IMPORT}")

    contract = details.get("recipe_contract")
    if not isinstance(contract, Mapping):
        errors.append("details.recipe_contract is required")
        contract = {}
    tasks = contract.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        errors.append("details.recipe_contract.tasks must be a non-empty list")
    if contract.get("live_metric_allowed") is not False:
        errors.append("details.recipe_contract.live_metric_allowed must be false")
    if not contract.get("next_gate"):
        errors.append("details.recipe_contract.next_gate is required")
    return errors


class PhysicsNeMoFNOGridAdapter(nn.Module):
    """Adapt PhysicsNeMo FNO tensors to the repo's standard (B,C,H,W) grid."""

    def __init__(
        self,
        fno: nn.Module,
        *,
        grid_shape: tuple[int, int],
        residual: bool = False,
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


def physicsnemo_import_status() -> dict[str, Any]:
    try:
        from physicsnemo.models.fno.fno import FNO  # noqa: F401
    except Exception as exc:
        return {
            "available": False,
            "pip_name": PHYSICSNEMO_PACKAGE,
            "import": PHYSICSNEMO_FNO_IMPORT,
            "source_url": PHYSICSNEMO_SOURCE_URL,
            "docs_url": PHYSICSNEMO_DOCS_URL,
            "recipe_docs_url": PHYSICSNEMO_RECIPE_DOCS_URL,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    version = "unknown"
    try:
        version = metadata.version(PHYSICSNEMO_PACKAGE)
    except metadata.PackageNotFoundError:
        pass
    return {
        "available": True,
        "pip_name": PHYSICSNEMO_PACKAGE,
        "import": PHYSICSNEMO_FNO_IMPORT,
        "declared_python_requires": ">=3.11,<=3.14",
        "source_url": PHYSICSNEMO_SOURCE_URL,
        "docs_url": PHYSICSNEMO_DOCS_URL,
        "recipe_docs_url": PHYSICSNEMO_RECIPE_DOCS_URL,
        "version": version,
    }


def live_recipe_runtime_status() -> dict[str, Any]:
    torch_version = str(getattr(torch, "__version__", "unknown"))
    try:
        import torchvision

        torchvision_version = str(getattr(torchvision, "__version__", "unknown"))
    except Exception as exc:
        torchvision_version = f"unavailable:{type(exc).__name__}"
    return {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch_version,
        "torchvision_version": torchvision_version,
    }


def load_physicsnemo_fno_class() -> type[nn.Module]:
    try:
        from physicsnemo.models.fno.fno import FNO
    except Exception as exc:
        raise MissingPhysicsNeMoError(
            "physicsnemo.models.fno.fno.FNO is required for --live-recipe. "
            "Install nvidia-physicsnemo in Python 3.11+ or use the official "
            "PhysicsNeMo container; keep --dry-run for a zero-dependency contract check."
        ) from exc
    return FNO


def physicsnemo_modes_for_grid(grid_shape: tuple[int, int], requested_modes: int) -> int:
    height, width = int(grid_shape[0]), int(grid_shape[1])
    modes = max(int(requested_modes), 1)
    if height == 1:
        return min(modes, max(1, width // 2))
    return min(modes, max(1, min(height, width) // 2))


def physicsnemo_dimension_for_grid(grid_shape: tuple[int, int]) -> int:
    return 1 if int(grid_shape[0]) == 1 else 2


def build_physicsnemo_fno_model(
    *,
    channels: int,
    grid_shape: tuple[int, int],
    latent_channels: int,
    fourier_modes: int,
    num_fno_layers: int,
    decoder_layers: int,
    decoder_layer_size: int,
    padding: int,
    residual: bool,
    fno_cls: type[nn.Module] | None = None,
) -> nn.Module:
    fno_cls = fno_cls or load_physicsnemo_fno_class()
    fno = fno_cls(
        in_channels=int(channels),
        out_channels=int(channels),
        decoder_layers=int(decoder_layers),
        decoder_layer_size=int(decoder_layer_size),
        dimension=physicsnemo_dimension_for_grid(grid_shape),
        latent_channels=int(latent_channels),
        num_fno_layers=int(num_fno_layers),
        num_fno_modes=physicsnemo_modes_for_grid(grid_shape, fourier_modes),
        padding=int(padding),
    )
    return PhysicsNeMoFNOGridAdapter(fno, grid_shape=grid_shape, residual=residual)


def _clone_tensor_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
        if isinstance(value, torch.Tensor)
    }


def train_physicsnemo_fno_group_model(
    currents: torch.Tensor,
    targets: torch.Tensor,
    *,
    latent_channels: int = 32,
    fourier_modes: int = 12,
    num_fno_layers: int = 4,
    decoder_layers: int = 1,
    decoder_layer_size: int = 32,
    padding: int = 5,
    residual: bool = False,
    epochs: int = 5,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 8,
    seed: int = 0,
    device: str | torch.device = "cpu",
    fno_cls: type[nn.Module] | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    if currents.shape != targets.shape:
        raise ValueError("currents and targets must have the same shape")
    if currents.dim() != 4:
        raise ValueError(f"Expected training tensors shaped (N,C,H,W), got {tuple(currents.shape)}")

    torch.manual_seed(int(seed))
    device = torch.device(device)
    grid_shape = (int(currents.shape[2]), int(currents.shape[3]))
    model = build_physicsnemo_fno_model(
        channels=int(currents.shape[1]),
        grid_shape=grid_shape,
        latent_channels=latent_channels,
        fourier_modes=fourier_modes,
        num_fno_layers=num_fno_layers,
        decoder_layers=decoder_layers,
        decoder_layer_size=decoder_layer_size,
        padding=padding,
        residual=residual,
        fno_cls=fno_cls,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    generator = torch.Generator().manual_seed(int(seed))
    best_loss = float("inf")
    best_state = _clone_tensor_state_dict(model)
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
            best_state = _clone_tensor_state_dict(model)
    model.load_state_dict(best_state)
    model.to("cpu")
    return model, {
        "model": "external_physicsnemo_recipe_validation_adapter",
        "implementation": PHYSICSNEMO_FNO_IMPORT,
        "train_frames": int(currents.shape[0]),
        "channels": int(currents.shape[1]),
        "height": int(currents.shape[2]),
        "width": int(currents.shape[3]),
        "dimension": physicsnemo_dimension_for_grid(grid_shape),
        "latent_channels": int(latent_channels),
        "fourier_modes": int(fourier_modes),
        "num_fno_modes": physicsnemo_modes_for_grid(grid_shape, fourier_modes),
        "num_fno_layers": int(num_fno_layers),
        "decoder_layers": int(decoder_layers),
        "decoder_layer_size": int(decoder_layer_size),
        "padding": int(padding),
        "residual": bool(residual),
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "batch_size": int(batch_size),
        "train_mse": best_loss,
    }


def train_physicsnemo_fno_groups(
    grouped_pairs: dict[tuple[str, int, int, int], tuple[torch.Tensor, torch.Tensor]],
    *,
    latent_channels: int,
    fourier_modes: int,
    num_fno_layers: int,
    decoder_layers: int,
    decoder_layer_size: int,
    padding: int,
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
        model, group_fit = train_physicsnemo_fno_group_model(
            currents,
            targets,
            latent_channels=latent_channels,
            fourier_modes=fourier_modes,
            num_fno_layers=num_fno_layers,
            decoder_layers=decoder_layers,
            decoder_layer_size=decoder_layer_size,
            padding=padding,
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
    return models, fit


def evaluate_physicsnemo_recipe_adapter(
    cfg: dict[str, Any],
    models: dict[tuple[str, int, int, int], nn.Module],
    *,
    tasks: list[str],
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
                    raise ValueError(f"No trained PhysicsNeMo recipe adapter for group {key}")
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
        raise RuntimeError("PhysicsNeMo recipe adapter received no eval pairs")

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


def _live_recipe_data_sources(args: argparse.Namespace, tasks: list[str]) -> dict[str, Any]:
    splits = [args.train_split, args.eval_split]
    return {
        task: {
            split: _split_source_record(args.data_root, task, split)
            for split in dict.fromkeys(splits)
        }
        for task in tasks
    }


def _live_recipe_command_record(args: argparse.Namespace, tasks: list[str]) -> list[str]:
    command = [
        "python",
        "scripts/run_physicsnemo_compatibility_smoke.py",
        "--live-recipe",
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
        "--latent-channels",
        str(args.latent_channels),
        "--fourier-modes",
        str(args.fourier_modes),
        "--num-fno-layers",
        str(args.num_fno_layers),
        "--decoder-layers",
        str(args.decoder_layers),
        "--decoder-layer-size",
        str(args.decoder_layer_size),
        "--padding",
        str(args.padding),
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
        "--tasks",
        *tasks,
    ]
    if args.data_root:
        command.extend(["--data-root", args.data_root])
    if args.residual:
        command.append("--residual")
    if args.dry_run:
        command.append("--dry-run")
    return command


def _live_recipe_common(args: argparse.Namespace, *, tasks: list[str]) -> dict[str, Any]:
    inspected_splits = list(dict.fromkeys([str(args.train_split), str(args.eval_split)]))
    return {
        "schema_version": 1,
        "measurement_type": LIVE_RECIPE_MEASUREMENT_TYPE,
        "run_name": args.name,
        "split": args.eval_split,
        "inspected_splits": inspected_splits,
        "claim_comparable": False,
        "published_numbers_directly_comparable": False,
        "held_out_test_used": False,
        "held_out_test_data_read": False,
        "source_refs": ["physicsnemo_official_repo", "physicsnemo_docs"],
        "extra": {
            "baseline": "external_physicsnemo_recipe_validation_adapter",
            "implementation": PHYSICSNEMO_FNO_IMPORT,
            "source_url": PHYSICSNEMO_SOURCE_URL,
            "docs_url": PHYSICSNEMO_DOCS_URL,
            "recipe_docs_url": PHYSICSNEMO_RECIPE_DOCS_URL,
            "physicsnemo": physicsnemo_import_status(),
            "runtime": live_recipe_runtime_status(),
            "task": tasks[0] if len(tasks) == 1 else list(tasks),
            "train_split": args.train_split,
            "split": args.eval_split,
            "max_train_samples": args.max_train_samples,
            "max_eval_samples": args.max_eval_samples,
            "max_pairs_per_task": args.max_pairs_per_task,
            "rollout_steps": args.rollout_steps,
            "train_stride": args.train_stride,
            "metric": args.metric,
            "latent_channels": args.latent_channels,
            "fourier_modes": args.fourier_modes,
            "num_fno_layers": args.num_fno_layers,
            "decoder_layers": args.decoder_layers,
            "decoder_layer_size": args.decoder_layer_size,
            "padding": args.padding,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "device": args.device,
            "residual": bool(args.residual),
            "command": _live_recipe_command_record(args, tasks),
        },
        "checkpoints": {},
        "config": args.config,
        "eval_config": args.config,
        "stages": ["external_physicsnemo_recipe_validation_adapter"],
    }


def write_live_recipe_dry_run_summary(args: argparse.Namespace, *, tasks: list[str]) -> Path:
    output_root = Path(args.output_root)
    run_dir = output_root / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    summary = {
        "status": "dry_run",
        "metrics": {},
        **_live_recipe_common(args, tasks=tasks),
        "details": {
            "contract": {
                "published_numbers_directly_comparable": False,
                "requires_optional_dependency": PHYSICSNEMO_PACKAGE,
                "optional_import": PHYSICSNEMO_FNO_IMPORT,
                "live_test_allowed": False,
                "live_validation_requires_python": ">=3.11,<=3.14",
                "train_split": args.train_split,
                "eval_split": args.eval_split,
            }
        },
        "duration_sec": 0.0,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "dry_run", "summary": str(summary_path)}, indent=2))
    return summary_path


def _write_live_recipe_group_manifest(path: Path, fit: dict[str, Any]) -> None:
    fieldnames = [
        "group",
        "model",
        "implementation",
        "train_frames",
        "channels",
        "height",
        "width",
        "dimension",
        "latent_channels",
        "fourier_modes",
        "num_fno_modes",
        "num_fno_layers",
        "decoder_layers",
        "decoder_layer_size",
        "padding",
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


def _append_live_recipe_results_row(
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
            "stages": "external_physicsnemo_recipe_validation_adapter",
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


def run_live_recipe_adapter(
    args: argparse.Namespace,
    *,
    fno_cls: type[nn.Module] | None = None,
) -> Path:
    inspected_splits = {str(args.train_split), str(args.eval_split)}
    if "test" in inspected_splits:
        raise RuntimeError("PhysicsNeMo live recipe adapter must not inspect split=test")

    cfg = _load_cfg(args.config)
    tasks = _selected_tasks(args, cfg)
    if args.dry_run:
        return write_live_recipe_dry_run_summary(args, tasks=tasks)

    fno_cls = fno_cls or load_physicsnemo_fno_class()
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
        raise RuntimeError("No training pairs collected for PhysicsNeMo recipe adapter")
    models, fit = train_physicsnemo_fno_groups(
        groups,
        latent_channels=args.latent_channels,
        fourier_modes=args.fourier_modes,
        num_fno_layers=args.num_fno_layers,
        decoder_layers=args.decoder_layers,
        decoder_layer_size=args.decoder_layer_size,
        padding=args.padding,
        residual=args.residual,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        fno_cls=fno_cls,
    )
    metrics = evaluate_physicsnemo_recipe_adapter(
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
    group_manifest = run_dir / "physicsnemo_groups.tsv"
    _write_live_recipe_group_manifest(group_manifest, fit)
    summary_path = run_dir / "summary.json"
    summary = {
        "status": LIVE_RECIPE_STATUS,
        "metrics": metrics,
        **_live_recipe_common(args, tasks=tasks),
        "summary_json": str(summary_path),
        "details": {
            "fit": fit,
            "group_manifest": str(group_manifest),
            "data_sources": _live_recipe_data_sources(args, tasks),
            "contract": {
                "published_numbers_directly_comparable": False,
                "claim_comparable": False,
                "held_out_test_policy": "Validation adapter only; split=test is blocked.",
                "requires_optional_dependency": PHYSICSNEMO_PACKAGE,
                "optional_import": PHYSICSNEMO_FNO_IMPORT,
            },
        },
        "duration_sec": finished - started,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _append_live_recipe_results_row(
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


def run_compatibility_smoke(args: argparse.Namespace) -> Path:
    summary = build_physicsnemo_smoke_summary(args)
    output_root = Path(args.output_root)
    run_dir = output_root / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    evidence_path = Path(args.evidence_json)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": summary["status"],
                "summary": str(summary_path),
                "evidence_json": str(evidence_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return summary_path


def check_compatibility_smoke(args: argparse.Namespace) -> bool:
    evidence_path = Path(args.evidence_json)
    if not evidence_path.exists():
        raise FileNotFoundError(f"PhysicsNeMo compatibility evidence not found: {evidence_path}")
    expected = build_physicsnemo_smoke_summary(args)
    actual = json.loads(evidence_path.read_text(encoding="utf-8"))
    if actual != expected:
        print(
            json.dumps(
                {
                    "status": "out_of_date",
                    "evidence_json": str(evidence_path),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return False
    print(
        json.dumps(
            {
                "status": "up_to_date",
                "evidence_json": str(evidence_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_multitask_heterogeneous_light_best.yaml")
    parser.add_argument("--name", default="physicsnemo_compatibility_smoke_light_v1")
    parser.add_argument("--output-root", default="reports/research/sota_loop/external_baselines")
    parser.add_argument("--data-root")
    parser.add_argument(
        "--evidence-json",
        default="docs/claim_evidence/physicsnemo_compatibility_smoke_light_v1.json",
    )
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--tasks", nargs="+", default=[])
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="val")
    parser.add_argument("--max-train-samples", type=int, default=2)
    parser.add_argument("--max-eval-samples", type=int, default=2)
    parser.add_argument("--max-pairs-per-task", type=int, default=64)
    parser.add_argument("--rollout-steps", type=int, default=16)
    parser.add_argument("--train-stride", type=int, default=4)
    parser.add_argument("--latent-channels", type=int, default=32)
    parser.add_argument("--fourier-modes", type=int, default=12)
    parser.add_argument("--num-fno-layers", type=int, default=4)
    parser.add_argument("--decoder-layers", type=int, default=1)
    parser.add_argument("--decoder-layer-size", type=int, default=32)
    parser.add_argument("--padding", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--metric", default="decoded_rollout_nrmse")
    parser.add_argument("--residual", action="store_true")
    parser.add_argument(
        "--live-recipe",
        action="store_true",
        help="Run the optional train/validation PhysicsNeMo FNO recipe adapter.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="For --live-recipe, write the validation contract without importing PhysicsNeMo.",
    )
    parser.add_argument(
        "--live-import",
        action="store_true",
        help="Optionally import physicsnemo and record local import status.",
    )
    parser.add_argument(
        "--require-live-import",
        action="store_true",
        help="Fail if --live-import cannot import physicsnemo.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare the deterministic dry smoke manifest with --evidence-json.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.require_live_import and not args.live_import:
        raise RuntimeError("--require-live-import requires --live-import")
    if args.live_recipe:
        run_live_recipe_adapter(args)
        return
    if args.check:
        if not check_compatibility_smoke(args):
            raise SystemExit(1)
        return
    run_compatibility_smoke(args)


if __name__ == "__main__":
    main()
