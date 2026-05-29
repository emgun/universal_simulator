#!/usr/bin/env python
from __future__ import annotations

"""Train/evaluate an optional NeuralOperator FNO baseline on the light-v1 protocol."""

import argparse
import csv
import importlib.metadata as metadata
import json
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
from ups.data.pdebench import get_pdebench_spec
from ups.eval.pdebench_runner import _aggregate_chunk_metrics, _flatten_field_step

NEURALOP_IMPORT = "neuralop.models.FNO"
NEURALOP_SOURCE_URL = "https://github.com/neuraloperator/neuraloperator"


class MissingNeuralOperatorError(RuntimeError):
    """Raised when a live external FNO run is requested without neuralop installed."""


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
    return models, fit


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
                    raise ValueError(f"No trained external FNO baseline for group {key}")
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
    return command


def _summary_common(args: argparse.Namespace, *, tasks: Sequence[str]) -> dict[str, Any]:
    return {
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
    tasks = _as_task_names(cfg, args.tasks or args.task)
    if args.dry_run:
        return write_dry_run_summary(args, tasks=tasks)
    if args.eval_split == "test" and not args.allow_held_out_test_eval:
        raise RuntimeError(
            "Live external FNO evaluation on split=test requires --allow-held-out-test-eval. "
            "Use --eval-split val while debugging adapter behavior."
        )

    load_neuraloperator_fno_class()
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
    models, fit = train_fno_groups(
        groups,
        hidden_channels=args.hidden_channels,
        fourier_modes=args.fourier_modes,
        n_layers=args.n_layers,
        residual=args.residual,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )
    metrics = evaluate_external_fno_baseline(
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
    group_manifest = run_dir / "fno_groups.tsv"
    _write_group_manifest(group_manifest, fit)
    summary_path = run_dir / "summary.json"
    summary = {
        "status": "complete",
        "metrics": metrics,
        **_summary_common(args, tasks=tasks),
        "details": {"fit": fit, "group_manifest": str(group_manifest)},
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
        default="reports/research/sota_loop/shared_context_decoded_eval/test_ledger.json",
    )
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-held-out-test-eval", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_baseline(args)


if __name__ == "__main__":
    main()
