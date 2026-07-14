#!/usr/bin/env python
from __future__ import annotations

"""Train a decoded-space residual refiner on train split, then validate it."""

import argparse
import copy
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts import run_light_experiment as light_runner
from scripts.calibrate_residual_gate import _test_guard_result
from scripts.fit_decoded_residual_gate import _cfg_for_split, _load_models, _task_names
from ups.core.latent_state import LatentState
from ups.data.latent_pairs import (
    infer_grid_shape,
    make_grid_coords,
    pdebench_condition_step,
    pdebench_conditioning_extras,
)
from ups.data.pdebench import PDEBenchConfig, PDEBenchDataset, get_pdebench_spec
from ups.eval.metrics import nrmse, relative_rrmse, spectral_energy_error
from ups.eval.pdebench_runner import _encode_grid_trajectory, _flatten_field_step
from ups.utils.config_loader import load_config_with_includes


class DecodedResidualRefiner(nn.Module):
    def __init__(self, *, input_dim: int, output_dim: int = 1, hidden_dim: int = 64) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def build_refiner_features(
    *,
    prediction: torch.Tensor,
    persistence: torch.Tensor,
    coords: torch.Tensor,
    horizon: int,
    rollout_steps: int,
    context_features: torch.Tensor | None = None,
) -> torch.Tensor:
    if prediction.shape != persistence.shape:
        raise ValueError("prediction and persistence shapes must match")
    if prediction.dim() != 3:
        raise ValueError(f"Expected flattened fields shaped (B, N, C), got {prediction.shape}")
    batch, nodes, _channels = prediction.shape
    coords_batch = coords
    if coords_batch.dim() == 2:
        coords_batch = coords_batch.unsqueeze(0)
    if coords_batch.shape[0] == 1 and batch != 1:
        coords_batch = coords_batch.expand(batch, -1, -1)
    if coords_batch.shape[:2] != (batch, nodes):
        raise ValueError(
            f"coords shape {tuple(coords.shape)} is incompatible with field shape {tuple(prediction.shape)}"
        )
    horizon_norm = torch.full(
        (batch, nodes, 1),
        float(horizon) / max(float(rollout_steps), 1.0),
        dtype=prediction.dtype,
        device=prediction.device,
    )
    horizon_log = torch.full(
        (batch, nodes, 1),
        torch.log1p(torch.tensor(float(horizon), dtype=prediction.dtype)).item(),
        dtype=prediction.dtype,
        device=prediction.device,
    )
    chunks = [
        prediction,
        persistence,
        prediction - persistence,
        coords_batch.to(device=prediction.device, dtype=prediction.dtype),
        horizon_norm,
        horizon_log,
    ]
    if context_features is not None:
        context = context_features.to(device=prediction.device, dtype=prediction.dtype)
        if context.dim() == 1:
            context = context.view(1, 1, -1).expand(batch, nodes, -1)
        elif context.dim() == 2:
            context = context.view(batch, 1, -1).expand(batch, nodes, -1)
        elif context.dim() != 3:
            raise ValueError("context_features must be shaped (D,), (B,D), or (B,N,D)")
        if context.shape[:2] != (batch, nodes):
            raise ValueError(
                f"context_features shape {tuple(context_features.shape)} is incompatible with field shape {tuple(prediction.shape)}"
            )
        chunks.append(context)
    features = torch.cat(tuple(chunks), dim=-1)
    return features.reshape(batch * nodes, features.shape[-1])


def train_refiner_from_tensors(
    features: torch.Tensor,
    target_delta: torch.Tensor,
    *,
    hidden_dim: int = 64,
    epochs: int = 500,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 0,
) -> tuple[DecodedResidualRefiner, dict[str, Any]]:
    if features.dim() != 2:
        raise ValueError("features must be a 2D tensor")
    if target_delta.dim() != 2:
        raise ValueError("target_delta must be a 2D tensor")
    if features.shape[0] != target_delta.shape[0]:
        raise ValueError("features and target_delta row counts must match")
    torch.manual_seed(int(seed))
    refiner = DecodedResidualRefiner(
        input_dim=int(features.shape[1]),
        output_dim=int(target_delta.shape[1]),
        hidden_dim=int(hidden_dim),
    )
    optimizer = torch.optim.AdamW(
        refiner.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    best_loss = float("inf")
    best_state = copy.deepcopy(refiner.state_dict())
    for _epoch in range(int(epochs)):
        pred_delta = refiner(features)
        loss = torch.mean((pred_delta - target_delta) ** 2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().item())
        if loss_value < best_loss:
            best_loss = loss_value
            best_state = copy.deepcopy(refiner.state_dict())
    refiner.load_state_dict(best_state)
    return refiner, {
        "model": "decoded_residual_refiner_mlp",
        "train_rows": int(features.shape[0]),
        "input_dim": int(features.shape[1]),
        "output_dim": int(target_delta.shape[1]),
        "hidden_dim": int(hidden_dim),
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "train_mse": best_loss,
    }


def subsample_refiner_tensors(
    tensors: Mapping[str, torch.Tensor], *, max_rows: int | None, seed: int
) -> dict[str, torch.Tensor]:
    row_count = int(tensors["features"].shape[0])
    if max_rows is None or int(max_rows) <= 0 or row_count <= int(max_rows):
        return {key: value for key, value in tensors.items()}
    generator = torch.Generator().manual_seed(int(seed))
    indices = torch.randperm(row_count, generator=generator)[: int(max_rows)]
    return {key: value.index_select(0, indices) for key, value in tensors.items()}


def evaluate_refiner_tensors(
    refiner: DecodedResidualRefiner,
    features: torch.Tensor,
    persistence: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    with torch.no_grad():
        refined = persistence + refiner(features)
    return {
        "mse": float(torch.mean((refined - target) ** 2).item()),
        "nrmse": float(nrmse(refined, target).item()),
        "rrmse": float(relative_rrmse(refined, target).item()),
        "spectral_energy_error": float(spectral_energy_error(refined.T, target.T).item()),
    }


def _load_dataset(
    cfg: Mapping[str, Any],
    *,
    task_name: str,
    split: str,
) -> PDEBenchDataset:
    data_cfg = cfg.get("data", {})
    return PDEBenchDataset(
        PDEBenchConfig(
            task=task_name,
            split=split,
            root=data_cfg.get("root"),
            param_keys=tuple(data_cfg.get("param_keys", ())),
            bc_keys=tuple(data_cfg.get("bc_keys", ())),
            max_samples=data_cfg.get("max_samples"),
        )
    )


def _task_context(task_name: str, task_names: Sequence[str]) -> torch.Tensor:
    context = torch.zeros(len(task_names), dtype=torch.float32)
    context[list(task_names).index(task_name)] = 1.0
    return context


def collect_refiner_tensors(
    cfg: Mapping[str, Any],
    *,
    checkpoint_source: str | Path,
    split: str,
    device: str | torch.device = "cpu",
    rollout_steps: int = 16,
    operator_checkpoint_names: Sequence[str] = light_runner.DEFAULT_OPERATOR_CHECKPOINTS,
) -> tuple[dict[str, torch.Tensor], dict[str, str], dict[str, Any]]:
    cfg_for_split = _cfg_for_split(cfg, split=split)
    device = torch.device(device)
    operator, encoder, decoder, checkpoints = _load_models(
        cfg_for_split,
        checkpoint_source=checkpoint_source,
        operator_checkpoint_names=operator_checkpoint_names,
        device=device,
    )
    data_cfg = cfg_for_split.get("data", {})
    task_names = _task_names(cfg_for_split)
    field_name = str(data_cfg.get("field_name", "u"))
    dt_tensor = torch.tensor(cfg_for_split.get("training", {}).get("dt", 0.1), device=device)
    skip_missing_tasks = bool(
        cfg_for_split.get("evaluation", {}).get(
            "skip_missing_tasks", data_cfg.get("skip_missing_tasks", False)
        )
    )
    feature_chunks: list[torch.Tensor] = []
    persistence_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    metadata = {
        "split": split,
        "task_count": 0,
        "sample_count": 0,
        "row_count": 0,
        "tasks": {},
        "skipped_missing_tasks": [],
    }
    with torch.no_grad():
        for task_name in task_names:
            task_family = get_pdebench_spec(task_name).family
            try:
                dataset = _load_dataset(cfg_for_split, task_name=task_name, split=split)
            except FileNotFoundError:
                if skip_missing_tasks:
                    metadata["skipped_missing_tasks"].append(task_name)
                    continue
                raise
            if len(dataset) == 0:
                continue
            metadata["task_count"] += 1
            task_rows = 0
            sample_fields = dataset[0]["fields"]
            grid_shape = infer_grid_shape(sample_fields)
            coords = make_grid_coords(grid_shape, device)
            base_cond: dict[str, torch.Tensor] = {}
            if bool(cfg_for_split.get("training", {}).get("auto_conditioning", False)):
                extras = pdebench_conditioning_extras(
                    task_name=task_name,
                    grid_shape=grid_shape,
                    task_vocab=task_names,
                )
                base_cond = {key: value.to(device) for key, value in extras.items()}
            param_vocab = tuple(data_cfg.get("param_keys", ()))
            bc_vocab = tuple(data_cfg.get("bc_keys", ()))
            for sample_index in range(len(dataset)):
                sample = dataset[sample_index]
                metadata["sample_count"] += 1
                fields = sample["fields"].float()
                latent_seq = _encode_grid_trajectory(
                    encoder,
                    fields,
                    coords,
                    grid_shape,
                    field_name=field_name,
                    device=device,
                )
                steps = min(int(rollout_steps), latent_seq.shape[0] - 1)
                if steps <= 0:
                    continue
                params = sample.get("params")
                bc = sample.get("bc")
                initial_cond = pdebench_condition_step(
                    params,
                    bc,
                    batch_size=1,
                    step=0,
                    extras=base_cond,
                    param_vocab=param_vocab,
                    bc_vocab=bc_vocab,
                )
                state = LatentState(
                    z=latent_seq[0:1].to(device),
                    t=torch.tensor(0.0, device=device),
                    cond={key: value.to(device) for key, value in initial_cond.items()},
                )
                for step in range(steps):
                    cond = pdebench_condition_step(
                        params,
                        bc,
                        batch_size=1,
                        step=step,
                        extras=base_cond,
                        param_vocab=param_vocab,
                        bc_vocab=bc_vocab,
                    )
                    state = LatentState(
                        z=state.z,
                        t=state.t,
                        cond={key: value.to(device) for key, value in cond.items()},
                    )
                    state = operator(state, dt_tensor)
                    decoded = decoder(coords, state.z, conditioning={})
                    prediction = decoded[field_name].detach().cpu()
                    persistence = _flatten_field_step(fields[step], grid_shape).cpu()
                    target = _flatten_field_step(fields[step + 1], grid_shape).cpu()
                    features = build_refiner_features(
                        prediction=prediction,
                        persistence=persistence,
                        coords=coords.cpu(),
                        horizon=step + 1,
                        rollout_steps=steps,
                        context_features=_task_context(task_name, task_names),
                    )
                    target_delta = (target - persistence).reshape(-1, target.shape[-1])
                    feature_chunks.append(features.cpu())
                    persistence_chunks.append(persistence.reshape(-1, persistence.shape[-1]).cpu())
                    target_chunks.append(target.reshape(-1, target.shape[-1]).cpu())
                    task_rows += int(features.shape[0])
            metadata["tasks"][task_name] = {
                "family": task_family,
                "samples": len(dataset),
                "rows": task_rows,
            }
            metadata["row_count"] += task_rows
    if not feature_chunks:
        raise RuntimeError(f"No decoded residual refiner rows collected for split {split}")
    tensors = {
        "features": torch.cat(feature_chunks, dim=0).float(),
        "persistence": torch.cat(persistence_chunks, dim=0).float(),
        "target": torch.cat(target_chunks, dim=0).float(),
    }
    tensors["target_delta"] = tensors["target"] - tensors["persistence"]
    return tensors, checkpoints, metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a decoded residual refiner on train split and validate on val"
    )
    parser.add_argument("--config", default="configs/train_multitask_heterogeneous_light_best.yaml")
    parser.add_argument("--checkpoint-source", required=True)
    parser.add_argument("--data-root", default="data/pdebench")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--eval-max-samples", type=int, default=32)
    parser.add_argument("--decoded-rollout-steps", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--operator-checkpoint-name", action="append", default=None)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-train-rows", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference-metric-value", type=float)
    parser.add_argument("--val-min-relative-improvement", type=float)
    parser.add_argument(
        "--output-json",
        default="reports/research/sota_loop/decoded_residual_refiner/fit_record.json",
    )
    parser.add_argument("--checkpoint-out")
    args = parser.parse_args()

    cfg = light_runner._apply_overrides(load_config_with_includes(args.config), args.override)
    cfg.setdefault("data", {})["root"] = args.data_root
    if args.eval_max_samples is not None:
        cfg["data"]["max_samples"] = int(args.eval_max_samples)
    operator_checkpoint_names = tuple(
        args.operator_checkpoint_name or light_runner.DEFAULT_OPERATOR_CHECKPOINTS
    )

    train_tensors, checkpoints, train_meta = collect_refiner_tensors(
        cfg,
        checkpoint_source=args.checkpoint_source,
        split=args.train_split,
        device=args.device,
        rollout_steps=args.decoded_rollout_steps,
        operator_checkpoint_names=operator_checkpoint_names,
    )
    full_train_row_count = int(train_tensors["features"].shape[0])
    train_tensors = subsample_refiner_tensors(
        train_tensors, max_rows=args.max_train_rows, seed=args.seed
    )
    train_meta["full_row_count"] = full_train_row_count
    train_meta["fit_row_count"] = int(train_tensors["features"].shape[0])
    refiner, fit = train_refiner_from_tensors(
        train_tensors["features"],
        train_tensors["target_delta"],
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    train_metrics = evaluate_refiner_tensors(
        refiner,
        train_tensors["features"],
        train_tensors["persistence"],
        train_tensors["target"],
    )
    val_tensors, _val_checkpoints, val_meta = collect_refiner_tensors(
        cfg,
        checkpoint_source=args.checkpoint_source,
        split=args.val_split,
        device=args.device,
        rollout_steps=args.decoded_rollout_steps,
        operator_checkpoint_names=operator_checkpoint_names,
    )
    validation_metrics = evaluate_refiner_tensors(
        refiner,
        val_tensors["features"],
        val_tensors["persistence"],
        val_tensors["target"],
    )
    validation_metric = float(validation_metrics["nrmse"])
    validation_guard = _test_guard_result(
        value=validation_metric,
        reference=args.reference_metric_value,
        min_relative_improvement=args.val_min_relative_improvement,
        mode="min",
    )
    checkpoint_path = Path(args.checkpoint_out) if args.checkpoint_out else None
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": refiner.state_dict(),
                "fit": fit,
                "config": {
                    "input_dim": refiner.input_dim,
                    "output_dim": refiner.output_dim,
                    "hidden_dim": refiner.hidden_dim,
                },
            },
            checkpoint_path,
        )
    record = {
        "model": "decoded_residual_refiner_mlp",
        "config": args.config,
        "checkpoint_source": args.checkpoint_source,
        "operator_checkpoint_names": list(operator_checkpoint_names),
        "checkpoints": checkpoints,
        "data_root": args.data_root,
        "train_split": args.train_split,
        "val_split": args.val_split,
        "eval_max_samples": args.eval_max_samples,
        "decoded_rollout_steps": args.decoded_rollout_steps,
        "max_train_rows": args.max_train_rows,
        "fit": fit,
        "train": {"metrics": train_metrics, "metadata": train_meta},
        "validation": {"metrics": validation_metrics, "metadata": val_meta},
        "validation_guard": validation_guard,
        "checkpoint_out": str(checkpoint_path) if checkpoint_path is not None else None,
        "held_out_test_policy": "No held-out test is run by this refiner; run a guarded test only if validation_guard.passed is true.",
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
