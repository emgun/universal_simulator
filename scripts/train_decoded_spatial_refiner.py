#!/usr/bin/env python
from __future__ import annotations

"""Train a spatial decoded residual refiner on train split, then validate it."""

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


class SpatialDecodedResidualRefiner(nn.Module):
    def __init__(
        self, *, input_channels: int, output_channels: int = 1, hidden_channels: int = 32
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.hidden_channels = int(hidden_channels)
        self.network = nn.Sequential(
            nn.Conv2d(self.input_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(self.hidden_channels, self.output_channels, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def flat_to_grid(field: torch.Tensor, grid_shape: tuple[int, int]) -> torch.Tensor:
    if field.dim() != 3:
        raise ValueError(f"Expected field shaped (B,N,C), got {tuple(field.shape)}")
    batch, nodes, channels = field.shape
    height, width = grid_shape
    if nodes != height * width:
        raise ValueError(f"Field has {nodes} nodes, expected {height * width}")
    return field.transpose(1, 2).reshape(batch, channels, height, width).contiguous()


def grid_to_flat(grid: torch.Tensor) -> torch.Tensor:
    if grid.dim() != 4:
        raise ValueError(f"Expected grid shaped (B,C,H,W), got {tuple(grid.shape)}")
    batch, channels, height, width = grid.shape
    return grid.reshape(batch, channels, height * width).transpose(1, 2).contiguous()


def build_spatial_refiner_input(
    *,
    prediction: torch.Tensor,
    persistence: torch.Tensor,
    coords: torch.Tensor,
    grid_shape: tuple[int, int],
    horizon: int,
    rollout_steps: int,
    context_features: torch.Tensor | None = None,
) -> torch.Tensor:
    if prediction.shape != persistence.shape:
        raise ValueError("prediction and persistence shapes must match")
    prediction_grid = flat_to_grid(prediction, grid_shape)
    persistence_grid = flat_to_grid(persistence, grid_shape)
    coords_batch = coords
    if coords_batch.dim() == 2:
        coords_batch = coords_batch.unsqueeze(0)
    if coords_batch.shape[0] == 1 and prediction.shape[0] != 1:
        coords_batch = coords_batch.expand(prediction.shape[0], -1, -1)
    coords_grid = flat_to_grid(coords_batch.to(dtype=prediction.dtype), grid_shape)
    horizon_norm = torch.full(
        (prediction_grid.shape[0], 1, *grid_shape),
        float(horizon) / max(float(rollout_steps), 1.0),
        dtype=prediction.dtype,
        device=prediction.device,
    )
    chunks = [
        prediction_grid,
        persistence_grid,
        prediction_grid - persistence_grid,
        coords_grid.to(device=prediction.device),
        horizon_norm,
    ]
    if context_features is not None:
        context = context_features.to(device=prediction.device, dtype=prediction.dtype)
        if context.dim() == 1:
            context = context.view(1, -1, 1, 1).expand(prediction_grid.shape[0], -1, *grid_shape)
        elif context.dim() == 2:
            context = context.view(prediction_grid.shape[0], -1, 1, 1).expand(-1, -1, *grid_shape)
        elif context.dim() == 4:
            pass
        else:
            raise ValueError("context_features must be shaped (D,), (B,D), or (B,D,H,W)")
        if context.shape[0] != prediction_grid.shape[0] or context.shape[-2:] != grid_shape:
            raise ValueError(
                f"context_features shape {tuple(context_features.shape)} is incompatible with grid shape {grid_shape}"
            )
        chunks.append(context)
    return torch.cat(tuple(chunks), dim=1)


def _as_groups(value: torch.Tensor | Sequence[torch.Tensor]) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    return [tensor for tensor in value]


def _batched_indices(row_count: int, *, batch_size: int, seed: int) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(int(seed))
    indices = torch.randperm(row_count, generator=generator)
    return [indices[start : start + batch_size] for start in range(0, row_count, batch_size)]


def train_spatial_refiner_from_tensors(
    features: torch.Tensor | Sequence[torch.Tensor],
    target_delta: torch.Tensor | Sequence[torch.Tensor],
    *,
    hidden_channels: int = 32,
    epochs: int = 300,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 16,
    seed: int = 0,
) -> tuple[SpatialDecodedResidualRefiner, dict[str, Any]]:
    feature_groups = _as_groups(features)
    target_groups = _as_groups(target_delta)
    if len(feature_groups) != len(target_groups):
        raise ValueError("features and target_delta group counts must match")
    input_channels = int(feature_groups[0].shape[1])
    output_channels = int(target_groups[0].shape[1])
    for feature_group, target_group in zip(feature_groups, target_groups):
        if feature_group.shape[0] != target_group.shape[0]:
            raise ValueError("features and target_delta frame counts must match")
        if int(feature_group.shape[1]) != input_channels:
            raise ValueError("all feature groups must have the same channel count")
        if int(target_group.shape[1]) != output_channels:
            raise ValueError("all target groups must have the same output channel count")

    torch.manual_seed(int(seed))
    refiner = SpatialDecodedResidualRefiner(
        input_channels=input_channels,
        output_channels=output_channels,
        hidden_channels=int(hidden_channels),
    )
    optimizer = torch.optim.AdamW(
        refiner.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    best_loss = float("inf")
    best_state = copy.deepcopy(refiner.state_dict())
    for epoch in range(int(epochs)):
        total_loss = 0.0
        batch_count = 0
        for group_index, (feature_group, target_group) in enumerate(
            zip(feature_groups, target_groups)
        ):
            for indices in _batched_indices(
                int(feature_group.shape[0]),
                batch_size=max(int(batch_size), 1),
                seed=int(seed) + epoch * 1009 + group_index,
            ):
                pred = refiner(feature_group.index_select(0, indices))
                loss = torch.mean((pred - target_group.index_select(0, indices)) ** 2)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().item())
                batch_count += 1
        mean_loss = total_loss / max(batch_count, 1)
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_state = copy.deepcopy(refiner.state_dict())
    refiner.load_state_dict(best_state)
    return refiner, {
        "model": "decoded_spatial_residual_refiner_cnn",
        "train_frames": sum(int(group.shape[0]) for group in feature_groups),
        "group_count": len(feature_groups),
        "input_channels": input_channels,
        "output_channels": output_channels,
        "hidden_channels": int(hidden_channels),
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "batch_size": int(batch_size),
        "train_mse": best_loss,
    }


def evaluate_spatial_refiner_tensors(
    refiner: SpatialDecodedResidualRefiner,
    features: torch.Tensor | Sequence[torch.Tensor],
    persistence: torch.Tensor | Sequence[torch.Tensor],
    target: torch.Tensor | Sequence[torch.Tensor],
) -> dict[str, float]:
    feature_groups = _as_groups(features)
    persistence_groups = _as_groups(persistence)
    target_groups = _as_groups(target)
    refined_flat: list[torch.Tensor] = []
    target_flat: list[torch.Tensor] = []
    with torch.no_grad():
        for feature_group, persistence_group, target_group in zip(
            feature_groups, persistence_groups, target_groups
        ):
            refined = persistence_group + refiner(feature_group)
            refined_flat.append(refined.reshape(-1))
            target_flat.append(target_group.reshape(-1))
    refined_all = torch.cat(refined_flat, dim=0)
    target_all = torch.cat(target_flat, dim=0)
    return {
        "mse": float(torch.mean((refined_all - target_all) ** 2).item()),
        "nrmse": float(nrmse(refined_all, target_all).item()),
        "rrmse": float(relative_rrmse(refined_all, target_all).item()),
        "spectral_energy_error": float(
            spectral_energy_error(refined_all.view(1, -1), target_all.view(1, -1)).item()
        ),
    }


def _task_context(task_name: str, task_names: Sequence[str]) -> torch.Tensor:
    context = torch.zeros(len(task_names), dtype=torch.float32)
    context[list(task_names).index(task_name)] = 1.0
    return context


def _load_dataset(cfg: Mapping[str, Any], *, task_name: str, split: str) -> PDEBenchDataset:
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


def _append_grouped(
    groups: dict[tuple[int, ...], list[torch.Tensor]], tensor: torch.Tensor
) -> None:
    groups.setdefault(tuple(tensor.shape[1:]), []).append(tensor.cpu())


def _stack_groups(groups: Mapping[tuple[int, ...], list[torch.Tensor]]) -> list[torch.Tensor]:
    return [torch.cat(chunks, dim=0).float() for _shape, chunks in sorted(groups.items())]


def _balanced_sample_counts(group_sizes: Sequence[int], max_frames: int) -> list[int]:
    total_frames = sum(int(size) for size in group_sizes)
    target_frames = min(int(max_frames), total_frames)
    if target_frames <= 0 or target_frames >= total_frames:
        return [int(size) for size in group_sizes]
    raw_counts = [target_frames * (int(size) / total_frames) for size in group_sizes]
    counts = [min(int(size), int(raw_count)) for size, raw_count in zip(group_sizes, raw_counts)]
    nonempty_count = sum(1 for size in group_sizes if int(size) > 0)
    if target_frames >= nonempty_count:
        for index, size in enumerate(group_sizes):
            if int(size) > 0 and counts[index] == 0:
                counts[index] = 1
    while sum(counts) > target_frames:
        candidates = [index for index, count in enumerate(counts) if count > 0]
        shrink_index = min(candidates, key=lambda index: (raw_counts[index], counts[index]))
        counts[shrink_index] -= 1
    remaining = target_frames - sum(counts)
    order = sorted(
        range(len(group_sizes)),
        key=lambda index: (raw_counts[index] - int(raw_counts[index]), raw_counts[index]),
        reverse=True,
    )
    while remaining > 0:
        progressed = False
        for index in order:
            if remaining <= 0:
                break
            if counts[index] < int(group_sizes[index]):
                counts[index] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    return counts


def subsample_spatial_tensors(
    tensors: Mapping[str, Sequence[torch.Tensor]], *, max_frames: int | None, seed: int
) -> dict[str, list[torch.Tensor]]:
    features = list(tensors["features"])
    persistence = list(tensors["persistence"])
    target = list(tensors["target"])
    target_delta = list(tensors["target_delta"])
    total_frames = sum(int(group.shape[0]) for group in features)
    if max_frames is None or int(max_frames) <= 0 or total_frames <= int(max_frames):
        return {
            "features": features,
            "persistence": persistence,
            "target": target,
            "target_delta": target_delta,
        }
    generator = torch.Generator().manual_seed(int(seed))
    sampled = {"features": [], "persistence": [], "target": [], "target_delta": []}
    sample_counts = _balanced_sample_counts(
        [int(feature_group.shape[0]) for feature_group in features],
        max_frames=int(max_frames),
    )
    for group_index, (feature_group, take) in enumerate(zip(features, sample_counts)):
        group_frames = int(feature_group.shape[0])
        if take <= 0:
            continue
        indices = torch.randperm(group_frames, generator=generator)[:take]
        sampled["features"].append(feature_group.index_select(0, indices))
        sampled["persistence"].append(persistence[group_index].index_select(0, indices))
        sampled["target"].append(target[group_index].index_select(0, indices))
        sampled["target_delta"].append(target_delta[group_index].index_select(0, indices))
    return sampled


def collect_spatial_tensors(
    cfg: Mapping[str, Any],
    *,
    checkpoint_source: str | Path,
    split: str,
    device: str | torch.device = "cpu",
    rollout_steps: int = 16,
    operator_checkpoint_names: Sequence[str] = light_runner.DEFAULT_OPERATOR_CHECKPOINTS,
) -> tuple[dict[str, list[torch.Tensor]], dict[str, str], dict[str, Any]]:
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
    feature_groups: dict[tuple[int, ...], list[torch.Tensor]] = {}
    persistence_groups: dict[tuple[int, ...], list[torch.Tensor]] = {}
    target_groups: dict[tuple[int, ...], list[torch.Tensor]] = {}
    target_delta_groups: dict[tuple[int, ...], list[torch.Tensor]] = {}
    metadata = {
        "split": split,
        "task_count": 0,
        "sample_count": 0,
        "frame_count": 0,
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
            task_frames = 0
            sample_fields = dataset.fields[0]
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
                    features = build_spatial_refiner_input(
                        prediction=prediction,
                        persistence=persistence,
                        coords=coords.cpu(),
                        grid_shape=grid_shape,
                        horizon=step + 1,
                        rollout_steps=steps,
                        context_features=_task_context(task_name, task_names),
                    )
                    persistence_grid = flat_to_grid(persistence, grid_shape)
                    target_grid = flat_to_grid(target, grid_shape)
                    target_delta_grid = target_grid - persistence_grid
                    _append_grouped(feature_groups, features)
                    _append_grouped(persistence_groups, persistence_grid)
                    _append_grouped(target_groups, target_grid)
                    _append_grouped(target_delta_groups, target_delta_grid)
                    task_frames += 1
            metadata["tasks"][task_name] = {
                "family": task_family,
                "samples": len(dataset),
                "frames": task_frames,
            }
            metadata["frame_count"] += task_frames
    if not feature_groups:
        raise RuntimeError(f"No decoded spatial refiner frames collected for split {split}")
    tensors = {
        "features": _stack_groups(feature_groups),
        "persistence": _stack_groups(persistence_groups),
        "target": _stack_groups(target_groups),
        "target_delta": _stack_groups(target_delta_groups),
    }
    return tensors, checkpoints, metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a spatial decoded residual refiner on train split and validate on val"
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
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-train-frames", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference-metric-value", type=float)
    parser.add_argument("--val-min-relative-improvement", type=float)
    parser.add_argument(
        "--output-json",
        default="reports/research/sota_loop/decoded_spatial_refiner/fit_record.json",
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
    train_tensors, checkpoints, train_meta = collect_spatial_tensors(
        cfg,
        checkpoint_source=args.checkpoint_source,
        split=args.train_split,
        device=args.device,
        rollout_steps=args.decoded_rollout_steps,
        operator_checkpoint_names=operator_checkpoint_names,
    )
    full_train_frame_count = sum(int(group.shape[0]) for group in train_tensors["features"])
    train_tensors = subsample_spatial_tensors(
        train_tensors, max_frames=args.max_train_frames, seed=args.seed
    )
    train_meta["full_frame_count"] = full_train_frame_count
    train_meta["fit_frame_count"] = sum(int(group.shape[0]) for group in train_tensors["features"])
    refiner, fit = train_spatial_refiner_from_tensors(
        train_tensors["features"],
        train_tensors["target_delta"],
        hidden_channels=args.hidden_channels,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    train_metrics = evaluate_spatial_refiner_tensors(
        refiner,
        train_tensors["features"],
        train_tensors["persistence"],
        train_tensors["target"],
    )
    val_tensors, _val_checkpoints, val_meta = collect_spatial_tensors(
        cfg,
        checkpoint_source=args.checkpoint_source,
        split=args.val_split,
        device=args.device,
        rollout_steps=args.decoded_rollout_steps,
        operator_checkpoint_names=operator_checkpoint_names,
    )
    validation_metrics = evaluate_spatial_refiner_tensors(
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
                    "input_channels": refiner.input_channels,
                    "output_channels": refiner.output_channels,
                    "hidden_channels": refiner.hidden_channels,
                },
            },
            checkpoint_path,
        )
    record = {
        "model": "decoded_spatial_residual_refiner_cnn",
        "config": args.config,
        "checkpoint_source": args.checkpoint_source,
        "operator_checkpoint_names": list(operator_checkpoint_names),
        "checkpoints": checkpoints,
        "data_root": args.data_root,
        "train_split": args.train_split,
        "val_split": args.val_split,
        "eval_max_samples": args.eval_max_samples,
        "decoded_rollout_steps": args.decoded_rollout_steps,
        "max_train_frames": args.max_train_frames,
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
