#!/usr/bin/env python
from __future__ import annotations

"""Fit a train-split decoded residual gate, then validate the frozen config."""

import argparse
import copy
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts import evaluate as evaluate_script
from scripts import run_light_experiment as light_runner
from scripts.calibrate_residual_gate import _test_guard_result
from ups.core.latent_state import LatentState
from ups.data.latent_pairs import (
    infer_grid_shape,
    make_grid_coords,
    pdebench_condition_step,
    pdebench_conditioning_extras,
)
from ups.data.pdebench import PDEBenchConfig, PDEBenchDataset, get_pdebench_spec
from ups.eval.pdebench_runner import (
    _encode_grid_trajectory,
    _flatten_field_step,
    _gate_features,
    evaluate_decoded_operator,
)
from ups.utils.config_loader import load_config_with_includes

DEFAULT_FEATURE_NAMES = (
    "horizon_norm",
    "horizon_log",
    "residual_rms",
    "persistence_rms",
    "prediction_rms",
)


def _logit(value: float, *, eps: float = 1e-6) -> float:
    clipped = min(max(float(value), eps), 1.0 - eps)
    return math.log(clipped / (1.0 - clipped))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _task_names(cfg: Mapping[str, Any]) -> list[str]:
    task_cfg = cfg.get("data", {}).get("task")
    if isinstance(task_cfg, str):
        return [task_cfg]
    if isinstance(task_cfg, (list, tuple)) and task_cfg:
        return [str(task) for task in task_cfg]
    raise ValueError("data.task must be a task name or non-empty list of task names")


def _source_checkpoint_dir(source: str | Path) -> Path:
    source_path = Path(source)
    checkpoint_dir = (
        source_path / "checkpoints" if (source_path / "checkpoints").is_dir() else source_path
    )
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint source directory not found: {source}")
    return checkpoint_dir


def least_squares_residual_alpha(
    *,
    prediction: torch.Tensor,
    persistence: torch.Tensor,
    target: torch.Tensor,
    min_alpha: float = 0.0,
    max_alpha: float = 1.0,
    eps: float = 1e-12,
) -> float:
    """Project target-persistence onto prediction-persistence and clamp to gate bounds."""

    residual = (prediction - persistence).reshape(-1).double()
    target_delta = (target - persistence).reshape(-1).double()
    denom = float(torch.dot(residual, residual).item())
    if denom <= eps:
        return float(min_alpha)
    alpha = float(torch.dot(target_delta, residual).item() / denom)
    return min(max(alpha, float(min_alpha)), float(max_alpha))


def predict_gate_alpha(
    features: Mapping[str, float],
    config: Mapping[str, Any],
    *,
    task_name: str | None = None,
    task_family: str | None = None,
    horizon: int | None = None,
) -> float:
    score = _logit(float(config.get("base_alpha", 0.5)))
    score += float(config.get("bias", 0.0))
    for name, weight in (config.get("feature_weights", {}) or {}).items():
        score += float(weight) * float(features.get(str(name), 0.0))
    if task_name is not None:
        score += float((config.get("task_bias", {}) or {}).get(str(task_name), 0.0))
    if task_family is not None:
        score += float((config.get("family_bias", {}) or {}).get(str(task_family), 0.0))
    if horizon is not None:
        horizon_bias = config.get("horizon_bias", {}) or {}
        score += float(horizon_bias.get(str(horizon), horizon_bias.get(horizon, 0.0)))
    alpha = _sigmoid(score)
    return min(
        max(alpha, float(config.get("min_alpha", 0.0))),
        float(config.get("max_alpha", 1.0)),
    )


def gate_config_override(config: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(config), sort_keys=True, separators=(",", ":"))
    return f"evaluation.decoded_persistence_residual_gate={encoded}"


def train_logistic_gate(
    rows: Sequence[Mapping[str, Any]],
    *,
    feature_names: Sequence[str] = DEFAULT_FEATURE_NAMES,
    epochs: int = 1000,
    learning_rate: float = 0.05,
    l2: float = 1e-4,
    seed: int = 0,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot fit decoded residual gate without train rows")
    feature_names = tuple(str(name) for name in feature_names)
    if not feature_names:
        raise ValueError("At least one feature name is required")

    x_values = [
        [float(row.get("features", {}).get(name, 0.0)) for name in feature_names] for row in rows
    ]
    y_values = [float(row["target_alpha"]) for row in rows]
    x_raw = torch.tensor(x_values, dtype=torch.float64)
    y = torch.tensor(y_values, dtype=torch.float64).clamp(0.0, 1.0)
    mean = x_raw.mean(dim=0)
    std = x_raw.std(dim=0, unbiased=False)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)
    x = (x_raw - mean) / std

    torch.manual_seed(int(seed))
    weights = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
    bias = torch.tensor(_logit(float(y.mean().item())), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([weights, bias], lr=float(learning_rate))
    for _ in range(int(epochs)):
        pred = torch.sigmoid(x @ weights + bias)
        loss = torch.mean((pred - y) ** 2)
        if l2:
            loss = loss + float(l2) * torch.mean(weights**2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        raw_weights = weights / std
        raw_bias = bias - torch.sum(weights * mean / std)
        pred = torch.sigmoid(x_raw @ raw_weights + raw_bias)
        train_mse = float(torch.mean((pred - y) ** 2).item())
        config = {
            "base_alpha": 0.5,
            "bias": float(raw_bias.item()),
            "min_alpha": 0.0,
            "max_alpha": 1.0,
            "feature_weights": {
                name: float(value.item()) for name, value in zip(feature_names, raw_weights)
            },
        }
        return {
            "model": "logistic_decoded_residual_gate",
            "config": config,
            "feature_names": list(feature_names),
            "feature_stats": {
                name: {"mean": float(mu.item()), "std": float(sigma.item())}
                for name, mu, sigma in zip(feature_names, mean, std)
            },
            "train_rows": len(rows),
            "train_mse": train_mse,
            "target_alpha_mean": float(y.mean().item()),
            "predicted_alpha_mean": float(pred.mean().item()),
        }


def _load_models(
    cfg: Mapping[str, Any],
    *,
    checkpoint_source: str | Path,
    operator_checkpoint_names: Sequence[str],
    device: torch.device,
) -> tuple[Any, Any, Any, dict[str, str]]:
    checkpoint_dir = _source_checkpoint_dir(checkpoint_source)
    operator_ckpt = light_runner._preferred_checkpoint(checkpoint_dir, operator_checkpoint_names)
    encoder_ckpt = light_runner._preferred_checkpoint(
        checkpoint_dir, ("encoder_joint.pt", "encoder.pt")
    )
    decoder_ckpt = light_runner._preferred_checkpoint(
        checkpoint_dir, ("decoder_joint.pt", "decoder.pt")
    )
    if operator_ckpt is None:
        raise FileNotFoundError(f"No operator checkpoint found in {checkpoint_dir}")
    if encoder_ckpt is None or decoder_ckpt is None:
        raise FileNotFoundError(
            f"Decoded gate fitting requires encoder/decoder checkpoints in {checkpoint_dir}"
        )

    operator = evaluate_script.make_operator(dict(cfg))
    encoder = evaluate_script.make_encoder(dict(cfg))
    decoder = evaluate_script.make_decoder(dict(cfg))
    evaluate_script._load_state_dict_compat(operator, str(operator_ckpt), prefix_to_strip="")
    evaluate_script._load_state_dict_compat(encoder, str(encoder_ckpt), prefix_to_strip="")
    evaluate_script._load_state_dict_compat(decoder, str(decoder_ckpt), prefix_to_strip="")
    operator.to(device).eval()
    encoder.to(device).eval()
    decoder.to(device).eval()
    return (
        operator,
        encoder,
        decoder,
        {
            "operator": str(operator_ckpt),
            "encoder": str(encoder_ckpt),
            "decoder": str(decoder_ckpt),
        },
    )


def _cfg_for_split(cfg: Mapping[str, Any], *, split: str) -> dict[str, Any]:
    updated = copy.deepcopy(dict(cfg))
    updated.setdefault("data", {})["split"] = split
    return updated


def collect_gate_rows(
    cfg: Mapping[str, Any],
    *,
    checkpoint_source: str | Path,
    split: str,
    device: str | torch.device = "cpu",
    rollout_steps: int = 16,
    operator_checkpoint_names: Sequence[str] = light_runner.DEFAULT_OPERATOR_CHECKPOINTS,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
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
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for task_name in task_names:
            task_family = get_pdebench_spec(task_name).family
            try:
                dataset = PDEBenchDataset(
                    PDEBenchConfig(
                        task=task_name,
                        split=split,
                        root=data_cfg.get("root"),
                        param_keys=tuple(data_cfg.get("param_keys", ())),
                        bc_keys=tuple(data_cfg.get("bc_keys", ())),
                        max_samples=data_cfg.get("max_samples"),
                    )
                )
            except FileNotFoundError:
                if skip_missing_tasks:
                    continue
                raise
            if len(dataset) == 0:
                continue
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
                    if field_name not in decoded:
                        raise KeyError(f"Decoder did not produce requested field '{field_name}'")
                    prediction = decoded[field_name].detach().cpu()
                    persistence = _flatten_field_step(fields[step], grid_shape).cpu()
                    target = _flatten_field_step(fields[step + 1], grid_shape).cpu()
                    horizon = step + 1
                    features = _gate_features(
                        pred_field=prediction,
                        persistence_field=persistence,
                        horizon=horizon,
                        rollout_steps=steps,
                    )
                    rows.append(
                        {
                            "sample_index": int(sample_index),
                            "task": task_name,
                            "family": task_family,
                            "horizon": horizon,
                            "features": features,
                            "target_alpha": least_squares_residual_alpha(
                                prediction=prediction,
                                persistence=persistence,
                                target=target,
                            ),
                        }
                    )
    if not rows:
        raise RuntimeError(f"No decoded residual gate rows collected for split {split}")
    return rows, checkpoints


def evaluate_gate_config(
    cfg: Mapping[str, Any],
    *,
    checkpoint_source: str | Path,
    split: str,
    gate_config: Mapping[str, Any],
    device: str | torch.device = "cpu",
    rollout_steps: int = 16,
    operator_checkpoint_names: Sequence[str] = light_runner.DEFAULT_OPERATOR_CHECKPOINTS,
) -> dict[str, Any]:
    cfg_for_split = _cfg_for_split(cfg, split=split)
    cfg_for_split.setdefault("evaluation", {})["decoded_persistence_residual_alpha"] = 0.0
    cfg_for_split["evaluation"]["decoded_persistence_residual_gate"] = dict(gate_config)
    device = torch.device(device)
    operator, encoder, decoder, checkpoints = _load_models(
        cfg_for_split,
        checkpoint_source=checkpoint_source,
        operator_checkpoint_names=operator_checkpoint_names,
        device=device,
    )
    report = evaluate_decoded_operator(
        cfg_for_split,
        encoder,
        operator,
        decoder,
        device=device,
        rollout_steps=rollout_steps,
    )
    return {
        "split": split,
        "metrics": report.metrics,
        "extra": report.extra or {},
        "checkpoints": checkpoints,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit a decoded residual gate on train rows and validate the frozen gate"
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
    parser.add_argument(
        "--operator-checkpoint-name",
        action="append",
        default=None,
        help="Operator checkpoint preference order; repeatable",
    )
    parser.add_argument("--feature-name", action="append", default=None)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference-metric-value", type=float)
    parser.add_argument("--val-min-relative-improvement", type=float)
    parser.add_argument(
        "--output-json",
        default="reports/research/sota_loop/learned_residual_gate/fit_record.json",
    )
    parser.add_argument("--export-selected-gate-config")
    args = parser.parse_args()

    cfg = light_runner._apply_overrides(load_config_with_includes(args.config), args.override)
    cfg.setdefault("data", {})["root"] = args.data_root
    if args.eval_max_samples is not None:
        cfg["data"]["max_samples"] = int(args.eval_max_samples)
    cfg.setdefault("evaluation", {})["skip_missing_tasks"] = bool(
        cfg.get("evaluation", {}).get(
            "skip_missing_tasks", cfg["data"].get("skip_missing_tasks", False)
        )
    )
    operator_checkpoint_names = tuple(
        args.operator_checkpoint_name or light_runner.DEFAULT_OPERATOR_CHECKPOINTS
    )
    feature_names = tuple(args.feature_name or DEFAULT_FEATURE_NAMES)

    train_rows, checkpoints = collect_gate_rows(
        cfg,
        checkpoint_source=args.checkpoint_source,
        split=args.train_split,
        device=args.device,
        rollout_steps=args.decoded_rollout_steps,
        operator_checkpoint_names=operator_checkpoint_names,
    )
    fit = train_logistic_gate(
        train_rows,
        feature_names=feature_names,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        seed=args.seed,
    )
    train_eval = evaluate_gate_config(
        cfg,
        checkpoint_source=args.checkpoint_source,
        split=args.train_split,
        gate_config=fit["config"],
        device=args.device,
        rollout_steps=args.decoded_rollout_steps,
        operator_checkpoint_names=operator_checkpoint_names,
    )
    validation_eval = evaluate_gate_config(
        cfg,
        checkpoint_source=args.checkpoint_source,
        split=args.val_split,
        gate_config=fit["config"],
        device=args.device,
        rollout_steps=args.decoded_rollout_steps,
        operator_checkpoint_names=operator_checkpoint_names,
    )
    validation_metric = float(validation_eval["metrics"]["decoded_rollout_nrmse"])
    validation_guard = _test_guard_result(
        value=validation_metric,
        reference=args.reference_metric_value,
        min_relative_improvement=args.val_min_relative_improvement,
        mode="min",
    )
    record = {
        "model": "learned_decoded_residual_gate",
        "config": args.config,
        "checkpoint_source": args.checkpoint_source,
        "operator_checkpoint_names": list(operator_checkpoint_names),
        "checkpoints": checkpoints,
        "data_root": args.data_root,
        "train_split": args.train_split,
        "val_split": args.val_split,
        "eval_max_samples": args.eval_max_samples,
        "decoded_rollout_steps": args.decoded_rollout_steps,
        "fit": fit,
        "train": train_eval,
        "validation": validation_eval,
        "validation_guard": validation_guard,
        "selected_override": gate_config_override(fit["config"]),
        "held_out_test_policy": "No held-out test is run by this fitter; run a guarded test only if validation_guard.passed is true.",
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    if args.export_selected_gate_config:
        selected_path = Path(args.export_selected_gate_config)
        selected_path.parent.mkdir(parents=True, exist_ok=True)
        selected_path.write_text(
            json.dumps(fit["config"], indent=2, sort_keys=True), encoding="utf-8"
        )
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
