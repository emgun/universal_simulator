#!/usr/bin/env python
from __future__ import annotations

"""Run the frozen, validation-only Darcy FNO parameter-conditioning ablation."""

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.run_external_neuraloperator_fno_baseline import (  # noqa: E402
    _begin_compute_tracking,
    _clone_tensor_state_dict,
    fno_modes_for_grid,
    load_neuraloperator_fno_class,
    neuraloperator_import_status,
)
from scripts.run_physical_conv_baseline import field_step_to_grid  # noqa: E402
from ups.data.baseline_runtime import (  # noqa: E402
    FROZEN_STRAT_V1_TRAINING_LOCK_SHA256,
    load_strat_v1_baseline_runtime,
)
from ups.data.latent_pairs import infer_grid_shape  # noqa: E402
from ups.data.manifests import canonical_sha256  # noqa: E402
from ups.data.pdebench import PDEBenchDataset  # noqa: E402
from ups.eval.pdebench_runner import _aggregate_chunk_metrics  # noqa: E402
from ups.eval.regime_metrics import (  # noqa: E402
    aligned_element_count,
    global_scale_regime_nrmse,
    regime_spread_ratio,
)

ARMS = ("U", "K")
RUNG_EPOCHS = (3, 6, 12, 24)
SEED = 17
SHUFFLE_SEED = 17017
TASK = "darcy2d"
PRIMARY_METRIC = "decoded_solution_nrmse"
DEFAULT_LOCK = REPO_ROOT / (
    "docs/data/releases/strat-v1/universal/"
    "9d43d283f04f5b8d17cf6126ad189075c53307e715d7d4f61af440c2fed155c1/"
    "training.lock.json"
)


@dataclass(frozen=True)
class BetaNormalizer:
    mean_log10: float
    std_log10: float
    count: int

    @classmethod
    def fit(cls, beta: torch.Tensor) -> BetaNormalizer:
        values = beta.detach().double().flatten()
        if values.numel() == 0 or not torch.isfinite(values).all() or (values <= 0).any():
            raise ValueError("training beta values must be finite, positive, and nonempty")
        logs = torch.log10(values)
        std = float(logs.std(unbiased=False).item())
        if not math.isfinite(std) or std <= 0:
            raise ValueError("training beta values must have nonzero log10 variance")
        return cls(float(logs.mean().item()), std, int(values.numel()))

    def transform(self, beta: torch.Tensor) -> torch.Tensor:
        values = beta.float()
        if not torch.isfinite(values).all() or (values <= 0).any():
            raise ValueError("beta values must be finite and positive")
        return (torch.log10(values) - self.mean_log10) / self.std_log10

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": "train_only_zscore_log10_beta_population_std",
            "mean_log10": self.mean_log10,
            "std_log10": self.std_log10,
            "fit_sample_count": self.count,
        }


class DarcyFNOAdapter(nn.Module):
    """FNO mapping conditioned inputs to one explicit Darcy solution channel."""

    def __init__(self, fno: nn.Module) -> None:
        super().__init__()
        self.fno = fno

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fno(inputs)


def build_model(
    *,
    input_channels: int,
    grid_shape: tuple[int, int],
    hidden_channels: int,
    fourier_modes: int,
    n_layers: int,
    fno_cls: type[nn.Module] | None = None,
) -> nn.Module:
    cls = fno_cls or load_neuraloperator_fno_class()
    return DarcyFNOAdapter(
        cls(
            n_modes=fno_modes_for_grid(grid_shape, fourier_modes),
            in_channels=int(input_channels),
            out_channels=1,
            hidden_channels=int(hidden_channels),
            n_layers=int(n_layers),
        )
    )


def conditioned_inputs(
    coefficients: torch.Tensor,
    beta: torch.Tensor,
    *,
    arm: str,
    normalizer: BetaNormalizer,
) -> torch.Tensor:
    """Construct U or K input without modifying the coefficient channel."""

    if coefficients.ndim != 4 or coefficients.shape[1] != 1:
        raise ValueError("Darcy coefficients must have shape (N,1,H,W)")
    beta = beta.flatten()
    if beta.numel() != coefficients.shape[0]:
        raise ValueError("one beta value is required per coefficient field")
    if arm == "U":
        return coefficients
    if arm != "K":
        raise ValueError(f"unknown ablation arm {arm!r}")
    normalized = normalizer.transform(beta).to(coefficients).view(-1, 1, 1, 1)
    shape = (-1, 1, coefficients.shape[2], coefficients.shape[3])
    beta_channel = normalized.expand(shape)
    presence_channel = torch.ones_like(beta_channel)
    return torch.cat((coefficients, beta_channel, presence_channel), dim=1)


def _collect(dataset: PDEBenchDataset) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    coefficients: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    beta: list[float] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        fields = sample["fields"].float()
        explicit_targets = sample.get("targets")
        if explicit_targets is None:
            raise ValueError("Darcy ablation requires explicit solution targets")
        explicit_targets = explicit_targets.float()
        if fields.shape[0] != 1 or explicit_targets.shape[0] != 1:
            raise ValueError("Darcy ablation requires steady one-input/one-target semantics")
        parameter = sample.get("params", {}).get("beta")
        if parameter is None or parameter.numel() != 1:
            raise ValueError("Darcy ablation requires one scalar beta per sample")
        grid_shape = infer_grid_shape(fields)
        coefficients.append(field_step_to_grid(fields[0], grid_shape).squeeze(0))
        targets.append(field_step_to_grid(explicit_targets[0], grid_shape).squeeze(0))
        beta.append(float(parameter.item()))
    if not coefficients:
        raise RuntimeError("Darcy ablation received no samples")
    return torch.stack(coefficients), torch.stack(targets), torch.tensor(beta)


def train_arm(
    coefficients: torch.Tensor,
    targets: torch.Tensor,
    beta: torch.Tensor,
    *,
    arm: str,
    normalizer: BetaNormalizer,
    hidden_channels: int,
    fourier_modes: int,
    n_layers: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    device: str | torch.device,
    fno_cls: type[nn.Module] | None = None,
) -> tuple[dict[int, nn.Module], dict[str, Any]]:
    """Train one continuous trajectory and snapshot exactly the frozen rungs."""

    torch.manual_seed(SEED)
    device = torch.device(device)
    inputs = conditioned_inputs(coefficients, beta, arm=arm, normalizer=normalizer)
    model = build_model(
        input_channels=int(inputs.shape[1]),
        grid_shape=(int(inputs.shape[2]), int(inputs.shape[3])),
        hidden_channels=hidden_channels,
        fourier_modes=fourier_modes,
        n_layers=n_layers,
        fno_cls=fno_cls,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    generator = torch.Generator().manual_seed(SEED)
    snapshots: dict[int, dict[str, torch.Tensor]] = {}
    epoch_mse: list[float] = []
    steps = 0
    for epoch in range(1, RUNG_EPOCHS[-1] + 1):
        order = torch.randperm(inputs.shape[0], generator=generator)
        loss_sum = 0.0
        batches = 0
        for start in range(0, inputs.shape[0], batch_size):
            indices = order[start : start + batch_size]
            x = inputs.index_select(0, indices).to(device)
            y = targets.index_select(0, indices).to(device)
            prediction = model(x)
            loss = torch.mean((prediction - y) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu())
            batches += 1
            steps += 1
        epoch_mse.append(loss_sum / max(batches, 1))
        if epoch in RUNG_EPOCHS:
            snapshots[epoch] = _clone_tensor_state_dict(model)
    rung_models: dict[int, nn.Module] = {}
    for epoch, state in snapshots.items():
        clone = build_model(
            input_channels=int(inputs.shape[1]),
            grid_shape=(int(inputs.shape[2]), int(inputs.shape[3])),
            hidden_channels=hidden_channels,
            fourier_modes=fourier_modes,
            n_layers=n_layers,
            fno_cls=fno_cls,
        ).cpu()
        clone.load_state_dict(state)
        rung_models[epoch] = clone
    fit = {
        "arm": arm,
        "seed": SEED,
        "rungs": list(RUNG_EPOCHS),
        "train_samples": int(inputs.shape[0]),
        "input_channels": int(inputs.shape[1]),
        "output_channels": 1,
        "epoch_train_mse": epoch_mse,
        "optimizer_steps": steps,
        "examples_seen": int(inputs.shape[0]) * RUNG_EPOCHS[-1],
        "batch_size": int(batch_size),
        "sample_order": "torch_randperm_continuous_generator_seed_17",
    }
    return rung_models, fit


def evaluate_arm(
    model: nn.Module,
    coefficients: torch.Tensor,
    targets: torch.Tensor,
    beta: torch.Tensor,
    *,
    arm: str,
    normalizer: BetaNormalizer,
    device: str | torch.device = "cpu",
    conditioning_beta: torch.Tensor | None = None,
) -> dict[str, Any]:
    channel_beta = beta if conditioning_beta is None else conditioning_beta
    inputs = conditioned_inputs(coefficients, channel_beta, arm=arm, normalizer=normalizer)
    with torch.no_grad():
        predictions = model.to(device).eval()(inputs.to(device)).cpu()
    pred_chunks = [predictions[index : index + 1] for index in range(len(predictions))]
    target_chunks = [targets[index : index + 1] for index in range(len(targets))]
    primary = float(_aggregate_chunk_metrics(pred_chunks, target_chunks)["nrmse"])
    regimes = []
    for value in sorted(set(float(item) for item in beta.tolist())):
        indices = torch.where(torch.isclose(beta, torch.tensor(value)))[0]
        regime_pred = [predictions[index : index + 1] for index in indices.tolist()]
        regime_target = [targets[index : index + 1] for index in indices.tolist()]
        raw = float(_aggregate_chunk_metrics(regime_pred, regime_target)["nrmse"])
        corrected = float(global_scale_regime_nrmse(regime_pred, regime_target, target_chunks))
        regimes.append(
            {
                "beta": value,
                "slice_normalized_nrmse": raw,
                "global_scale_nrmse": corrected,
                "element_count": aligned_element_count(regime_pred, regime_target),
                "spread_ratio_to_primary": regime_spread_ratio(corrected, primary),
            }
        )
    return {
        "primary_metric": PRIMARY_METRIC,
        "primary_value": primary,
        "per_beta": regimes,
        "maximum_corrected_spread_ratio": max(
            item["spread_ratio_to_primary"] for item in regimes
        ),
        "predictions": predictions,
    }


def deterministic_beta_permutation(count: int, *, seed: int = SHUFFLE_SEED) -> torch.Tensor:
    if count < 2:
        raise ValueError("shuffled-beta ablation requires at least two samples")
    generator = torch.Generator().manual_seed(int(seed))
    permutation = torch.randperm(count, generator=generator)
    if torch.equal(permutation, torch.arange(count)):
        permutation = torch.roll(permutation, 1)
    return permutation


def beta_diagnostics(
    *,
    selected_models: dict[str, nn.Module],
    coefficients: torch.Tensor,
    targets: torch.Tensor,
    beta: torch.Tensor,
    normalizer: BetaNormalizer,
    device: str | torch.device,
) -> dict[str, Any]:
    permutation = deterministic_beta_permutation(len(beta))
    shuffled_beta = beta.index_select(0, permutation)
    shuffled = evaluate_arm(
        selected_models["K"], coefficients, targets, beta,
        arm="K", normalizer=normalizer, device=device, conditioning_beta=shuffled_beta,
    )
    regimes = torch.tensor(sorted(set(float(item) for item in beta.tolist())))
    sensitivity: dict[str, Any] = {}
    for arm in ARMS:
        predictions = []
        for value in regimes:
            counterfactual = torch.full_like(beta, float(value))
            inputs = conditioned_inputs(
                coefficients, counterfactual, arm=arm, normalizer=normalizer
            )
            with torch.no_grad():
                predictions.append(selected_models[arm].to(device).eval()(inputs.to(device)).cpu())
        stacked = torch.stack(predictions)
        # Referencing one counterfactual avoids floating-point mean noise and
        # makes the structural invariance of arm U exactly auditable.
        deltas = stacked - stacked[:1]
        rms = float(torch.sqrt(torch.mean(deltas.square())).item())
        scale = float(torch.sqrt(torch.mean(stacked.square())).item())
        sensitivity[arm] = {
            "counterfactual_beta_values": regimes.tolist(),
            "prediction_rms_from_first_beta": rms,
            "relative_prediction_rms_from_first_beta": rms / max(scale, 1e-12),
        }
    permutation_sha = hashlib.sha256(
        json.dumps(permutation.tolist(), separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "counterfactual_beta_sensitivity": sensitivity,
        "deterministic_shuffled_beta": {
            "seed": SHUFFLE_SEED,
            "permutation_sha256": permutation_sha,
            "fixed_point_count": int((permutation == torch.arange(len(beta))).sum()),
            "primary_metric": PRIMARY_METRIC,
            "primary_value": shuffled["primary_value"],
            "maximum_corrected_spread_ratio": shuffled["maximum_corrected_spread_ratio"],
            "per_beta_true_regime": shuffled["per_beta"],
        },
    }


def _checkpoint(
    path: Path, model: nn.Module, *, arm: str, epoch: int, fit: dict[str, Any]
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "model_family": "neuraloperator_fno",
        "arm": arm,
        "epoch": epoch,
        "state_dict": _clone_tensor_state_dict(model),
        "fit": fit,
    }
    torch.save(payload, path)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "epoch": epoch,
    }


def git_commit() -> str:
    value = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    if len(value) != 40:
        raise RuntimeError("could not resolve an exact git commit")
    return value


def run(args: argparse.Namespace, *, fno_cls: type[nn.Module] | None = None) -> Path:
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    runtime = load_strat_v1_baseline_runtime(
        args.training_lock,
        args.data_root,
        expected_lock_sha256=FROZEN_STRAT_V1_TRAINING_LOCK_SHA256,
    )
    darcy_runtime = runtime.tasks[TASK]
    train_dataset = PDEBenchDataset(
        runtime.dataset_config(TASK, "train", condition_on_regime=True)
    )
    valid_dataset = PDEBenchDataset(
        runtime.dataset_config(TASK, "val", condition_on_regime=True)
    )
    train_coefficients, train_targets, train_beta = _collect(train_dataset)
    valid_coefficients, valid_targets, valid_beta = _collect(valid_dataset)
    normalizer = BetaNormalizer.fit(train_beta)
    output_dir.mkdir(parents=True)
    _begin_compute_tracking(args.device)
    started = time.time()
    arm_results: dict[str, Any] = {}
    selected_models: dict[str, nn.Module] = {}
    for arm in ARMS:
        _begin_compute_tracking(args.device)
        arm_started = time.time()
        rung_models, fit = train_arm(
            train_coefficients,
            train_targets,
            train_beta,
            arm=arm,
            normalizer=normalizer,
            hidden_channels=args.hidden_channels,
            fourier_modes=args.fourier_modes,
            n_layers=args.n_layers,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            device=args.device,
            fno_cls=fno_cls,
        )
        history = []
        checkpoints = {}
        for epoch in RUNG_EPOCHS:
            evaluation = evaluate_arm(
                rung_models[epoch], valid_coefficients, valid_targets, valid_beta,
                arm=arm, normalizer=normalizer, device=args.device,
            )
            predictions = evaluation.pop("predictions")
            del predictions
            history.append({"epoch": epoch, **evaluation})
            checkpoints[str(epoch)] = _checkpoint(
                output_dir / f"arm_{arm}_epoch_{epoch}.pt",
                rung_models[epoch], arm=arm, epoch=epoch, fit=fit,
            )
        selected = min(history, key=lambda item: (item["primary_value"], item["epoch"]))
        selected_epoch = int(selected["epoch"])
        selected_models[arm] = rung_models[selected_epoch]
        parameters = list(selected_models[arm].parameters())
        compute = {
            "duration_sec": time.time() - arm_started,
            "device": args.device,
            "total_parameter_count": sum(item.numel() for item in parameters),
            "trainable_parameter_count": sum(
                item.numel() for item in parameters if item.requires_grad
            ),
            "optimizer_steps": fit["optimizer_steps"],
            "examples_seen": fit["examples_seen"],
        }
        resolved_device = torch.device(args.device)
        if resolved_device.type == "cuda" and torch.cuda.is_available():
            compute.update(
                {
                    "cuda_device_name": torch.cuda.get_device_name(resolved_device),
                    "peak_cuda_memory_bytes": int(
                        torch.cuda.max_memory_allocated(resolved_device)
                    ),
                }
            )
        arm_results[arm] = {
            "conditioning": (
                "coefficient_only" if arm == "U" else
                "coefficient_plus_normalized_log10_beta_constant_plus_presence"
            ),
            "fit": fit,
            "validation_history": history,
            "selection": {
                "rule": "minimum_finite_validation_primary_earliest_tie",
                "metric": PRIMARY_METRIC,
                "selected_epoch": selected_epoch,
                "selected_value": selected["primary_value"],
            },
            "checkpoints": {
                "rungs": checkpoints,
                "selected": checkpoints[str(selected_epoch)],
            },
            "compute": compute,
        }
    diagnostics = beta_diagnostics(
        selected_models=selected_models,
        coefficients=valid_coefficients,
        targets=valid_targets,
        beta=valid_beta,
        normalizer=normalizer,
        device=args.device,
    )
    lock_objects = {
        item.object_id: {
            "role": item.role,
            "path": item.path,
            "sha256": item.checksums["sha256"],
            "size_bytes": item.size_bytes,
        }
        for item in runtime.lock.objects
        if item.object_id in {"darcy2d-train", "darcy2d-valid"}
    }
    summary = {
        "schema_version": 1,
        "artifact_id": "strat-v1-darcy-fno-conditioning-ablation",
        "status": "complete_validation_only",
        "task": TASK,
        "semantics": "steady_coefficient_to_explicit_solution_target",
        "held_out_reads": 0,
        "source": {"git_commit": git_commit()},
        "training_lock": {
            "path": str(runtime.lock_path),
            "lock_sha256": runtime.lock.lock_sha256,
            "selection_sha256": runtime.selection_sha256,
            "darcy_objects": lock_objects,
        },
        "sample_counts": {"train": len(train_beta), "valid": len(valid_beta)},
        "regime_counts": {
            "train": [vars(item) for item in darcy_runtime.train.regimes],
            "valid": [vars(item) for item in darcy_runtime.valid.regimes],
        },
        "beta_normalization": normalizer.as_dict(),
        "matched_design": {
            "arms": list(ARMS),
            "seed": SEED,
            "rungs": list(RUNG_EPOCHS),
            "same_architecture_hidden_modes_layers": True,
            "same_data_order_updates": True,
            "selection_role": "valid",
            "selection_metric": PRIMARY_METRIC,
        },
        "architecture": {
            "implementation": "neuralop.models.FNO",
            "dependency": neuraloperator_import_status(),
            "hidden_channels": args.hidden_channels,
            "fourier_modes": args.fourier_modes,
            "n_layers": args.n_layers,
            "output_channels": 1,
        },
        "arms": arm_results,
        "diagnostics": diagnostics,
        "compute": {"duration_sec": time.time() - started, "device": args.device},
    }
    summary["artifact_sha256"] = canonical_sha256(summary)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen validation-only Darcy FNO conditioning ablation"
    )
    parser.add_argument("--training-lock", default=str(DEFAULT_LOCK))
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument("--fourier-modes", type=int, default=16)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    print(json.dumps({"summary": str(run(args))}, indent=2))


if __name__ == "__main__":
    main()
