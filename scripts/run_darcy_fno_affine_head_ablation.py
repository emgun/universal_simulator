#!/usr/bin/env python
from __future__ import annotations

"""Run the frozen validation-only Darcy K-long versus affine-head ablation."""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.run_darcy_fno_conditioning_ablation import (  # noqa: E402
    BetaNormalizer,
    _collect,
    deterministic_beta_permutation,
)
from scripts.run_external_neuraloperator_fno_baseline import (  # noqa: E402
    _begin_compute_tracking,
    _clone_tensor_state_dict,
    fno_modes_for_grid,
    load_neuraloperator_fno_class,
    neuraloperator_import_status,
)
from ups.data.baseline_runtime import (  # noqa: E402
    FROZEN_STRAT_V1_TRAINING_LOCK_SHA256,
    load_strat_v1_baseline_runtime,
)
from ups.data.manifests import canonical_sha256  # noqa: E402
from ups.data.pdebench import PDEBenchDataset  # noqa: E402
from ups.eval.pdebench_runner import _aggregate_chunk_metrics  # noqa: E402
from ups.eval.regime_metrics import (  # noqa: E402
    aligned_element_count,
    global_scale_regime_nrmse,
    regime_spread_ratio,
)
from ups.training.resumable_checkpoint import (  # noqa: E402
    CheckpointBindings,
    TrainingProgress,
    checkpoint_record_path,
    load_training_checkpoint,
    save_training_checkpoint,
    verify_checkpoint_record,
)

ARMS = ("K-long", "A-affine")
RUNG_EPOCHS = (3, 6, 12, 24, 48, 96, 192)
SEED = 17
SHUFFLE_SEED = 17017
TASK = "darcy2d"
PRIMARY_METRIC = "decoded_solution_nrmse"
EXPECTED_DARCY_OBJECTS = {
    "darcy2d-train": "47945f27fa1f56f856733d3bc1aa1b0b5f498669a73cdb7352940292d71d09fe",
    "darcy2d-valid": "2b345a587f6f95a9ff4a12f6cce80ac4c8c83540a03c2a11f87ffdc91be1b595",
}
DEFAULT_LOCK = REPO_ROOT / (
    "docs/data/releases/strat-v1/universal/"
    "9d43d283f04f5b8d17cf6126ad189075c53307e715d7d4f61af440c2fed155c1/"
    "training.lock.json"
)


@dataclass(frozen=True)
class RawBetaNormalizer:
    mean: float
    std: float
    count: int

    @classmethod
    def fit(cls, beta: torch.Tensor) -> RawBetaNormalizer:
        values = beta.detach().double().flatten()
        if values.numel() == 0 or not torch.isfinite(values).all() or (values <= 0).any():
            raise ValueError("training beta values must be finite, positive, and nonempty")
        std = float(values.std(unbiased=False).item())
        if not math.isfinite(std) or std <= 0:
            raise ValueError("training beta values must have nonzero raw variance")
        return cls(float(values.mean().item()), std, int(values.numel()))

    def transform(self, beta: torch.Tensor) -> torch.Tensor:
        values = beta.float()
        if not torch.isfinite(values).all() or (values <= 0).any():
            raise ValueError("beta values must be finite and positive")
        return (values - self.mean) / self.std

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": "train_only_zscore_raw_beta_population_std",
            "mean": self.mean,
            "std": self.std,
            "fit_sample_count": self.count,
        }


class DarcyFNO(nn.Module):
    """Thin adapter retaining an explicit, auditable FNO output contract."""

    def __init__(self, fno: nn.Module) -> None:
        super().__init__()
        self.fno = fno

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fno(inputs)


def build_model(
    *,
    arm: str,
    grid_shape: tuple[int, int],
    hidden_channels: int,
    fourier_modes: int,
    n_layers: int,
    fno_cls: type[nn.Module] | None = None,
) -> nn.Module:
    cls = fno_cls or load_neuraloperator_fno_class()
    if arm == "K-long":
        input_channels, output_channels = 3, 1
    elif arm == "A-affine":
        input_channels, output_channels = 2, 2
    else:
        raise ValueError(f"unknown ablation arm {arm!r}")
    return DarcyFNO(
        cls(
            n_modes=fno_modes_for_grid(grid_shape, fourier_modes),
            in_channels=input_channels,
            out_channels=output_channels,
            hidden_channels=int(hidden_channels),
            n_layers=int(n_layers),
        )
    )


def model_inputs(
    coefficients: torch.Tensor, beta: torch.Tensor, *, arm: str, log_normalizer: BetaNormalizer
) -> torch.Tensor:
    if coefficients.ndim != 4 or coefficients.shape[1] != 1:
        raise ValueError("Darcy coefficients must have shape (N,1,H,W)")
    if beta.numel() != coefficients.shape[0]:
        raise ValueError("one beta value is required per coefficient field")
    presence = torch.ones_like(coefficients)
    if arm == "A-affine":
        return torch.cat((coefficients, presence), dim=1)
    if arm != "K-long":
        raise ValueError(f"unknown ablation arm {arm!r}")
    z_log = log_normalizer.transform(beta).to(coefficients).view(-1, 1, 1, 1)
    return torch.cat((coefficients, z_log.expand_as(coefficients), presence), dim=1)


def predict(
    model: nn.Module,
    coefficients: torch.Tensor,
    beta: torch.Tensor,
    *,
    arm: str,
    log_normalizer: BetaNormalizer,
    raw_normalizer: RawBetaNormalizer,
) -> torch.Tensor:
    outputs = model(model_inputs(coefficients, beta, arm=arm, log_normalizer=log_normalizer))
    if arm == "K-long":
        if outputs.shape[1] != 1:
            raise ValueError("K-long FNO must emit one solution channel")
        return outputs
    if outputs.shape[1] != 2:
        raise ValueError("A-affine FNO must emit exactly two basis fields")
    z_raw = raw_normalizer.transform(beta).to(outputs).view(-1, 1, 1, 1)
    return outputs[:, :1] + z_raw * outputs[:, 1:2]


def _model_spec(*, arm: str, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "family": "neuraloperator_fno",
        "arm": arm,
        "hidden_channels": int(args.hidden_channels),
        "fourier_modes": int(args.fourier_modes),
        "n_layers": int(args.n_layers),
        "input_channels": 3 if arm == "K-long" else 2,
        "output_channels": 1 if arm == "K-long" else 2,
    }


def _record_dict(path: Path) -> dict[str, Any]:
    record = verify_checkpoint_record(path)
    return {
        "path": str(path),
        "record_path": str(checkpoint_record_path(path)),
        "sha256": record.checkpoint_sha256,
        "size_bytes": record.checkpoint_bytes,
        "parent_checkpoint_sha256": record.parent_checkpoint_sha256,
        "record_self_sha256": record.self_hash["value"],
    }


def train_arm(
    coefficients: torch.Tensor,
    targets: torch.Tensor,
    beta: torch.Tensor,
    *,
    arm: str,
    log_normalizer: BetaNormalizer,
    raw_normalizer: RawBetaNormalizer,
    hidden_channels: int,
    fourier_modes: int,
    n_layers: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    device: str | torch.device,
    fno_cls: type[nn.Module] | None = None,
    checkpoint_dir: Path | None = None,
    checkpoint_bindings: CheckpointBindings | None = None,
    resume: bool = False,
) -> tuple[dict[int, nn.Module], dict[str, Any]]:
    """Train one continuous, deterministic trajectory through all D2 rungs."""

    torch.manual_seed(SEED)
    device = torch.device(device)
    model = build_model(
        arm=arm,
        grid_shape=(int(coefficients.shape[2]), int(coefficients.shape[3])),
        hidden_channels=hidden_channels,
        fourier_modes=fourier_modes,
        n_layers=n_layers,
        fno_cls=fno_cls,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(SEED)
    snapshots: dict[int, dict[str, torch.Tensor]] = {}
    epoch_mse: list[float] = []
    steps = 0
    examples_seen = 0
    start_epoch = 1
    parent_sha: str | None = None
    checkpoint_records: dict[str, Any] = {}
    resumed_from: dict[str, Any] | None = None
    if (checkpoint_dir is None) != (checkpoint_bindings is None):
        raise ValueError("checkpoint_dir and checkpoint_bindings must be provided together")
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        present: list[tuple[int, Path]] = []
        gap_seen = False
        prior_sha: str | None = None
        for epoch in RUNG_EPOCHS:
            path = checkpoint_dir / f"arm_{arm}_epoch_{epoch}.pt"
            pair_exists = path.exists() or checkpoint_record_path(path).exists()
            if not pair_exists:
                gap_seen = True
                continue
            if gap_seen:
                raise RuntimeError(f"non-contiguous resumable checkpoint chain for arm {arm}")
            record = verify_checkpoint_record(path)
            if record.parent_checkpoint_sha256 != prior_sha:
                raise RuntimeError(f"invalid resumable checkpoint lineage for arm {arm}")
            checkpoint_records[str(epoch)] = _record_dict(path)
            present.append((epoch, path))
            prior_sha = record.checkpoint_sha256
        if present and not resume:
            raise FileExistsError(
                f"checkpoints already exist for arm {arm}; explicit resume required"
            )
        if resume and present:
            for index, (epoch, path) in enumerate(present):
                clone = build_model(
                    arm=arm,
                    grid_shape=(int(coefficients.shape[2]), int(coefficients.shape[3])),
                    hidden_channels=hidden_channels,
                    fourier_modes=fourier_modes,
                    n_layers=n_layers,
                    fno_cls=fno_cls,
                ).cpu()
                clone_optimizer = torch.optim.AdamW(
                    clone.parameters(), lr=learning_rate, weight_decay=weight_decay
                )
                clone_generator = torch.Generator().manual_seed(SEED)
                expected_parent = None if index == 0 else present[index - 1][1]
                expected_parent_sha = (
                    None
                    if expected_parent is None
                    else verify_checkpoint_record(expected_parent).checkpoint_sha256
                )
                loaded = load_training_checkpoint(
                    path,
                    model=clone,
                    optimizer=clone_optimizer,
                    sampler_generator=clone_generator,
                    expected_bindings=checkpoint_bindings,
                    expected_parent_checkpoint_sha256=expected_parent_sha,
                    map_location="cpu",
                )
                if loaded.progress.completed_epoch != epoch:
                    raise RuntimeError("checkpoint progress does not match its rung")
                snapshots[epoch] = _clone_tensor_state_dict(clone)
            latest_epoch, latest_path = present[-1]
            latest_parent = None if len(present) == 1 else present[-2][1]
            latest_parent_sha = (
                None
                if latest_parent is None
                else verify_checkpoint_record(latest_parent).checkpoint_sha256
            )
            loaded = load_training_checkpoint(
                latest_path,
                model=model,
                optimizer=optimizer,
                sampler_generator=generator,
                expected_bindings=checkpoint_bindings,
                expected_parent_checkpoint_sha256=latest_parent_sha,
                map_location=device,
            )
            epoch_mse = [float(item) for item in loaded.progress.history]
            steps = loaded.progress.steps
            examples_seen = loaded.progress.examples
            start_epoch = latest_epoch + 1
            parent_sha = loaded.record.checkpoint_sha256
            resumed_from = checkpoint_records[str(latest_epoch)]
    for epoch in range(start_epoch, RUNG_EPOCHS[-1] + 1):
        order = torch.randperm(coefficients.shape[0], generator=generator)
        loss_sum = 0.0
        batches = 0
        for start in range(0, coefficients.shape[0], batch_size):
            indices = order[start : start + batch_size]
            x = coefficients.index_select(0, indices).to(device)
            y = targets.index_select(0, indices).to(device)
            b = beta.index_select(0, indices).to(device)
            prediction = predict(
                model,
                x,
                b,
                arm=arm,
                log_normalizer=log_normalizer,
                raw_normalizer=raw_normalizer,
            )
            loss = torch.mean((prediction - y) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu())
            batches += 1
            steps += 1
            examples_seen += int(indices.numel())
        epoch_mse.append(loss_sum / max(batches, 1))
        if epoch in RUNG_EPOCHS:
            snapshots[epoch] = _clone_tensor_state_dict(model)
            if checkpoint_dir is not None and checkpoint_bindings is not None:
                path = checkpoint_dir / f"arm_{arm}_epoch_{epoch}.pt"
                record = save_training_checkpoint(
                    path,
                    model=model,
                    optimizer=optimizer,
                    progress=TrainingProgress(
                        completed_epoch=epoch,
                        steps=steps,
                        examples=examples_seen,
                        history=epoch_mse,
                    ),
                    sampler_generator=generator,
                    bindings=checkpoint_bindings,
                    parent_checkpoint_sha256=parent_sha,
                )
                parent_sha = record.checkpoint_sha256
                checkpoint_records[str(epoch)] = _record_dict(path)
    models: dict[int, nn.Module] = {}
    for epoch, state in snapshots.items():
        clone = build_model(
            arm=arm,
            grid_shape=(int(coefficients.shape[2]), int(coefficients.shape[3])),
            hidden_channels=hidden_channels,
            fourier_modes=fourier_modes,
            n_layers=n_layers,
            fno_cls=fno_cls,
        ).cpu()
        clone.load_state_dict(state)
        models[epoch] = clone
    fit = {
        "arm": arm,
        "seed": SEED,
        "rungs": list(RUNG_EPOCHS),
        "train_samples": int(coefficients.shape[0]),
        "input_channels": 3 if arm == "K-long" else 2,
        "output_channels": 1 if arm == "K-long" else 2,
        "reconstruction": (
            "direct_scalar" if arm == "K-long" else "h0_plus_zscore_raw_beta_times_h1"
        ),
        "epoch_train_mse": epoch_mse,
        "optimizer_steps": steps,
        "examples_seen": examples_seen,
        "batch_size": int(batch_size),
        "sample_order": "torch_randperm_continuous_generator_seed_17",
        "resumable_checkpoints": checkpoint_records,
        "resume_provenance": {
            "resumed": resumed_from is not None,
            "resumed_from": resumed_from,
            "trajectory_end_epoch": RUNG_EPOCHS[-1],
        },
    }
    return models, fit


def evaluate_arm(
    model: nn.Module,
    coefficients: torch.Tensor,
    targets: torch.Tensor,
    beta: torch.Tensor,
    *,
    arm: str,
    log_normalizer: BetaNormalizer,
    raw_normalizer: RawBetaNormalizer,
    device: str | torch.device = "cpu",
    conditioning_beta: torch.Tensor | None = None,
    batch_size: int = 8,
) -> dict[str, Any]:
    channel_beta = beta if conditioning_beta is None else conditioning_beta
    predictions = _predict_batched(
        model,
        coefficients,
        channel_beta,
        arm=arm,
        log_normalizer=log_normalizer,
        raw_normalizer=raw_normalizer,
        device=device,
        batch_size=batch_size,
    )
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
        "maximum_corrected_spread_ratio": max(item["spread_ratio_to_primary"] for item in regimes),
        "predictions": predictions,
    }


def _predict_batched(
    model: nn.Module,
    coefficients: torch.Tensor,
    beta: torch.Tensor,
    *,
    arm: str,
    log_normalizer: BetaNormalizer,
    raw_normalizer: RawBetaNormalizer,
    device: str | torch.device,
    batch_size: int,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError("evaluation batch_size must be positive")
    if len(coefficients) != len(beta):
        raise ValueError("evaluation coefficients and beta must have equal length")
    resolved_device = torch.device(device)
    chunks: list[torch.Tensor] = []
    runtime_model = model.to(resolved_device).eval()
    try:
        with torch.no_grad():
            for start in range(0, len(coefficients), batch_size):
                stop = min(start + batch_size, len(coefficients))
                chunks.append(
                    predict(
                        runtime_model,
                        coefficients[start:stop].to(resolved_device),
                        beta[start:stop].to(resolved_device),
                        arm=arm,
                        log_normalizer=log_normalizer,
                        raw_normalizer=raw_normalizer,
                    ).cpu()
                )
    finally:
        model.to("cpu")
        if resolved_device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
    return (
        torch.cat(chunks, dim=0)
        if chunks
        else coefficients.new_empty((0, 1, *coefficients.shape[2:]))
    )


def beta_diagnostics(
    *,
    selected_models: Mapping[str, nn.Module],
    coefficients: torch.Tensor,
    targets: torch.Tensor,
    beta: torch.Tensor,
    log_normalizer: BetaNormalizer,
    raw_normalizer: RawBetaNormalizer,
    device: str | torch.device,
    batch_size: int = 8,
) -> dict[str, Any]:
    permutation = deterministic_beta_permutation(len(beta), seed=SHUFFLE_SEED)
    shuffled_beta = beta.index_select(0, permutation)
    shuffled: dict[str, Any] = {}
    sensitivity: dict[str, Any] = {}
    regimes = torch.tensor(sorted(set(float(item) for item in beta.tolist())))
    for arm in ARMS:
        reference_eval = evaluate_arm(
            selected_models[arm],
            coefficients,
            targets,
            beta,
            arm=arm,
            log_normalizer=log_normalizer,
            raw_normalizer=raw_normalizer,
            device=device,
            batch_size=batch_size,
        )
        shuffled_eval = evaluate_arm(
            selected_models[arm],
            coefficients,
            targets,
            beta,
            arm=arm,
            log_normalizer=log_normalizer,
            raw_normalizer=raw_normalizer,
            device=device,
            conditioning_beta=shuffled_beta,
            batch_size=batch_size,
        )
        reference_primary = float(reference_eval["primary_value"])
        shuffled_eval["absolute_degradation_vs_true_beta"] = (
            shuffled_eval["primary_value"] - reference_primary
        )
        shuffled_eval["relative_degradation_vs_true_beta"] = (
            shuffled_eval["primary_value"] - reference_primary
        ) / max(reference_primary, 1e-12)
        shuffled_eval["true_beta_primary_value"] = reference_primary
        shuffled_eval.pop("predictions")
        shuffled[arm] = shuffled_eval
        predictions = []
        for value in regimes:
            counterfactual = torch.full_like(beta, float(value))
            predictions.append(
                _predict_batched(
                    selected_models[arm],
                    coefficients,
                    counterfactual,
                    arm=arm,
                    log_normalizer=log_normalizer,
                    raw_normalizer=raw_normalizer,
                    device=device,
                    batch_size=batch_size,
                )
            )
        stacked = torch.stack(predictions)
        rms = float(torch.sqrt(torch.mean((stacked - stacked[:1]).square())).item())
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
            "arms": shuffled,
        },
    }


def git_commit() -> str:
    value = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    if len(value) != 40:
        raise RuntimeError("could not resolve an exact git commit")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_fingerprint() -> str:
    paths = (
        Path(__file__),
        REPO_ROOT / "scripts/run_darcy_fno_conditioning_ablation.py",
        REPO_ROOT / "src/ups/training/resumable_checkpoint.py",
    )
    return canonical_sha256(
        {
            "git_commit": git_commit(),
            "files": {str(path.relative_to(REPO_ROOT)): _file_sha256(path) for path in paths},
        }
    )


def _resolve_plan_fingerprint(args: argparse.Namespace) -> str:
    explicit = getattr(args, "plan_sha256", None)
    plan_path = getattr(args, "plan_path", None)
    if explicit:
        if len(explicit) != 64 or any(
            character not in "0123456789abcdef" for character in explicit
        ):
            raise ValueError("--plan-sha256 must be a lowercase SHA-256 digest")
        return explicit
    if plan_path:
        path = Path(plan_path)
        plan = json.loads(path.read_text(encoding="utf-8"))
        recorded = plan.get("plan_sha256")
        if recorded != canonical_sha256({k: v for k, v in plan.items() if k != "plan_sha256"}):
            raise ValueError("--plan-path does not contain a valid self-hashed plan")
        return str(recorded)
    raise ValueError("one of --plan-path or --plan-sha256 is required")


def _runtime_evidence(device: str | torch.device, dependency: Mapping[str, Any]) -> dict[str, Any]:
    resolved = torch.device(device)
    evidence: dict[str, Any] = {
        "python": sys.version,
        "torch": torch.__version__,
        "neuraloperator": dict(dependency),
        "device_type": resolved.type,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    if resolved.type == "cuda" and torch.cuda.is_available():
        evidence["cuda_device_name"] = torch.cuda.get_device_name(resolved)
    return evidence


def configure_deterministic_runtime() -> None:
    """Configure strict Torch/CUDA determinism before any CUDA context exists."""

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _exact_darcy_objects(runtime: Any) -> dict[str, Any]:
    records = {
        item.object_id: {
            "role": item.role,
            "path": item.path,
            "sha256": item.checksums["sha256"],
            "size_bytes": item.size_bytes,
        }
        for item in runtime.lock.objects
        if item.object_id in EXPECTED_DARCY_OBJECTS
    }
    observed = {key: value["sha256"] for key, value in records.items()}
    if observed != EXPECTED_DARCY_OBJECTS:
        raise ValueError(f"Darcy object hashes do not match frozen D2 contract: {observed}")
    return records


def run(args: argparse.Namespace, *, fno_cls: type[nn.Module] | None = None) -> Path:
    output_dir = Path(args.output_dir)
    resume = bool(getattr(args, "resume", False))
    if output_dir.exists() and not resume:
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    if resume and not output_dir.is_dir():
        raise FileNotFoundError("--resume requires an existing output directory")
    if resume and (output_dir / "summary.json").exists():
        raise FileExistsError("refusing to resume a completed output directory")
    plan_fingerprint = _resolve_plan_fingerprint(args)
    configure_deterministic_runtime()
    runtime = load_strat_v1_baseline_runtime(
        args.training_lock,
        args.data_root,
        expected_lock_sha256=FROZEN_STRAT_V1_TRAINING_LOCK_SHA256,
    )
    darcy_objects = _exact_darcy_objects(runtime)
    darcy_runtime = runtime.tasks[TASK]
    train_dataset = PDEBenchDataset(runtime.dataset_config(TASK, "train", condition_on_regime=True))
    valid_dataset = PDEBenchDataset(runtime.dataset_config(TASK, "val", condition_on_regime=True))
    train_coefficients, train_targets, train_beta = _collect(train_dataset)
    valid_coefficients, valid_targets, valid_beta = _collect(valid_dataset)
    log_normalizer = BetaNormalizer.fit(train_beta)
    raw_normalizer = RawBetaNormalizer.fit(train_beta)
    output_dir.mkdir(parents=True, exist_ok=resume)
    dependency = neuraloperator_import_status()
    source_fingerprint = _source_fingerprint()
    lock_file_sha256 = _file_sha256(runtime.lock_path)
    data_fingerprint = canonical_sha256(
        {
            "lock_sha256": runtime.lock.lock_sha256,
            "lock_file_sha256": lock_file_sha256,
            "selection_sha256": runtime.selection_sha256,
            "darcy_objects": darcy_objects,
        }
    )
    runtime_evidence = _runtime_evidence(args.device, dependency)
    runtime_fingerprint = canonical_sha256(runtime_evidence)
    run_identity = {
        "schema_version": 1,
        "plan_fingerprint": plan_fingerprint,
        "data_fingerprint": data_fingerprint,
        "source_fingerprint": source_fingerprint,
        "runtime_fingerprint": runtime_fingerprint,
    }
    identity_path = output_dir / "run_identity.json"
    if resume:
        observed_identity = json.loads(identity_path.read_text(encoding="utf-8"))
        if observed_identity != run_identity:
            raise ValueError("resume run identity does not match the frozen D2 contract")
    else:
        identity_path.write_text(
            json.dumps(run_identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    _begin_compute_tracking(args.device)
    started = time.time()
    arm_results: dict[str, Any] = {}
    selected_models: dict[str, nn.Module] = {}
    for arm in ARMS:
        _begin_compute_tracking(args.device)
        arm_started = time.time()
        normalizer_spec = log_normalizer.as_dict() if arm == "K-long" else raw_normalizer.as_dict()
        bindings = CheckpointBindings(
            model_spec=_model_spec(arm=arm, args=args),
            optimizer_spec={
                "implementation": "torch.optim.AdamW",
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "objective": "raw_solution_mse",
                "batch_size": args.batch_size,
                "seed": SEED,
                "sample_order": "torch_randperm_continuous_generator_seed_17",
            },
            normalizer_spec=normalizer_spec,
            plan_fingerprint=plan_fingerprint,
            data_fingerprint=data_fingerprint,
            source_fingerprint=source_fingerprint,
            runtime_fingerprint=runtime_fingerprint,
        )
        rung_models, fit = train_arm(
            train_coefficients,
            train_targets,
            train_beta,
            arm=arm,
            log_normalizer=log_normalizer,
            raw_normalizer=raw_normalizer,
            hidden_channels=args.hidden_channels,
            fourier_modes=args.fourier_modes,
            n_layers=args.n_layers,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            device=args.device,
            fno_cls=fno_cls,
            checkpoint_dir=output_dir / "checkpoints" / arm,
            checkpoint_bindings=bindings,
            resume=resume,
        )
        history = []
        checkpoints = {}
        for epoch in RUNG_EPOCHS:
            evaluation = evaluate_arm(
                rung_models[epoch],
                valid_coefficients,
                valid_targets,
                valid_beta,
                arm=arm,
                log_normalizer=log_normalizer,
                raw_normalizer=raw_normalizer,
                device=args.device,
                batch_size=args.batch_size,
            )
            evaluation.pop("predictions")
            history.append({"epoch": epoch, **evaluation})
            checkpoints[str(epoch)] = {
                **fit["resumable_checkpoints"][str(epoch)],
                "epoch": epoch,
            }
        finite = [item for item in history if math.isfinite(item["primary_value"])]
        if not finite:
            raise RuntimeError(f"arm {arm} has no finite validation rung")
        selected = min(finite, key=lambda item: (item["primary_value"], item["epoch"]))
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
                    "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(resolved_device)),
                }
            )
        arm_results[arm] = {
            "conditioning": (
                "coefficient_plus_train_zscore_log10_beta_plus_presence_direct_solution"
                if arm == "K-long"
                else "coefficient_plus_presence_two_basis_affine_train_zscore_raw_beta"
            ),
            "fit": fit,
            "validation_history": history,
            "selection": {
                "rule": "minimum_finite_validation_primary_earliest_tie",
                "metric": PRIMARY_METRIC,
                "selected_epoch": selected_epoch,
                "selected_value": selected["primary_value"],
            },
            "checkpoints": {"rungs": checkpoints, "selected": checkpoints[str(selected_epoch)]},
            "compute": compute,
        }
    diagnostics = beta_diagnostics(
        selected_models=selected_models,
        coefficients=valid_coefficients,
        targets=valid_targets,
        beta=valid_beta,
        log_normalizer=log_normalizer,
        raw_normalizer=raw_normalizer,
        device=args.device,
        batch_size=args.batch_size,
    )
    summary = {
        "schema_version": 1,
        "artifact_id": "strat-v1-darcy-fno-affine-head-ablation-d2",
        "status": "complete_validation_only",
        "task": TASK,
        "semantics": "steady_coefficient_and_beta_to_explicit_solution_target",
        "held_out_reads": 0,
        "source": {"git_commit": git_commit()},
        "integrity_bindings": run_identity,
        "training_lock": {
            "path": str(runtime.lock_path),
            "lock_sha256": runtime.lock.lock_sha256,
            "lock_file_sha256": lock_file_sha256,
            "selection_sha256": runtime.selection_sha256,
            "darcy_objects": darcy_objects,
        },
        "sample_counts": {"train": len(train_beta), "valid": len(valid_beta)},
        "regime_counts": {
            "train": [vars(item) for item in darcy_runtime.train.regimes],
            "valid": [vars(item) for item in darcy_runtime.valid.regimes],
        },
        "beta_normalization": {
            "K-long": log_normalizer.as_dict(),
            "A-affine": raw_normalizer.as_dict(),
        },
        "matched_design": {
            "arms": list(ARMS),
            "seed": SEED,
            "rungs": list(RUNG_EPOCHS),
            "same_data_order_updates": True,
            "same_optimizer_and_raw_solution_mse": True,
            "selection_role": "valid",
            "selection_metric": PRIMARY_METRIC,
        },
        "architecture": {
            "implementation": "neuralop.models.FNO",
            "dependency": dependency,
            "hidden_channels": args.hidden_channels,
            "fourier_modes": args.fourier_modes,
            "n_layers": args.n_layers,
            "arm_specs": {arm: _model_spec(arm=arm, args=args) for arm in ARMS},
        },
        "optimizer": {
            "implementation": "torch.optim.AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "objective": "raw_solution_mse",
            "batch_size": args.batch_size,
        },
        "arms": arm_results,
        "diagnostics": diagnostics,
        "runtime": runtime_evidence,
        "compute": {"duration_sec": time.time() - started, "device": args.device},
    }
    summary["artifact_sha256"] = canonical_sha256(summary)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen validation-only Darcy FNO affine-head ablation"
    )
    parser.add_argument("--training-lock", default=str(DEFAULT_LOCK))
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    plan = parser.add_mutually_exclusive_group(required=True)
    plan.add_argument("--plan-path")
    plan.add_argument("--plan-sha256")
    parser.add_argument("--resume", action="store_true")
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
