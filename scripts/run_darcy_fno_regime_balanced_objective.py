#!/usr/bin/env python
from __future__ import annotations

"""Run the validation-only Darcy D3 regime-balanced objective ablation."""

import argparse
import hashlib
import json
import math
import sys
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts import run_darcy_fno_affine_head_ablation as d2  # noqa: E402
from scripts.run_darcy_fno_conditioning_ablation import BetaNormalizer, _collect  # noqa: E402
from ups.data.baseline_runtime import (  # noqa: E402
    FROZEN_STRAT_V1_TRAINING_LOCK_SHA256,
    load_strat_v1_baseline_runtime,
)
from ups.data.manifests import canonical_sha256  # noqa: E402
from ups.data.pdebench import PDEBenchDataset  # noqa: E402
from ups.training.resumable_checkpoint import (  # noqa: E402
    CheckpointBindings,
    TrainingProgress,
    checkpoint_record_path,
    load_training_checkpoint,
    save_training_checkpoint,
    verify_checkpoint_record,
)

ARMS = ("R-mean", "B-minimax")
RUNG_EPOCHS = (3, 6, 12, 24, 48, 96, 192, 384)
EXPECTED_BETAS = (0.01, 0.1, 1.0, 10.0, 100.0)
SEED = 17
BATCH_SIZE = 10
SAMPLES_PER_REGIME_PER_BATCH = 2
TASK = "darcy2d"
DEFAULT_LOCK = d2.DEFAULT_LOCK


def regime_complete_batches(
    beta: torch.Tensor, *, generator: torch.Generator
) -> Iterator[torch.Tensor]:
    """Yield deterministic batches with exactly two samples from every regime.

    Every sample is consumed exactly once per epoch. The frozen Darcy training
    root has equal counts divisible by two; fail closed for any other shape.
    """

    flat = beta.detach().cpu().flatten()
    groups: list[torch.Tensor] = []
    for value in EXPECTED_BETAS:
        indices = torch.where(
            torch.isclose(flat, torch.tensor(value, dtype=flat.dtype), rtol=1e-6, atol=1e-8)
        )[0]
        if len(indices) == 0 or len(indices) % SAMPLES_PER_REGIME_PER_BATCH:
            raise ValueError("each Darcy beta regime must have a positive even sample count")
        groups.append(indices[torch.randperm(len(indices), generator=generator)])
    counts = {len(group) for group in groups}
    if len(counts) != 1 or sum(len(group) for group in groups) != len(flat):
        raise ValueError("Darcy training samples must be exactly balanced across five regimes")
    count = counts.pop()
    for start in range(0, count, SAMPLES_PER_REGIME_PER_BATCH):
        yield torch.cat([group[start : start + SAMPLES_PER_REGIME_PER_BATCH] for group in groups])


def regime_objective(
    prediction: torch.Tensor, target: torch.Tensor, beta: torch.Tensor, *, arm: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the registered D3 objective and ordered per-regime raw MSE."""

    if prediction.shape != target.shape or len(beta) != len(prediction):
        raise ValueError("prediction, target, and beta must be sample-aligned")
    losses = []
    for value in EXPECTED_BETAS:
        mask = torch.isclose(beta.flatten(), beta.new_tensor(value), rtol=1e-6, atol=1e-8)
        if int(mask.sum()) != SAMPLES_PER_REGIME_PER_BATCH:
            raise ValueError("objective requires exactly two samples from every beta regime")
        losses.append(torch.mean((prediction[mask] - target[mask]).square()))
    per_regime = torch.stack(losses)
    mean = per_regime.mean()
    if arm == "R-mean":
        return mean, per_regime
    if arm == "B-minimax":
        return 0.5 * mean + 0.5 * per_regime.max(), per_regime
    raise ValueError(f"unknown D3 arm {arm!r}")


def _beta_diagnostics(
    models: dict[str, nn.Module],
    coefficients: torch.Tensor,
    targets: torch.Tensor,
    beta: torch.Tensor,
    *,
    normalizer: BetaNormalizer,
    raw_normalizer: d2.RawBetaNormalizer,
    device: str | torch.device,
) -> dict[str, Any]:
    permutation = d2.deterministic_beta_permutation(len(beta), seed=d2.SHUFFLE_SEED)
    shuffled_beta = beta.index_select(0, permutation)
    shuffled: dict[str, Any] = {}
    sensitivity: dict[str, Any] = {}
    for arm, model in models.items():
        reference = d2.evaluate_arm(
            model,
            coefficients,
            targets,
            beta,
            arm="K-long",
            log_normalizer=normalizer,
            raw_normalizer=raw_normalizer,
            device=device,
            batch_size=BATCH_SIZE,
        )
        shuffled_eval = d2.evaluate_arm(
            model,
            coefficients,
            targets,
            beta,
            arm="K-long",
            log_normalizer=normalizer,
            raw_normalizer=raw_normalizer,
            device=device,
            conditioning_beta=shuffled_beta,
            batch_size=BATCH_SIZE,
        )
        reference_primary = float(reference["primary_value"])
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
        for value in EXPECTED_BETAS:
            predictions.append(
                d2._predict_batched(
                    model,
                    coefficients,
                    torch.full_like(beta, value),
                    arm="K-long",
                    log_normalizer=normalizer,
                    raw_normalizer=raw_normalizer,
                    device=device,
                    batch_size=BATCH_SIZE,
                )
            )
        stacked = torch.stack(predictions)
        rms = float(torch.sqrt(torch.mean((stacked - stacked[:1]).square())).item())
        scale = float(torch.sqrt(torch.mean(stacked.square())).item())
        sensitivity[arm] = {
            "counterfactual_beta_values": list(EXPECTED_BETAS),
            "prediction_rms_from_first_beta": rms,
            "relative_prediction_rms_from_first_beta": rms / max(scale, 1e-12),
        }
    permutation_sha = hashlib.sha256(
        json.dumps(permutation.tolist(), separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "counterfactual_beta_sensitivity": sensitivity,
        "deterministic_shuffled_beta": {
            "seed": d2.SHUFFLE_SEED,
            "permutation_sha256": permutation_sha,
            "fixed_point_count": int((permutation == torch.arange(len(beta))).sum()),
            "arms": shuffled,
        },
    }


def _record(path: Path) -> dict[str, Any]:
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
    normalizer: BetaNormalizer,
    hidden_channels: int,
    fourier_modes: int,
    n_layers: int,
    learning_rate: float,
    weight_decay: float,
    device: str | torch.device,
    fno_cls: type[nn.Module] | None = None,
    checkpoint_dir: Path | None = None,
    checkpoint_bindings: CheckpointBindings | None = None,
    resume: bool = False,
) -> tuple[dict[int, nn.Module], dict[str, Any]]:
    """Train one deterministic D3 trajectory through all registered rungs."""

    if arm not in ARMS:
        raise ValueError(f"unknown D3 arm {arm!r}")
    torch.manual_seed(SEED)
    resolved_device = torch.device(device)
    model = d2.build_model(
        arm="K-long",
        grid_shape=(int(coefficients.shape[2]), int(coefficients.shape[3])),
        hidden_channels=hidden_channels,
        fourier_modes=fourier_modes,
        n_layers=n_layers,
        fno_cls=fno_cls,
    ).to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(SEED)
    raw_normalizer = d2.RawBetaNormalizer.fit(beta)
    snapshots: dict[int, dict[str, torch.Tensor]] = {}
    history: list[float] = []
    steps = examples = 0
    start_epoch = 1
    parent_sha: str | None = None
    records: dict[str, Any] = {}
    if (checkpoint_dir is None) != (checkpoint_bindings is None):
        raise ValueError("checkpoint_dir and checkpoint_bindings must be provided together")
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        present = []
        gap = False
        prior_sha = None
        for epoch in RUNG_EPOCHS:
            path = checkpoint_dir / f"arm_{arm}_epoch_{epoch}.pt"
            exists = path.exists() or checkpoint_record_path(path).exists()
            if not exists:
                gap = True
                continue
            if gap:
                raise RuntimeError(f"non-contiguous checkpoint chain for {arm}")
            record = verify_checkpoint_record(path)
            if record.parent_checkpoint_sha256 != prior_sha:
                raise RuntimeError(f"invalid checkpoint lineage for {arm}")
            records[str(epoch)] = _record(path)
            present.append((epoch, path))
            prior_sha = record.checkpoint_sha256
        if present and not resume:
            raise FileExistsError(f"checkpoints already exist for {arm}; use --resume")
        if resume and present:
            for index, (completed_epoch, completed_path) in enumerate(present):
                clone = d2.build_model(
                    arm="K-long",
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
                expected = (
                    None
                    if index == 0
                    else verify_checkpoint_record(present[index - 1][1]).checkpoint_sha256
                )
                loaded_clone = load_training_checkpoint(
                    completed_path,
                    model=clone,
                    optimizer=clone_optimizer,
                    sampler_generator=clone_generator,
                    expected_bindings=checkpoint_bindings,
                    expected_parent_checkpoint_sha256=expected,
                    map_location="cpu",
                )
                if loaded_clone.progress.completed_epoch != completed_epoch:
                    raise RuntimeError("checkpoint progress does not match its D3 rung")
                snapshots[completed_epoch] = d2._clone_tensor_state_dict(clone)
            epoch, path = present[-1]
            expected_parent = (
                None
                if len(present) == 1
                else verify_checkpoint_record(present[-2][1]).checkpoint_sha256
            )
            loaded = load_training_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                sampler_generator=generator,
                expected_bindings=checkpoint_bindings,
                expected_parent_checkpoint_sha256=expected_parent,
                map_location=resolved_device,
            )
            history = [float(item) for item in loaded.progress.history]
            steps, examples = loaded.progress.steps, loaded.progress.examples
            start_epoch, parent_sha = epoch + 1, loaded.record.checkpoint_sha256
    for epoch in range(start_epoch, RUNG_EPOCHS[-1] + 1):
        loss_sum = 0.0
        batches = 0
        for indices in regime_complete_batches(beta, generator=generator):
            x = coefficients.index_select(0, indices).to(resolved_device)
            y = targets.index_select(0, indices).to(resolved_device)
            b = beta.index_select(0, indices).to(resolved_device)
            prediction = d2.predict(
                model,
                x,
                b,
                arm="K-long",
                log_normalizer=normalizer,
                raw_normalizer=raw_normalizer,
            )
            loss, _ = regime_objective(prediction, y, b, arm=arm)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu())
            batches += 1
            steps += 1
            examples += len(indices)
        history.append(loss_sum / batches)
        if epoch in RUNG_EPOCHS:
            snapshots[epoch] = d2._clone_tensor_state_dict(model)
            if checkpoint_dir is not None and checkpoint_bindings is not None:
                path = checkpoint_dir / f"arm_{arm}_epoch_{epoch}.pt"
                record = save_training_checkpoint(
                    path,
                    model=model,
                    optimizer=optimizer,
                    progress=TrainingProgress(epoch, steps, examples, history),
                    sampler_generator=generator,
                    bindings=checkpoint_bindings,
                    parent_checkpoint_sha256=parent_sha,
                )
                parent_sha = record.checkpoint_sha256
                records[str(epoch)] = _record(path)
    models = {}
    for epoch, state in snapshots.items():
        clone = d2.build_model(
            arm="K-long",
            grid_shape=(int(coefficients.shape[2]), int(coefficients.shape[3])),
            hidden_channels=hidden_channels,
            fourier_modes=fourier_modes,
            n_layers=n_layers,
            fno_cls=fno_cls,
        ).cpu()
        clone.load_state_dict(state)
        models[epoch] = clone
    return models, {
        "arm": arm,
        "objective": (
            "mean_per_regime_raw_mse"
            if arm == "R-mean"
            else "half_mean_plus_half_max_per_regime_raw_mse"
        ),
        "epoch_train_objective": history,
        "optimizer_steps": steps,
        "examples_seen": examples,
        "rungs": list(RUNG_EPOCHS),
        "batch_contract": {
            "batch_size": BATCH_SIZE,
            "samples_per_regime": 2,
            "regimes": list(EXPECTED_BETAS),
        },
        "resumable_checkpoints": records,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", default="cpu")
    return parser


def run(args: argparse.Namespace, *, fno_cls: type[nn.Module] | None = None) -> Path:
    if args.batch_size != BATCH_SIZE:
        raise ValueError(f"D3 requires the frozen regime-complete batch size {BATCH_SIZE}")
    output = Path(args.output_dir)
    if output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {output}")
    if args.resume and (output / "summary.json").exists():
        raise FileExistsError("refusing to resume a completed D3 output directory")
    output.mkdir(parents=True, exist_ok=args.resume)
    plan_fingerprint = d2._resolve_plan_fingerprint(args)
    d2.configure_deterministic_runtime()
    runtime = load_strat_v1_baseline_runtime(
        args.training_lock,
        args.data_root,
        expected_lock_sha256=FROZEN_STRAT_V1_TRAINING_LOCK_SHA256,
    )
    train = PDEBenchDataset(runtime.dataset_config(TASK, "train", condition_on_regime=True))
    valid = PDEBenchDataset(runtime.dataset_config(TASK, "val", condition_on_regime=True))
    train_x, train_y, train_beta = _collect(train)
    valid_x, valid_y, valid_beta = _collect(valid)
    normalizer = BetaNormalizer.fit(train_beta)
    raw_normalizer = d2.RawBetaNormalizer.fit(train_beta)
    darcy_objects = d2._exact_darcy_objects(runtime)
    lock_file_sha256 = d2._file_sha256(runtime.lock_path)
    data_fingerprint = canonical_sha256(
        {
            "lock_sha256": runtime.lock.lock_sha256,
            "lock_file_sha256": lock_file_sha256,
            "selection_sha256": runtime.selection_sha256,
            "darcy_objects": darcy_objects,
        }
    )
    source_fingerprint = canonical_sha256(
        {"git_commit": d2.git_commit(), "runner_sha256": d2._file_sha256(Path(__file__))}
    )
    dependency = d2.neuraloperator_import_status()
    runtime_evidence = d2._runtime_evidence(args.device, dependency)
    runtime_fingerprint = canonical_sha256(runtime_evidence)
    run_identity = {
        "schema_version": 1,
        "plan_fingerprint": plan_fingerprint,
        "data_fingerprint": data_fingerprint,
        "source_fingerprint": source_fingerprint,
        "runtime_fingerprint": runtime_fingerprint,
    }
    identity_path = output / "run_identity.json"
    if args.resume:
        if json.loads(identity_path.read_text(encoding="utf-8")) != run_identity:
            raise ValueError("resume run identity does not match the frozen D3 contract")
    else:
        identity_path.write_text(
            json.dumps(run_identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    started = time.time()
    results = {}
    selected_models: dict[str, nn.Module] = {}
    for arm in ARMS:
        arm_started = time.time()
        bindings = CheckpointBindings(
            model_spec={**d2._model_spec(arm="K-long", args=args), "d3_arm": arm},
            optimizer_spec={
                "implementation": "torch.optim.AdamW",
                "objective": arm,
                "batch_size": BATCH_SIZE,
                "seed": SEED,
            },
            normalizer_spec=normalizer.as_dict(),
            plan_fingerprint=plan_fingerprint,
            data_fingerprint=data_fingerprint,
            source_fingerprint=source_fingerprint,
            runtime_fingerprint=runtime_fingerprint,
        )
        models, fit = train_arm(
            train_x,
            train_y,
            train_beta,
            arm=arm,
            normalizer=normalizer,
            hidden_channels=args.hidden_channels,
            fourier_modes=args.fourier_modes,
            n_layers=args.n_layers,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=args.device,
            fno_cls=fno_cls,
            checkpoint_dir=output / "checkpoints" / arm,
            checkpoint_bindings=bindings,
            resume=args.resume,
        )
        history = []
        for epoch in RUNG_EPOCHS:
            metrics = d2.evaluate_arm(
                models[epoch],
                valid_x,
                valid_y,
                valid_beta,
                arm="K-long",
                log_normalizer=normalizer,
                raw_normalizer=raw_normalizer,
                device=args.device,
                batch_size=BATCH_SIZE,
            )
            metrics.pop("predictions")
            history.append({"epoch": epoch, **metrics})
        selected = min(
            (item for item in history if math.isfinite(item["primary_value"])),
            key=lambda item: (item["primary_value"], item["epoch"]),
        )
        selected_epoch = int(selected["epoch"])
        selected_models[arm] = models[selected_epoch]
        checkpoints = {
            str(epoch): {**fit["resumable_checkpoints"][str(epoch)], "epoch": epoch}
            for epoch in RUNG_EPOCHS
        }
        parameters = list(models[selected_epoch].parameters())
        results[arm] = {
            "conditioning": "coefficient_plus_train_zscore_log10_beta_plus_presence_direct_solution",
            "fit": fit,
            "validation_history": history,
            "selection": {
                "rule": "minimum_finite_validation_primary_earliest_tie",
                "metric": d2.PRIMARY_METRIC,
                "selected_epoch": selected_epoch,
                "selected_value": selected["primary_value"],
            },
            "checkpoints": {
                "rungs": checkpoints,
                "selected": checkpoints[str(selected_epoch)],
            },
            "compute": {
                "duration_sec": time.time() - arm_started,
                "device": args.device,
                "total_parameter_count": sum(item.numel() for item in parameters),
                "trainable_parameter_count": sum(
                    item.numel() for item in parameters if item.requires_grad
                ),
                "optimizer_steps": fit["optimizer_steps"],
                "examples_seen": fit["examples_seen"],
            },
        }
    diagnostics = _beta_diagnostics(
        selected_models,
        valid_x,
        valid_y,
        valid_beta,
        normalizer=normalizer,
        raw_normalizer=raw_normalizer,
        device=args.device,
    )
    summary = {
        "schema_version": 1,
        "artifact_id": "strat-v1-darcy-fno-regime-balanced-objective-d3",
        "status": "complete_validation_only",
        "task": TASK,
        "semantics": "steady_coefficient_and_beta_to_explicit_solution_target",
        "held_out_reads": 0,
        "source": {"git_commit": d2.git_commit()},
        "integrity_bindings": {
            **run_identity,
        },
        "training_lock": {
            "path": str(runtime.lock_path),
            "lock_sha256": runtime.lock.lock_sha256,
            "lock_file_sha256": lock_file_sha256,
            "selection_sha256": runtime.selection_sha256,
            "darcy_objects": darcy_objects,
        },
        "sample_counts": {"train": len(train_beta), "valid": len(valid_beta)},
        "regime_counts": {
            "train": [vars(item) for item in runtime.tasks[TASK].train.regimes],
            "valid": [vars(item) for item in runtime.tasks[TASK].valid.regimes],
        },
        "beta_normalization": {arm: normalizer.as_dict() for arm in ARMS},
        "matched_design": {
            "arms": list(ARMS),
            "seed": SEED,
            "rungs": list(RUNG_EPOCHS),
            "regime_complete_batch_size": BATCH_SIZE,
            "samples_per_regime_per_batch": SAMPLES_PER_REGIME_PER_BATCH,
            "same_model_optimizer_order_and_updates": True,
            "selection_role": "valid",
            "selection_metric": d2.PRIMARY_METRIC,
        },
        "architecture": {
            "implementation": "neuralop.models.FNO",
            "dependency": dependency,
            "hidden_channels": args.hidden_channels,
            "fourier_modes": args.fourier_modes,
            "n_layers": args.n_layers,
            "arm_specs": {
                arm: {**d2._model_spec(arm="K-long", args=args), "d3_arm": arm} for arm in ARMS
            },
        },
        "optimizer": {
            "implementation": "torch.optim.AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": BATCH_SIZE,
            "objectives": {
                "R-mean": "mean_per_regime_raw_mse",
                "B-minimax": "0.5_mean_plus_0.5_max_per_regime_raw_mse",
            },
        },
        "arms": results,
        "diagnostics": diagnostics,
        "runtime": runtime_evidence,
        "compute": {"duration_sec": time.time() - started, "device": args.device},
    }
    summary["artifact_sha256"] = canonical_sha256(summary)
    path = output / "summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> None:
    print(json.dumps({"summary": str(run(build_parser().parse_args(argv)))}, indent=2))


if __name__ == "__main__":
    main()
