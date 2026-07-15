#!/usr/bin/env python
from __future__ import annotations

"""Run the single-arm, validation-only Darcy conditioned-UNO D4 experiment."""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts import run_darcy_fno_affine_head_ablation as d2  # noqa: E402
from scripts import run_darcy_fno_regime_balanced_objective as d3  # noqa: E402
from scripts import run_external_neuraloperator_uno_baseline as uno  # noqa: E402
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

ARM = "U-conditioned"
TASK = "darcy2d"
RUNG_EPOCHS = d3.RUNG_EPOCHS
SEED = d3.SEED
BATCH_SIZE = d3.BATCH_SIZE
DEFAULT_LOCK = d3.DEFAULT_LOCK


def configure_deterministic_runtime() -> None:
    """Keep the D2 determinism lock, tolerating unsupported UNO interpolation kernels.

    NeuralOperator UNO uses bicubic interpolation whose CUDA backward kernel is not
    deterministic in the pinned Torch runtime.  Warning-only mode preserves fixed
    seeds, deterministic cuDNN, and deterministic alternatives where available
    without making that upstream kernel incompatibility fatal.
    """

    d2.configure_deterministic_runtime()
    torch.use_deterministic_algorithms(True, warn_only=True)


def build_model(
    *,
    grid_shape: tuple[int, int],
    hidden_channels: int = 16,
    fourier_modes: int = 16,
    n_layers: int = 4,
    lifting_channels: int = 32,
    projection_channels: int = 32,
    channel_mlp_skip: str = "linear",
    uno_cls: type[nn.Module] | None = None,
) -> nn.Module:
    return uno.build_neuraloperator_uno_model(
        in_channels=3,
        out_channels=1,
        grid_shape=grid_shape,
        hidden_channels=hidden_channels,
        fourier_modes=fourier_modes,
        n_layers=n_layers,
        lifting_channels=lifting_channels,
        projection_channels=projection_channels,
        channel_mlp_skip=channel_mlp_skip,
        identity_scaling=False,
        residual=False,
        uno_cls=uno_cls,
    )


def model_spec(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "implementation": uno.NEURALOP_IMPORT,
        "in_channels": 3,
        "out_channels": 1,
        "hidden_channels": args.hidden_channels,
        "fourier_modes": args.fourier_modes,
        "n_layers": args.n_layers,
        "lifting_channels": args.lifting_channels,
        "projection_channels": args.projection_channels,
        "channel_mlp_skip": args.channel_mlp_skip,
        "identity_scaling": False,
        "residual": False,
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
    normalizer: BetaNormalizer,
    args: argparse.Namespace,
    checkpoint_dir: Path,
    checkpoint_bindings: CheckpointBindings,
    uno_cls: type[nn.Module] | None = None,
) -> tuple[dict[int, nn.Module], dict[str, Any]]:
    torch.manual_seed(SEED)
    device = torch.device(args.device)

    def fresh() -> nn.Module:
        return build_model(
            grid_shape=(int(coefficients.shape[2]), int(coefficients.shape[3])),
            hidden_channels=args.hidden_channels,
            fourier_modes=args.fourier_modes,
            n_layers=args.n_layers,
            lifting_channels=args.lifting_channels,
            projection_channels=args.projection_channels,
            channel_mlp_skip=args.channel_mlp_skip,
            uno_cls=uno_cls,
        )

    model = fresh().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    generator = torch.Generator().manual_seed(SEED)
    raw_normalizer = d2.RawBetaNormalizer.fit(beta)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    snapshots: dict[int, dict[str, torch.Tensor]] = {}
    records: dict[str, Any] = {}
    history: list[float] = []
    steps = examples = 0
    start_epoch = 1
    parent_sha: str | None = None
    present: list[tuple[int, Path]] = []
    gap = False
    prior_sha = None
    for epoch in RUNG_EPOCHS:
        path = checkpoint_dir / f"arm_{ARM}_epoch_{epoch}.pt"
        exists = path.exists() or checkpoint_record_path(path).exists()
        if not exists:
            gap = True
            continue
        if gap:
            raise RuntimeError("non-contiguous D4 checkpoint chain")
        record = verify_checkpoint_record(path)
        if record.parent_checkpoint_sha256 != prior_sha:
            raise RuntimeError("invalid D4 checkpoint lineage")
        records[str(epoch)] = _record(path)
        present.append((epoch, path))
        prior_sha = record.checkpoint_sha256
    if present and not args.resume:
        raise FileExistsError("D4 checkpoints exist; use --resume")
    if args.resume and present:
        for index, (epoch, path) in enumerate(present):
            clone = fresh().cpu()
            clone_optimizer = torch.optim.AdamW(
                clone.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
            )
            loaded = load_training_checkpoint(
                path,
                model=clone,
                optimizer=clone_optimizer,
                sampler_generator=torch.Generator().manual_seed(SEED),
                expected_bindings=checkpoint_bindings,
                expected_parent_checkpoint_sha256=(
                    None
                    if index == 0
                    else verify_checkpoint_record(present[index - 1][1]).checkpoint_sha256
                ),
            )
            if loaded.progress.completed_epoch != epoch:
                raise RuntimeError("checkpoint progress does not match D4 rung")
            snapshots[epoch] = d2._clone_tensor_state_dict(clone)
        epoch, path = present[-1]
        loaded = load_training_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            sampler_generator=generator,
            expected_bindings=checkpoint_bindings,
            expected_parent_checkpoint_sha256=(
                None
                if len(present) == 1
                else verify_checkpoint_record(present[-2][1]).checkpoint_sha256
            ),
            map_location=device,
        )
        history = [float(value) for value in loaded.progress.history]
        steps, examples = loaded.progress.steps, loaded.progress.examples
        start_epoch, parent_sha = epoch + 1, loaded.record.checkpoint_sha256
    for epoch in range(start_epoch, RUNG_EPOCHS[-1] + 1):
        total = 0.0
        batches = 0
        model.train()
        for indices in d3.regime_complete_batches(beta, generator=generator):
            x = coefficients.index_select(0, indices).to(device)
            y = targets.index_select(0, indices).to(device)
            b = beta.index_select(0, indices).to(device)
            prediction = d2.predict(
                model,
                x,
                b,
                arm="K-long",
                log_normalizer=normalizer,
                raw_normalizer=raw_normalizer,
            )
            loss, _ = d3.regime_objective(prediction, y, b, arm="R-mean")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu())
            batches += 1
            steps += 1
            examples += len(indices)
        history.append(total / batches)
        if epoch in RUNG_EPOCHS:
            snapshots[epoch] = d2._clone_tensor_state_dict(model)
            path = checkpoint_dir / f"arm_{ARM}_epoch_{epoch}.pt"
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
        clone = fresh().cpu()
        clone.load_state_dict(state)
        models[epoch] = clone
    return models, {
        "arm": ARM,
        "objective": "mean_per_regime_raw_mse",
        "epoch_train_objective": history,
        "optimizer_steps": steps,
        "examples_seen": examples,
        "rungs": list(RUNG_EPOCHS),
        "batch_contract": {"batch_size": BATCH_SIZE, "samples_per_regime": 2},
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
    parser.add_argument("--lifting-channels", type=int, default=32)
    parser.add_argument("--projection-channels", type=int, default=32)
    parser.add_argument("--channel-mlp-skip", default="linear")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", default="cpu")
    return parser


def run(args: argparse.Namespace, *, uno_cls: type[nn.Module] | None = None) -> Path:
    if args.batch_size != BATCH_SIZE:
        raise ValueError(f"D4 requires batch size {BATCH_SIZE}")
    output = Path(args.output_dir)
    if output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {output}")
    if args.resume and (output / "summary.json").exists():
        raise FileExistsError("refusing to resume completed D4 output")
    plan_fingerprint = d2._resolve_plan_fingerprint(args)
    configure_deterministic_runtime()
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
    lock_file_sha = d2._file_sha256(runtime.lock_path)
    data_fingerprint = canonical_sha256(
        {
            "lock_sha256": runtime.lock.lock_sha256,
            "lock_file_sha256": lock_file_sha,
            "selection_sha256": runtime.selection_sha256,
            "darcy_objects": darcy_objects,
        }
    )
    source_fingerprint = canonical_sha256(
        {
            "git_commit": d2.git_commit(),
            "runner_sha256": d2._file_sha256(Path(__file__)),
            "uno_runner_sha256": d2._file_sha256(Path(uno.__file__)),
        }
    )
    dependency = uno.neuraloperator_import_status()
    runtime_evidence = d2._runtime_evidence(args.device, dependency)
    runtime_fingerprint = canonical_sha256(runtime_evidence)
    identity = {
        "schema_version": 1,
        "plan_fingerprint": plan_fingerprint,
        "data_fingerprint": data_fingerprint,
        "source_fingerprint": source_fingerprint,
        "runtime_fingerprint": runtime_fingerprint,
    }
    output.mkdir(parents=True, exist_ok=args.resume)
    identity_path = output / "run_identity.json"
    if args.resume:
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError("resume identity does not match D4 contract")
    else:
        identity_path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
    bindings = CheckpointBindings(
        model_spec=model_spec(args),
        optimizer_spec={
            "implementation": "torch.optim.AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "objective": "mean_per_regime_raw_mse",
            "batch_size": BATCH_SIZE,
            "seed": SEED,
        },
        normalizer_spec=normalizer.as_dict(),
        plan_fingerprint=plan_fingerprint,
        data_fingerprint=data_fingerprint,
        source_fingerprint=source_fingerprint,
        runtime_fingerprint=runtime_fingerprint,
    )
    started = time.time()
    models, fit = train_arm(
        train_x,
        train_y,
        train_beta,
        normalizer=normalizer,
        args=args,
        checkpoint_dir=output / "checkpoints" / ARM,
        checkpoint_bindings=bindings,
        uno_cls=uno_cls,
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
    finite = [row for row in history if math.isfinite(row["primary_value"])]
    if not finite:
        raise RuntimeError("D4 has no finite validation rung")
    selected = min(finite, key=lambda row: (row["primary_value"], row["epoch"]))
    selected_epoch = int(selected["epoch"])
    diagnostics = d3._beta_diagnostics(
        {ARM: models[selected_epoch]},
        valid_x,
        valid_y,
        valid_beta,
        normalizer=normalizer,
        raw_normalizer=raw_normalizer,
        device=args.device,
    )
    checkpoints = {
        str(epoch): {**fit["resumable_checkpoints"][str(epoch)], "epoch": epoch}
        for epoch in RUNG_EPOCHS
    }
    parameters = list(models[selected_epoch].parameters())
    summary = {
        "schema_version": 1,
        "artifact_id": "strat-v1-darcy-conditioned-uno-ablation-d4",
        "status": "complete_validation_only",
        "task": TASK,
        "semantics": "steady_coefficient_and_beta_to_explicit_solution_target",
        "held_out_reads": 0,
        "source": {"git_commit": d2.git_commit()},
        "integrity_bindings": identity,
        "training_lock": {
            "path": str(runtime.lock_path),
            "lock_sha256": runtime.lock.lock_sha256,
            "lock_file_sha256": lock_file_sha,
            "selection_sha256": runtime.selection_sha256,
            "darcy_objects": darcy_objects,
        },
        "sample_counts": {"train": len(train_beta), "valid": len(valid_beta)},
        "regime_counts": {
            "train": [vars(item) for item in runtime.tasks[TASK].train.regimes],
            "valid": [vars(item) for item in runtime.tasks[TASK].valid.regimes],
        },
        "beta_normalization": normalizer.as_dict(),
        "historical_control": {
            "artifact_id": "strat-v1-darcy-fno-regime-balanced-objective-d3",
            "arm": "R-mean",
            "selected_epoch": 384,
            "plateau_epoch": 384,
            "primary_value": 0.11694165553982801,
            "beta100_global_scale_nrmse": 0.25762147974587746,
            "maximum_corrected_spread_ratio": 2.2029915564017024,
        },
        "matched_design": {
            "live_arms": [ARM],
            "historical_control_arm": "R-mean",
            "seed": SEED,
            "rungs": list(RUNG_EPOCHS),
            "regime_complete_batch_size": BATCH_SIZE,
            "samples_per_regime_per_batch": 2,
            "selection_role": "valid",
            "selection_metric": d2.PRIMARY_METRIC,
        },
        "architecture": {**model_spec(args), "dependency": dependency},
        "optimizer": {
            "implementation": "torch.optim.AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": BATCH_SIZE,
            "objective": "mean_per_regime_raw_mse",
        },
        "arms": {
            ARM: {
                "conditioning": "coefficient_plus_train_zscore_log10_beta_plus_presence_direct_solution",
                "fit": fit,
                "validation_history": history,
                "selection": {
                    "rule": "minimum_finite_validation_primary_earliest_tie",
                    "metric": d2.PRIMARY_METRIC,
                    "selected_epoch": selected_epoch,
                    "selected_value": selected["primary_value"],
                },
                "checkpoints": {"rungs": checkpoints, "selected": checkpoints[str(selected_epoch)]},
                "compute": {
                    "duration_sec": time.time() - started,
                    "device": args.device,
                    "total_parameter_count": sum(p.numel() for p in parameters),
                    "trainable_parameter_count": sum(
                        p.numel() for p in parameters if p.requires_grad
                    ),
                    "optimizer_steps": fit["optimizer_steps"],
                    "examples_seen": fit["examples_seen"],
                },
            }
        },
        "diagnostics": diagnostics,
        "runtime": runtime_evidence,
        "compute": {"duration_sec": time.time() - started, "device": args.device},
    }
    summary["artifact_sha256"] = canonical_sha256(summary)
    path = output / "summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return path


def main() -> None:
    path = run(build_parser().parse_args())
    print(json.dumps({"summary": str(path)}))


if __name__ == "__main__":
    main()
