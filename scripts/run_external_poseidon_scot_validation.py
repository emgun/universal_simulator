#!/usr/bin/env python
from __future__ import annotations

"""Run validation-only Poseidon ScOT transfer through the light-v1 adapter."""

import argparse
import hashlib
import importlib
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts import run_external_neuraloperator_fno_baseline as fno_runner
from scripts.run_external_poseidon_transfer_adapter import (
    POSEIDON_MODEL_IMPORT,
    POSEIDON_SOURCE_URL,
    light_step_to_poseidon_pixels,
    poseidon_checkpoint_handle,
    poseidon_pixels_to_repo_flat,
    poseidon_source_snapshot,
)
from scripts.run_physical_conv_baseline import _add_rollout_metrics
from ups.data.latent_pairs import infer_channel_count, infer_grid_shape
from ups.data.pdebench import get_pdebench_spec
from ups.eval.pdebench_runner import _aggregate_chunk_metrics, _flatten_field_step

MEASUREMENT_TYPE = "poseidon_scot_validation_measurement"
ALLOWED_STATUSES = {"validation_model_measurement_complete", "invalid"}
DEFAULT_CHECKPOINT_FILE = "model.safetensors"


class MissingPoseidonDependencyError(RuntimeError):
    """Raised when live ScOT validation is requested without official Poseidon import."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _field_step_count(fields: torch.Tensor) -> int:
    if fields.dim() >= 3 and fields.shape[0] > 1:
        return int(fields.shape[0])
    return 1


def _insert_poseidon_path(poseidon_repo: Path | None) -> str:
    if poseidon_repo is None:
        return ""
    if not poseidon_repo.exists():
        raise MissingPoseidonDependencyError(f"Poseidon repo does not exist: {poseidon_repo}")
    repo_path = str(poseidon_repo)
    sys.path.insert(0, repo_path)
    return repo_path


def _remove_poseidon_path(path: str) -> None:
    if not path:
        return
    try:
        sys.path.remove(path)
    except ValueError:
        pass


def resolve_checkpoint_file(
    *,
    checkpoint_handle: str,
    filename: str = DEFAULT_CHECKPOINT_FILE,
    expected_sha256: str = "",
) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download, model_info

    info = model_info(checkpoint_handle, files_metadata=True)
    path = Path(hf_hub_download(repo_id=checkpoint_handle, filename=filename))
    sha256 = _sha256_file(path)
    if expected_sha256 and expected_sha256 != sha256:
        raise RuntimeError(
            f"Checkpoint SHA256 mismatch for {checkpoint_handle}/{filename}: "
            f"expected {expected_sha256}, got {sha256}"
        )
    sibling_size = None
    for sibling in info.siblings:
        if sibling.rfilename == filename:
            sibling_size = getattr(sibling, "size", None)
            break
    return {
        "handle": checkpoint_handle,
        "repo_sha": info.sha,
        "filename": filename,
        "path": str(path),
        "bytes": path.stat().st_size,
        "hub_size": sibling_size,
        "sha256": sha256,
        "expected_sha256": expected_sha256,
        "sha256_status": "matched" if expected_sha256 else "recorded",
    }


def load_poseidon_scot_model(
    *,
    poseidon_repo: Path | None,
    checkpoint_handle: str,
    image_size: int | None,
    channels: int,
    device: str | torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    inserted_path = _insert_poseidon_path(poseidon_repo)
    try:
        module = importlib.import_module("scOT.model")
        scot_cls = module.ScOT
        config_cls = module.ScOTConfig
        config = config_cls.from_pretrained(checkpoint_handle)
        original_config = {
            "image_size": int(config.image_size),
            "num_channels": int(config.num_channels),
            "num_out_channels": int(config.num_out_channels),
            "use_conditioning": bool(config.use_conditioning),
            "learn_residual": bool(config.learn_residual),
            "embed_dim": int(config.embed_dim),
            "depths": list(config.depths),
            "num_heads": list(config.num_heads),
            "window_size": int(config.window_size),
        }
        if image_size is not None:
            config.image_size = int(image_size)
        config.num_channels = int(channels)
        config.num_out_channels = int(channels)
        config.channel_slice_list_normalized_loss = None
        model = scot_cls.from_pretrained(
            checkpoint_handle,
            config=config,
            ignore_mismatched_sizes=True,
        )
    except Exception as exc:
        raise MissingPoseidonDependencyError(
            "Official Poseidon ScOT is required for live validation. "
            "Pass --poseidon-repo pointing at the official checkout."
        ) from exc
    finally:
        _remove_poseidon_path(inserted_path)
    model.to(torch.device(device)).eval()
    return model, {
        "implementation": POSEIDON_MODEL_IMPORT,
        "checkpoint_handle": checkpoint_handle,
        "original_config": original_config,
        "effective_config": {
            "image_size": int(config.image_size),
            "num_channels": int(config.num_channels),
            "num_out_channels": int(config.num_out_channels),
            "use_conditioning": bool(config.use_conditioning),
            "learn_residual": bool(config.learn_residual),
            "channel_slice_list_normalized_loss": config.channel_slice_list_normalized_loss,
        },
        "embedding_recovery_replaced": True,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
    }


def _model_predict_pixels(
    model: nn.Module,
    pixels: torch.Tensor,
    *,
    time_value: float,
    device: torch.device,
) -> torch.Tensor:
    pixels = pixels.to(device)
    time_tensor = torch.full(
        (int(pixels.shape[0]),),
        float(time_value),
        dtype=pixels.dtype,
        device=device,
    )
    with torch.no_grad():
        output = model(pixel_values=pixels, time=time_tensor)
    if hasattr(output, "output"):
        prediction = output.output
    elif isinstance(output, tuple):
        prediction = output[0]
    else:
        prediction = output
    return prediction.detach().cpu()


def evaluate_poseidon_scot_validation(
    cfg: Mapping[str, Any],
    model: nn.Module,
    *,
    tasks: Sequence[str],
    split: str,
    data_root: str | None,
    max_eval_samples: int,
    rollout_steps: int,
    image_size: int,
    time_value: float,
    device: str | torch.device,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    device = torch.device(device)
    total_pred: list[torch.Tensor] = []
    total_target: list[torch.Tensor] = []
    per_task_pred: dict[str, list[torch.Tensor]] = {}
    per_task_target: dict[str, list[torch.Tensor]] = {}
    per_family_pred: dict[str, list[torch.Tensor]] = {}
    per_family_target: dict[str, list[torch.Tensor]] = {}
    records: list[dict[str, Any]] = []

    for task in tasks:
        dataset = fno_runner._dataset(
            cfg,
            task=task,
            split=split,
            data_root=data_root,
            max_samples=max_eval_samples,
        )
        family = get_pdebench_spec(task).family
        task_pairs = 0
        task_channels: int | None = None
        task_grid_shape: tuple[int, int] | None = None
        for sample_idx in range(len(dataset)):
            fields = dataset[sample_idx]["fields"].float()
            grid_shape = infer_grid_shape(fields)
            channels = infer_channel_count(fields, grid_shape)
            if channels != 1:
                raise ValueError(
                    f"Poseidon-T scalar validation currently expects one channel; "
                    f"task={task} has {channels}"
                )
            task_channels = channels
            task_grid_shape = grid_shape
            max_steps = min(_field_step_count(fields) - 1, int(rollout_steps))
            for step in range(max_steps):
                pixels = light_step_to_poseidon_pixels(
                    fields[step],
                    grid_shape,
                    image_size=image_size,
                )
                pred_pixels = _model_predict_pixels(
                    model,
                    pixels,
                    time_value=time_value,
                    device=device,
                )
                pred = poseidon_pixels_to_repo_flat(pred_pixels, grid_shape)
                target = _flatten_field_step(fields[step + 1].float(), grid_shape).cpu()
                if not torch.isfinite(pred).all():
                    raise RuntimeError(f"Non-finite Poseidon prediction for task={task}")
                total_pred.append(pred)
                total_target.append(target)
                per_task_pred.setdefault(task, []).append(pred)
                per_task_target.setdefault(task, []).append(target)
                per_family_pred.setdefault(family, []).append(pred)
                per_family_target.setdefault(family, []).append(target)
                task_pairs += 1
        records.append(
            {
                "task": task,
                "split": split,
                "family": family,
                "sample_count": len(dataset),
                "pairs_evaluated": task_pairs,
                "repo_inferred_grid_shape": list(task_grid_shape or (0, 0)),
                "repo_inferred_channels": task_channels,
                "poseidon_image_size": int(image_size),
                "time_value": float(time_value),
                "teacher_forced_steps": True,
            }
        )

    if not total_pred:
        raise RuntimeError("Poseidon validation received no eval pairs")
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
    return metrics, records


def _command_record(args: argparse.Namespace) -> list[str]:
    command = [
        "python",
        "scripts/run_external_poseidon_scot_validation.py",
        "--config",
        args.config,
        "--name",
        args.name,
        "--output-root",
        args.output_root,
        "--eval-split",
        args.eval_split,
        "--max-eval-samples",
        str(args.max_eval_samples),
        "--rollout-steps",
        str(args.rollout_steps),
        "--poseidon-model-size",
        args.poseidon_model_size,
        "--checkpoint-file",
        args.checkpoint_file,
        "--device",
        args.device,
        "--time-value",
        str(args.time_value),
    ]
    if args.data_root:
        command.extend(["--data-root", args.data_root])
    if args.poseidon_repo:
        command.extend(["--poseidon-repo", args.poseidon_repo])
    if args.image_size:
        command.extend(["--image-size", str(args.image_size)])
    if args.expected_checkpoint_sha256:
        command.extend(["--expected-checkpoint-sha256", args.expected_checkpoint_sha256])
    if args.allow_held_out_test_eval:
        command.append("--allow-held-out-test-eval")
    tasks = list(args.tasks or args.task)
    if tasks:
        command.append("--tasks")
        command.extend(str(task) for task in tasks)
    return command


def validate_poseidon_scot_summary(summary: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if summary.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if summary.get("status") not in ALLOWED_STATUSES:
        errors.append(f"status must be one of {sorted(ALLOWED_STATUSES)}")
    if summary.get("measurement_type") != MEASUREMENT_TYPE:
        errors.append(f"measurement_type must be {MEASUREMENT_TYPE}")
    if summary.get("split") == "test" and summary.get("held_out_test_used") is not True:
        errors.append("test split summaries must mark held_out_test_used true")
    if summary.get("split") != "test" and summary.get("held_out_test_used") is not False:
        errors.append("validation summaries must mark held_out_test_used false")
    if summary.get("claim_comparable") is not False:
        errors.append("validation Poseidon measurements are not claim comparable")
    if summary.get("published_numbers_directly_comparable") is not False:
        errors.append("published_numbers_directly_comparable must be false")
    metrics = summary.get("metrics")
    if not isinstance(metrics, Mapping) or "decoded_rollout_nrmse" not in metrics:
        errors.append("metrics.decoded_rollout_nrmse is required")
    checkpoint = summary.get("details", {}).get("pretrained_checkpoint")
    if not isinstance(checkpoint, Mapping) or not checkpoint.get("sha256"):
        errors.append("details.pretrained_checkpoint.sha256 is required")
    model_info = summary.get("details", {}).get("model")
    if (
        not isinstance(model_info, Mapping)
        or model_info.get("embedding_recovery_replaced") is not True
    ):
        errors.append("details.model.embedding_recovery_replaced must be true")
    return errors


def run_poseidon_scot_validation(args: argparse.Namespace) -> Path:
    if args.eval_split == "test" and not args.allow_held_out_test_eval:
        raise RuntimeError(
            "Live Poseidon ScOT evaluation on split=test requires --allow-held-out-test-eval. "
            "Use --eval-split val while debugging transfer behavior."
        )
    checkpoint_handle = poseidon_checkpoint_handle(args.poseidon_model_size)
    checkpoint = resolve_checkpoint_file(
        checkpoint_handle=checkpoint_handle,
        filename=args.checkpoint_file,
        expected_sha256=args.expected_checkpoint_sha256,
    )
    cfg = fno_runner._load_cfg(args.config)
    tasks = fno_runner._as_task_names(cfg, args.tasks or args.task)
    model, model_info = load_poseidon_scot_model(
        poseidon_repo=Path(args.poseidon_repo) if args.poseidon_repo else None,
        checkpoint_handle=checkpoint_handle,
        image_size=args.image_size if args.image_size else None,
        channels=1,
        device=args.device,
    )
    image_size = int(model_info["effective_config"]["image_size"])
    started = time.time()
    metrics, records = evaluate_poseidon_scot_validation(
        cfg,
        model,
        tasks=tasks,
        split=args.eval_split,
        data_root=args.data_root,
        max_eval_samples=args.max_eval_samples,
        rollout_steps=args.rollout_steps,
        image_size=image_size,
        time_value=args.time_value,
        device=args.device,
    )
    finished = time.time()
    output_root = Path(args.output_root)
    run_dir = output_root / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": "validation_model_measurement_complete",
        "measurement_type": MEASUREMENT_TYPE,
        "run_name": args.name,
        "config": args.config,
        "eval_config": args.config,
        "split": args.eval_split,
        "metrics": metrics,
        "claim_comparable": False,
        "published_numbers_directly_comparable": False,
        "held_out_test_used": args.eval_split == "test",
        "held_out_test_data_read": args.eval_split == "test",
        "stages": ["external_poseidon_scot_validation"],
        "extra": {
            "baseline": "external_poseidon_scot",
            "implementation": POSEIDON_MODEL_IMPORT,
            "source_url": POSEIDON_SOURCE_URL,
            "task": tasks[0] if len(tasks) == 1 else list(tasks),
            "split": args.eval_split,
            "max_eval_samples": args.max_eval_samples,
            "rollout_steps": args.rollout_steps,
            "image_size": image_size,
            "time_value": args.time_value,
            "device": args.device,
            "metric": "decoded_rollout_nrmse",
            "allow_held_out_test_eval": bool(args.allow_held_out_test_eval),
            "command": _command_record(args),
        },
        "details": {
            "poseidon_source": poseidon_source_snapshot(
                Path(args.poseidon_repo) if args.poseidon_repo else None
            ),
            "pretrained_checkpoint": checkpoint,
            "model": model_info,
            "evaluation_records": records,
            "contract": {
                "validation_split_only": args.eval_split != "test",
                "teacher_forced_light_v1_steps": True,
                "embedding_recovery_replaced_for_scalar_light_v1": True,
                "requires_finetuning_before_held_out_test": True,
                "published_numbers_directly_comparable": False,
            },
        },
        "duration_sec": finished - started,
    }
    errors = validate_poseidon_scot_summary(summary)
    if errors:
        summary["status"] = "invalid"
        summary["validation_errors"] = errors
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "status": summary["status"],
                "main_metric": {
                    "decoded_rollout_nrmse": metrics["decoded_rollout_nrmse"],
                },
            },
            indent=2,
        )
    )
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_multitask_heterogeneous_light_best.yaml")
    parser.add_argument("--name", default="poseidon_scot_val_light_v1")
    parser.add_argument("--output-root", default="reports/research/sota_loop/external_baselines")
    parser.add_argument("--data-root")
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--tasks", nargs="+", default=[])
    parser.add_argument("--eval-split", default="val")
    parser.add_argument("--max-eval-samples", type=int, default=2)
    parser.add_argument("--rollout-steps", type=int, default=4)
    parser.add_argument("--poseidon-model-size", default="T")
    parser.add_argument("--checkpoint-file", default=DEFAULT_CHECKPOINT_FILE)
    parser.add_argument("--expected-checkpoint-sha256", default="")
    parser.add_argument("--poseidon-repo")
    parser.add_argument("--image-size", type=int, default=0)
    parser.add_argument("--time-value", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-held-out-test-eval", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_poseidon_scot_validation(args)


if __name__ == "__main__":
    main()
