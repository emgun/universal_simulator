#!/usr/bin/env python
from __future__ import annotations

"""Validate the light-v1 data adapter needed for Poseidon ScOT transfer.

This is not a Poseidon performance measurement. It writes a validation-only
manifest for the data/shape/provenance gate that must pass before an official
pretrained ScOT checkpoint can be evaluated against the light-v1 protocol.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts import run_external_neuraloperator_fno_baseline as fno_runner
from scripts.run_physical_conv_baseline import field_step_to_grid, grid_to_flat
from ups.data.latent_pairs import infer_channel_count, infer_grid_shape
from ups.data.pdebench import PDEBenchConfig, PDEBenchDataset, get_pdebench_spec
from ups.eval.pdebench_runner import _aggregate_chunk_metrics, _flatten_field_step

POSEIDON_MODEL_IMPORT = "scOT.model.ScOT"
POSEIDON_CONFIG_IMPORT = "scOT.model.ScOTConfig"
POSEIDON_SOURCE_URL = "https://github.com/camlab-ethz/poseidon"
POSEIDON_PRETRAINED_MODEL_TEMPLATE = "camlab-ethz/Poseidon-{model_size}"
POSEIDON_PRETRAINING_COLLECTION = (
    "https://huggingface.co/collections/camlab-ethz/poseidon-664fa125729c53d8607e209a"
)
POSEIDON_DOWNSTREAM_COLLECTION = (
    "https://huggingface.co/collections/camlab-ethz/"
    "poseidon-downstream-tasks-664fa237cd6b0c097971ef14"
)

ALLOWED_STATUSES = {"validation_adapter_manifest_complete", "invalid"}
ADAPTER_MEASUREMENT_TYPE = "poseidon_validation_adapter_manifest"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(path: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=True,
        )
    except Exception:
        return "unknown"
    return proc.stdout.strip()


def _field_step_count(fields: torch.Tensor) -> int:
    if fields.dim() >= 3 and fields.shape[0] > 1:
        return int(fields.shape[0])
    return 1


def poseidon_checkpoint_handle(model_size: str) -> str:
    normalized = str(model_size).upper()
    if normalized not in {"T", "B", "L"}:
        raise ValueError("Poseidon model size must be one of T, B, or L")
    return POSEIDON_PRETRAINED_MODEL_TEMPLATE.format(model_size=normalized)


def poseidon_source_snapshot(poseidon_repo: Path | None) -> dict[str, Any]:
    exists = poseidon_repo is not None and poseidon_repo.exists()
    required_files = [
        "README.md",
        "pyproject.toml",
        "scOT/model.py",
        "scOT/train.py",
        "scOT/inference.py",
        "scOT/problems/base.py",
    ]
    files: dict[str, dict[str, Any]] = {}
    for relative in required_files:
        path = poseidon_repo / relative if exists and poseidon_repo is not None else Path(relative)
        files[relative] = {
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else 0,
        }
    return {
        "source_id": "poseidon_official_repo",
        "source_url": POSEIDON_SOURCE_URL,
        "repo_path": str(poseidon_repo) if poseidon_repo is not None else "",
        "available": bool(exists),
        "commit": _git_commit(poseidon_repo) if exists and poseidon_repo is not None else "missing",
        "required_files": files,
    }


def poseidon_import_status(poseidon_repo: Path | None) -> dict[str, Any]:
    inserted_path = ""
    if poseidon_repo is not None and poseidon_repo.exists():
        inserted_path = str(poseidon_repo)
        sys.path.insert(0, inserted_path)
    try:
        from scOT.model import ScOT, ScOTConfig  # noqa: F401
    except Exception as exc:
        return {
            "available": False,
            "import": POSEIDON_MODEL_IMPORT,
            "config_import": POSEIDON_CONFIG_IMPORT,
            "source_url": POSEIDON_SOURCE_URL,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    finally:
        if inserted_path:
            try:
                sys.path.remove(inserted_path)
            except ValueError:
                pass
    return {
        "available": True,
        "import": POSEIDON_MODEL_IMPORT,
        "config_import": POSEIDON_CONFIG_IMPORT,
        "source_url": POSEIDON_SOURCE_URL,
    }


def light_step_to_poseidon_pixels(
    field_step: torch.Tensor,
    grid_shape: tuple[int, int],
    *,
    image_size: int,
) -> torch.Tensor:
    """Convert one repo field step into Poseidon-style square pixel values."""

    grid = field_step_to_grid(field_step.float(), grid_shape)
    return F.interpolate(
        grid,
        size=(int(image_size), int(image_size)),
        mode="bilinear",
        align_corners=False,
    ).contiguous()


def poseidon_pixels_to_repo_flat(
    pixel_values: torch.Tensor,
    grid_shape: tuple[int, int],
) -> torch.Tensor:
    """Map Poseidon square output pixels back to the repo flattened metric shape."""

    grid = F.interpolate(
        pixel_values.float(),
        size=(int(grid_shape[0]), int(grid_shape[1])),
        mode="bilinear",
        align_corners=False,
    )
    return grid_to_flat(grid)


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


def _inspect_task_split(
    *,
    cfg: Mapping[str, Any],
    task: str,
    split: str,
    data_root: str | None,
    max_samples: int,
    max_steps: int,
    image_size: int,
) -> tuple[dict[str, Any], list[torch.Tensor], list[torch.Tensor]]:
    dataset = PDEBenchDataset(
        PDEBenchConfig(
            task=task,
            split=split,
            root=data_root or cfg.get("data", {}).get("root"),
            max_samples=max_samples,
        )
    )
    pred_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    pixel_shape: list[int] | None = None
    raw_sample_shape: list[int] | None = None
    grid_shape: tuple[int, int] | None = None
    channels = 0
    steps_inspected = 0
    sample_count = min(int(max_samples), len(dataset))
    for sample_idx in range(sample_count):
        fields = dataset[sample_idx]["fields"].float()
        raw_sample_shape = list(fields.shape)
        grid_shape = infer_grid_shape(fields)
        channels = infer_channel_count(fields, grid_shape)
        for step in range(min(_field_step_count(fields), int(max_steps))):
            original = _flatten_field_step(fields[step].float(), grid_shape)
            pixels = light_step_to_poseidon_pixels(
                fields[step],
                grid_shape,
                image_size=image_size,
            )
            roundtrip = poseidon_pixels_to_repo_flat(pixels, grid_shape)
            pixel_shape = list(pixels.shape)
            pred_chunks.append(roundtrip)
            target_chunks.append(original)
            steps_inspected += 1
    if not pred_chunks or grid_shape is None or raw_sample_shape is None or pixel_shape is None:
        raise RuntimeError(f"No adapter samples inspected for task={task} split={split}")
    stats = _aggregate_chunk_metrics(pred_chunks, target_chunks)
    spec = get_pdebench_spec(task)
    record = {
        "task": task,
        "split": split,
        "family": spec.family,
        "traits": list(spec.traits),
        "sample_count_inspected": sample_count,
        "steps_inspected": steps_inspected,
        "raw_sample_shape": raw_sample_shape,
        "repo_inferred_grid_shape": list(grid_shape),
        "repo_inferred_channels": channels,
        "poseidon_pixel_shape": pixel_shape,
        "adapter_policy": {
            "input": "repo field step",
            "to_poseidon": "field_step_to_grid then bilinear resize to square pixel_values",
            "from_poseidon": "bilinear resize back to repo-inferred grid then grid_to_flat",
            "image_size": int(image_size),
            "lossy_shape_adapter": True,
        },
        "roundtrip": {
            "mse": stats["mse"],
            "mae": stats["mae"],
            "nrmse": stats["nrmse"],
            "rrmse": stats["rrmse"],
            "spectral_energy_error": stats["spectral_energy_error"],
        },
        "source": _split_source_record(data_root or cfg.get("data", {}).get("root"), task, split),
    }
    return record, pred_chunks, target_chunks


def inspect_poseidon_adapter(
    *,
    cfg: Mapping[str, Any],
    tasks: Sequence[str],
    splits: Sequence[str],
    data_root: str | None,
    max_samples: int,
    max_steps: int,
    image_size: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    records: list[dict[str, Any]] = []
    all_pred: list[torch.Tensor] = []
    all_target: list[torch.Tensor] = []
    for task in tasks:
        for split in splits:
            record, pred, target = _inspect_task_split(
                cfg=cfg,
                task=task,
                split=split,
                data_root=data_root,
                max_samples=max_samples,
                max_steps=max_steps,
                image_size=image_size,
            )
            records.append(record)
            all_pred.extend(pred)
            all_target.extend(target)
    stats = _aggregate_chunk_metrics(all_pred, all_target)
    metrics = {
        "adapter_roundtrip_mse": stats["mse"],
        "adapter_roundtrip_mae": stats["mae"],
        "adapter_roundtrip_nrmse": stats["nrmse"],
        "adapter_roundtrip_rrmse": stats["rrmse"],
        "adapter_roundtrip_spectral_energy_error": stats["spectral_energy_error"],
    }
    for record in records:
        key_prefix = f"adapter_roundtrip_{record['task']}_{record['split']}"
        metrics[f"{key_prefix}_nrmse"] = float(record["roundtrip"]["nrmse"])
        metrics[f"{key_prefix}_mse"] = float(record["roundtrip"]["mse"])
    return records, metrics


def _command_record(args: argparse.Namespace) -> list[str]:
    command = [
        "python",
        "scripts/run_external_poseidon_transfer_adapter.py",
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
        "--max-samples",
        str(args.max_samples),
        "--max-steps",
        str(args.max_steps),
        "--image-size",
        str(args.image_size),
        "--poseidon-model-size",
        args.poseidon_model_size,
        "--foundation-contract-json",
        args.foundation_contract_json,
    ]
    if args.data_root:
        command.extend(["--data-root", args.data_root])
    if args.poseidon_repo:
        command.extend(["--poseidon-repo", args.poseidon_repo])
    if args.pretrained_checkpoint_sha256:
        command.extend(["--pretrained-checkpoint-sha256", args.pretrained_checkpoint_sha256])
    tasks = list(args.tasks or args.task)
    if tasks:
        command.append("--tasks")
        command.extend(str(task) for task in tasks)
    return command


def validate_poseidon_adapter_summary(summary: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if summary.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if summary.get("status") not in ALLOWED_STATUSES:
        errors.append(f"status must be one of {sorted(ALLOWED_STATUSES)}")
    if summary.get("measurement_type") != ADAPTER_MEASUREMENT_TYPE:
        errors.append(f"measurement_type must be {ADAPTER_MEASUREMENT_TYPE}")
    if summary.get("claim_comparable") is not False:
        errors.append("claim_comparable must be false for adapter manifests")
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
        errors.append("validation adapter manifests must not inspect split=test")

    metrics = summary.get("metrics")
    if not isinstance(metrics, Mapping) or "adapter_roundtrip_nrmse" not in metrics:
        errors.append("metrics.adapter_roundtrip_nrmse is required")
    elif "decoded_rollout_nrmse" in metrics:
        errors.append("adapter manifests must not report decoded_rollout_nrmse")

    details = summary.get("details")
    if not isinstance(details, Mapping):
        errors.append("details must be an object")
        details = {}
    if not details.get("adapter_records"):
        errors.append("details.adapter_records is required")
    checkpoint = details.get("pretrained_checkpoint")
    if not isinstance(checkpoint, Mapping) or not checkpoint.get("handle"):
        errors.append("details.pretrained_checkpoint.handle is required")
    elif checkpoint.get("requires_hash_before_model_metric") is not True:
        errors.append("pretrained checkpoint must require hash before model metric")
    source = details.get("poseidon_source")
    if not isinstance(source, Mapping) or not source.get("commit"):
        errors.append("details.poseidon_source.commit is required")
    return errors


def run_adapter_manifest(args: argparse.Namespace) -> Path:
    inspected_splits = [str(args.train_split), str(args.eval_split)]
    if "test" in inspected_splits:
        raise RuntimeError("Validation-only Poseidon adapter manifest must not inspect split=test")

    cfg = fno_runner._load_cfg(args.config)
    tasks = fno_runner._as_task_names(cfg, args.tasks or args.task)
    started = time.time()
    records, metrics = inspect_poseidon_adapter(
        cfg=cfg,
        tasks=tasks,
        splits=list(dict.fromkeys(inspected_splits)),
        data_root=args.data_root,
        max_samples=args.max_samples,
        max_steps=args.max_steps,
        image_size=args.image_size,
    )
    finished = time.time()

    output_root = Path(args.output_root)
    run_dir = output_root / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    checkpoint_handle = poseidon_checkpoint_handle(args.poseidon_model_size)
    poseidon_repo = Path(args.poseidon_repo) if args.poseidon_repo else None
    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": "validation_adapter_manifest_complete",
        "measurement_type": ADAPTER_MEASUREMENT_TYPE,
        "run_name": args.name,
        "config": args.config,
        "eval_config": args.config,
        "split": args.eval_split,
        "inspected_splits": list(dict.fromkeys(inspected_splits)),
        "metrics": metrics,
        "claim_comparable": False,
        "published_numbers_directly_comparable": False,
        "held_out_test_used": False,
        "held_out_test_data_read": False,
        "stages": ["external_poseidon_transfer_adapter_manifest"],
        "extra": {
            "baseline": "external_poseidon_transfer_adapter",
            "implementation": POSEIDON_MODEL_IMPORT,
            "source_url": POSEIDON_SOURCE_URL,
            "pretraining_collection": POSEIDON_PRETRAINING_COLLECTION,
            "downstream_collection": POSEIDON_DOWNSTREAM_COLLECTION,
            "task": tasks[0] if len(tasks) == 1 else list(tasks),
            "train_split": args.train_split,
            "split": args.eval_split,
            "max_samples": args.max_samples,
            "max_steps": args.max_steps,
            "image_size": args.image_size,
            "metric": "adapter_roundtrip_nrmse",
            "foundation_contract_json": args.foundation_contract_json,
            "command": _command_record(args),
        },
        "details": {
            "adapter_records": records,
            "poseidon_source": poseidon_source_snapshot(poseidon_repo),
            "poseidon_import": poseidon_import_status(poseidon_repo),
            "pretrained_checkpoint": {
                "handle": checkpoint_handle,
                "model_size": str(args.poseidon_model_size).upper(),
                "sha256": args.pretrained_checkpoint_sha256 or "",
                "sha256_status": "provided" if args.pretrained_checkpoint_sha256 else "pending",
                "requires_hash_before_model_metric": True,
            },
            "contract": {
                "foundation_contract_json": args.foundation_contract_json,
                "validation_split_only": True,
                "produces_model_metric": False,
                "published_numbers_directly_comparable": False,
                "next_gate": (
                    "load ScOT.from_pretrained with checkpoint hash, run validation "
                    "decoded_rollout_nrmse, then decide whether to spend held-out test budget"
                ),
            },
        },
        "duration_sec": finished - started,
    }
    errors = validate_poseidon_adapter_summary(summary)
    if errors:
        summary["status"] = "invalid"
        summary["validation_errors"] = errors
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": summary["status"],
                "summary": str(summary_path),
                "adapter_roundtrip_nrmse": metrics["adapter_roundtrip_nrmse"],
            },
            indent=2,
        )
    )
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_multitask_heterogeneous_light_best.yaml")
    parser.add_argument("--name", default="poseidon_transfer_adapter_manifest_light_v1")
    parser.add_argument("--output-root", default="reports/research/sota_loop/external_baselines")
    parser.add_argument("--data-root")
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--tasks", nargs="+", default=[])
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="val")
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--poseidon-model-size", default="T")
    parser.add_argument("--poseidon-repo")
    parser.add_argument("--pretrained-checkpoint-sha256", default="")
    parser.add_argument(
        "--foundation-contract-json",
        default="docs/claim_evidence/foundation_transfer_readiness_light_v1.json",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_adapter_manifest(args)


if __name__ == "__main__":
    main()
