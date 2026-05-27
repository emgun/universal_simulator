#!/usr/bin/env python
from __future__ import annotations

"""Run lightweight UPS experiments with repeatable configs, overrides, and summaries."""

import argparse
import copy
import csv
import json
import math
import os
import shutil
import sys
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import h5py
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts import evaluate as evaluate_script
from scripts import train as train_script
from ups.eval.pdebench_runner import evaluate_decoded_operator, evaluate_latent_operator
from ups.eval.promotion import (
    evaluate_promotion_rules,
    parse_promotion_rule,
    promotion_rules_from_config,
)
from ups.utils.config_loader import load_config_with_includes
from ups.utils.monitoring import init_monitoring_session

STAGE_FUNCTIONS = {
    "operator": train_script.train_operator,
    "decoder": train_script.train_decoder,
    "operator_decoded": train_script.train_operator_decoded,
    "joint_codec_operator": train_script.train_joint_codec_operator,
    "diff_residual": train_script.train_diffusion,
    "consistency_distill": train_script.train_consistency,
    "steady_prior": train_script.train_steady_prior,
}

DEFAULT_OPERATOR_CHECKPOINTS = ("operator_joint.pt", "operator_decoded.pt", "operator.pt")


def _operator_checkpoint_names_for_stages(stages: Sequence[str]) -> tuple[str, ...]:
    for stage in reversed(stages):
        if stage == "joint_codec_operator":
            return DEFAULT_OPERATOR_CHECKPOINTS
        if stage == "operator_decoded":
            return ("operator_decoded.pt", "operator.pt", "operator_joint.pt")
        if stage == "operator":
            return ("operator.pt", "operator_decoded.pt", "operator_joint.pt")
    return DEFAULT_OPERATOR_CHECKPOINTS


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _parse_override(text: str) -> tuple[str, Any]:
    if "=" not in text:
        raise ValueError(f"Invalid override '{text}'. Expected key=value")
    key, raw = text.split("=", 1)
    return key.strip(), yaml.safe_load(raw)


def _set_dotpath(cfg: dict[str, Any], path: str, value: Any) -> None:
    cursor: dict[str, Any] = cfg
    parts = [part for part in path.split(".") if part]
    if not parts:
        raise ValueError(f"Invalid override path '{path}'")
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def _apply_overrides(cfg: dict[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    updated = copy.deepcopy(cfg)
    for text in overrides:
        key, value = _parse_override(text)
        _set_dotpath(updated, key, value)
    return updated


def _task_names(cfg: dict[str, Any]) -> list[str]:
    task_cfg = cfg.get("data", {}).get("task")
    if isinstance(task_cfg, str):
        return [task_cfg]
    if isinstance(task_cfg, (list, tuple)):
        return [str(task) for task in task_cfg]
    raise ValueError(
        "Experiment configs require data.task to be a task name or a list of task names"
    )


def _all_tasks(*task_groups: Iterable[str]) -> list[str]:
    ordered: list[str] = []
    seen = set()
    for group in task_groups:
        for task in group:
            if task in seen:
                continue
            seen.add(task)
            ordered.append(task)
    return ordered


def _split_name(cfg: dict[str, Any], default: str = "train") -> str:
    return str(cfg.get("data", {}).get("split", default))


def _override_data_root(cfg: dict[str, Any], root: Path) -> dict[str, Any]:
    updated = copy.deepcopy(cfg)
    updated.setdefault("data", {})["root"] = str(root)
    return updated


def _prepare_runtime_cfg(
    cfg: dict[str, Any],
    *,
    checkpoint_dir: Path,
    log_dir: Path,
    disable_wandb: bool,
) -> dict[str, Any]:
    updated = copy.deepcopy(cfg)
    updated.setdefault("checkpoint", {})["dir"] = str(checkpoint_dir)
    updated.setdefault("training", {})["log_path"] = str(log_dir / "training.jsonl")
    updated.setdefault("logging", {}).setdefault("wandb", {})
    if disable_wandb:
        updated["logging"]["wandb"]["enabled"] = False
    return updated


def _prepare_eval_cfg(
    train_cfg: dict[str, Any],
    eval_cfg: dict[str, Any] | None,
    *,
    log_dir: Path,
    disable_wandb: bool,
) -> dict[str, Any]:
    if eval_cfg is None:
        prepared = copy.deepcopy(train_cfg)
    else:
        prepared = _deep_merge(train_cfg, eval_cfg)
    prepared.setdefault("training", {})
    prepared["training"]["dt"] = train_cfg.get("training", {}).get(
        "dt", prepared["training"].get("dt", 0.1)
    )
    prepared["training"].setdefault(
        "batch_size", train_cfg.get("training", {}).get("batch_size", 1)
    )
    prepared["training"].setdefault("num_workers", 0)
    prepared["training"].setdefault("pin_memory", False)
    prepared["training"]["log_path"] = str(log_dir / "evaluation.jsonl")
    prepared.setdefault("logging", {}).setdefault("wandb", {})
    if disable_wandb:
        prepared["logging"]["wandb"]["enabled"] = False
    return prepared


def _dedupe_text(values: Iterable[str]) -> list[str]:
    seen: dict[str, None] = {}
    for value in values:
        text = str(value).strip()
        if text:
            seen.setdefault(text, None)
    return list(seen)


def _split_env_list(text: str) -> list[str]:
    return [item.strip() for item in text.replace(";", ",").split(",") if item.strip()]


def _configure_wandb(
    cfg: dict[str, Any],
    *,
    enabled: bool,
    run_name: str,
    project: str,
    entity: str,
    group: str,
    tags: Sequence[str],
    job_type: str,
) -> dict[str, Any]:
    updated = copy.deepcopy(cfg)
    wandb_cfg = updated.setdefault("logging", {}).setdefault("wandb", {})
    wandb_cfg["enabled"] = bool(enabled)
    if enabled:
        wandb_cfg["run_name"] = run_name
        if project:
            wandb_cfg["project"] = project
        if entity:
            wandb_cfg["entity"] = entity
        if group:
            wandb_cfg["group"] = group
        if job_type:
            wandb_cfg["job_type"] = job_type
        existing_tags = wandb_cfg.get("tags", [])
        if isinstance(existing_tags, str):
            existing_tags = [existing_tags]
        wandb_cfg["tags"] = _dedupe_text([*existing_tags, *tags, "light-experiment"])
    return updated


def _read_wandb_run_records(log_dir: Path) -> list[dict[str, Any]]:
    path = log_dir / "wandb_runs.jsonl"
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _tracking_payload(
    *,
    allow_wandb: bool,
    train_cfg: dict[str, Any],
    log_dir: Path,
) -> dict[str, Any]:
    wandb_cfg = train_cfg.get("logging", {}).get("wandb", {})
    runs = _read_wandb_run_records(log_dir)
    return {
        "wandb": {
            "requested": bool(allow_wandb),
            "enabled": bool(wandb_cfg.get("enabled")),
            "mode": os.environ.get("WANDB_MODE", ""),
            "project": wandb_cfg.get("project", ""),
            "entity": wandb_cfg.get("entity", ""),
            "group": wandb_cfg.get("group", ""),
            "run_name": wandb_cfg.get("run_name", ""),
            "tags": wandb_cfg.get("tags", []),
            "job_type": wandb_cfg.get("job_type", ""),
            "run_count": len(runs),
            "runs": runs,
        }
    }


def _summary_wandb_payload(summary: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in summary.get("metrics", {}).items():
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            payload[f"summary/{key}"] = float(value)
    for key, value in summary.get("extra", {}).items():
        if isinstance(value, bool):
            payload[f"summary_extra/{key}"] = int(value)
        elif isinstance(value, (int, float)) and math.isfinite(float(value)):
            payload[f"summary_extra/{key}"] = float(value)
        elif isinstance(value, str):
            payload[f"summary_extra/{key}"] = value
    for split, split_payload in summary.get("extra_evaluations", {}).items():
        safe_split = _safe_artifact_name(str(split))
        for key, value in split_payload.get("metrics", {}).items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                payload[f"summary_{safe_split}/{key}"] = float(value)
    payload["summary/run_name"] = str(summary.get("run_name", ""))
    payload["summary/stages"] = ",".join(str(stage) for stage in summary.get("stages", ()))
    payload["summary/skip_training"] = int(bool(summary.get("skip_training", False)))
    if "duration_sec" in summary:
        payload["summary/duration_sec"] = float(summary["duration_sec"])
    return payload


def _log_summary_to_wandb(
    *,
    allow_wandb: bool,
    train_cfg: dict[str, Any],
    log_dir: Path,
    summary: dict[str, Any],
) -> None:
    if not allow_wandb or not train_cfg.get("logging", {}).get("wandb", {}).get("enabled"):
        return
    session = init_monitoring_session(
        train_cfg,
        component="benchmark-summary",
        file_path=str(log_dir / "benchmark_summary.jsonl"),
    )
    try:
        session.log(_summary_wandb_payload(summary))
    finally:
        session.finish()


def _preferred_checkpoint(checkpoint_dir: Path, names: Sequence[str]) -> Path | None:
    for name in names:
        candidate = checkpoint_dir / name
        if candidate.exists():
            return candidate
    return None


def _copy_checkpoint_source(source: Path, checkpoint_dir: Path) -> list[Path]:
    source_dir = source / "checkpoints" if (source / "checkpoints").is_dir() else source
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint source directory not found: {source}")
    copied: list[Path] = []
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(source_dir.glob("*.pt")):
        target = checkpoint_dir / path.name
        shutil.copy2(path, target)
        copied.append(target)
    if not copied:
        raise FileNotFoundError(f"No .pt checkpoints found in {source_dir}")
    return copied


def _synthetic_task_shape(task: str) -> tuple[int, ...]:
    if "2d" in task:
        return (4, 4)
    return (8,)


def _task_seed(task: str, split: str) -> float:
    return float(sum(ord(ch) for ch in f"{task}:{split}") % 31)


def _make_1d_series(task: str, split: str, samples: int, steps: int, width: int) -> torch.Tensor:
    xs = torch.linspace(0.0, 2.0 * math.pi, width, dtype=torch.float32)
    data = torch.zeros(samples, steps, width, dtype=torch.float32)
    base_shift = _task_seed(task, split) * 0.03
    for sample_idx in range(samples):
        amplitude = 0.5 + 0.1 * sample_idx + base_shift
        for step_idx in range(steps):
            phase = 0.35 * step_idx + 0.12 * sample_idx + base_shift
            wave = amplitude * torch.sin(xs + phase)
            if "advection" in task:
                wave = amplitude * torch.sin(xs - phase)
            elif "burgers" in task:
                wave = wave + 0.2 * torch.sin(2.0 * xs + phase)
            data[sample_idx, step_idx] = wave
    return data


def _make_2d_series(
    task: str, split: str, samples: int, steps: int, height: int, width: int
) -> torch.Tensor:
    ys = torch.linspace(0.0, 1.0, height, dtype=torch.float32)
    xs = torch.linspace(0.0, 1.0, width, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    data = torch.zeros(samples, steps, 1, height, width, dtype=torch.float32)
    base_shift = _task_seed(task, split) * 0.02
    for sample_idx in range(samples):
        amplitude = 0.4 + 0.08 * sample_idx + base_shift
        for step_idx in range(steps):
            phase = 0.25 * step_idx + 0.1 * sample_idx + base_shift
            field = (
                amplitude
                * torch.sin(2.0 * math.pi * (grid_x + phase))
                * torch.cos(2.0 * math.pi * (grid_y - phase))
            )
            if "darcy" in task:
                field = field + 0.15 * grid_x
            elif "navier" in task:
                field = field + 0.1 * torch.sin(4.0 * math.pi * grid_y + phase)
            data[sample_idx, step_idx, 0] = field
    return data


def _make_metadata_dataset(samples: int, steps: int, index: int) -> torch.Tensor:
    base = torch.linspace(0.0, 1.0, steps, dtype=torch.float32).view(1, steps, 1)
    return base.repeat(samples, 1, 1) + 0.05 * float(index)


def bootstrap_synthetic_pdebench(
    root: Path,
    *,
    tasks: Sequence[str],
    splits: Sequence[str],
    param_keys: Sequence[str] = (),
    bc_keys: Sequence[str] = (),
    samples: int = 2,
    steps: int = 4,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        shape = _synthetic_task_shape(task)
        for split in splits:
            file_path = root / f"{task}_{split}.h5"
            if file_path.exists():
                continue
            with h5py.File(file_path, "w") as handle:
                if len(shape) == 1:
                    fields = _make_1d_series(task, split, samples, steps, shape[0])
                else:
                    fields = _make_2d_series(task, split, samples, steps, shape[0], shape[1])
                handle.create_dataset("data", data=fields.numpy())
                for idx, key in enumerate(param_keys):
                    handle.create_dataset(
                        key, data=_make_metadata_dataset(samples, steps, idx).numpy()
                    )
                for idx, key in enumerate(bc_keys):
                    handle.create_dataset(
                        key,
                        data=_make_metadata_dataset(samples, steps, idx + len(param_keys)).numpy(),
                    )


def _clone_eval_cfg(
    cfg: dict[str, Any], *, tasks: Sequence[str] | None = None, split: str | None = None
) -> dict[str, Any]:
    cloned = copy.deepcopy(cfg)
    data_cfg = cloned.setdefault("data", {})
    if tasks:
        data_cfg["task"] = list(tasks) if len(tasks) > 1 else str(tasks[0])
    if split is not None:
        data_cfg["split"] = split
    return cloned


def _safe_artifact_name(text: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in text.strip())
    return safe or "eval"


def _evaluate_once(
    cfg: dict[str, Any],
    *,
    checkpoint_dir: Path,
    operator_checkpoint_names: Sequence[str],
    decoded: bool,
    device: str,
    decoded_rollout_steps: int | None,
    transfer_tasks: Sequence[str],
    transfer_split: str | None,
    cli_promotion_rules: Sequence[str],
) -> dict[str, Any]:
    operator_ckpt = _preferred_checkpoint(checkpoint_dir, operator_checkpoint_names)
    if operator_ckpt is None:
        raise FileNotFoundError(f"No operator checkpoint found in {checkpoint_dir}")

    operator = evaluate_script.make_operator(cfg)
    evaluate_script._load_state_dict_compat(operator, str(operator_ckpt), prefix_to_strip="")
    result = evaluate_latent_operator(cfg, operator, device=device, return_details=True)
    report, details = result  # type: ignore[misc]

    if decoded:
        encoder_ckpt = _preferred_checkpoint(checkpoint_dir, ("encoder_joint.pt", "encoder.pt"))
        decoder_ckpt = _preferred_checkpoint(checkpoint_dir, ("decoder_joint.pt", "decoder.pt"))
        if encoder_ckpt is None or decoder_ckpt is None:
            raise FileNotFoundError(
                "Decoded evaluation requested but encoder/decoder checkpoints are missing"
            )
        encoder = evaluate_script.make_encoder(cfg)
        decoder = evaluate_script.make_decoder(cfg)
        evaluate_script._load_state_dict_compat(encoder, str(encoder_ckpt), prefix_to_strip="")
        evaluate_script._load_state_dict_compat(decoder, str(decoder_ckpt), prefix_to_strip="")
        decoded_report = evaluate_decoded_operator(
            cfg,
            encoder,
            operator,
            decoder,
            device=device,
            rollout_steps=decoded_rollout_steps,
        )
        report.metrics.update(decoded_report.metrics)
        if report.extra is None:
            report.extra = {}
        if decoded_report.extra:
            report.extra.update({f"decoded_{k}": v for k, v in decoded_report.extra.items()})

    if transfer_tasks:
        transfer_cfg = _clone_eval_cfg(cfg, tasks=transfer_tasks, split=transfer_split)
        transfer_report = evaluate_latent_operator(
            transfer_cfg, operator, device=device, return_details=False
        )
        report.metrics.update(
            {f"transfer_{key}": value for key, value in transfer_report.metrics.items()}
        )
        if report.extra is None:
            report.extra = {}
        report.extra["transfer_tasks"] = list(transfer_tasks)
        report.extra["transfer_split"] = _split_name(transfer_cfg)
        if decoded:
            encoder_ckpt = _preferred_checkpoint(checkpoint_dir, ("encoder_joint.pt", "encoder.pt"))
            decoder_ckpt = _preferred_checkpoint(checkpoint_dir, ("decoder_joint.pt", "decoder.pt"))
            assert encoder_ckpt is not None and decoder_ckpt is not None
            encoder = evaluate_script.make_encoder(transfer_cfg)
            decoder = evaluate_script.make_decoder(transfer_cfg)
            evaluate_script._load_state_dict_compat(encoder, str(encoder_ckpt), prefix_to_strip="")
            evaluate_script._load_state_dict_compat(decoder, str(decoder_ckpt), prefix_to_strip="")
            transfer_decoded_report = evaluate_decoded_operator(
                transfer_cfg,
                encoder,
                operator,
                decoder,
                device=device,
                rollout_steps=decoded_rollout_steps,
            )
            report.metrics.update(
                {f"transfer_{key}": value for key, value in transfer_decoded_report.metrics.items()}
            )

    promotion_rules = promotion_rules_from_config(cfg)
    promotion_rules.extend(parse_promotion_rule(rule) for rule in cli_promotion_rules)
    if promotion_rules:
        promotion_result = evaluate_promotion_rules(report.metrics, promotion_rules)
        if report.extra is None:
            report.extra = {}
        report.extra["promotion_passed"] = promotion_result.passed
        report.extra["promotion_rule_count"] = len(promotion_rules)
        report.extra["promotion_failed_rules"] = promotion_result.failed_rules
        report.extra["promotion_missing_metrics"] = promotion_result.missing_metrics

    return {
        "metrics": report.metrics,
        "extra": report.extra or {},
        "details": details,
        "checkpoints": {
            "operator": str(operator_ckpt),
            "encoder": (
                str(_preferred_checkpoint(checkpoint_dir, ("encoder_joint.pt", "encoder.pt")))
                if decoded
                else None
            ),
            "decoder": (
                str(_preferred_checkpoint(checkpoint_dir, ("decoder_joint.pt", "decoder.pt")))
                if decoded
                else None
            ),
        },
    }


def _main_metric(metrics: dict[str, float]) -> tuple[str, float]:
    for key in (
        "decoded_rollout_nrmse",
        "transfer_decoded_rollout_nrmse",
        "decoded_step1_nrmse",
        "mse",
    ):
        if key in metrics:
            return key, float(metrics[key])
    first_key = next(iter(metrics))
    return first_key, float(metrics[first_key])


def _append_results_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_name",
        "timestamp",
        "stages",
        "decoded",
        "train_split",
        "eval_split",
        "transfer_tasks",
        "promotion_passed",
        "main_metric_name",
        "main_metric_value",
        "summary_json",
        "wandb_run_ids",
        "wandb_urls",
    ]
    row_map: dict[str, dict[str, Any]] = {}
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for existing in reader:
                run_name = existing.get("run_name")
                if run_name:
                    row_map[run_name] = dict(existing)
    row_map[str(row["run_name"])] = row
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(row_map[name] for name in sorted(row_map))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run lightweight UPS experiments with resolved configs and summaries"
    )
    parser.add_argument("--config", required=True, help="Training config path")
    parser.add_argument(
        "--eval-config", help="Optional evaluation config path; defaults to the training config"
    )
    parser.add_argument(
        "--name", required=True, help="Experiment name used for the output directory"
    )
    parser.add_argument(
        "--output-root",
        default="reports/light_experiments",
        help="Root directory for experiment outputs",
    )
    parser.add_argument(
        "--stage",
        action="append",
        default=None,
        choices=sorted(STAGE_FUNCTIONS),
        help="Training stage(s) to run in order",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training stages and evaluate existing/copied checkpoints",
    )
    parser.add_argument(
        "--checkpoint-source",
        help="Directory containing checkpoints or a run directory with checkpoints/",
    )
    parser.add_argument(
        "--override", action="append", default=[], help="Config override like latent.dim=16"
    )
    parser.add_argument(
        "--eval-override",
        action="append",
        default=[],
        help="Eval-only override like data.split=val",
    )
    parser.add_argument(
        "--extra-eval-split",
        action="append",
        default=[],
        help="Additional data split to evaluate with the same trained checkpoints; can be repeated",
    )
    parser.add_argument(
        "--transfer-task",
        action="append",
        default=[],
        help="Optional transfer evaluation task; can be repeated",
    )
    parser.add_argument("--transfer-split", help="Optional split override for transfer evaluation")
    parser.add_argument(
        "--promotion-rule",
        action="append",
        default=[],
        help="Optional promotion rule like max:family_*_decoded_rollout_nrmse<=0.3",
    )
    parser.add_argument(
        "--decoded",
        action="store_true",
        help="Run decoded evaluation when encoder/decoder checkpoints are available",
    )
    parser.add_argument(
        "--decoded-rollout-steps",
        type=int,
        default=None,
        help="Optional rollout cap for decoded evaluation",
    )
    parser.add_argument("--device", default="cpu", help="Evaluation device")
    parser.add_argument(
        "--bootstrap-synthetic",
        action="store_true",
        help="Create tiny PDEBench-style HDF5 files if data is unavailable",
    )
    parser.add_argument(
        "--synthetic-root",
        help="Where to place synthetic PDEBench files; defaults to <run_dir>/synthetic_pdebench",
    )
    parser.add_argument(
        "--synthetic-samples",
        type=int,
        default=2,
        help="Number of synthetic trajectories per task/split",
    )
    parser.add_argument(
        "--synthetic-steps",
        type=int,
        default=4,
        help="Number of synthetic time steps per trajectory",
    )
    parser.add_argument(
        "--keep-existing-synthetic",
        action="store_true",
        help="Reuse an existing synthetic root without regenerating files",
    )
    parser.add_argument(
        "--allow-wandb",
        action="store_true",
        help="Keep W&B enabled instead of forcing it off for lightweight runs",
    )
    parser.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT", ""),
        help="W&B project override when --allow-wandb is set",
    )
    parser.add_argument(
        "--wandb-entity",
        default=os.environ.get("WANDB_ENTITY", ""),
        help="W&B entity override when --allow-wandb is set",
    )
    parser.add_argument(
        "--wandb-group",
        default=os.environ.get("WANDB_GROUP", ""),
        help="W&B group for this experiment batch",
    )
    parser.add_argument(
        "--wandb-tag", action="append", default=[], help="Extra W&B tag; can be repeated"
    )
    parser.add_argument(
        "--wandb-job-type",
        default=os.environ.get("WANDB_JOB_TYPE", "light-experiment"),
        help="W&B job type",
    )
    args = parser.parse_args()

    stages = [] if args.skip_training else (args.stage or ["operator"])
    output_root = Path(args.output_root)
    run_dir = output_root / args.name
    checkpoint_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = _apply_overrides(load_config_with_includes(args.config), args.override)
    eval_source = load_config_with_includes(args.eval_config) if args.eval_config else {}
    eval_cfg = (
        _apply_overrides(eval_source, args.eval_override)
        if args.eval_config or args.eval_override
        else None
    )
    train_cfg = _configure_wandb(
        train_cfg,
        enabled=args.allow_wandb,
        run_name=args.name,
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        tags=[*_split_env_list(os.environ.get("WANDB_TAGS", "")), *args.wandb_tag],
        job_type=args.wandb_job_type,
    )
    if eval_cfg is not None:
        eval_cfg = _configure_wandb(
            eval_cfg,
            enabled=args.allow_wandb,
            run_name=args.name,
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            tags=[*_split_env_list(os.environ.get("WANDB_TAGS", "")), *args.wandb_tag],
            job_type=args.wandb_job_type,
        )

    if args.bootstrap_synthetic:
        synthetic_root = (
            Path(args.synthetic_root) if args.synthetic_root else (run_dir / "synthetic_pdebench")
        )
        train_tasks = _task_names(train_cfg)
        eval_tasks = (
            _task_names(eval_cfg)
            if eval_cfg is not None and eval_cfg.get("data", {}).get("task") is not None
            else train_tasks
        )
        all_tasks = _all_tasks(train_tasks, eval_tasks, args.transfer_task)
        splits = {_split_name(train_cfg), _split_name(eval_cfg or train_cfg)}
        splits.update(str(split) for split in args.extra_eval_split)
        if args.transfer_task:
            splits.add(args.transfer_split or _split_name(eval_cfg or train_cfg))
        if not args.keep_existing_synthetic:
            bootstrap_synthetic_pdebench(
                synthetic_root,
                tasks=all_tasks,
                splits=sorted(splits),
                param_keys=tuple(train_cfg.get("data", {}).get("param_keys", ())),
                bc_keys=tuple(train_cfg.get("data", {}).get("bc_keys", ())),
                samples=args.synthetic_samples,
                steps=args.synthetic_steps,
            )
        train_cfg = _override_data_root(train_cfg, synthetic_root)
        if eval_cfg is not None:
            eval_cfg = _override_data_root(eval_cfg, synthetic_root)

    train_cfg = _prepare_runtime_cfg(
        train_cfg,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        disable_wandb=not args.allow_wandb,
    )
    eval_cfg = _prepare_eval_cfg(
        train_cfg,
        eval_cfg,
        log_dir=log_dir,
        disable_wandb=not args.allow_wandb,
    )

    train_cfg_path = run_dir / "resolved_train.yaml"
    eval_cfg_path = run_dir / "resolved_eval.yaml"
    train_cfg_path.write_text(yaml.safe_dump(train_cfg, sort_keys=False), encoding="utf-8")
    eval_cfg_path.write_text(yaml.safe_dump(eval_cfg, sort_keys=False), encoding="utf-8")

    copied_checkpoints: list[Path] = []
    if args.checkpoint_source:
        copied_checkpoints = _copy_checkpoint_source(Path(args.checkpoint_source), checkpoint_dir)

    train_script.set_seed(train_cfg)
    started = time.time()
    for stage in stages:
        STAGE_FUNCTIONS[stage](train_cfg)

    summary = _evaluate_once(
        eval_cfg,
        checkpoint_dir=checkpoint_dir,
        operator_checkpoint_names=_operator_checkpoint_names_for_stages(stages),
        decoded=args.decoded,
        device=args.device,
        decoded_rollout_steps=args.decoded_rollout_steps,
        transfer_tasks=args.transfer_task,
        transfer_split=args.transfer_split,
        cli_promotion_rules=args.promotion_rule,
    )
    summary["run_name"] = args.name
    summary["stages"] = stages
    summary["skip_training"] = bool(args.skip_training)
    if copied_checkpoints:
        summary["checkpoint_source"] = str(Path(args.checkpoint_source))
        summary["copied_checkpoints"] = [str(path) for path in copied_checkpoints]
    summary["config"] = str(train_cfg_path)
    summary["eval_config"] = str(eval_cfg_path)

    extra_evaluations: dict[str, Any] = {}
    for split in args.extra_eval_split:
        split_name = str(split)
        split_cfg = _clone_eval_cfg(eval_cfg, split=split_name)
        split_summary = _evaluate_once(
            split_cfg,
            checkpoint_dir=checkpoint_dir,
            operator_checkpoint_names=_operator_checkpoint_names_for_stages(stages),
            decoded=args.decoded,
            device=args.device,
            decoded_rollout_steps=args.decoded_rollout_steps,
            transfer_tasks=args.transfer_task,
            transfer_split=args.transfer_split,
            cli_promotion_rules=args.promotion_rule,
        )
        split_summary["run_name"] = args.name
        split_summary["split"] = split_name
        split_summary["stages"] = stages
        split_summary["config"] = str(train_cfg_path)
        split_summary["eval_config"] = str(eval_cfg_path)
        split_path = run_dir / f"summary_{_safe_artifact_name(split_name)}.json"
        split_path.write_text(json.dumps(split_summary, indent=2), encoding="utf-8")
        extra_evaluations[split_name] = {
            "summary": str(split_path),
            "metrics": split_summary["metrics"],
            "extra": split_summary["extra"],
        }

    finished = time.time()
    summary["duration_sec"] = finished - started
    if extra_evaluations:
        summary["extra_evaluations"] = extra_evaluations

    _log_summary_to_wandb(
        allow_wandb=args.allow_wandb,
        train_cfg=train_cfg,
        log_dir=log_dir,
        summary=summary,
    )
    summary["tracking"] = _tracking_payload(
        allow_wandb=args.allow_wandb, train_cfg=train_cfg, log_dir=log_dir
    )
    for split in args.extra_eval_split:
        split_path = run_dir / f"summary_{_safe_artifact_name(str(split))}.json"
        split_summary = json.loads(split_path.read_text(encoding="utf-8"))
        split_summary["tracking"] = summary["tracking"]
        split_path.write_text(json.dumps(split_summary, indent=2), encoding="utf-8")

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    main_metric_name, main_metric_value = _main_metric(summary["metrics"])
    results_row = {
        "run_name": args.name,
        "timestamp": int(finished),
        "stages": ",".join(stages),
        "decoded": bool(args.decoded),
        "train_split": _split_name(train_cfg),
        "eval_split": _split_name(eval_cfg),
        "transfer_tasks": ",".join(args.transfer_task),
        "promotion_passed": summary["extra"].get("promotion_passed"),
        "main_metric_name": main_metric_name,
        "main_metric_value": main_metric_value,
        "summary_json": str(summary_path),
        "wandb_run_ids": ",".join(
            str(run.get("id", "")) for run in summary["tracking"]["wandb"]["runs"] if run.get("id")
        ),
        "wandb_urls": ",".join(
            str(run.get("url", ""))
            for run in summary["tracking"]["wandb"]["runs"]
            if run.get("url")
        ),
    }
    _append_results_row(output_root / "results.tsv", results_row)

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "summary": str(summary_path),
                "main_metric": {main_metric_name: main_metric_value},
                "extra_evaluations": {
                    split: payload["summary"] for split, payload in extra_evaluations.items()
                },
                "promotion_passed": summary["extra"].get("promotion_passed"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
