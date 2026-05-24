from __future__ import annotations

"""Physical-space persistence baselines for demo scorecards."""

from collections.abc import Sequence
from typing import Any

from ups.data.latent_pairs import infer_grid_shape
from ups.data.pdebench import PDEBenchConfig, PDEBenchDataset, get_pdebench_spec
from ups.eval.pdebench_runner import _aggregate_chunk_metrics, _flatten_field_step
from ups.eval.reports import MetricReport


def _task_names(cfg: dict[str, Any]) -> list[str]:
    task_cfg = cfg.get("data", {}).get("task")
    if isinstance(task_cfg, str):
        return [task_cfg]
    if isinstance(task_cfg, (list, tuple)) and task_cfg:
        return [str(task) for task in task_cfg]
    raise ValueError("Persistence baseline requires data.task to be a task name or non-empty list")


def _add_rollout_metrics(
    metrics: dict[str, float],
    *,
    prefix: str,
    pred_chunks: list,
    target_chunks: list,
) -> None:
    if not pred_chunks:
        return
    stats = _aggregate_chunk_metrics(pred_chunks, target_chunks)
    metrics[f"{prefix}decoded_rollout_nrmse"] = stats["nrmse"]
    metrics[f"{prefix}decoded_rollout_rrmse"] = stats["rrmse"]
    metrics[f"{prefix}decoded_rollout_mse"] = stats["mse"]
    metrics[f"{prefix}decoded_rollout_mae"] = stats["mae"]


def evaluate_persistence_decoded(
    cfg: dict[str, Any],
    *,
    rollout_steps: int | None = None,
    tasks: Sequence[str] | None = None,
) -> MetricReport:
    """Evaluate a physical-space persistence baseline on PDEBench-like HDF5 data.

    The baseline predicts the previous physical field as the next field. It is
    intentionally simple, deterministic, and cheap; it provides a minimum
    non-learned baseline for the demo scorecard.
    """

    data_cfg = cfg.get("data", {})
    task_names = list(tasks or _task_names(cfg))
    total_pred = []
    total_target = []
    horizon_pred: dict[int, list] = {1: [], 4: [], 16: []}
    horizon_target: dict[int, list] = {1: [], 4: [], 16: []}
    per_task_pred: dict[str, list] = {}
    per_task_target: dict[str, list] = {}
    per_task_step1_pred: dict[str, list] = {}
    per_task_step1_target: dict[str, list] = {}
    per_family_pred: dict[str, list] = {}
    per_family_target: dict[str, list] = {}
    per_family_step1_pred: dict[str, list] = {}
    per_family_step1_target: dict[str, list] = {}

    for task_name in task_names:
        task_family = get_pdebench_spec(task_name).family
        dataset = PDEBenchDataset(
            PDEBenchConfig(
                task=task_name,
                split=data_cfg.get("split", "train"),
                root=data_cfg.get("root"),
                param_keys=tuple(data_cfg.get("param_keys", ())),
                bc_keys=tuple(data_cfg.get("bc_keys", ())),
                max_samples=data_cfg.get("max_samples"),
            )
        )
        if len(dataset) == 0:
            continue
        grid_shape = infer_grid_shape(dataset.fields[0])
        for idx in range(len(dataset)):
            fields = dataset[idx]["fields"].float()
            max_steps = int(fields.shape[0]) - 1
            if max_steps <= 0:
                continue
            steps = max_steps if rollout_steps is None else min(max_steps, int(rollout_steps))
            for step in range(steps):
                pred_field = _flatten_field_step(fields[step], grid_shape).cpu()
                target_field = _flatten_field_step(fields[step + 1], grid_shape).cpu()
                total_pred.append(pred_field)
                total_target.append(target_field)
                per_task_pred.setdefault(task_name, []).append(pred_field)
                per_task_target.setdefault(task_name, []).append(target_field)
                per_family_pred.setdefault(task_family, []).append(pred_field)
                per_family_target.setdefault(task_family, []).append(target_field)
                horizon = step + 1
                if horizon in horizon_pred:
                    horizon_pred[horizon].append(pred_field)
                    horizon_target[horizon].append(target_field)
                if horizon == 1:
                    per_task_step1_pred.setdefault(task_name, []).append(pred_field)
                    per_task_step1_target.setdefault(task_name, []).append(target_field)
                    per_family_step1_pred.setdefault(task_family, []).append(pred_field)
                    per_family_step1_target.setdefault(task_family, []).append(target_field)

    if not total_pred:
        raise RuntimeError("Persistence baseline received no valid rollout steps")

    rollout_stats = _aggregate_chunk_metrics(total_pred, total_target)
    metrics = {
        "decoded_mse": rollout_stats["mse"],
        "decoded_mae": rollout_stats["mae"],
        "decoded_nrmse": rollout_stats["nrmse"],
        "decoded_rrmse": rollout_stats["rrmse"],
        "decoded_spectral_energy_error": rollout_stats["spectral_energy_error"],
        "decoded_rollout_mse": rollout_stats["mse"],
        "decoded_rollout_mae": rollout_stats["mae"],
        "decoded_rollout_nrmse": rollout_stats["nrmse"],
        "decoded_rollout_rrmse": rollout_stats["rrmse"],
        "decoded_rollout_spectral_energy_error": rollout_stats["spectral_energy_error"],
        "mse": rollout_stats["mse"],
        "mae": rollout_stats["mae"],
        "rmse": rollout_stats["mse"] ** 0.5,
    }
    if horizon_pred[1]:
        step1_stats = _aggregate_chunk_metrics(horizon_pred[1], horizon_target[1])
        metrics["decoded_step1_nrmse"] = step1_stats["nrmse"]
        metrics["decoded_step1_rrmse"] = step1_stats["rrmse"]
    for horizon in (4, 16):
        if horizon_pred[horizon]:
            horizon_stats = _aggregate_chunk_metrics(horizon_pred[horizon], horizon_target[horizon])
            metrics[f"decoded_h{horizon}_nrmse"] = horizon_stats["nrmse"]
            metrics[f"decoded_h{horizon}_rrmse"] = horizon_stats["rrmse"]
    for task_name, pred_chunks in per_task_pred.items():
        _add_rollout_metrics(
            metrics,
            prefix=f"task_{task_name}_",
            pred_chunks=pred_chunks,
            target_chunks=per_task_target[task_name],
        )
        if task_name in per_task_step1_pred:
            step1_stats = _aggregate_chunk_metrics(
                per_task_step1_pred[task_name],
                per_task_step1_target[task_name],
            )
            metrics[f"task_{task_name}_decoded_step1_nrmse"] = step1_stats["nrmse"]
    for family_name, pred_chunks in per_family_pred.items():
        _add_rollout_metrics(
            metrics,
            prefix=f"family_{family_name}_",
            pred_chunks=pred_chunks,
            target_chunks=per_family_target[family_name],
        )
        if family_name in per_family_step1_pred:
            step1_stats = _aggregate_chunk_metrics(
                per_family_step1_pred[family_name],
                per_family_step1_target[family_name],
            )
            metrics[f"family_{family_name}_decoded_step1_nrmse"] = step1_stats["nrmse"]

    task_extra: str | list[str] = task_names[0] if len(task_names) == 1 else task_names
    return MetricReport(
        metrics=metrics,
        extra={
            "baseline": "persistence",
            "task": task_extra,
            "split": data_cfg.get("split", "train"),
            "samples": len(total_pred),
        },
    )
