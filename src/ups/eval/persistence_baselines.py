from __future__ import annotations

"""Physical-space persistence baselines for protocol scorecards."""

import math
import re
from collections.abc import Sequence
from typing import Any

import torch

from ups.data.latent_pairs import infer_grid_shape
from ups.data.pdebench import PDEBenchConfig, PDEBenchDataset, get_pdebench_spec
from ups.eval.pdebench_runner import _aggregate_chunk_metrics, _flatten_field_step
from ups.eval.regime_metrics import global_scale_regime_nrmse
from ups.eval.reports import MetricReport

_REGIME_KEYS = {
    "advection1d": "beta",
    "burgers1d": "nu",
    "darcy2d": "beta",
}
_CONTRACT_TASKS = ("advection1d", "burgers1d", "darcy2d")


def _task_names(cfg: dict[str, Any]) -> list[str]:
    task_cfg = cfg.get("data", {}).get("task")
    if isinstance(task_cfg, str):
        return [task_cfg]
    if isinstance(task_cfg, (list, tuple)) and task_cfg:
        return [str(task) for task in task_cfg]
    raise ValueError("Persistence baseline requires data.task to be a task name or non-empty list")


def _regime_value(sample: dict[str, Any], *, task: str, strict_contract: bool) -> float | str:
    key = _REGIME_KEYS.get(task)
    value = (sample.get("params") or {}).get(key) if key else None
    if value is None:
        if strict_contract:
            raise ValueError(f"{task} is missing required regime metadata {key!r}")
        return "unknown"
    tensor = torch.as_tensor(value).detach().cpu().reshape(-1)
    if tensor.numel() != 1 or not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{task} regime metadata {key!r} must be one finite scalar")
    return float(tensor.item())


def _regime_label(value: float | str) -> str:
    # Regime datasets are commonly stored as float32. Six significant digits
    # recovers their reviewed physical labels (0.1, 0.002, 100) without leaking
    # binary representation noise into stable metric keys.
    return value if isinstance(value, str) else format(value, ".6g")


def _regime_slug(label: str) -> str:
    slug = label.lower().replace("-", "neg").replace("+", "pos").replace(".", "p")
    slug = re.sub(r"[^a-z0-9]+", "_", slug).strip("_")
    if not slug:
        raise ValueError(f"Unable to form a metric-safe regime label from {label!r}")
    return slug


def _stats(pred: list[torch.Tensor], target: list[torch.Tensor]) -> dict[str, float]:
    return _aggregate_chunk_metrics(pred, target)


def evaluate_persistence_decoded(
    cfg: dict[str, Any],
    *,
    rollout_steps: int | None = None,
    tasks: Sequence[str] | None = None,
    strict_contract: bool = False,
) -> MetricReport:
    """Evaluate initial-state persistence and the steady identity operator.

    For temporal tasks, a persistence *rollout* holds the observed initial field
    fixed for every requested horizon. Darcy is not temporal: its minimal
    non-learned analogue is the identity operator from coefficient to solution.
    """

    data_cfg = cfg.get("data", {})
    task_names = list(tasks or _task_names(cfg))
    if len(task_names) != len(set(task_names)):
        raise ValueError("Persistence baseline task list contains duplicates")
    if strict_contract and tuple(task_names) != _CONTRACT_TASKS:
        raise ValueError(
            "strat-v1 persistence requires tasks in canonical order: " + ", ".join(_CONTRACT_TASKS)
        )
    steps_requested = 16 if rollout_steps is None and strict_contract else rollout_steps
    if strict_contract and steps_requested != 16:
        raise ValueError("strat-v1 persistence requires exactly 16 temporal rollout steps")

    task_pred: dict[str, list[torch.Tensor]] = {}
    task_target: dict[str, list[torch.Tensor]] = {}
    task_horizon_pred: dict[str, dict[int, list[torch.Tensor]]] = {}
    task_horizon_target: dict[str, dict[int, list[torch.Tensor]]] = {}
    task_regime_pred: dict[str, dict[str, list[torch.Tensor]]] = {}
    task_regime_target: dict[str, dict[str, list[torch.Tensor]]] = {}
    sample_counts: dict[str, int] = {}

    for task_name in task_names:
        spec = get_pdebench_spec(task_name)
        configured_params = tuple(str(key) for key in data_cfg.get("param_keys", ()))
        regime_key = _REGIME_KEYS.get(task_name)
        param_keys = tuple(
            dict.fromkeys((*configured_params, *((regime_key,) if regime_key else ())))
        )
        dataset = PDEBenchDataset(
            PDEBenchConfig(
                task=task_name,
                split=data_cfg.get("split", "train"),
                root=data_cfg.get("root"),
                data_lock_path=data_cfg.get("data_lock_path"),
                data_lock_sha256=data_cfg.get("data_lock_sha256"),
                selection_sha256=data_cfg.get("selection_sha256"),
                param_keys=param_keys,
                bc_keys=tuple(data_cfg.get("bc_keys", ())),
                max_samples=data_cfg.get("max_samples"),
            )
        )
        if len(dataset) == 0:
            raise RuntimeError(f"Persistence baseline received no {task_name} samples")

        sample_counts[task_name] = len(dataset)
        for idx in range(len(dataset)):
            sample = dataset[idx]
            fields = sample["fields"].float()
            grid_shape = infer_grid_shape(fields)
            regime = _regime_label(
                _regime_value(sample, task=task_name, strict_contract=strict_contract)
            )

            if spec.mapping_kind == "steady_operator":
                solution = sample["targets"].float()
                if fields.shape[0] != 1 or solution.shape[0] != 1:
                    raise ValueError(
                        "Steady operator baseline requires one coefficient and one solution"
                    )
                predictions = [_flatten_field_step(fields[0], grid_shape).cpu()]
                targets_for_sample = [_flatten_field_step(solution[0], grid_shape).cpu()]
            else:
                available_steps = int(fields.shape[0]) - 1
                if available_steps <= 0:
                    raise ValueError(f"{task_name} sample {idx} has no prediction horizon")
                steps = available_steps if steps_requested is None else int(steps_requested)
                if steps <= 0:
                    raise ValueError("rollout_steps must be positive")
                if available_steps < steps:
                    if strict_contract:
                        raise ValueError(
                            f"{task_name} sample {idx} has {available_steps} horizons; {steps} required"
                        )
                    steps = available_steps
                initial = _flatten_field_step(fields[0], grid_shape).cpu()
                predictions = [initial for _ in range(steps)]
                targets_for_sample = [
                    _flatten_field_step(fields[horizon], grid_shape).cpu()
                    for horizon in range(1, steps + 1)
                ]
                for horizon, (prediction, target) in enumerate(
                    zip(predictions, targets_for_sample, strict=True), start=1
                ):
                    task_horizon_pred.setdefault(task_name, {}).setdefault(horizon, []).append(
                        prediction
                    )
                    task_horizon_target.setdefault(task_name, {}).setdefault(horizon, []).append(
                        target
                    )

            task_pred.setdefault(task_name, []).extend(predictions)
            task_target.setdefault(task_name, []).extend(targets_for_sample)
            task_regime_pred.setdefault(task_name, {}).setdefault(regime, []).extend(predictions)
            task_regime_target.setdefault(task_name, {}).setdefault(regime, []).extend(
                targets_for_sample
            )

    missing_tasks = [task for task in task_names if not task_pred.get(task)]
    if missing_tasks:
        raise RuntimeError(
            "Persistence baseline produced no predictions for: " + ", ".join(missing_tasks)
        )

    metrics: dict[str, float] = {}
    details: dict[str, Any] = {"tasks": {}}
    primary_values: list[float] = []
    temporal_horizon_values: dict[int, list[float]] = {}
    for task_name in task_names:
        spec = get_pdebench_spec(task_name)
        task_stats = _stats(task_pred[task_name], task_target[task_name])
        if spec.mapping_kind == "steady_operator":
            primary_name = "decoded_solution_nrmse"
        else:
            primary_name = "decoded_rollout_nrmse"
        metrics[f"task_{task_name}_{primary_name}"] = task_stats["nrmse"]
        primary_values.append(task_stats["nrmse"])

        horizon_details: dict[str, float] = {}
        if spec.mapping_kind != "steady_operator":
            for horizon in sorted(task_horizon_pred[task_name]):
                horizon_stats = _stats(
                    task_horizon_pred[task_name][horizon],
                    task_horizon_target[task_name][horizon],
                )
                metrics[f"task_{task_name}_decoded_h{horizon}_nrmse"] = horizon_stats["nrmse"]
                horizon_details[str(horizon)] = horizon_stats["nrmse"]
                temporal_horizon_values.setdefault(horizon, []).append(horizon_stats["nrmse"])

        regime_details: dict[str, float] = {}
        suffix = (
            "decoded_solution_nrmse"
            if spec.mapping_kind == "steady_operator"
            else "decoded_rollout_nrmse"
        )
        seen_slugs: set[str] = set()
        for regime in sorted(
            task_regime_pred[task_name], key=lambda value: (value == "unknown", value)
        ):
            slug = _regime_slug(regime)
            if slug in seen_slugs:
                raise ValueError(f"{task_name} regime labels collide after metric slugging")
            seen_slugs.add(slug)
            regime_stats = _stats(
                task_regime_pred[task_name][regime],
                task_regime_target[task_name][regime],
            )
            metrics[f"task_{task_name}_regime_{slug}_{suffix}"] = regime_stats["nrmse"]
            global_scale_key = suffix.replace("_nrmse", "_global_scale_nrmse")
            metrics[f"task_{task_name}_regime_{slug}_{global_scale_key}"] = (
                global_scale_regime_nrmse(
                    task_regime_pred[task_name][regime],
                    task_regime_target[task_name][regime],
                    task_target[task_name],
                )
            )
            regime_details[regime] = regime_stats["nrmse"]

        details["tasks"][task_name] = {
            "mapping_kind": spec.mapping_kind,
            "primary_metric": primary_name,
            "primary_nrmse": task_stats["nrmse"],
            "sample_count": sample_counts[task_name],
            "per_horizon_nrmse": horizon_details,
            "per_regime_nrmse": regime_details,
            "regime_normalization": {
                "slice_normalized": "regime target RMS",
                "global_scale": "complete task target RMS",
            },
        }

    metrics["macro_primary_nrmse"] = math.fsum(primary_values) / len(primary_values)
    for horizon, values in sorted(temporal_horizon_values.items()):
        metrics[f"temporal_macro_decoded_h{horizon}_nrmse"] = math.fsum(values) / len(values)

    # Preserve the legacy generic evaluator surface for diagnostic callers. The
    # strict strat-v1 path intentionally uses the semantically explicit macro and
    # task keys above, so steady Darcy is never mislabeled as a rollout there.
    if not strict_contract:
        pooled_pred = [chunk for task in task_names for chunk in task_pred[task]]
        pooled_target = [chunk for task in task_names for chunk in task_target[task]]
        pooled_stats = _stats(pooled_pred, pooled_target)
        metrics["decoded_rollout_nrmse"] = pooled_stats["nrmse"]
        first_horizon_pred = [
            chunk for task in task_names for chunk in task_horizon_pred.get(task, {}).get(1, [])
        ]
        first_horizon_target = [
            chunk for task in task_names for chunk in task_horizon_target.get(task, {}).get(1, [])
        ]
        if first_horizon_pred:
            metrics["decoded_step1_nrmse"] = _stats(first_horizon_pred, first_horizon_target)[
                "nrmse"
            ]

    task_extra: str | list[str] = task_names[0] if len(task_names) == 1 else task_names
    return MetricReport(
        metrics=metrics,
        extra={
            "baseline": "persistence",
            "task": task_extra,
            "split": data_cfg.get("split", "train"),
            "samples_by_task": sample_counts,
            "temporal_rollout_semantics": "initial_state_held_constant",
            "steady_operator_semantics": "coefficient_identity_to_solution",
            "physical_parameter_conditioning": False,
            "inferred_parameter_context": False,
            "regime_metadata_reporting_only": True,
            "details": details,
        },
    )
