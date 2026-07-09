#!/usr/bin/env python
from __future__ import annotations

"""Validate model-side beta transport-head summary boundaries."""

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

AGGREGATE_GATE = 0.35078329353213156
ADVECTION_GATE = 0.4866576789288726
ADVECTION_H16_GATE = 0.44444171136384397
BURGERS_GATE = 0.15674926288225416
DARCY_GATE = 0.2071060212271272


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _as_mapping(value: Any, name: str, errors: list[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(f"{name} must be a mapping")
        return {}
    return value


def _contains_test_split(command: Any) -> bool:
    if isinstance(command, str):
        text = command
    elif isinstance(command, Sequence):
        text = " ".join(str(part) for part in command)
    else:
        return False
    return "split=test" in text or "--split test" in text or "data.split=test" in text


def _metric_gate(
    metrics: Mapping[str, Any],
    key: str,
    gate: float,
    errors: list[str],
    *,
    required: bool = True,
) -> None:
    if key not in metrics:
        if required:
            errors.append(f"metrics.{key} is required")
        return
    value = float(metrics[key])
    if value > gate:
        errors.append(f"metrics.{key}={value} must be <= {gate}")


def validate_summary(summary: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    extra = _as_mapping(summary.get("extra"), "summary.extra", errors)
    metrics = _as_mapping(summary.get("metrics"), "summary.metrics", errors)

    if bool(summary.get("held_out_test_used", extra.get("held_out_test_used", False))):
        errors.append("held_out_test_used must be false")
    if bool(summary.get("held_out_test_data_read", extra.get("held_out_test_data_read", False))):
        errors.append("held_out_test_data_read must be false")
    if _contains_test_split(summary.get("command") or summary.get("command_args")):
        errors.append("command must not use split=test")

    head = _as_mapping(
        extra.get("model_side_transport_head"), "extra.model_side_transport_head", errors
    )
    if not head:
        errors.append("extra.model_side_transport_head is required")
    elif not bool(head.get("enabled", False)):
        errors.append("extra.model_side_transport_head.enabled must be true")
    tasks = (
        [str(task) for task in head.get("tasks", [])]
        if isinstance(head.get("tasks", []), Sequence)
        else []
    )
    if tasks != ["advection1d"]:
        errors.append("extra.model_side_transport_head.tasks must be exactly ['advection1d']")
    required_params = set(str(name) for name in head.get("required_params", []))
    if "beta" not in required_params:
        errors.append("extra.model_side_transport_head.required_params must include beta")
    if head.get("mode") != "periodic_roll":
        errors.append("extra.model_side_transport_head.mode must be periodic_roll")
    if head.get("apply_at") != "decoded_rollout":
        errors.append("extra.model_side_transport_head.apply_at must be decoded_rollout")

    head_metrics = _as_mapping(
        extra.get("model_side_transport_head_metrics"),
        "extra.model_side_transport_head_metrics",
        errors,
    )
    if int(head_metrics.get("beta_missing_count", 0) or 0) > 0:
        errors.append("extra.model_side_transport_head_metrics.beta_missing_count must be 0")

    for key in (
        "decoded_context_roll_shift_estimator",
        "decoded_observed_roll_shift_estimator",
        "decoded_prediction_roll_shift_estimator",
        "decoded_data_conditioned_roll_shift_estimator",
    ):
        if extra.get(key):
            errors.append(f"extra.{key} must be empty for model-side transport-head evidence")

    _metric_gate(metrics, "decoded_rollout_nrmse", AGGREGATE_GATE, errors)
    _metric_gate(metrics, "task_advection1d_decoded_rollout_nrmse", ADVECTION_GATE, errors)
    _metric_gate(metrics, "task_advection1d_decoded_h16_nrmse", ADVECTION_H16_GATE, errors)
    _metric_gate(metrics, "task_burgers1d_decoded_rollout_nrmse", BURGERS_GATE, errors)
    _metric_gate(metrics, "task_darcy2d_decoded_rollout_nrmse", DARCY_GATE, errors)
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_json", help="Path to summary.json")
    args = parser.parse_args()
    summary = _load_json(args.summary_json)
    errors = validate_summary(summary)
    payload = {"passed": not errors, "errors": errors, "summary_json": args.summary_json}
    print(json.dumps(payload, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
