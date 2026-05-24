from __future__ import annotations

"""Promotion-rule utilities for gating benchmark candidates."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any


@dataclass(frozen=True)
class PromotionRule:
    metric: str
    operator: str
    threshold: float
    label: str | None = None
    reducer: str = "value"

    def describe(self) -> str:
        prefix = "" if self.reducer == "value" else f"{self.reducer}:"
        return self.label or f"{prefix}{self.metric}{self.operator}{self.threshold:g}"


@dataclass(frozen=True)
class PromotionResult:
    passed: bool
    failed_rules: list[str]
    missing_metrics: list[str]


def parse_promotion_rule(text: str) -> PromotionRule:
    for operator in ("<=", ">=", "<", ">"):
        if operator in text:
            metric, threshold = text.split(operator, 1)
            metric = metric.strip()
            threshold = threshold.strip()
            if metric and threshold:
                reducer = "value"
                for candidate in ("max", "min", "mean"):
                    prefix = f"{candidate}:"
                    if metric.startswith(prefix):
                        reducer = candidate
                        metric = metric[len(prefix) :].strip()
                        break
                return PromotionRule(
                    metric=metric, operator=operator, threshold=float(threshold), reducer=reducer
                )
            break
    raise ValueError(
        f"Invalid promotion rule '{text}'. Expected forms like decoded_rollout_nrmse<=0.2"
    )


def promotion_rules_from_config(cfg: Mapping[str, Any]) -> list[PromotionRule]:
    rules_cfg = cfg.get("evaluation", {}).get("promotion", {}).get("rules", [])
    rules: list[PromotionRule] = []
    for entry in rules_cfg:
        if isinstance(entry, str):
            rules.append(parse_promotion_rule(entry))
        elif isinstance(entry, Mapping):
            rules.append(
                PromotionRule(
                    metric=str(entry["metric"]),
                    operator=str(entry.get("operator", entry.get("op", "<="))),
                    threshold=float(entry["threshold"]),
                    label=str(entry["label"]) if "label" in entry else None,
                    reducer=str(entry.get("reducer", "value")),
                )
            )
        else:
            raise TypeError(f"Unsupported promotion rule config entry: {entry!r}")
    return rules


def _reduce_metric_values(values: Sequence[float], reducer: str) -> float:
    if reducer == "value":
        return float(values[0])
    if reducer == "max":
        return max(values)
    if reducer == "min":
        return min(values)
    if reducer == "mean":
        return sum(values) / len(values)
    raise ValueError(f"Unsupported promotion reducer '{reducer}'")


def _resolve_rule_values(
    metrics: Mapping[str, float], rule: PromotionRule
) -> tuple[list[str], list[float]]:
    if any(ch in rule.metric for ch in "*?[]"):
        keys = sorted(key for key in metrics if fnmatch(key, rule.metric))
    else:
        keys = [rule.metric] if rule.metric in metrics else []
    values = [float(metrics[key]) for key in keys]
    return keys, values


def evaluate_promotion_rules(
    metrics: Mapping[str, float], rules: Sequence[PromotionRule]
) -> PromotionResult:
    failed_rules: list[str] = []
    missing_metrics: list[str] = []

    for rule in rules:
        matched_keys, values = _resolve_rule_values(metrics, rule)
        if not values:
            missing_metrics.append(
                rule.metric if rule.reducer == "value" else f"{rule.reducer}:{rule.metric}"
            )
            failed_rules.append(rule.describe())
            continue
        value = _reduce_metric_values(values, rule.reducer)
        if rule.operator == "<=":
            passed = value <= rule.threshold
        elif rule.operator == ">=":
            passed = value >= rule.threshold
        elif rule.operator == "<":
            passed = value < rule.threshold
        elif rule.operator == ">":
            passed = value > rule.threshold
        else:
            raise ValueError(f"Unsupported promotion operator '{rule.operator}'")

        if not passed:
            scope = ""
            if len(matched_keys) > 1:
                scope = f", matched={matched_keys}"
            failed_rules.append(f"{rule.describe()} (actual={value:.6g}{scope})")

    return PromotionResult(
        passed=not failed_rules and not missing_metrics,
        failed_rules=failed_rules,
        missing_metrics=missing_metrics,
    )
