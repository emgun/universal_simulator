#!/usr/bin/env python
from __future__ import annotations

"""Audit whether current UPS evidence is ready for a universal SOTA claim."""

import argparse
import glob
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

DEFAULT_DIAGNOSTIC_RUN_FRAGMENTS = (
    "gate_hook",
    "residual_alpha",
    "roll_shift",
    "observed_shift",
    "transport_gate",
    "transport_horizon_gate",
    "transport_residual_gate",
)


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON in {path}")
    return payload


def _maybe_load_json(path: str | Path) -> dict[str, Any]:
    if not path:
        return {}
    json_path = Path(path)
    if not json_path.exists():
        return {}
    return _load_json(json_path)


def _non_empty(value: Any) -> bool:
    if value is None or value == "":
        return False
    if isinstance(value, (list, tuple)):
        return any(_non_empty(item) for item in value)
    if isinstance(value, Mapping):
        return any(_non_empty(item) for item in value.values())
    return bool(str(value).strip())


def _join_values(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value if _non_empty(item))
    return value


def _metric_value(row: Mapping[str, Any], metric_name: str) -> float | None:
    value = row.get(f"metric:{metric_name}", row.get(metric_name))
    if value is None or value == "":
        return None
    return float(value)


def _rows(scorecard: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = scorecard.get("rows", [])
    if not isinstance(rows, list):
        return []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _wandb_urls(summary: Mapping[str, Any]) -> str:
    tracking = summary.get("tracking", {})
    wandb = tracking.get("wandb", {}) if isinstance(tracking, Mapping) else {}
    runs = wandb.get("runs", []) if isinstance(wandb, Mapping) else []
    if not isinstance(runs, list):
        return ""
    urls = [str(run.get("url", "")) for run in runs if isinstance(run, Mapping) and run.get("url")]
    return ",".join(urls)


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _summary_cost_fields(summary: Mapping[str, Any]) -> dict[str, Any]:
    extra = summary.get("extra", {})
    cost = summary.get("cost")
    if not isinstance(cost, Mapping) and isinstance(extra, Mapping):
        cost = extra.get("cost")
    if not isinstance(cost, Mapping):
        return {}

    hourly_usd = _as_float(cost.get("hourly_usd", cost.get("cost_per_hour_usd")))
    wall_clock_hours = _as_float(
        cost.get("wall_clock_hours", cost.get("duration_hours", cost.get("runtime_hours")))
    )
    gpu_count = _as_float(cost.get("gpu_count", cost.get("gpus")))
    gpu_hours = _as_float(cost.get("gpu_hours"))
    estimated_usd = _as_float(
        cost.get("estimated_usd", cost.get("estimated_cost_usd", cost.get("cost_usd")))
    )
    if gpu_hours is None and wall_clock_hours is not None and gpu_count is not None:
        gpu_hours = wall_clock_hours * gpu_count
    if estimated_usd is None and wall_clock_hours is not None and hourly_usd is not None:
        estimated_usd = wall_clock_hours * hourly_usd

    return {
        "cost_provider": cost.get("provider", ""),
        "cost_instance_type": cost.get("instance_type", cost.get("machine_type", "")),
        "cost_gpu_type": cost.get("gpu_type", cost.get("gpu_model", "")),
        "cost_gpu_count": gpu_count,
        "cost_wall_clock_hours": wall_clock_hours,
        "cost_gpu_hours": gpu_hours,
        "cost_estimated_usd": estimated_usd,
    }


def _row_from_summary(summary: Mapping[str, Any], summary_path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run_name": summary.get("run_name", summary_path.parent.name),
        "split": summary.get("split", summary.get("eval_split", "")),
        "summary_json": str(summary_path),
        "duration_sec": summary.get("duration_sec"),
        "tracking_wandb_urls": _wandb_urls(summary),
        "artifact_urls": summary.get("artifact_urls", ""),
        "artifact_handles": summary.get("artifact_handles", ""),
        **_summary_cost_fields(summary),
    }
    metrics = summary.get("metrics", {})
    if isinstance(metrics, Mapping):
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                row[f"metric:{key}"] = float(value)
    return row


def _summary_rows(patterns: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = []
    for pattern in patterns:
        for item in sorted(glob.glob(pattern)):
            path = Path(item)
            if not path.exists():
                continue
            rows.append(_row_from_summary(_load_json(path), path))
    return rows


def _claim_evidence_rows(claim_evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = claim_evidence.get("candidate_evidence", [])
    if not isinstance(records, list):
        return []
    rows: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        row = {
            str(key): _join_values(value)
            for key, value in record.items()
            if key not in {"cost", "metrics"} and not isinstance(value, Mapping)
        }
        row.update(_summary_cost_fields(record))
        metrics = record.get("metrics", {})
        if isinstance(metrics, Mapping):
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    row[f"metric:{key}"] = float(value)
        if row.get("run_name"):
            rows.append(row)
    return rows


def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name: dict[str, dict[str, Any]] = {}
    for row in rows:
        run_name = str(row.get("run_name", ""))
        if not run_name:
            continue
        if run_name not in by_name:
            by_name[run_name] = row
            continue
        merged = dict(by_name[run_name])
        for key, value in row.items():
            if value not in (None, ""):
                merged[key] = value
            elif key not in merged:
                merged[key] = value
        by_name[run_name] = merged
    return list(by_name.values())


def _best_row(rows: list[dict[str, Any]], metric_name: str) -> dict[str, Any] | None:
    metric_rows = [(row, _metric_value(row, metric_name)) for row in rows]
    candidates = [(row, value) for row, value in metric_rows if value is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[1])[0]


def _claim_eligible_rows(
    rows: list[dict[str, Any]],
    diagnostic_fragments: tuple[str, ...],
    *,
    claim_split: str,
) -> list[dict[str, Any]]:
    eligible = []
    for row in rows:
        run_name = str(row.get("run_name", ""))
        if any(fragment and fragment in run_name for fragment in diagnostic_fragments):
            continue
        if str(row.get("split", "")) != claim_split:
            continue
        eligible.append(row)
    return eligible


def _baseline_row(rows: list[dict[str, Any]], baseline_run_name: str) -> dict[str, Any] | None:
    for row in rows:
        if row.get("run_name") == baseline_run_name:
            return row
    return None


def _has_any_metric(row: Mapping[str, Any] | None, names: tuple[str, ...]) -> bool:
    if row is None:
        return False
    return any(_metric_value(row, name) is not None for name in names)


def _has_task_metrics(row: Mapping[str, Any] | None) -> bool:
    return all(
        _has_any_metric(row, (f"task_{task}_decoded_rollout_nrmse",))
        for task in ("advection1d", "burgers1d", "darcy2d")
    )


def _has_spectral_metrics(row: Mapping[str, Any] | None) -> bool:
    return _has_any_metric(
        row,
        (
            "decoded_rollout_spectral_energy_error",
            "decoded_spectral_energy_error",
            "spectral_error",
        ),
    )


def _has_cost_or_throughput(row: Mapping[str, Any] | None) -> bool:
    if row is None:
        return False
    return any(
        row.get(key) not in (None, "")
        for key in (
            "duration_sec",
            "cost_estimated_usd",
            "cost_wall_clock_hours",
            "cost_gpu_hours",
        )
    )


def _has_wandb_or_artifact_handles(row: Mapping[str, Any] | None) -> bool:
    if row is None:
        return False
    if str(row.get("tracking_wandb_urls", "")).strip():
        return True
    if str(row.get("artifact_urls", "")).strip():
        return True
    if str(row.get("artifact_handles", "")).strip():
        return True
    return False


def _documentation_status(
    claim_evidence: Mapping[str, Any],
    best: Mapping[str, Any] | None,
    *,
    metric_name: str,
    claim_split: str,
) -> dict[str, Any]:
    doc = claim_evidence.get("claim_documentation", {})
    if not isinstance(doc, Mapping):
        return {"present": False, "validated": False, "run_name": "", "reason": "missing"}
    if best is None:
        return {
            "present": True,
            "validated": False,
            "run_name": str(doc.get("run_name", "")),
            "reason": "no claim-eligible best row",
        }

    checkpoints = doc.get("checkpoints", {})
    required = {
        "status": str(doc.get("status", "")) == "complete",
        "run_name": str(doc.get("run_name", "")) == str(best.get("run_name", "")),
        "split": str(doc.get("split", "")) == claim_split,
        "summary_json": _non_empty(doc.get("summary_json")),
        "commit": _non_empty(doc.get("commit")),
        "command": _non_empty(doc.get("command")) or _non_empty(doc.get("commands")),
        "checkpoints": isinstance(checkpoints, Mapping)
        and all(_non_empty(checkpoints.get(name)) for name in ("operator", "encoder", "decoder")),
        "artifact_handles": _non_empty(doc.get("artifact_handles"))
        or _non_empty(doc.get("artifact_urls")),
    }
    doc_metric_name = str(doc.get("metric_name", metric_name))
    best_metric = _metric_value(best, metric_name)
    doc_metric = _as_float(doc.get("metric_value"))
    required["metric_name"] = doc_metric_name == metric_name
    required["metric_value"] = (
        best_metric is not None
        and doc_metric is not None
        and abs(float(best_metric) - float(doc_metric)) <= 1.0e-12
    )

    if _non_empty(doc.get("summary_json")) and _non_empty(best.get("summary_json")):
        required["summary_json"] = str(doc.get("summary_json")) == str(best.get("summary_json"))

    failed = [key for key, passed in required.items() if not passed]
    return {
        "present": True,
        "validated": not failed,
        "run_name": str(doc.get("run_name", "")),
        "reason": "complete" if not failed else f"failed_fields={failed}",
    }


def _strong_baseline_status(
    claim_evidence: Mapping[str, Any],
    best: Mapping[str, Any] | None,
    *,
    metric_name: str,
    claim_split: str,
) -> dict[str, Any]:
    comparison = claim_evidence.get("strong_baseline_comparison", {})
    if not isinstance(comparison, Mapping):
        return {"present": False, "validated": False, "reason": "missing"}
    if best is None:
        return {"present": True, "validated": False, "reason": "no claim-eligible best row"}
    comparison_status = str(comparison.get("status", ""))
    if comparison_status not in {"complete", "compared"}:
        return {
            "present": True,
            "validated": False,
            "reason": str(comparison.get("reason", f"status={comparison_status or 'missing'}")),
        }
    required = {
        "claim_run_name": str(comparison.get("claim_run_name", ""))
        == str(best.get("run_name", "")),
        "split": str(comparison.get("split", "")) == claim_split,
        "metric_name": str(comparison.get("metric_name", metric_name)) == metric_name,
        "baseline_run_name": _non_empty(comparison.get("baseline_run_name")),
        "baseline_metric_value": _as_float(comparison.get("baseline_metric_value")) is not None,
        "candidate_metric_value": _as_float(comparison.get("candidate_metric_value")) is not None,
        "artifact_handles": _non_empty(comparison.get("artifact_handles"))
        or _non_empty(comparison.get("artifact_urls")),
    }
    failed = [key for key, passed in required.items() if not passed]
    return {
        "present": True,
        "validated": not failed,
        "reason": "complete" if not failed else f"failed_fields={failed}",
    }


def _check(key: str, passed: bool, evidence: str) -> dict[str, Any]:
    return {"key": key, "passed": bool(passed), "evidence": evidence}


def _status_from_checks(checks: list[Mapping[str, Any]]) -> str:
    return "sota_ready" if all(bool(check.get("passed")) for check in checks) else "not_sota_ready"


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    light_scorecard = _maybe_load_json(args.light_scorecard_json)
    transport_status = _maybe_load_json(args.transport_status_json)
    transfer_scorecard = _maybe_load_json(args.transfer_scorecard_json)
    claim_evidence_json = getattr(
        args,
        "claim_evidence_json",
        "",
    )
    claim_evidence = _maybe_load_json(claim_evidence_json)

    candidate_summary_globs = tuple(getattr(args, "candidate_summary_glob", None) or ())
    rows = _dedupe_rows(
        [
            *_rows(light_scorecard),
            *_summary_rows(candidate_summary_globs),
            *_claim_evidence_rows(claim_evidence),
        ]
    )
    baseline = _baseline_row(rows, args.baseline_run_name)
    diagnostic_fragments = tuple(
        getattr(args, "exclude_diagnostic_fragment", None) or DEFAULT_DIAGNOSTIC_RUN_FRAGMENTS
    )
    claim_split = str(getattr(args, "claim_split", "test"))
    eligible_rows = _claim_eligible_rows(
        rows,
        diagnostic_fragments,
        claim_split=claim_split,
    )
    eligible_rows = [row for row in eligible_rows if row.get("run_name") != args.baseline_run_name]
    best_overall = _best_row(rows, args.metric_name)
    best = _best_row(eligible_rows, args.metric_name)

    baseline_value = _metric_value(baseline, args.metric_name) if baseline else None
    best_overall_value = _metric_value(best_overall, args.metric_name) if best_overall else None
    best_value = _metric_value(best, args.metric_name) if best else None
    improvement_fraction = None
    if baseline_value is not None and best_value is not None and baseline_value != 0.0:
        improvement_fraction = (baseline_value - best_value) / abs(baseline_value)

    transport_literal_achieved = transport_status.get("status") == "literal_achieved"
    passes_min_improvement = improvement_fraction is not None and improvement_fraction >= float(
        args.min_improvement
    )
    transfer_validated = (
        transfer_scorecard.get("status")
        in {
            "transfer_validated",
            "partial_transfer_validated",
        }
        and int(transfer_scorecard.get("evaluated_task_count", 0) or 0) >= 2
    )
    scorecard_complete = (
        _has_task_metrics(best) and _has_spectral_metrics(best) and _has_cost_or_throughput(best)
    )
    handles_confirmed = bool(args.artifact_handles_confirmed) or _has_wandb_or_artifact_handles(
        best
    )
    documentation_status = _documentation_status(
        claim_evidence,
        best,
        metric_name=args.metric_name,
        claim_split=claim_split,
    )
    strong_baseline_status = _strong_baseline_status(
        claim_evidence,
        best,
        metric_name=args.metric_name,
        claim_split=claim_split,
    )
    documentation_confirmed = bool(args.documentation_confirmed) or bool(
        documentation_status["validated"]
    )
    strong_baseline_compared = bool(args.strong_baseline_compared) or bool(
        strong_baseline_status["validated"]
    )

    readiness_checks = [
        _check(
            "official_transport_objective_achieved",
            transport_literal_achieved,
            f"transport status={transport_status.get('status', 'missing')}",
        ),
        _check(
            "light_v1_min_improvement",
            passes_min_improvement,
            (
                f"best={best_value}, baseline={baseline_value}, "
                f"improvement_fraction={improvement_fraction}, required={args.min_improvement}"
            ),
        ),
        _check(
            "claim_eligible_light_v1_candidate",
            best is not None,
            (
                f"claim_eligible_run_count={len(eligible_rows)}, "
                f"excluded_diagnostic_fragments={list(diagnostic_fragments)}, "
                f"claim_split={claim_split}"
            ),
        ),
        _check(
            "transfer_signal_present",
            transfer_validated,
            (
                f"transfer status={transfer_scorecard.get('status', 'missing')}, "
                f"evaluated_task_count={transfer_scorecard.get('evaluated_task_count', 0)}"
            ),
        ),
        _check(
            "medium_or_larger_confirmation",
            bool(args.medium_confirmed),
            "explicit confirmation flag required for medium-or-larger split evidence",
        ),
        _check(
            "strong_baseline_comparison",
            strong_baseline_compared,
            (
                "explicit confirmation flag or complete claim evidence entry required for "
                f"reproduced or fair strong baseline comparison; {strong_baseline_status['reason']}"
            ),
        ),
        _check(
            "scorecard_metrics_complete",
            scorecard_complete,
            (
                f"per_task_metrics={_has_task_metrics(best)}, "
                f"spectral_metric={_has_spectral_metrics(best)}, "
                f"cost_or_throughput={_has_cost_or_throughput(best)}"
            ),
        ),
        _check(
            "wandb_or_artifact_handles",
            handles_confirmed,
            "best row has W&B/artifact handles or explicit confirmation flag is set",
        ),
        _check(
            "claim_documentation_confirmed",
            documentation_confirmed,
            (
                "explicit confirmation flag or validated claim evidence entry required for exact "
                f"split/command/commit/checkpoint claim docs; {documentation_status['reason']}"
            ),
        ),
    ]
    blocking_reasons = [
        str(check["key"]) for check in readiness_checks if not bool(check["passed"])
    ]
    status = _status_from_checks(readiness_checks)

    record: dict[str, Any] = {
        "status": status,
        "sota_ready": status == "sota_ready",
        "transport_objective": {
            "source": str(args.transport_status_json),
            "status": transport_status.get("status", "missing"),
            "literal_achieved": transport_literal_achieved,
        },
        "light_v1": {
            "scorecard_json": str(args.light_scorecard_json),
            "candidate_summary_globs": list(candidate_summary_globs),
            "claim_split": claim_split,
            "baseline_run_name": args.baseline_run_name,
            "baseline_metric_name": args.metric_name,
            "baseline_metric_value": baseline_value,
            "best_overall_run_name": best_overall.get("run_name") if best_overall else "",
            "best_overall_metric_value": best_overall_value,
            "excluded_diagnostic_fragments": list(diagnostic_fragments),
            "claim_eligible_run_count": len(eligible_rows),
            "best_run_name": best.get("run_name") if best else "",
            "best_metric_value": best_value,
            "baseline_improvement_fraction": improvement_fraction,
            "min_improvement_required": float(args.min_improvement),
            "passes_min_improvement_gate": passes_min_improvement,
            "per_task_metrics_present": _has_task_metrics(best),
            "spectral_metric_present": _has_spectral_metrics(best),
            "cost_or_throughput_present": _has_cost_or_throughput(best),
            "wandb_or_artifact_handles_present": _has_wandb_or_artifact_handles(best),
        },
        "transfer": {
            "scorecard_json": str(args.transfer_scorecard_json),
            "status": transfer_scorecard.get("status", "missing"),
            "calibration_scope": transfer_scorecard.get("calibration_scope", ""),
            "evaluated_task_count": transfer_scorecard.get("evaluated_task_count", 0),
            "skipped_task_count": transfer_scorecard.get("skipped_task_count", 0),
            "mean_validation_nrmse": transfer_scorecard.get("mean_validation_nrmse"),
        },
        "claim_evidence": {
            "source": str(claim_evidence_json),
            "present": bool(claim_evidence),
        },
        "claim_documentation": documentation_status,
        "strong_baseline_comparison": strong_baseline_status,
        "readiness_checks": readiness_checks,
        "blocking_reasons": blocking_reasons,
        "next_recommended_path": (
            "Train or evaluate a learned general PDE operator/refiner gate that improves the "
            "full light-v1 decoded rollout score by at least the configured threshold, then "
            "confirm on a medium-or-larger split and compare against a strong neural baseline."
        ),
    }

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit universal SOTA readiness")
    parser.add_argument(
        "--light-scorecard-json",
        default="reports/demo/light_latest/scorecard.json",
        help="Light-v1 demo scorecard JSON",
    )
    parser.add_argument(
        "--transport-status-json",
        default="reports/research/sota_loop/transport_objective_status.json",
        help="Official transport objective status JSON",
    )
    parser.add_argument(
        "--transfer-scorecard-json",
        default="reports/research/sota_loop/inferred_transfer_scorecard/scorecard.json",
        help="Inferred transport transfer scorecard JSON",
    )
    parser.add_argument("--baseline-run-name", default="persistence_light_v1_test")
    parser.add_argument("--metric-name", default="decoded_rollout_nrmse")
    parser.add_argument("--min-improvement", type=float, default=0.2)
    parser.add_argument(
        "--claim-split",
        default="test",
        help="Required split metadata for claim-eligible candidate summaries.",
    )
    parser.add_argument("--medium-confirmed", action="store_true")
    parser.add_argument("--strong-baseline-compared", action="store_true")
    parser.add_argument("--artifact-handles-confirmed", action="store_true")
    parser.add_argument("--documentation-confirmed", action="store_true")
    parser.add_argument(
        "--claim-evidence-json",
        default="docs/claim_evidence/universal_sota_claim_evidence.json",
        help="Optional machine-readable claim evidence manifest.",
    )
    parser.add_argument(
        "--exclude-diagnostic-fragment",
        action="append",
        default=None,
        help=(
            "Run-name fragment to exclude from universal-SOTA claim eligibility. "
            "Defaults to known diagnostic transport sidecars."
        ),
    )
    parser.add_argument(
        "--candidate-summary-glob",
        action="append",
        default=["reports/light_experiments_remote/ups_light*/summary.json"],
        help="Optional summary.json glob for additional light-v1 claim candidates.",
    )
    parser.add_argument(
        "--output-json",
        default="reports/research/sota_loop/universal_sota_status.json",
    )
    args = parser.parse_args()
    print(json.dumps(run_audit(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
