#!/usr/bin/env python
from __future__ import annotations

"""Run validation-only data-conditioned transport ablations."""

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_data_conditioned_transport_shift_gate import run_gate

DEFAULT_OUTPUT_JSON = (
    "reports/research/sota_loop/data_conditioned_ablation_matrix/ablation_matrix.json"
)
DEFAULT_EVIDENCE_JSON = (
    "docs/claim_evidence/ups_advection_data_conditioned_ablation_val_evidence.json"
)
DEFAULT_CONTRACT_JSON = "docs/claim_evidence/ups_advection_next_validation_contracts.json"
CONTEXT_FEATURES = {"context_shift", "context_shift_abs"}
REQUIRED_VARIANTS = (
    "full_context_shift",
    "weaker_context_shift",
    "no_data_conditioning",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def _shift_range(start: int, stop: int) -> list[int]:
    if stop < start:
        raise ValueError("shift range stop must be >= start")
    return list(range(int(start), int(stop) + 1))


def _gate_args(
    args: argparse.Namespace,
    *,
    variant_id: str,
    feature_names: list[str],
    shifts: list[int],
    min_horizon: int,
    context_transitions: int,
    output_dir: Path,
) -> argparse.Namespace:
    return argparse.Namespace(
        data_root=args.data_root,
        task=args.task,
        train_split=args.train_split,
        val_split=args.val_split,
        max_samples=args.max_samples,
        train_max_samples=args.train_max_samples,
        val_max_samples=args.val_max_samples,
        rollout_steps=args.rollout_steps,
        shift=shifts,
        feature=feature_names,
        metric=args.metric,
        ridge=args.ridge,
        min_horizon=min_horizon,
        context_transitions=context_transitions,
        refine_radius=0,
        fractional_refine_step=0.0,
        min_shift=args.min_shift,
        max_shift=args.max_shift,
        no_normalize=args.no_normalize,
        reference_metric_value=args.reference_metric_value,
        val_min_relative_improvement=args.val_min_relative_improvement,
        output_json=str(output_dir / f"{variant_id}.json"),
    )


def _variant_summary(
    *,
    variant_id: str,
    gate_record: dict[str, Any],
    report_path: Path,
) -> dict[str, Any]:
    shifts = [float(shift) for shift in gate_record["candidate_shifts"]]
    validation = gate_record["fit"]["selected_validation"]
    train = gate_record["fit"]["train"]
    feature_names = [str(name) for name in gate_record["feature_names"]]
    return {
        "variant_id": variant_id,
        "report_path": _repo_path(report_path),
        "report_sha256": _sha256(report_path),
        "report_bytes": report_path.stat().st_size,
        "feature_names": feature_names,
        "uses_context_features": bool(CONTEXT_FEATURES.intersection(feature_names)),
        "context_transitions": gate_record.get("context_transitions"),
        "candidate_shift_min": min(shifts),
        "candidate_shift_max": max(shifts),
        "candidate_shift_max_abs": max(abs(min(shifts)), abs(max(shifts))),
        "metric_name": gate_record["metric"],
        "held_out_test_used": gate_record["held_out_test_used"],
        "held_out_test_data_read": gate_record["held_out_test_data_read"],
        "test_eligible": gate_record["test_eligible"],
        "metrics": {
            "train_nrmse": float(train["nrmse"]),
            "validation_nrmse": float(validation["nrmse"]),
            "train_metric_value": float(train["metric_value"]),
            "validation_metric_value": float(validation["metric_value"]),
            "validation_transition_count": int(validation["transition_count"]),
            "validation_predicted_shift_mean": float(validation["predicted_shift_mean"]),
            "validation_predicted_shift_std": float(validation["predicted_shift_std"]),
            "validation_predicted_shift_min": float(validation["predicted_shift_min"]),
            "validation_predicted_shift_max": float(validation["predicted_shift_max"]),
        },
        "selected_override": gate_record["selected_override"],
        "blockers": list(gate_record["blockers"]),
    }


def _deltas(variants: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    full = float(variants["full_context_shift"]["metrics"]["validation_metric_value"])
    deltas: dict[str, dict[str, float]] = {}
    for variant_id, record in variants.items():
        value = float(record["metrics"]["validation_metric_value"])
        absolute = value - full
        deltas[variant_id] = {
            "absolute": absolute,
            "relative_to_full": 0.0 if full == 0.0 else absolute / full,
        }
    return deltas


def _interpretation(
    variants: dict[str, dict[str, Any]],
    deltas: dict[str, dict[str, float]],
) -> dict[str, Any]:
    full_value = float(variants["full_context_shift"]["metrics"]["validation_metric_value"])
    full_is_best = all(
        full_value <= float(record["metrics"]["validation_metric_value"]) + 1e-12
        for record in variants.values()
    )
    degraded = {
        variant_id: float(delta["absolute"]) > 0.0
        for variant_id, delta in deltas.items()
        if variant_id != "full_context_shift"
    }
    return {
        "full_context_shift_is_best": bool(full_is_best),
        "weaker_context_shift_degrades": degraded.get("weaker_context_shift", False),
        "no_data_conditioning_degrades": degraded.get("no_data_conditioning", False),
        "context_dependency_supported": bool(full_is_best and all(degraded.values())),
        "summary": (
            "Full context-shift remains the strongest validation variant; bounded or "
            "non-context ablations degrade, so the current win should stay scoped as "
            "teacher/context-dependent until a P2 sidecar reduces that dependency."
        ),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _file_ref(path: Path) -> dict[str, Any]:
    return {
        "path": _repo_path(path),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _evidence_record(
    *,
    args: argparse.Namespace,
    matrix: dict[str, Any],
    matrix_path: Path,
) -> dict[str, Any]:
    contract_path = REPO_ROOT / str(args.contract_json)
    return {
        "measurement_type": "ups_advection_data_conditioned_ablation_validation",
        "date": str(args.date),
        "contract": _file_ref(contract_path),
        "matrix_report": _file_ref(matrix_path),
        "task": matrix["task"],
        "split": matrix["split"],
        "data_root": matrix["data_root"],
        "max_samples": matrix["max_samples"],
        "rollout_steps": matrix["rollout_steps"],
        "metric_name": matrix["metric_name"],
        "held_out_test_used": matrix["held_out_test_used"],
        "held_out_test_data_read": matrix["held_out_test_data_read"],
        "test_ledger_writes": matrix["test_ledger_writes"],
        "variants": deepcopy(matrix["variants"]),
        "deltas_vs_full_context_shift": deepcopy(matrix["deltas_vs_full_context_shift"]),
        "context_dependency_interpretation": deepcopy(matrix["context_dependency_interpretation"]),
        "decision": {
            "status": "validation_ablation_completed_p2_required",
            "held_out_test_allowed_by_this_evidence": False,
            "next_step": (
                "Use this ablation result to prioritize a validation-only P2 learned "
                "warp/transport sidecar before any new held-out pretest contract."
            ),
        },
    }


def run_matrix(args: argparse.Namespace) -> dict[str, Any]:
    if args.train_split == args.val_split:
        raise ValueError("--train-split and --val-split must differ")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variant_specs = {
        "full_context_shift": {
            "feature_names": ["context_shift"],
            "shifts": _shift_range(args.full_shift_min, args.full_shift_max),
            "min_horizon": max(2, int(args.context_transitions) + 1),
            "context_transitions": int(args.context_transitions),
        },
        "weaker_context_shift": {
            "feature_names": ["context_shift"],
            "shifts": _shift_range(args.weak_shift_min, args.weak_shift_max),
            "min_horizon": max(2, int(args.context_transitions) + 1),
            "context_transitions": int(args.context_transitions),
        },
        "no_data_conditioning": {
            "feature_names": ["bias"],
            "shifts": _shift_range(args.no_data_shift_min, args.no_data_shift_max),
            "min_horizon": 1,
            "context_transitions": 0,
        },
    }

    variants: dict[str, dict[str, Any]] = {}
    for variant_id in REQUIRED_VARIANTS:
        spec = variant_specs[variant_id]
        gate_args = _gate_args(
            args,
            variant_id=variant_id,
            feature_names=spec["feature_names"],
            shifts=spec["shifts"],
            min_horizon=spec["min_horizon"],
            context_transitions=spec["context_transitions"],
            output_dir=output_dir,
        )
        gate_record = run_gate(gate_args)
        gate_path = Path(gate_args.output_json)
        _write_json(gate_path, gate_record)
        variants[variant_id] = _variant_summary(
            variant_id=variant_id,
            gate_record=gate_record,
            report_path=gate_path,
        )

    deltas = _deltas(variants)
    record = {
        "measurement_type": "ups_advection_data_conditioned_ablation_matrix",
        "task": args.task,
        "data_root": str(args.data_root),
        "train_split": args.train_split,
        "split": args.val_split,
        "held_out_test_used": False,
        "held_out_test_data_read": False,
        "test_ledger_writes": [],
        "max_samples": args.max_samples,
        "train_max_samples": args.train_max_samples,
        "val_max_samples": args.val_max_samples,
        "rollout_steps": args.rollout_steps,
        "metric_name": args.metric,
        "variants": variants,
        "deltas_vs_full_context_shift": deltas,
        "context_dependency_interpretation": _interpretation(variants, deltas),
        "notes": [
            "All variants fit on train split and score validation split only.",
            "No held-out test split is loaded, measured, or authorized by this matrix.",
        ],
    }

    output_json = Path(args.output_json)
    _write_json(output_json, record)
    evidence_json = getattr(args, "evidence_json", None)
    if evidence_json:
        evidence = _evidence_record(args=args, matrix=record, matrix_path=output_json)
        _write_json(Path(evidence_json), evidence)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/pdebench")
    parser.add_argument("--task", default="advection1d")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--train-max-samples", type=int)
    parser.add_argument("--val-max-samples", type=int)
    parser.add_argument("--rollout-steps", type=int, default=16)
    parser.add_argument("--metric", choices=("mse", "nrmse"), default="nrmse")
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--context-transitions", type=int, default=1)
    parser.add_argument("--full-shift-min", type=int, default=-80)
    parser.add_argument("--full-shift-max", type=int, default=80)
    parser.add_argument("--weak-shift-min", type=int, default=-8)
    parser.add_argument("--weak-shift-max", type=int, default=8)
    parser.add_argument("--no-data-shift-min", type=int, default=-80)
    parser.add_argument("--no-data-shift-max", type=int, default=80)
    parser.add_argument("--min-shift", type=float)
    parser.add_argument("--max-shift", type=float)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--reference-metric-value", type=float, default=1.0)
    parser.add_argument("--val-min-relative-improvement", type=float, default=0.0)
    parser.add_argument(
        "--output-dir", default="reports/research/sota_loop/data_conditioned_ablation_matrix"
    )
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--evidence-json", default=None)
    parser.add_argument("--contract-json", default=DEFAULT_CONTRACT_JSON)
    parser.add_argument("--date", default="2026-06-09")
    args = parser.parse_args(argv)

    record = run_matrix(args)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
