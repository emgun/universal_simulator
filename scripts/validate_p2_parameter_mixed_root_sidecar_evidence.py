#!/usr/bin/env python
from __future__ import annotations

"""Validate mixed-root full-task P2 parameter sidecar evidence."""

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.validate_next_validation_contracts import load_json
from scripts.validate_p2_parameter_conditioned_sidecar_evidence import (
    validate_evidence as validate_source_sidecar_evidence,
)
from scripts.validate_p2_parameter_decoded_sidecar_evidence import (
    validate_evidence as validate_prior_decoded_evidence,
)

DEFAULT_EVIDENCE_JSON = (
    "docs/claim_evidence/ups_advection_p2_parameter_mixed_root_sidecar_val_evidence.json"
)
EXPECTED_MEASUREMENT_TYPE = "ups_advection_p2_parameter_mixed_root_sidecar_validation"
EXPECTED_DECISION_STATUS = "mixed_root_full_task_validation_supports_p2_parameter_sidecar"
EXPECTED_SUMMARY_PATH = (
    "docs/claim_evidence/artifacts/"
    "ups_advection_p2_parameter_mixed_root_sidecar_val_summary.json"
)
EXPECTED_PRIOR_DECODED_EVIDENCE = (
    "docs/claim_evidence/ups_advection_p2_parameter_decoded_sidecar_val_evidence.json"
)
EXPECTED_SOURCE_EVIDENCE = (
    "docs/claim_evidence/ups_advection_p2_parameter_conditioned_sidecar_val_evidence.json"
)
EXPECTED_CHECKPOINT_SOURCE = "reports/research/sota_loop/learned_capacity_gate/ups_light_local_joint_rollout4_residual_ft_val"
EXPECTED_TASKS = ["burgers1d", "advection1d", "darcy2d"]
EXPECTED_TASK_ROOTS = {"advection1d": "data/pdebench_official_advection_light"}
EXPECTED_COEFFICIENTS = {
    "param:beta": 10.236877359639507,
    "bias": -0.08098891730605368,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_mapping(value: Any, label: str, errors: list[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(f"{label} must be an object")
        return {}
    return value


def _as_sequence(value: Any, label: str, errors: list[str]) -> Sequence[Any]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        errors.append(f"{label} must be a sequence")
        return ()
    return value


def _validate_file_ref(
    source: Mapping[str, Any],
    *,
    expected_path: str,
    label: str,
    root: Path,
    errors: list[str],
) -> Path | None:
    if source.get("path") != expected_path:
        errors.append(f"{label}.path must be {expected_path}")
        return None
    path = root / expected_path
    if not path.exists():
        errors.append(f"{label}.path does not exist: {path}")
        return None
    if source.get("sha256") != _sha256(path):
        errors.append(f"{label}.sha256 must match file bytes")
    if isinstance(source.get("bytes"), int) and source.get("bytes") != path.stat().st_size:
        errors.append(f"{label}.bytes must match file size")
    return path


def _contains_arg(command_args: Sequence[Any], needle: str) -> bool:
    return any(str(arg) == needle or needle in str(arg) for arg in command_args)


def _metric(summary: Mapping[str, Any], name: str) -> float | None:
    metrics = summary.get("metrics")
    if not isinstance(metrics, Mapping):
        return None
    value = metrics.get(name)
    if not isinstance(value, (int, float)):
        return None
    return float(value)


def _close(a: Any, b: Any, *, tol: float = 1e-12) -> bool:
    return (
        isinstance(a, (int, float))
        and isinstance(b, (int, float))
        and abs(float(a) - float(b)) <= tol
    )


def _runtime_estimator(estimator: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in estimator.items() if key != "config_key"}


def validate_evidence(evidence: Mapping[str, Any], *, root: Path | None = None) -> list[str]:
    errors: list[str] = []
    repo_root = root or Path.cwd()

    if evidence.get("measurement_type") != EXPECTED_MEASUREMENT_TYPE:
        errors.append(f"measurement_type must be {EXPECTED_MEASUREMENT_TYPE}")
    if evidence.get("tasks") != EXPECTED_TASKS:
        errors.append(f"tasks must be {EXPECTED_TASKS}")
    if evidence.get("split") != "val":
        errors.append("split must be val")
    if evidence.get("base_data_root") != "data/pdebench":
        errors.append("base_data_root must be data/pdebench")
    if evidence.get("task_roots") != EXPECTED_TASK_ROOTS:
        errors.append(f"task_roots must be {EXPECTED_TASK_ROOTS}")
    if evidence.get("held_out_test_used") is not False:
        errors.append("held_out_test_used must be false")
    if evidence.get("held_out_test_data_read") is not False:
        errors.append("held_out_test_data_read must be false")
    if evidence.get("test_ledger_writes") != []:
        errors.append("test_ledger_writes must be empty")

    command_args = _as_sequence(evidence.get("command_args"), "command_args", errors)
    if _contains_arg(command_args, "--extra-eval-split"):
        errors.append("command_args must not include --extra-eval-split")
    if _contains_arg(command_args, "data.split=test"):
        errors.append("command_args must not set data.split=test")
    for required in (
        "data.split=val",
        "data.max_samples=32",
        "data.param_keys=[beta]",
        "data.task_roots={advection1d: data/pdebench_official_advection_light}",
        "evaluation.skip_missing_tasks=false",
    ):
        if not _contains_arg(command_args, required):
            errors.append(f"command_args must include {required}")

    summary_path = _validate_file_ref(
        _as_mapping(evidence.get("summary_report"), "summary_report", errors),
        expected_path=EXPECTED_SUMMARY_PATH,
        label="summary_report",
        root=repo_root,
        errors=errors,
    )
    summary: dict[str, Any] = {}
    if summary_path is not None:
        summary = load_json(summary_path)

    prior_path = _validate_file_ref(
        _as_mapping(
            evidence.get("prior_decoded_sidecar_evidence"),
            "prior_decoded_sidecar_evidence",
            errors,
        ),
        expected_path=EXPECTED_PRIOR_DECODED_EVIDENCE,
        label="prior_decoded_sidecar_evidence",
        root=repo_root,
        errors=errors,
    )
    if prior_path is not None:
        errors.extend(validate_prior_decoded_evidence(load_json(prior_path), root=repo_root))

    source_path = _validate_file_ref(
        _as_mapping(evidence.get("source_sidecar_evidence"), "source_sidecar_evidence", errors),
        expected_path=EXPECTED_SOURCE_EVIDENCE,
        label="source_sidecar_evidence",
        root=repo_root,
        errors=errors,
    )
    if source_path is not None:
        errors.extend(validate_source_sidecar_evidence(load_json(source_path), root=repo_root))

    estimator = _as_mapping(evidence.get("estimator"), "estimator", errors)
    if estimator.get("config_key") != "evaluation.decoded_data_conditioned_roll_shift_estimator":
        errors.append("estimator.config_key must point at decoded data-conditioned estimator")
    if estimator.get("feature_names") != ["param:beta", "bias"]:
        errors.append("estimator.feature_names must be ['param:beta', 'bias']")
    if estimator.get("coefficients") != EXPECTED_COEFFICIENTS:
        errors.append("estimator.coefficients must match locked train-fit beta coefficients")
    if estimator.get("tasks") != ["advection1d"]:
        errors.append("estimator.tasks must be ['advection1d']")
    if estimator.get("mode") != "roll_persistence":
        errors.append("estimator.mode must be roll_persistence")

    if summary:
        if summary.get("checkpoint_source") != EXPECTED_CHECKPOINT_SOURCE:
            errors.append("summary.checkpoint_source must match expected frozen checkpoint")
        if summary.get("skip_training") is not True:
            errors.append("summary.skip_training must be true")
        extra = _as_mapping(summary.get("extra"), "summary.extra", errors)
        if extra.get("decoded_task") != EXPECTED_TASKS:
            errors.append("summary.extra.decoded_task must include the full task mix")
        if extra.get("decoded_split") != "val":
            errors.append("summary.extra.decoded_split must be val")
        if extra.get("decoded_task_roots") != EXPECTED_TASK_ROOTS:
            errors.append("summary.extra.decoded_task_roots must match evidence task_roots")
        if extra.get("decoded_skipped_missing_tasks") != []:
            errors.append("summary.extra.decoded_skipped_missing_tasks must be empty")
        if extra.get("decoded_skip_missing_tasks") is not False:
            errors.append("summary.extra.decoded_skip_missing_tasks must be false")
        if extra.get("decoded_decoded_context_roll_shift_estimator") != {}:
            errors.append("summary must not enable decoded_context_roll_shift_estimator")
        if extra.get("decoded_decoded_observed_roll_shift_estimator") != {}:
            errors.append("summary must not enable decoded_observed_roll_shift_estimator")
        if extra.get("decoded_decoded_prediction_roll_shift_estimator") != {}:
            errors.append("summary must not enable decoded_prediction_roll_shift_estimator")
        if extra.get("decoded_decoded_data_conditioned_roll_shift_estimator") != _runtime_estimator(
            estimator
        ):
            errors.append("summary estimator must match evidence estimator")

        metrics = _as_mapping(evidence.get("metrics"), "metrics", errors)
        for key in (
            "decoded_rollout_nrmse",
            "task_advection1d_decoded_rollout_nrmse",
            "task_burgers1d_decoded_rollout_nrmse",
            "task_darcy2d_decoded_rollout_nrmse",
            "task_advection1d_decoded_h16_nrmse",
            "decoded_data_conditioned_roll_shift_mean",
        ):
            if not _close(metrics.get(key), _metric(summary, key)):
                errors.append(f"metrics.{key} must match summary")
        reference = metrics.get("reference_context_phase_validation_nrmse")
        current = metrics.get("decoded_rollout_nrmse")
        delta = metrics.get("absolute_delta_vs_reference_context_phase")
        improvement = metrics.get("improvement_fraction_vs_reference_context_phase")
        if not _close(delta, float(current) - float(reference), tol=1e-12):
            errors.append("metrics.absolute_delta_vs_reference_context_phase mismatch")
        expected_improvement = (float(reference) - float(current)) / float(reference)
        if not _close(improvement, expected_improvement, tol=1e-12):
            errors.append("metrics.improvement_fraction_vs_reference_context_phase mismatch")
        if not (isinstance(current, (int, float)) and float(current) < float(reference)):
            errors.append("mixed-root validation must improve on reference context-phase metric")

    scope = _as_mapping(evidence.get("decoded_scope"), "decoded_scope", errors)
    if scope.get("uses_decoded_evaluator") is not True:
        errors.append("decoded_scope.uses_decoded_evaluator must be true")
    if scope.get("decoded_rollout_steps") != 16:
        errors.append("decoded_scope.decoded_rollout_steps must be 16")
    if scope.get("max_samples") != 32:
        errors.append("decoded_scope.max_samples must be 32")
    if scope.get("skipped_missing_tasks") != []:
        errors.append("decoded_scope.skipped_missing_tasks must be empty")
    if scope.get("full_task_mix_validation") is not True:
        errors.append("decoded_scope.full_task_mix_validation must be true")
    if scope.get("same_exact_data_root_as_primary_light_v1") is not False:
        errors.append("decoded_scope.same_exact_data_root_as_primary_light_v1 must be false")
    if scope.get("mixed_root_advection_beta_provenance") is not True:
        errors.append("decoded_scope.mixed_root_advection_beta_provenance must be true")
    if scope.get("full_multitask_primary_claim_replacement") is not False:
        errors.append("decoded_scope.full_multitask_primary_claim_replacement must be false")

    teacher = _as_mapping(
        evidence.get("teacher_context_dependency"),
        "teacher_context_dependency",
        errors,
    )
    if teacher.get("uses_observed_context_transitions") is not False:
        errors.append("teacher_context_dependency.uses_observed_context_transitions must be false")
    if teacher.get("uses_beta_metadata") is not True:
        errors.append("teacher_context_dependency.uses_beta_metadata must be true")
    if teacher.get("uses_source_identity_as_learned_key") is not False:
        errors.append(
            "teacher_context_dependency.uses_source_identity_as_learned_key must be false"
        )

    decision = _as_mapping(evidence.get("decision"), "decision", errors)
    if decision.get("status") != EXPECTED_DECISION_STATUS:
        errors.append(f"decision.status must be {EXPECTED_DECISION_STATUS}")
    if decision.get("held_out_test_allowed_by_this_evidence") is not False:
        errors.append("decision.held_out_test_allowed_by_this_evidence must be false")
    if decision.get("primary_claim_replacement") is not False:
        errors.append("decision.primary_claim_replacement must be false")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-json", type=Path, default=Path(DEFAULT_EVIDENCE_JSON))
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)

    evidence = load_json(args.repo_root / args.evidence_json)
    errors = validate_evidence(evidence, root=args.repo_root)
    record = {
        "evidence_json": str(args.evidence_json),
        "errors": errors,
        "status": "valid" if not errors else "invalid",
    }
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
