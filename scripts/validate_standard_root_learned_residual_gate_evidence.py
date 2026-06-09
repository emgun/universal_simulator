#!/usr/bin/env python
from __future__ import annotations

"""Validate standard-root learned residual-gate validation evidence."""

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
from scripts.validate_p2_parameter_canonical_root_sidecar_evidence import (
    validate_evidence as validate_p2_canonical_evidence,
)

DEFAULT_EVIDENCE_JSON = (
    "docs/claim_evidence/ups_standard_root_learned_residual_gate_val_evidence.json"
)
EXPECTED_MEASUREMENT_TYPE = "ups_standard_root_learned_residual_gate_validation"
EXPECTED_DECISION_STATUS = "negative_standard_root_learned_residual_gate_probe_recorded"
EXPECTED_TASKS = ["burgers1d", "advection1d", "darcy2d"]
EXPECTED_CHECKPOINT_SOURCE = "reports/research/sota_loop/learned_capacity_gate/ups_light_local_joint_rollout4_residual_ft_val"
EXPECTED_FIT_REPORT = (
    "docs/claim_evidence/artifacts/ups_standard_root_learned_residual_gate_fit_record.json"
)
EXPECTED_PERSISTENCE_REPORT = (
    "docs/claim_evidence/artifacts/ups_standard_root_persistence_val_summary.json"
)
EXPECTED_OPERATOR_REPORT = (
    "docs/claim_evidence/artifacts/ups_standard_root_operator_val_summary.json"
)
EXPECTED_P2_EVIDENCE = (
    "docs/claim_evidence/ups_advection_p2_parameter_canonical_root_sidecar_val_evidence.json"
)


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


def _metric(mapping: Mapping[str, Any], key: str) -> float | None:
    metrics = mapping.get("metrics")
    if not isinstance(metrics, Mapping):
        return None
    value = metrics.get(key)
    if not isinstance(value, (int, float)):
        return None
    return float(value)


def _close(left: Any, right: Any, *, tol: float = 1e-12) -> bool:
    return (
        isinstance(left, (int, float))
        and isinstance(right, (int, float))
        and abs(float(left) - float(right)) <= tol
    )


def _validate_command(
    command_args: Sequence[Any],
    *,
    label: str,
    require_data_root_override: bool,
    errors: list[str],
) -> None:
    if _contains_arg(command_args, "--extra-eval-split"):
        errors.append(f"{label} must not include --extra-eval-split")
    if _contains_arg(command_args, "data.split=test"):
        errors.append(f"{label} must not set data.split=test")
    if _contains_arg(command_args, "data.task_roots"):
        errors.append(f"{label} must not include data.task_roots")
    if _contains_arg(command_args, "data.param_keys=[beta]"):
        errors.append(f"{label} must not request beta param_keys")
    if require_data_root_override and not _contains_arg(command_args, "data.root=data/pdebench"):
        errors.append(f"{label} must include data.root=data/pdebench")
    if "--data-root" in [str(arg) for arg in command_args] and not _contains_arg(
        command_args, "data/pdebench"
    ):
        errors.append(f"{label} --data-root must be data/pdebench")


def validate_evidence(evidence: Mapping[str, Any], *, root: Path | None = None) -> list[str]:
    errors: list[str] = []
    repo_root = root or Path.cwd()

    if evidence.get("measurement_type") != EXPECTED_MEASUREMENT_TYPE:
        errors.append(f"measurement_type must be {EXPECTED_MEASUREMENT_TYPE}")
    if evidence.get("tasks") != EXPECTED_TASKS:
        errors.append(f"tasks must be {EXPECTED_TASKS}")
    if evidence.get("train_split") != "train":
        errors.append("train_split must be train")
    if evidence.get("val_split") != "val":
        errors.append("val_split must be val")
    if evidence.get("data_root") != "data/pdebench":
        errors.append("data_root must be data/pdebench")
    if evidence.get("checkpoint_source") != EXPECTED_CHECKPOINT_SOURCE:
        errors.append("checkpoint_source must match expected frozen checkpoint")
    if evidence.get("held_out_test_used") is not False:
        errors.append("held_out_test_used must be false")
    if evidence.get("held_out_test_data_read") is not False:
        errors.append("held_out_test_data_read must be false")
    if evidence.get("test_ledger_writes") != []:
        errors.append("test_ledger_writes must be empty")

    fit_command_args = _as_sequence(evidence.get("fit_command_args"), "fit_command_args", errors)
    _validate_command(
        fit_command_args,
        label="fit_command_args",
        require_data_root_override=False,
        errors=errors,
    )
    if not _contains_arg(fit_command_args, "scripts/fit_decoded_residual_gate.py"):
        errors.append("fit_command_args must run scripts/fit_decoded_residual_gate.py")
    if not _contains_arg(fit_command_args, "0.11122069865007121"):
        errors.append("fit_command_args must use canonical P2 reference metric")

    baseline_commands = _as_mapping(
        evidence.get("baseline_command_args"), "baseline_command_args", errors
    )
    for name in ("persistence", "operator"):
        command = _as_sequence(baseline_commands.get(name), f"baseline_command_args.{name}", errors)
        _validate_command(
            command,
            label=f"baseline_command_args.{name}",
            require_data_root_override=True,
            errors=errors,
        )

    fit_report_path = _validate_file_ref(
        _as_mapping(evidence.get("fit_report"), "fit_report", errors),
        expected_path=EXPECTED_FIT_REPORT,
        label="fit_report",
        root=repo_root,
        errors=errors,
    )
    fit_report: dict[str, Any] = {}
    if fit_report_path is not None:
        fit_report = load_json(fit_report_path)

    baseline_refs = _as_mapping(evidence.get("baseline_reports"), "baseline_reports", errors)
    persistence_path = _validate_file_ref(
        _as_mapping(baseline_refs.get("persistence"), "baseline_reports.persistence", errors),
        expected_path=EXPECTED_PERSISTENCE_REPORT,
        label="baseline_reports.persistence",
        root=repo_root,
        errors=errors,
    )
    operator_path = _validate_file_ref(
        _as_mapping(baseline_refs.get("operator"), "baseline_reports.operator", errors),
        expected_path=EXPECTED_OPERATOR_REPORT,
        label="baseline_reports.operator",
        root=repo_root,
        errors=errors,
    )
    persistence_summary = load_json(persistence_path) if persistence_path is not None else {}
    operator_summary = load_json(operator_path) if operator_path is not None else {}

    p2_path = _validate_file_ref(
        _as_mapping(
            evidence.get("prior_p2_canonical_root_evidence"),
            "prior_p2_canonical_root_evidence",
            errors,
        ),
        expected_path=EXPECTED_P2_EVIDENCE,
        label="prior_p2_canonical_root_evidence",
        root=repo_root,
        errors=errors,
    )
    p2_evidence: dict[str, Any] = {}
    if p2_path is not None:
        p2_evidence = load_json(p2_path)
        errors.extend(validate_p2_canonical_evidence(p2_evidence, root=repo_root))

    metrics = _as_mapping(evidence.get("metrics"), "metrics", errors)
    if fit_report:
        if fit_report.get("model") != "learned_decoded_residual_gate":
            errors.append("fit_report.model must be learned_decoded_residual_gate")
        if fit_report.get("data_root") != "data/pdebench":
            errors.append("fit_report.data_root must be data/pdebench")
        if fit_report.get("train_split") != "train":
            errors.append("fit_report.train_split must be train")
        if fit_report.get("val_split") != "val":
            errors.append("fit_report.val_split must be val")
        if fit_report.get("eval_max_samples") != 32:
            errors.append("fit_report.eval_max_samples must be 32")
        if fit_report.get("decoded_rollout_steps") != 16:
            errors.append("fit_report.decoded_rollout_steps must be 16")
        validation = _as_mapping(fit_report.get("validation"), "fit_report.validation", errors)
        val_extra = _as_mapping(validation.get("extra"), "fit_report.validation.extra", errors)
        if val_extra.get("task_roots") != {}:
            errors.append("fit_report.validation.extra.task_roots must be empty")
        for key in (
            "decoded_context_roll_shift_estimator",
            "decoded_data_conditioned_roll_shift_estimator",
            "decoded_observed_roll_shift_estimator",
            "decoded_prediction_roll_shift_estimator",
        ):
            if val_extra.get(key) != {}:
                errors.append(f"fit_report.validation.extra.{key} must be empty")
        guard = _as_mapping(
            fit_report.get("validation_guard"), "fit_report.validation_guard", errors
        )
        if guard.get("passed") is not False:
            errors.append("fit_report.validation_guard.passed must be false")
        if not _close(guard.get("reference_metric_value"), 0.11122069865007121):
            errors.append("fit_report.validation_guard.reference_metric_value must be canonical P2")
        val_metrics = _as_mapping(
            validation.get("metrics"), "fit_report.validation.metrics", errors
        )
        for evidence_key, fit_key in (
            ("learned_gate_val_decoded_rollout_nrmse", "decoded_rollout_nrmse"),
            (
                "learned_gate_task_advection1d_decoded_rollout_nrmse",
                "task_advection1d_decoded_rollout_nrmse",
            ),
            (
                "learned_gate_task_burgers1d_decoded_rollout_nrmse",
                "task_burgers1d_decoded_rollout_nrmse",
            ),
            (
                "learned_gate_task_darcy2d_decoded_rollout_nrmse",
                "task_darcy2d_decoded_rollout_nrmse",
            ),
        ):
            if not _close(metrics.get(evidence_key), val_metrics.get(fit_key)):
                errors.append(f"metrics.{evidence_key} must match fit report")

    if persistence_summary and not _close(
        metrics.get("persistence_val_decoded_rollout_nrmse"),
        _metric(persistence_summary, "decoded_rollout_nrmse"),
    ):
        errors.append("metrics.persistence_val_decoded_rollout_nrmse must match persistence report")
    if operator_summary and not _close(
        metrics.get("operator_val_decoded_rollout_nrmse"),
        _metric(operator_summary, "decoded_rollout_nrmse"),
    ):
        errors.append("metrics.operator_val_decoded_rollout_nrmse must match operator report")
    if p2_evidence:
        p2_metrics = _as_mapping(p2_evidence.get("metrics"), "prior_p2.metrics", errors)
        if not _close(
            metrics.get("canonical_p2_val_decoded_rollout_nrmse"),
            p2_metrics.get("decoded_rollout_nrmse"),
        ):
            errors.append("metrics.canonical_p2_val_decoded_rollout_nrmse must match P2 evidence")

    learned = metrics.get("learned_gate_val_decoded_rollout_nrmse")
    persistence = metrics.get("persistence_val_decoded_rollout_nrmse")
    operator = metrics.get("operator_val_decoded_rollout_nrmse")
    p2 = metrics.get("canonical_p2_val_decoded_rollout_nrmse")
    context = metrics.get("context_phase_reference_val_decoded_rollout_nrmse")
    for label, reference, delta_key, improvement_key in (
        (
            "persistence",
            persistence,
            "absolute_delta_vs_persistence",
            "improvement_fraction_vs_persistence",
        ),
        ("operator", operator, "absolute_delta_vs_operator", "improvement_fraction_vs_operator"),
        (
            "canonical_p2",
            p2,
            "absolute_delta_vs_canonical_p2",
            "improvement_fraction_vs_canonical_p2",
        ),
        (
            "context_phase_reference",
            context,
            "absolute_delta_vs_context_phase_reference",
            "improvement_fraction_vs_context_phase_reference",
        ),
    ):
        if not isinstance(learned, (int, float)) or not isinstance(reference, (int, float)):
            continue
        if not _close(metrics.get(delta_key), float(learned) - float(reference)):
            errors.append(f"metrics.{delta_key} mismatch")
        expected_improvement = (float(reference) - float(learned)) / float(reference)
        if not _close(metrics.get(improvement_key), expected_improvement):
            errors.append(f"metrics.{improvement_key} mismatch")
        if label == "persistence" and not float(learned) < float(reference):
            errors.append("learned gate must improve persistence baseline")
        if label == "canonical_p2" and not float(learned) > float(reference):
            errors.append("learned gate evidence must remain negative versus canonical P2")

    scope = _as_mapping(evidence.get("model_scope"), "model_scope", errors)
    expected_scope = {
        "standard_root_validation": True,
        "uses_generated_root": False,
        "uses_task_roots": False,
        "uses_beta_metadata": False,
        "uses_observed_context_transitions": False,
        "uses_roll_shift_sidecar": False,
        "uses_learned_decoded_residual_gate": True,
        "full_task_mix_validation": True,
        "max_samples": 32,
        "decoded_rollout_steps": 16,
        "full_multitask_primary_claim_replacement": False,
    }
    for key, expected in expected_scope.items():
        if scope.get(key) != expected:
            errors.append(f"model_scope.{key} must be {str(expected).lower()}")

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
        "passed": not errors,
    }
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
