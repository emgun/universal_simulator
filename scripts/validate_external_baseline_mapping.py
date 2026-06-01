from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ALLOWED_MAPPING_STATUSES = {
    "selected_reproduction_path_not_measured",
    "external_reproduction_measured",
}

ALLOWED_SELECTED_STATUSES = {
    "selected_not_yet_measured",
    "measured_complete",
}

ALLOWED_IMPLEMENTATION_STATUSES = {
    "external_adapter_not_implemented",
    "external_adapter_available_measurement_pending",
    "external_adapter_measured_complete",
}

ALLOWED_FOUNDATION_TRANSFER_STATUSES = {
    "contract_defined_measurement_pending",
}

ALLOWED_FOUNDATION_ADAPTER_STATUSES = {
    "validation_adapter_manifest_complete",
}

ALLOWED_FOUNDATION_VALIDATION_STATUSES = {
    "validation_model_measurement_complete",
}

ALLOWED_FOUNDATION_FINETUNE_STATUSES = {
    "validation_finetune_measurement_complete",
}

REQUIRED_PROTOCOL_KEYS = {
    "data_root",
    "split",
    "metric_name",
    "metric_value",
    "max_eval_samples",
    "rollout_steps",
    "task_set",
    "task_metric_keys",
}

REQUIRED_SCOPED_VARIANT_KEYS = {
    "artifact_handle",
    "artifact_sha256",
    "evidence_json",
    "external_paper_reproduction",
    "metric_name",
    "metric_value",
    "not_autonomous_rollout_claim",
    "published_numbers_directly_comparable",
    "run_name",
    "same_exact_inference_contract_as_primary",
    "split",
    "status",
    "variant_id",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return data


def _as_mapping(value: Any, label: str, errors: list[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(f"{label} must be an object")
        return {}
    return value


def _as_list(value: Any, label: str, errors: list[str]) -> list[Any]:
    if not isinstance(value, list):
        errors.append(f"{label} must be a list")
        return []
    return value


def _task_name_from_metric_key(metric_key: str) -> str | None:
    prefix = "task_"
    suffix = "_decoded_rollout_nrmse"
    if not metric_key.startswith(prefix) or not metric_key.endswith(suffix):
        return None
    return metric_key[len(prefix) : -len(suffix)]


def _claim_task_metric_keys(claim_evidence: Mapping[str, Any], errors: list[str]) -> list[str]:
    records = _as_list(claim_evidence.get("candidate_evidence"), "candidate_evidence", errors)
    if len(records) != 1:
        errors.append("candidate_evidence must contain exactly one committed claim row")
        return []
    record = _as_mapping(records[0], "candidate_evidence[0]", errors)
    metrics = _as_mapping(record.get("metrics"), "candidate_evidence[0].metrics", errors)
    return sorted(key for key in metrics if key.startswith("task_"))


def _claim_task_set(task_metric_keys: list[str], errors: list[str]) -> list[str]:
    tasks: list[str] = []
    for key in task_metric_keys:
        task = _task_name_from_metric_key(key)
        if task is None:
            errors.append(f"task metric key has unsupported shape: {key}")
            continue
        tasks.append(task)
    return sorted(tasks)


def _close_enough(left: Any, right: Any) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-15)
    return left == right


def _validate_test_measurements(
    *,
    measurements: list[Any],
    label: str,
    protocol: Mapping[str, Any],
    errors: list[str],
) -> None:
    for index, measurement_value in enumerate(measurements):
        measurement = _as_mapping(
            measurement_value,
            f"{label}[{index}]",
            errors,
        )
        if measurement.get("split") != protocol.get("split"):
            errors.append(f"{label}[{index}].split must match claim_protocol.split")
        if measurement.get("metric_name") != protocol.get("metric_name"):
            errors.append(f"{label}[{index}].metric_name must match claim_protocol.metric_name")
        if measurement.get("held_out_test_used") is not True:
            errors.append(f"{label}[{index}].held_out_test_used must be true")
        if measurement.get("claim_comparable") is not True:
            errors.append(f"{label}[{index}].claim_comparable must be true")
        if measurement.get("published_numbers_directly_comparable") is not False:
            errors.append(f"{label}[{index}].published_numbers_directly_comparable must be false")
        for key in ("measurement_key", "evidence_json", "artifact_handle"):
            if not measurement.get(key):
                errors.append(f"{label}[{index}].{key} is required")


def _validate_scoped_claim_variants(
    *,
    mapping_variants_value: Any,
    claim_evidence: Mapping[str, Any],
    protocol: Mapping[str, Any],
    errors: list[str],
) -> None:
    claim_variants = _as_list(
        claim_evidence.get("scoped_claim_variants", []),
        "claim_evidence.scoped_claim_variants",
        errors,
    )
    mapping_variants = _as_list(
        mapping_variants_value,
        "scoped_claim_variants",
        errors,
    )
    claim_by_id: dict[str, Mapping[str, Any]] = {}
    for index, claim_variant_value in enumerate(claim_variants):
        claim_variant = _as_mapping(
            claim_variant_value,
            f"claim_evidence.scoped_claim_variants[{index}]",
            errors,
        )
        variant_id = claim_variant.get("variant_id")
        if isinstance(variant_id, str) and variant_id:
            claim_by_id[variant_id] = claim_variant

    mapped_ids: set[str] = set()
    for index, mapping_variant_value in enumerate(mapping_variants):
        variant = _as_mapping(mapping_variant_value, f"scoped_claim_variants[{index}]", errors)
        variant_id = variant.get("variant_id")
        label = (
            f"scoped_claim_variants[{variant_id}]"
            if isinstance(variant_id, str) and variant_id
            else f"scoped_claim_variants[{index}]"
        )
        if not isinstance(variant_id, str) or not variant_id:
            errors.append(f"{label}.variant_id is required")
            continue
        if variant_id in mapped_ids:
            errors.append(f"duplicate scoped_claim_variants variant_id: {variant_id}")
        mapped_ids.add(variant_id)

        claim_variant = claim_by_id.get(variant_id)
        if claim_variant is None:
            errors.append(f"{label} has no matching claim evidence variant")
            continue

        for key in REQUIRED_SCOPED_VARIANT_KEYS:
            if key not in variant:
                errors.append(f"{label}.{key} is required")

        checks = {
            "run_name": claim_variant.get("run_name"),
            "split": claim_variant.get("split"),
            "metric_name": claim_variant.get("metric_name"),
            "metric_value": claim_variant.get("metric_value"),
            "evidence_json": claim_variant.get("evidence_json"),
            "artifact_sha256": claim_variant.get("artifact_sha256"),
            "status": claim_variant.get("status"),
        }
        for key, expected in checks.items():
            if expected is not None and not _close_enough(variant.get(key), expected):
                errors.append(f"{label}.{key} must match claim evidence")

        artifact_handles = _as_list(
            claim_variant.get("artifact_handles"),
            f"claim_evidence.scoped_claim_variants[{variant_id}].artifact_handles",
            errors,
        )
        if variant.get("artifact_handle") not in artifact_handles:
            errors.append(f"{label}.artifact_handle must match claim evidence")

        if variant.get("split") != protocol.get("split"):
            errors.append(f"{label}.split must match claim_protocol.split")
        if variant.get("metric_name") != protocol.get("metric_name"):
            errors.append(f"{label}.metric_name must match claim_protocol.metric_name")
        if variant.get("run_name") == protocol.get("run_name"):
            errors.append(f"{label}.run_name must not replace the primary claim run")
        if variant.get("same_exact_inference_contract_as_primary") is not False:
            errors.append(f"{label}.same_exact_inference_contract_as_primary must be false")
        if variant.get("not_autonomous_rollout_claim") is not True:
            errors.append(f"{label}.not_autonomous_rollout_claim must be true")
        if variant.get("published_numbers_directly_comparable") is not False:
            errors.append(f"{label}.published_numbers_directly_comparable must be false")
        if variant.get("external_paper_reproduction") is not False:
            errors.append(f"{label}.external_paper_reproduction must be false")
        if not variant.get("comparability_note"):
            errors.append(f"{label}.comparability_note is required")

    missing_ids = sorted(set(claim_by_id) - mapped_ids)
    for variant_id in missing_ids:
        errors.append(f"scoped_claim_variants missing claim evidence variant: {variant_id}")


def _validate_foundation_transfer_contract(
    contract: Mapping[str, Any],
    errors: list[str],
) -> None:
    if contract.get("status") not in ALLOWED_FOUNDATION_TRANSFER_STATUSES:
        errors.append(
            "foundation_transfer_contract.status must be one of "
            f"{sorted(ALLOWED_FOUNDATION_TRANSFER_STATUSES)}"
        )
    if contract.get("measurement_type") != "foundation_transfer_readiness_contract":
        errors.append(
            "foundation_transfer_contract.measurement_type must be "
            "foundation_transfer_readiness_contract"
        )
    if contract.get("held_out_test_used") is not False:
        errors.append("foundation_transfer_contract.held_out_test_used must be false")
    if contract.get("held_out_test_data_read") is not False:
        errors.append("foundation_transfer_contract.held_out_test_data_read must be false")
    if contract.get("claim_comparable") is not False:
        errors.append("foundation_transfer_contract.claim_comparable must be false")
    if contract.get("published_numbers_directly_comparable") is not False:
        errors.append(
            "foundation_transfer_contract.published_numbers_directly_comparable must be false"
        )
    for key in ("run_name", "evidence_json", "artifact_handle", "artifact_sha256"):
        if not contract.get(key):
            errors.append(f"foundation_transfer_contract.{key} is required")

    inspected_splits = _as_list(
        contract.get("inspected_splits"),
        "foundation_transfer_contract.inspected_splits",
        errors,
    )
    if "test" in inspected_splits:
        errors.append("foundation_transfer_contract.inspected_splits must not include test")
    if not inspected_splits:
        errors.append("foundation_transfer_contract.inspected_splits is required")

    source_commits = _as_mapping(
        contract.get("source_commits"),
        "foundation_transfer_contract.source_commits",
        errors,
    )
    for key in ("poseidon_official_repo", "cno_official_repo"):
        if not source_commits.get(key):
            errors.append(f"foundation_transfer_contract.source_commits.{key} is required")

    blockers = _as_list(
        contract.get("measurement_blockers"),
        "foundation_transfer_contract.measurement_blockers",
        errors,
    )
    if not blockers:
        errors.append("foundation_transfer_contract.measurement_blockers is required")
    if "foundation_measurement_ready" in blockers:
        errors.append(
            "foundation_transfer_contract.measurement_blockers must not include "
            "foundation_measurement_ready"
        )
    if not contract.get("next_validation_gate"):
        errors.append("foundation_transfer_contract.next_validation_gate is required")


def _validate_foundation_transfer_adapter_gate(
    gate: Mapping[str, Any],
    errors: list[str],
) -> None:
    if gate.get("status") not in ALLOWED_FOUNDATION_ADAPTER_STATUSES:
        errors.append(
            "foundation_transfer_adapter_gate.status must be one of "
            f"{sorted(ALLOWED_FOUNDATION_ADAPTER_STATUSES)}"
        )
    if gate.get("measurement_type") != "poseidon_validation_adapter_manifest":
        errors.append(
            "foundation_transfer_adapter_gate.measurement_type must be "
            "poseidon_validation_adapter_manifest"
        )
    if gate.get("held_out_test_used") is not False:
        errors.append("foundation_transfer_adapter_gate.held_out_test_used must be false")
    if gate.get("held_out_test_data_read") is not False:
        errors.append("foundation_transfer_adapter_gate.held_out_test_data_read must be false")
    if gate.get("claim_comparable") is not False:
        errors.append("foundation_transfer_adapter_gate.claim_comparable must be false")
    if gate.get("published_numbers_directly_comparable") is not False:
        errors.append(
            "foundation_transfer_adapter_gate.published_numbers_directly_comparable "
            "must be false"
        )
    for key in ("run_name", "summary_json", "evidence_json", "artifact_handle", "artifact_sha256"):
        if not gate.get(key):
            errors.append(f"foundation_transfer_adapter_gate.{key} is required")

    inspected_splits = _as_list(
        gate.get("inspected_splits"),
        "foundation_transfer_adapter_gate.inspected_splits",
        errors,
    )
    if "test" in inspected_splits:
        errors.append("foundation_transfer_adapter_gate.inspected_splits must not include test")
    if not inspected_splits:
        errors.append("foundation_transfer_adapter_gate.inspected_splits is required")

    metrics = _as_mapping(
        gate.get("metrics"),
        "foundation_transfer_adapter_gate.metrics",
        errors,
    )
    if "adapter_roundtrip_nrmse" not in metrics:
        errors.append(
            "foundation_transfer_adapter_gate.metrics.adapter_roundtrip_nrmse is required"
        )
    if "decoded_rollout_nrmse" in metrics:
        errors.append("foundation_transfer_adapter_gate must not report decoded_rollout_nrmse")

    source_commits = _as_mapping(
        gate.get("source_commits"),
        "foundation_transfer_adapter_gate.source_commits",
        errors,
    )
    if not source_commits.get("poseidon_official_repo"):
        errors.append(
            "foundation_transfer_adapter_gate.source_commits.poseidon_official_repo " "is required"
        )
    checkpoint = _as_mapping(
        gate.get("pretrained_checkpoint"),
        "foundation_transfer_adapter_gate.pretrained_checkpoint",
        errors,
    )
    if not checkpoint.get("handle"):
        errors.append("foundation_transfer_adapter_gate.pretrained_checkpoint.handle is required")
    if checkpoint.get("requires_hash_before_model_metric") is not True:
        errors.append(
            "foundation_transfer_adapter_gate.pretrained_checkpoint must require hash "
            "before model metric"
        )
    if not gate.get("next_validation_gate"):
        errors.append("foundation_transfer_adapter_gate.next_validation_gate is required")


def _validate_foundation_transfer_validation_measurement(
    measurement: Mapping[str, Any],
    errors: list[str],
) -> None:
    if measurement.get("status") not in ALLOWED_FOUNDATION_VALIDATION_STATUSES:
        errors.append(
            "foundation_transfer_validation_measurement.status must be one of "
            f"{sorted(ALLOWED_FOUNDATION_VALIDATION_STATUSES)}"
        )
    if measurement.get("measurement_type") != "poseidon_scot_validation_measurement":
        errors.append(
            "foundation_transfer_validation_measurement.measurement_type must be "
            "poseidon_scot_validation_measurement"
        )
    if measurement.get("split") == "test":
        errors.append("foundation_transfer_validation_measurement.split must not be test")
    if measurement.get("held_out_test_used") is not False:
        errors.append("foundation_transfer_validation_measurement.held_out_test_used must be false")
    if measurement.get("held_out_test_data_read") is not False:
        errors.append(
            "foundation_transfer_validation_measurement.held_out_test_data_read must be false"
        )
    if measurement.get("claim_comparable") is not False:
        errors.append("foundation_transfer_validation_measurement.claim_comparable must be false")
    if measurement.get("published_numbers_directly_comparable") is not False:
        errors.append(
            "foundation_transfer_validation_measurement.published_numbers_directly_comparable "
            "must be false"
        )
    for key in ("run_name", "summary_json", "evidence_json", "artifact_handle", "artifact_sha256"):
        if not measurement.get(key):
            errors.append(f"foundation_transfer_validation_measurement.{key} is required")

    metrics = _as_mapping(
        measurement.get("metrics"),
        "foundation_transfer_validation_measurement.metrics",
        errors,
    )
    if "decoded_rollout_nrmse" not in metrics:
        errors.append(
            "foundation_transfer_validation_measurement.metrics.decoded_rollout_nrmse "
            "is required"
        )
    for task in ("advection1d", "burgers1d", "darcy2d"):
        key = f"task_{task}_decoded_rollout_nrmse"
        if key not in metrics:
            errors.append(f"foundation_transfer_validation_measurement.metrics.{key} is required")

    source_commits = _as_mapping(
        measurement.get("source_commits"),
        "foundation_transfer_validation_measurement.source_commits",
        errors,
    )
    if not source_commits.get("poseidon_official_repo"):
        errors.append(
            "foundation_transfer_validation_measurement.source_commits.poseidon_official_repo "
            "is required"
        )
    checkpoint = _as_mapping(
        measurement.get("pretrained_checkpoint"),
        "foundation_transfer_validation_measurement.pretrained_checkpoint",
        errors,
    )
    if not checkpoint.get("handle"):
        errors.append(
            "foundation_transfer_validation_measurement.pretrained_checkpoint.handle is required"
        )
    if checkpoint.get("sha256_status") != "matched":
        errors.append(
            "foundation_transfer_validation_measurement.pretrained_checkpoint.sha256_status "
            "must be matched"
        )
    if not checkpoint.get("sha256"):
        errors.append(
            "foundation_transfer_validation_measurement.pretrained_checkpoint.sha256 is required"
        )
    if measurement.get("requires_finetuning_before_held_out_test") is not True:
        errors.append(
            "foundation_transfer_validation_measurement.requires_finetuning_before_held_out_test "
            "must be true"
        )


def _validate_foundation_transfer_finetune_validation_measurement(
    measurement: Mapping[str, Any],
    errors: list[str],
) -> None:
    if measurement.get("status") not in ALLOWED_FOUNDATION_FINETUNE_STATUSES:
        errors.append(
            "foundation_transfer_finetune_validation_measurement.status must be one of "
            f"{sorted(ALLOWED_FOUNDATION_FINETUNE_STATUSES)}"
        )
    if measurement.get("measurement_type") != "poseidon_scot_finetune_validation_measurement":
        errors.append(
            "foundation_transfer_finetune_validation_measurement.measurement_type must be "
            "poseidon_scot_finetune_validation_measurement"
        )
    if measurement.get("train_split") == "test":
        errors.append(
            "foundation_transfer_finetune_validation_measurement.train_split must not be test"
        )
    if measurement.get("split") == "test":
        errors.append("foundation_transfer_finetune_validation_measurement.split must not be test")
    if measurement.get("held_out_test_used") is not False:
        errors.append(
            "foundation_transfer_finetune_validation_measurement.held_out_test_used must be false"
        )
    if measurement.get("held_out_test_data_read") is not False:
        errors.append(
            "foundation_transfer_finetune_validation_measurement.held_out_test_data_read "
            "must be false"
        )
    if measurement.get("claim_comparable") is not False:
        errors.append(
            "foundation_transfer_finetune_validation_measurement.claim_comparable must be false"
        )
    if measurement.get("published_numbers_directly_comparable") is not False:
        errors.append(
            "foundation_transfer_finetune_validation_measurement."
            "published_numbers_directly_comparable must be false"
        )
    for key in ("run_name", "summary_json", "evidence_json", "artifact_handle", "artifact_sha256"):
        if not measurement.get(key):
            errors.append(f"foundation_transfer_finetune_validation_measurement.{key} is required")

    metrics = _as_mapping(
        measurement.get("metrics"),
        "foundation_transfer_finetune_validation_measurement.metrics",
        errors,
    )
    if "decoded_rollout_nrmse" not in metrics:
        errors.append(
            "foundation_transfer_finetune_validation_measurement.metrics."
            "decoded_rollout_nrmse is required"
        )
    for task in ("advection1d", "burgers1d", "darcy2d"):
        key = f"task_{task}_decoded_rollout_nrmse"
        if key not in metrics:
            errors.append(
                f"foundation_transfer_finetune_validation_measurement.metrics.{key} " "is required"
            )

    training = _as_mapping(
        measurement.get("training"),
        "foundation_transfer_finetune_validation_measurement.training",
        errors,
    )
    if int(training.get("train_pairs", 0)) <= 0:
        errors.append(
            "foundation_transfer_finetune_validation_measurement.training.train_pairs "
            "must be positive"
        )
    if training.get("train_split") == "test":
        errors.append(
            "foundation_transfer_finetune_validation_measurement.training.train_split "
            "must not be test"
        )

    trainable = _as_mapping(
        measurement.get("trainable_parameters"),
        "foundation_transfer_finetune_validation_measurement.trainable_parameters",
        errors,
    )
    if trainable.get("adapter_mode") != "scalar_layers":
        errors.append(
            "foundation_transfer_finetune_validation_measurement.trainable_parameters."
            "adapter_mode must be scalar_layers"
        )
    if int(trainable.get("trainable_parameter_count", 0)) <= 0:
        errors.append(
            "foundation_transfer_finetune_validation_measurement.trainable_parameters."
            "trainable_parameter_count must be positive"
        )

    source_commits = _as_mapping(
        measurement.get("source_commits"),
        "foundation_transfer_finetune_validation_measurement.source_commits",
        errors,
    )
    if not source_commits.get("poseidon_official_repo"):
        errors.append(
            "foundation_transfer_finetune_validation_measurement.source_commits."
            "poseidon_official_repo is required"
        )
    checkpoint = _as_mapping(
        measurement.get("pretrained_checkpoint"),
        "foundation_transfer_finetune_validation_measurement.pretrained_checkpoint",
        errors,
    )
    if checkpoint.get("sha256_status") != "matched":
        errors.append(
            "foundation_transfer_finetune_validation_measurement.pretrained_checkpoint."
            "sha256_status must be matched"
        )
    if not checkpoint.get("sha256"):
        errors.append(
            "foundation_transfer_finetune_validation_measurement.pretrained_checkpoint."
            "sha256 is required"
        )

    decision = _as_mapping(
        measurement.get("decision"),
        "foundation_transfer_finetune_validation_measurement.decision",
        errors,
    )
    if decision.get("result") != "stopped_scalar_only_path_no_held_out_test":
        errors.append(
            "foundation_transfer_finetune_validation_measurement.decision.result must be "
            "stopped_scalar_only_path_no_held_out_test"
        )


def validate_mapping(
    mapping: Mapping[str, Any],
    claim_evidence: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []

    if mapping.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if mapping.get("mapping_status") not in ALLOWED_MAPPING_STATUSES:
        errors.append(
            "mapping_status must be one of "
            f"{sorted(ALLOWED_MAPPING_STATUSES)}, got {mapping.get('mapping_status')!r}"
        )

    terminology = _as_mapping(mapping.get("terminology"), "terminology", errors)
    for key in ("claim_protocol", "external_paper_reproduction"):
        if not terminology.get(key):
            errors.append(f"terminology.{key} is required")

    claim_doc = _as_mapping(
        claim_evidence.get("claim_documentation"), "claim_documentation", errors
    )
    strong_baseline = _as_mapping(
        claim_evidence.get("strong_baseline_comparison"),
        "strong_baseline_comparison",
        errors,
    )
    protocol = _as_mapping(mapping.get("claim_protocol"), "claim_protocol", errors)
    for key in REQUIRED_PROTOCOL_KEYS:
        if key not in protocol:
            errors.append(f"claim_protocol.{key} is required")

    task_metric_keys = _claim_task_metric_keys(claim_evidence, errors)
    task_set = _claim_task_set(task_metric_keys, errors)
    protocol_checks = {
        "run_name": claim_doc.get("run_name"),
        "split": claim_doc.get("split"),
        "metric_name": claim_doc.get("metric_name"),
        "metric_value": claim_doc.get("metric_value"),
        "summary_json": claim_doc.get("summary_json"),
        "held_out_test_ledger_json": claim_doc.get("held_out_test_ledger_json"),
        "artifact_sha256": claim_doc.get("artifact_sha256"),
    }
    for key, expected in protocol_checks.items():
        if expected is not None and not _close_enough(protocol.get(key), expected):
            errors.append(f"claim_protocol.{key} must match claim_documentation.{key}")

    if sorted(protocol.get("task_metric_keys", [])) != task_metric_keys:
        errors.append("claim_protocol.task_metric_keys must match candidate_evidence metrics")
    if sorted(protocol.get("task_set", [])) != task_set:
        errors.append("claim_protocol.task_set must match candidate_evidence task metrics")

    command = str(claim_doc.get("command", ""))
    rollout_steps = protocol.get("rollout_steps")
    max_eval_samples = protocol.get("max_eval_samples")
    if isinstance(rollout_steps, int) and f"--decoded-rollout-steps {rollout_steps}" not in command:
        errors.append("claim_protocol.rollout_steps must match the claim command")
    if isinstance(max_eval_samples, int) and f"data.max_samples={max_eval_samples}" not in command:
        errors.append("claim_protocol.max_eval_samples must match the claim command")

    _validate_scoped_claim_variants(
        mapping_variants_value=mapping.get("scoped_claim_variants"),
        claim_evidence=claim_evidence,
        protocol=protocol,
        errors=errors,
    )

    local_baseline = _as_mapping(
        mapping.get("local_strong_baseline"), "local_strong_baseline", errors
    )
    local_checks = {
        "run_name": strong_baseline.get("baseline_run_name"),
        "baseline_family": strong_baseline.get("baseline_family"),
        "split": strong_baseline.get("split"),
        "metric_name": strong_baseline.get("metric_name"),
        "metric_value": strong_baseline.get("baseline_metric_value"),
        "summary_json": strong_baseline.get("baseline_summary_json"),
        "artifact_sha256": strong_baseline.get("baseline_artifact_sha256"),
    }
    for key, expected in local_checks.items():
        if expected is not None and not _close_enough(local_baseline.get(key), expected):
            errors.append(f"local_strong_baseline.{key} must match strong_baseline_comparison")
    if local_baseline.get("is_external_paper_reproduction") is not False:
        errors.append("local_strong_baseline.is_external_paper_reproduction must be false")

    sources = _as_list(mapping.get("external_sources"), "external_sources", errors)
    source_ids: set[str] = set()
    source_kinds: set[str] = set()
    for index, source_value in enumerate(sources):
        source = _as_mapping(source_value, f"external_sources[{index}]", errors)
        source_id = source.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            errors.append(f"external_sources[{index}].source_id is required")
            continue
        if source_id in source_ids:
            errors.append(f"duplicate external source_id: {source_id}")
        source_ids.add(source_id)
        source_kinds.add(str(source.get("kind", "")))
        if source.get("primary_source") is not True:
            errors.append(f"external_sources[{index}] must be marked primary_source=true")
        if not str(source.get("url", "")).startswith("https://"):
            errors.append(f"external_sources[{index}].url must be an https URL")
    if "official_repo" not in source_kinds:
        errors.append("external_sources must include at least one official_repo")
    if "paper" not in source_kinds:
        errors.append("external_sources must include at least one paper")

    candidates = _as_list(mapping.get("baseline_candidates"), "baseline_candidates", errors)
    candidates_by_id: dict[str, Mapping[str, Any]] = {}
    selected_primary_count = 0
    for index, candidate_value in enumerate(candidates):
        candidate = _as_mapping(candidate_value, f"baseline_candidates[{index}]", errors)
        candidate_id = candidate.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            errors.append(f"baseline_candidates[{index}].candidate_id is required")
            continue
        if candidate_id in candidates_by_id:
            errors.append(f"duplicate baseline candidate_id: {candidate_id}")
        candidates_by_id[candidate_id] = candidate
        if candidate.get("status") == "selected_primary_reproduction_path":
            selected_primary_count += 1
            implementation_status = candidate.get("implementation_status")
            if implementation_status not in ALLOWED_IMPLEMENTATION_STATUSES:
                errors.append(
                    f"{candidate_id}.implementation_status must be one of "
                    f"{sorted(ALLOWED_IMPLEMENTATION_STATUSES)}"
                )
            command_template = _as_list(
                candidate.get("reproduction_command_template"),
                f"{candidate_id}.reproduction_command_template",
                errors,
            )
            if "scripts/run_external_neuraloperator_fno_baseline.py" not in command_template:
                errors.append(
                    f"{candidate_id}.reproduction_command_template must use the external FNO runner"
                )
            if selected_mapping := candidate.get("protocol_mapping"):
                if (
                    isinstance(selected_mapping, Mapping)
                    and selected_mapping.get("eval_split") == "test"
                    and "--allow-held-out-test-eval" not in command_template
                ):
                    errors.append(
                        f"{candidate_id}.reproduction_command_template must explicitly opt into "
                        "held-out test evaluation"
                    )
        for source_ref in _as_list(
            candidate.get("source_refs"),
            f"baseline_candidates[{index}].source_refs",
            errors,
        ):
            if source_ref not in source_ids:
                errors.append(f"{candidate_id} references unknown source_ref: {source_ref}")
    if selected_primary_count != 1:
        errors.append("exactly one baseline candidate must be selected_primary_reproduction_path")

    selected_path = _as_mapping(
        mapping.get("selected_reproduction_path"), "selected_reproduction_path", errors
    )
    selected_id = selected_path.get("candidate_id")
    if selected_path.get("status") not in ALLOWED_SELECTED_STATUSES:
        errors.append(
            "selected_reproduction_path.status must be one of "
            f"{sorted(ALLOWED_SELECTED_STATUSES)}"
        )
    if selected_id not in candidates_by_id:
        errors.append(f"selected_reproduction_path.candidate_id is unknown: {selected_id!r}")
        selected_candidate: Mapping[str, Any] = {}
    else:
        selected_candidate = candidates_by_id[str(selected_id)]
        if selected_candidate.get("status") != "selected_primary_reproduction_path":
            errors.append("selected_reproduction_path must point to the selected primary candidate")
    for source_ref in _as_list(
        selected_path.get("source_refs"),
        "selected_reproduction_path.source_refs",
        errors,
    ):
        if source_ref not in source_ids:
            errors.append(f"selected_reproduction_path references unknown source_ref: {source_ref}")

    selected_mapping = _as_mapping(
        selected_candidate.get("protocol_mapping"),
        "selected candidate protocol_mapping",
        errors,
    )
    selected_checks = {
        "data_root": protocol.get("data_root"),
        "eval_split": protocol.get("split"),
        "metric_name": protocol.get("metric_name"),
        "max_eval_samples": protocol.get("max_eval_samples"),
        "rollout_steps": protocol.get("rollout_steps"),
        "lower_is_better": protocol.get("lower_is_better"),
    }
    for key, expected in selected_checks.items():
        if expected is not None and selected_mapping.get(key) != expected:
            errors.append(f"selected candidate protocol_mapping.{key} must match claim_protocol")
    if sorted(selected_mapping.get("tasks", [])) != sorted(protocol.get("task_set", [])):
        errors.append(
            "selected candidate protocol_mapping.tasks must match claim_protocol.task_set"
        )
    if selected_mapping.get("published_numbers_directly_comparable") is not False:
        errors.append("selected candidate must not mark published numbers as directly comparable")

    if mapping.get("mapping_status") == "external_reproduction_measured":
        if selected_path.get("status") != "measured_complete":
            errors.append(
                "external_reproduction_measured requires selected_reproduction_path measured_complete"
            )
        if selected_candidate.get("implementation_status") != "external_adapter_measured_complete":
            errors.append(
                "external_reproduction_measured requires selected primary implementation_status "
                "external_adapter_measured_complete"
            )
        test_measurements = _as_list(
            selected_candidate.get("test_measurements"),
            "selected candidate test_measurements",
            errors,
        )
        if not test_measurements:
            errors.append(
                "external_reproduction_measured requires selected primary test_measurements"
            )
        _validate_test_measurements(
            measurements=test_measurements,
            label="test_measurements",
            protocol=protocol,
            errors=errors,
        )

    for candidate_id, candidate in candidates_by_id.items():
        if candidate is selected_candidate:
            continue
        if candidate.get("implementation_status") != "external_adapter_measured_complete":
            continue
        test_measurements = _as_list(
            candidate.get("test_measurements"),
            f"{candidate_id}.test_measurements",
            errors,
        )
        if not test_measurements:
            errors.append(f"{candidate_id}.test_measurements is required for measured candidate")
        _validate_test_measurements(
            measurements=test_measurements,
            label=f"{candidate_id}.test_measurements",
            protocol=protocol,
            errors=errors,
        )

    contract = _as_mapping(mapping.get("reproduction_contract"), "reproduction_contract", errors)
    if contract.get("allows_test_tuning") is not False:
        errors.append("reproduction_contract.allows_test_tuning must be false")
    if contract.get("requires_train_only_model_selection") is not True:
        errors.append("reproduction_contract.requires_train_only_model_selection must be true")
    required_artifacts = set(
        item
        for item in _as_list(
            contract.get("required_output_artifacts"),
            "reproduction_contract.required_output_artifacts",
            errors,
        )
        if isinstance(item, str)
    )
    for required in ("summary_json", "artifact_sha256", "held_out_ledger_reference"):
        if required not in required_artifacts:
            errors.append(f"reproduction_contract.required_output_artifacts missing {required}")

    foundation_transfer_contract = _as_mapping(
        mapping.get("foundation_transfer_contract"),
        "foundation_transfer_contract",
        errors,
    )
    _validate_foundation_transfer_contract(foundation_transfer_contract, errors)

    foundation_transfer_adapter_gate = _as_mapping(
        mapping.get("foundation_transfer_adapter_gate"),
        "foundation_transfer_adapter_gate",
        errors,
    )
    _validate_foundation_transfer_adapter_gate(foundation_transfer_adapter_gate, errors)

    foundation_transfer_validation_measurement = _as_mapping(
        mapping.get("foundation_transfer_validation_measurement"),
        "foundation_transfer_validation_measurement",
        errors,
    )
    _validate_foundation_transfer_validation_measurement(
        foundation_transfer_validation_measurement,
        errors,
    )

    foundation_transfer_finetune_validation_measurement = _as_mapping(
        mapping.get("foundation_transfer_finetune_validation_measurement"),
        "foundation_transfer_finetune_validation_measurement",
        errors,
    )
    _validate_foundation_transfer_finetune_validation_measurement(
        foundation_transfer_finetune_validation_measurement,
        errors,
    )

    decision = _as_mapping(mapping.get("comparability_decision"), "comparability_decision", errors)
    if decision.get("published_numbers_directly_comparable") is not False:
        errors.append("comparability_decision.published_numbers_directly_comparable must be false")
    external_claim_status = decision.get("external_claim_status")
    if mapping.get("mapping_status") == "external_reproduction_measured":
        if external_claim_status != "external_reproduction_measured":
            errors.append(
                "comparability_decision.external_claim_status must be "
                "external_reproduction_measured when mapping_status is external_reproduction_measured"
            )
    elif external_claim_status != "external_reproduction_path_selected_not_measured":
        errors.append(
            "comparability_decision.external_claim_status must remain "
            "external_reproduction_path_selected_not_measured until a measured run is recorded"
        )

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mapping-json",
        type=Path,
        default=Path("docs/claim_evidence/external_baseline_mapping.json"),
    )
    parser.add_argument(
        "--claim-evidence-json",
        type=Path,
        default=Path("docs/claim_evidence/universal_sota_claim_evidence.json"),
    )
    args = parser.parse_args(argv)

    mapping = load_json(args.mapping_json)
    claim_evidence = load_json(args.claim_evidence_json)
    errors = validate_mapping(mapping, claim_evidence)
    result = {
        "status": "valid" if not errors else "invalid",
        "mapping_json": str(args.mapping_json),
        "claim_evidence_json": str(args.claim_evidence_json),
        "errors": errors,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
