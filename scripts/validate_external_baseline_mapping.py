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

    decision = _as_mapping(mapping.get("comparability_decision"), "comparability_decision", errors)
    if decision.get("published_numbers_directly_comparable") is not False:
        errors.append("comparability_decision.published_numbers_directly_comparable must be false")
    if decision.get("external_claim_status") != "external_reproduction_path_selected_not_measured":
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
