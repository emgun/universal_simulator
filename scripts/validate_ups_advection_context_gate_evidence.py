#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import tarfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

DEFAULT_EVIDENCE_JSON = "docs/claim_evidence/ups_advection_context_delay_val_gate_evidence.json"
FORBIDDEN_TEST_COMMAND_TOKENS = (
    "--extra-eval-split test",
    "--eval-split test",
    "--allow-held-out-test-eval",
    "--held-out-test-ledger-json",
    "data.split=test",
)


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


def _close_enough(left: Any, right: Any) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-15)
    return left == right


def _summary_member_name(run: Mapping[str, Any]) -> str:
    summary_json = str(run.get("summary_json", ""))
    run_dir = Path(summary_json).parent.name
    return f"{run_dir}/summary.json"


def _artifact_summaries(
    *,
    artifact_path: Path,
    expected_members: set[str],
    errors: list[str],
) -> dict[str, Mapping[str, Any]]:
    if not artifact_path.exists():
        errors.append(f"artifact.path does not exist: {artifact_path}")
        return {}

    summaries: dict[str, Mapping[str, Any]] = {}
    try:
        with tarfile.open(artifact_path, "r:gz") as archive:
            members = set(archive.getnames())
            missing = sorted(expected_members - members)
            unexpected = sorted(members - expected_members)
            for member in missing:
                errors.append(f"artifact missing summary member: {member}")
            for member in unexpected:
                errors.append(f"artifact contains unexpected member: {member}")
            for member in sorted(expected_members & members):
                extracted = archive.extractfile(member)
                if extracted is None:
                    errors.append(f"artifact member is not a file: {member}")
                    continue
                loaded = json.loads(extracted.read().decode("utf-8"))
                if not isinstance(loaded, Mapping):
                    errors.append(f"artifact member must contain a JSON object: {member}")
                    continue
                summaries[member] = loaded
    except (tarfile.TarError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        errors.append(f"artifact could not be read as summary tarball: {exc}")
    return summaries


def _command_uses_test(command: str) -> bool:
    normalized = " ".join(command.split())
    return any(token in normalized for token in FORBIDDEN_TEST_COMMAND_TOKENS)


def _validate_top_level(evidence: Mapping[str, Any], errors: list[str]) -> None:
    if evidence.get("measurement_type") != "ups_advection_context_delay_validation_gate":
        errors.append("measurement_type must be ups_advection_context_delay_validation_gate")
    if evidence.get("held_out_test_used") is not False:
        errors.append("held_out_test_used must be false")
    if evidence.get("test_split_accessed") is not False:
        errors.append("test_split_accessed must be false")
    if evidence.get("split") != "val":
        errors.append("split must be val")
    if evidence.get("decoded_rollout_steps") != 16:
        errors.append("decoded_rollout_steps must be 16")

    selection_rule = _as_mapping(
        evidence.get("selection_rule"),
        "selection_rule",
        errors,
    )
    if selection_rule.get("held_out_test_requires_separate_protocol_review") is not True:
        errors.append("selection_rule.held_out_test_requires_separate_protocol_review must be true")
    if selection_rule.get("direction") != "lower_is_better":
        errors.append("selection_rule.direction must be lower_is_better")


def _validate_artifact(
    *,
    evidence: Mapping[str, Any],
    runs: list[Mapping[str, Any]],
    repo_root: Path,
    errors: list[str],
) -> dict[str, Mapping[str, Any]]:
    artifact = _as_mapping(evidence.get("artifact"), "artifact", errors)
    artifact_path_value = artifact.get("path")
    if not artifact_path_value:
        errors.append("artifact.path is required")
        return {}
    artifact_path = repo_root / str(artifact_path_value)
    if artifact_path.exists():
        actual_sha = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        if actual_sha != artifact.get("sha256"):
            errors.append("artifact.sha256 does not match artifact bytes")

    expected_members = {_summary_member_name(run) for run in runs}
    listed_contents = set(
        str(value) for value in _as_list(artifact.get("contents"), "artifact.contents", errors)
    )
    if listed_contents != expected_members:
        errors.append("artifact.contents must match run summary members")

    return _artifact_summaries(
        artifact_path=artifact_path,
        expected_members=expected_members,
        errors=errors,
    )


def _validate_run(
    *,
    run: Mapping[str, Any],
    index: int,
    summaries: Mapping[str, Mapping[str, Any]],
    errors: list[str],
) -> None:
    label = f"runs[{index}]"
    command = str(run.get("command", ""))
    if _command_uses_test(command):
        errors.append(f"{label}.command must not request held-out test evaluation")
    if "--override data.split=val" not in command:
        errors.append(f"{label}.command must pin data.split=val")

    for key in (
        "decoded_rollout_nrmse",
        "task_advection1d_decoded_rollout_nrmse",
        "task_burgers1d_decoded_rollout_nrmse",
        "task_darcy2d_decoded_rollout_nrmse",
    ):
        if not isinstance(run.get(key), (int, float)):
            errors.append(f"{label}.{key} must be numeric")

    summary = summaries.get(_summary_member_name(run))
    if summary is None:
        return
    if summary.get("run_name") != run.get("name"):
        errors.append(f"{label}.name must match artifact summary run_name")
    metrics = _as_mapping(summary.get("metrics"), f"{label}.summary.metrics", errors)
    for key in (
        "decoded_rollout_nrmse",
        "task_advection1d_decoded_rollout_nrmse",
        "task_burgers1d_decoded_rollout_nrmse",
        "task_darcy2d_decoded_rollout_nrmse",
    ):
        if not _close_enough(run.get(key), metrics.get(key)):
            errors.append(f"{label}.{key} must match artifact summary metric")


def _validate_selection(
    evidence: Mapping[str, Any], runs: list[Mapping[str, Any]], errors: list[str]
) -> None:
    if not runs:
        errors.append("runs must not be empty")
        return

    selection_rule = _as_mapping(
        evidence.get("selection_rule"),
        "selection_rule",
        errors,
    )
    baseline_name = selection_rule.get("baseline_run")
    by_name = {str(run.get("name")): run for run in runs}
    baseline = by_name.get(str(baseline_name))
    if baseline is None:
        errors.append("selection_rule.baseline_run must reference a run")
        return

    best = min(runs, key=lambda run: float(run.get("decoded_rollout_nrmse", float("inf"))))
    reported = _as_mapping(
        evidence.get("best_validation_candidate"),
        "best_validation_candidate",
        errors,
    )
    if reported.get("name") != best.get("name"):
        errors.append(
            "best_validation_candidate.name must identify the lowest validation metric run"
        )

    baseline_metric = float(baseline.get("decoded_rollout_nrmse", float("nan")))
    best_metric = float(best.get("decoded_rollout_nrmse", float("nan")))
    improvement = (baseline_metric - best_metric) / baseline_metric
    minimum = float(selection_rule.get("minimum_useful_relative_improvement", 0.0))
    if float(reported.get("relative_improvement_vs_current", float("nan"))) < minimum:
        errors.append(
            "best_validation_candidate.relative_improvement_vs_current must clear the selection rule"
        )
    if not _close_enough(reported.get("relative_improvement_vs_current"), improvement):
        errors.append("best_validation_candidate.relative_improvement_vs_current is inconsistent")
    if not _close_enough(reported.get("decoded_rollout_nrmse"), best.get("decoded_rollout_nrmse")):
        errors.append("best_validation_candidate.decoded_rollout_nrmse must match best run")


def _validate_context_delay_monotonicity(
    runs: list[Mapping[str, Any]],
    errors: list[str],
) -> None:
    transport_only = [
        run
        for run in runs
        if run.get("families") == ["transport"] and isinstance(run.get("context_transitions"), int)
    ]
    by_delay = {
        int(run["context_transitions"]): float(run["decoded_rollout_nrmse"])
        for run in transport_only
    }
    required_delays = [1, 2, 4, 8]
    if sorted(by_delay) != required_delays:
        errors.append("transport-only context-delay runs must include CT1, CT2, CT4, and CT8")
        return
    values = [by_delay[delay] for delay in required_delays]
    if any(left > right for left, right in zip(values, values[1:])):
        errors.append(
            "transport-only context-delay metrics must be non-decreasing as delay increases"
        )


def validate_evidence(evidence: Mapping[str, Any], *, repo_root: Path) -> list[str]:
    errors: list[str] = []
    _validate_top_level(evidence, errors)

    run_values = _as_list(evidence.get("runs"), "runs", errors)
    runs = [_as_mapping(run, f"runs[{index}]", errors) for index, run in enumerate(run_values)]
    summaries = _validate_artifact(
        evidence=evidence,
        runs=runs,
        repo_root=repo_root,
        errors=errors,
    )
    for index, run in enumerate(runs):
        _validate_run(run=run, index=index, summaries=summaries, errors=errors)
    _validate_selection(evidence, runs, errors)
    _validate_context_delay_monotonicity(runs, errors)

    decision = _as_mapping(evidence.get("decision"), "decision", errors)
    if decision.get("status") != "validation_gate_cleared_for_protocol_review":
        errors.append("decision.status must be validation_gate_cleared_for_protocol_review")
    if "held-out" not in str(decision.get("held_out_test_next_step", "")):
        errors.append("decision.held_out_test_next_step must describe held-out policy")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate UPS advection context-delay validation-gate evidence"
    )
    parser.add_argument("--evidence-json", default=DEFAULT_EVIDENCE_JSON)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    evidence = load_json(repo_root / args.evidence_json)
    errors = validate_evidence(evidence, repo_root=repo_root)
    record = {
        "evidence_json": args.evidence_json,
        "errors": errors,
        "status": "valid" if not errors else "invalid",
    }
    print(json.dumps(record, indent=2, sort_keys=True))
    raise SystemExit(0 if not errors else 2)


if __name__ == "__main__":
    main()
