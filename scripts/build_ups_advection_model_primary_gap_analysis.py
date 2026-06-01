#!/usr/bin/env python
from __future__ import annotations

"""Build a no-heldout-rerun gap analysis for the failed UPS advection candidate."""

import argparse
import hashlib
import json
import sys
import tarfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.validate_ups_advection_model_gate_evidence import load_json

DEFAULT_OUTPUT_JSON = "docs/claim_evidence/ups_advection_model_primary_gap_analysis.json"
DEFAULT_HELDOUT_EVIDENCE_JSON = (
    "docs/claim_evidence/ups_advection_model_primary_heldout_light_v1_evidence.json"
)
DEFAULT_CLAIM_EVIDENCE_JSON = "docs/claim_evidence/universal_sota_claim_evidence.json"
EXPECTED_MEASUREMENT_TYPE = "ups_advection_model_primary_gap_analysis"

METRIC_KEYS = (
    "decoded_rollout_nrmse",
    "decoded_step1_nrmse",
    "decoded_h4_nrmse",
    "decoded_h16_nrmse",
    "task_advection1d_decoded_rollout_nrmse",
    "task_advection1d_decoded_step1_nrmse",
    "task_advection1d_decoded_h4_nrmse",
    "task_advection1d_decoded_h16_nrmse",
    "task_burgers1d_decoded_rollout_nrmse",
    "task_burgers1d_decoded_step1_nrmse",
    "task_burgers1d_decoded_h4_nrmse",
    "task_burgers1d_decoded_h16_nrmse",
    "task_darcy2d_decoded_rollout_nrmse",
    "task_darcy2d_decoded_step1_nrmse",
    "task_darcy2d_decoded_h4_nrmse",
    "task_darcy2d_decoded_h16_nrmse",
)

ROLLOUT_KEYS = tuple(key for key in METRIC_KEYS if "rollout" in key)
TASK_ROLLOUT_KEYS = tuple(key for key in ROLLOUT_KEYS if key.startswith("task_"))
HORIZON_KEYS = tuple(key for key in METRIC_KEYS if "_h" in key or "_step1_" in key)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_tar_json(path: Path, member: str) -> dict[str, Any]:
    with tarfile.open(path, mode="r:gz") as archive:
        extracted = archive.extractfile(member)
        if extracted is None:
            raise FileNotFoundError(f"{member} not found in {path}")
        payload = json.load(extracted)
    if not isinstance(payload, dict):
        raise TypeError(f"{member} in {path} must contain a JSON object")
    return payload


def _metric_subset(metrics: Mapping[str, Any]) -> dict[str, float]:
    return {
        key: float(metrics[key])
        for key in METRIC_KEYS
        if isinstance(metrics.get(key), (int, float))
    }


def _delta_block(
    left: Mapping[str, float], right: Mapping[str, float]
) -> dict[str, dict[str, float]]:
    block: dict[str, dict[str, float]] = {}
    for key in METRIC_KEYS:
        if key not in left or key not in right:
            continue
        left_value = float(left[key])
        right_value = float(right[key])
        absolute = left_value - right_value
        block[key] = {
            "left": left_value,
            "right": right_value,
            "absolute": absolute,
            "relative_to_right": absolute / right_value if right_value else 0.0,
        }
    return block


def _validation_to_test_gap(
    validation: Mapping[str, float], test: Mapping[str, float]
) -> dict[str, dict[str, float]]:
    block: dict[str, dict[str, float]] = {}
    for key in METRIC_KEYS:
        if key not in validation or key not in test:
            continue
        validation_value = float(validation[key])
        test_value = float(test[key])
        absolute = test_value - validation_value
        block[key] = {
            "validation": validation_value,
            "test": test_value,
            "absolute": absolute,
            "relative_to_validation": absolute / validation_value if validation_value else 0.0,
        }
    return block


def _source_json(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _claim_summary_path(claim_evidence: Mapping[str, Any], suffix: str) -> str:
    documentation = claim_evidence.get("claim_documentation", {})
    if not isinstance(documentation, Mapping):
        raise TypeError("claim_documentation must be an object")
    files = documentation.get("artifact_files", [])
    if not isinstance(files, list):
        raise TypeError("claim_documentation.artifact_files must be a list")
    matches = [
        str(item.get("path"))
        for item in files
        if isinstance(item, Mapping) and str(item.get("path", "")).endswith(suffix)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one claim artifact path ending with {suffix}")
    return matches[0]


def _current_candidate(claim_evidence: Mapping[str, Any]) -> Mapping[str, Any]:
    candidates = claim_evidence.get("candidate_evidence", [])
    if not isinstance(candidates, list) or not candidates:
        raise TypeError("candidate_evidence must be a non-empty list")
    candidate = candidates[0]
    if not isinstance(candidate, Mapping):
        raise TypeError("candidate_evidence[0] must be an object")
    return candidate


def _dominant_task_regression(delta: Mapping[str, Mapping[str, float]]) -> dict[str, float | str]:
    task_deltas = {key: value for key, value in delta.items() if key in TASK_ROLLOUT_KEYS}
    key, value = max(task_deltas.items(), key=lambda item: item[1]["absolute"])
    return {
        "metric": key,
        "absolute": value["absolute"],
        "relative_to_ct8": value["relative_to_right"],
    }


def build_analysis(
    *,
    root: Path,
    heldout_evidence_json: Path,
    claim_evidence_json: Path,
) -> dict[str, Any]:
    heldout_evidence_path = root / heldout_evidence_json
    claim_evidence_path = root / claim_evidence_json
    heldout_evidence = load_json(heldout_evidence_path)
    claim_evidence = load_json(claim_evidence_path)

    artifact = heldout_evidence["artifact"]
    if not isinstance(artifact, Mapping):
        raise TypeError("heldout artifact must be an object")
    artifact_path = root / str(artifact["path"])
    contents = list(artifact["contents"])
    candidate_val_summary = _load_tar_json(artifact_path, str(contents[0]))
    candidate_test_summary = _load_tar_json(artifact_path, str(contents[1]))
    candidate_val_metrics = _metric_subset(candidate_val_summary["metrics"])
    candidate_test_metrics = _metric_subset(candidate_test_summary["metrics"])

    current_candidate = _current_candidate(claim_evidence)
    ct8_val_path = root / _claim_summary_path(claim_evidence, "/summary.json")
    ct8_test_path = root / str(current_candidate["summary_json"])
    ct8_val_summary = load_json(ct8_val_path)
    ct8_test_summary = load_json(ct8_test_path)
    ct8_val_metrics = _metric_subset(ct8_val_summary["metrics"])
    ct8_test_metrics = _metric_subset(ct8_test_summary["metrics"])

    candidate_vs_ct8_validation = _delta_block(candidate_val_metrics, ct8_val_metrics)
    candidate_vs_ct8_test = _delta_block(candidate_test_metrics, ct8_test_metrics)
    candidate_gap = _validation_to_test_gap(candidate_val_metrics, candidate_test_metrics)
    ct8_gap = _validation_to_test_gap(ct8_val_metrics, ct8_test_metrics)
    candidate_test_vs_ct8_test_horizon = {
        key: candidate_vs_ct8_test[key] for key in HORIZON_KEYS if key in candidate_vs_ct8_test
    }

    return {
        "schema_version": 1,
        "measurement_type": EXPECTED_MEASUREMENT_TYPE,
        "date": "2026-06-01",
        "analysis_scope": "no_heldout_rerun_validation_to_test_gap_analysis",
        "new_held_out_test_command_executed": False,
        "held_out_test_data_reaccessed": False,
        "uses_existing_held_out_test_summary": True,
        "source_files": {
            "heldout_evidence": _source_json(heldout_evidence_path, root),
            "claim_evidence": _source_json(claim_evidence_path, root),
            "candidate_artifact": {
                "path": artifact["path"],
                "sha256": _sha256(artifact_path),
                "recorded_sha256": artifact["sha256"],
                "bytes": artifact_path.stat().st_size,
                "recorded_bytes": artifact["bytes"],
            },
            "ct8_validation_summary": _source_json(ct8_val_path, root),
            "ct8_test_summary": _source_json(ct8_test_path, root),
        },
        "candidate": {
            "run_name": heldout_evidence["run_name"],
            "checkpoint_source": heldout_evidence["checkpoint_source"],
            "checkpoint_preference_stage": heldout_evidence["checkpoint_preference_stage"],
            "validation_metrics": candidate_val_metrics,
            "test_metrics": candidate_test_metrics,
            "validation_to_test_gap": candidate_gap,
        },
        "current_ct8_primary": {
            "run_name": current_candidate["run_name"],
            "validation_summary_json": str(ct8_val_path.relative_to(root)),
            "test_summary_json": current_candidate["summary_json"],
            "validation_metrics": ct8_val_metrics,
            "test_metrics": ct8_test_metrics,
            "validation_to_test_gap": ct8_gap,
        },
        "candidate_vs_current_ct8_primary": {
            "lower_is_better": True,
            "validation_delta_candidate_minus_ct8": candidate_vs_ct8_validation,
            "test_delta_candidate_minus_ct8": candidate_vs_ct8_test,
            "test_horizon_delta_candidate_minus_ct8": candidate_test_vs_ct8_test_horizon,
            "dominant_test_regression": _dominant_task_regression(candidate_vs_ct8_test),
            "candidate_beats_current_ct8_primary": False,
        },
        "diagnosis": {
            "primary_failure_mode": (
                "The no-context candidate improved the prior no-context validation baseline, "
                "but it was not competitive with the CT8 primary validation contract and "
                "failed held-out primarily on advection."
            ),
            "horizon_signal": (
                "On held-out advection, the no-context candidate is slightly better than CT8 "
                "at h4 but much worse at h16, which points to long-horizon transport phase "
                "tracking rather than Burgers or Darcy as the next target."
            ),
            "unchanged_task_signal": (
                "Burgers and Darcy held-out rollout errors are effectively unchanged versus CT8; "
                "the failed promotion is dominated by advection."
            ),
        },
        "decision": {
            "status": "gap_analysis_complete_no_promotion_no_rerun",
            "candidate_promoted": False,
            "do_not_repeat_held_out_test": True,
            "next_validation_only_gate": (
                "Before any future primary-contract held-out spend, require validation-only "
                "evidence that improves CT8 or a frozen no-context robustness threshold on "
                "overall, advection rollout, and long-horizon advection h16."
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=Path(DEFAULT_OUTPUT_JSON))
    parser.add_argument(
        "--heldout-evidence-json",
        type=Path,
        default=Path(DEFAULT_HELDOUT_EVIDENCE_JSON),
    )
    parser.add_argument(
        "--claim-evidence-json",
        type=Path,
        default=Path(DEFAULT_CLAIM_EVIDENCE_JSON),
    )
    args = parser.parse_args(argv)

    analysis = build_analysis(
        root=Path.cwd(),
        heldout_evidence_json=args.heldout_evidence_json,
        claim_evidence_json=args.claim_evidence_json,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(analysis, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": "wrote", "output_json": str(args.output_json)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
