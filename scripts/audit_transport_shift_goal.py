#!/usr/bin/env python
from __future__ import annotations

"""Audit the benchmark-clean constant transport-shift goal from result artifacts."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import h5py


REQUIREMENTS = (
    "real_light_v1_train_val_accessed",
    "train_only_shift_fit",
    "validation_sota_guard_passed",
    "exactly_one_held_out_test_after_validation",
    "results_recorded",
)

PASS_STATUSES_BY_MODE = {
    "report": None,
    "test-ready": {"test_ready", "achieved"},
    "achieved": {"achieved"},
}


def _load_optional_json(path: str | None) -> tuple[dict[str, Any] | None, str | None]:
    if not path:
        return None, None
    json_path = Path(path)
    if not json_path.exists():
        return None, f"missing: {json_path}"
    return json.loads(json_path.read_text(encoding="utf-8")), str(json_path)


def _requirement(name: str, status: str, evidence: str) -> dict[str, str]:
    if name not in REQUIREMENTS:
        raise ValueError(f"unknown requirement: {name}")
    return {"name": name, "status": status, "evidence": evidence}


def _extract_best_shifts(gate: Mapping[str, Any] | None) -> dict[str, int]:
    if not gate:
        return {}
    diagnostic = gate.get("diagnostic", {})
    best_shifts = diagnostic.get("best_shifts", {})
    return {str(split): int(shift) for split, shift in best_shifts.items()}


def _json_safe_attrs(attrs: Mapping[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in attrs.items():
        safe[key] = value.tolist() if hasattr(value, "tolist") else value
    return safe


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inspect_hdf5_split(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        datasets: dict[str, Any] = {}
        for key, value in handle.items():
            if isinstance(value, h5py.Dataset):
                datasets[key] = {
                    "shape": [int(dim) for dim in value.shape],
                    "dtype": str(value.dtype),
                    "attrs": _json_safe_attrs(value.attrs),
                }
        sample_aligned = [
            key
            for key, value in datasets.items()
            if key != "data" and datasets.get("data") and value["shape"][:1] == datasets["data"]["shape"][:1]
        ]
        return {
            "path": str(path),
            "exists": True,
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
            "file_attrs": _json_safe_attrs(handle.attrs),
            "datasets": datasets,
            "sample_aligned_auxiliary_datasets": sample_aligned,
            "has_parameter_metadata": bool(handle.attrs) or any(
                key != "data" or bool(value["attrs"]) for key, value in datasets.items()
            ),
        }


def _inspect_data_schema(data_root: str, task: str, splits: list[str]) -> dict[str, Any]:
    root = Path(data_root)
    split_records: dict[str, Any] = {}
    for split in splits:
        path = root / f"{task}_{split}.h5"
        if not path.exists():
            split_records[split] = {"path": str(path), "exists": False}
            continue
        split_records[split] = _inspect_hdf5_split(path)
    return {
        "data_root": str(root),
        "task": task,
        "splits": split_records,
        "parameter_metadata_available": any(
            bool(record.get("has_parameter_metadata")) for record in split_records.values() if record.get("exists")
        ),
    }


def _test_result_count(gate: Mapping[str, Any] | None) -> int:
    if not gate:
        return 0
    test_payload = gate.get("test")
    if not test_payload:
        return 0
    if isinstance(test_payload, list):
        return len(test_payload)
    return 1


def audit_goal(args: argparse.Namespace) -> dict[str, Any]:
    gate, gate_path = _load_optional_json(args.official_gate_json)
    compatibility, compatibility_path = _load_optional_json(args.compatible_window_selection_json)
    data_schema = _inspect_data_schema(args.data_root, args.task, args.schema_splits)

    missing_inputs = [
        message
        for message in (
            None if gate is not None else f"missing official gate JSON: {args.official_gate_json}",
            None
            if compatibility is not None
            else f"missing compatibility JSON: {args.compatible_window_selection_json}",
        )
        if message
    ]

    validation_guard = ((gate or {}).get("fit") or {}).get("validation_guard") or {}
    guard_passed = bool(validation_guard.get("passed"))
    test_eligible = bool((gate or {}).get("test_eligible"))
    test_result_count = _test_result_count(gate)
    has_test_result = test_result_count > 0
    leaked_test_result = has_test_result and not test_eligible
    multiple_test_results = test_result_count > 1
    compatible = bool((compatibility or {}).get("compatible"))
    common_shifts = list((compatibility or {}).get("common_shifts") or [])
    best_shifts = _extract_best_shifts(gate)

    if missing_inputs:
        status = "missing_evidence"
        test_allowed = False
        blockers = missing_inputs
    elif leaked_test_result:
        status = "invalid_test_leakage"
        test_allowed = False
        blockers = ["held-out test result is present even though validation did not authorize test evaluation"]
    elif multiple_test_results:
        status = "invalid_multiple_tests"
        test_allowed = False
        blockers = ["more than one held-out test result is present"]
    elif not compatible:
        status = "blocked_incompatible_splits"
        test_allowed = False
        blockers = ["full-source train/val/test scans have no common constant shift"]
    elif not guard_passed:
        status = "validation_failed"
        test_allowed = False
        blockers = ["train-fitted constant shift failed the validation SOTA guard"]
    elif test_eligible and not has_test_result:
        status = "test_ready"
        test_allowed = True
        blockers = ["validation passed and exactly one held-out test still needs to be run"]
    elif test_eligible and has_test_result:
        status = "achieved"
        test_allowed = False
        blockers = []
    else:
        status = "blocked"
        test_allowed = False
        blockers = ["gate did not authorize held-out test"]

    requirements = [
        _requirement(
            "real_light_v1_train_val_accessed",
            "satisfied" if gate else "missing",
            gate_path or "official train/val gate artifact missing",
        ),
        _requirement(
            "train_only_shift_fit",
            "satisfied" if gate and ((gate.get("fit") or {}).get("selected_train_shift") is not None) else "missing",
            f"selected_train_shift={((gate or {}).get('fit') or {}).get('selected_train_shift')}",
        ),
        _requirement(
            "validation_sota_guard_passed",
            "satisfied" if guard_passed else "failed",
            f"validation_guard={validation_guard}",
        ),
        _requirement(
            "exactly_one_held_out_test_after_validation",
            "satisfied" if has_test_result and test_eligible and test_result_count == 1 else "violated"
            if leaked_test_result or multiple_test_results
            else "blocked",
            "one test result present after gate eligibility"
            if has_test_result and test_eligible and test_result_count == 1
            else "test result present before gate eligibility"
            if leaked_test_result
            else f"{test_result_count} held-out test results present"
            if multiple_test_results
            else "held-out test not run because gate did not pass",
        ),
        _requirement(
            "results_recorded",
            "satisfied" if gate and compatibility else "partial",
            ", ".join(path for path in (gate_path, compatibility_path) if path) or "missing result artifacts",
        ),
    ]

    return {
        "status": status,
        "test_allowed": test_allowed,
        "blockers": blockers,
        "requirements": requirements,
        "constant_shift": {
            "official_gate_json": gate_path,
            "compatible_window_selection_json": compatibility_path,
            "best_shifts": best_shifts,
            "selected_train_shift": ((gate or {}).get("fit") or {}).get("selected_train_shift"),
            "selected_validation": ((gate or {}).get("fit") or {}).get("selected_validation"),
            "validation_guard": validation_guard,
            "test_eligible": test_eligible,
            "has_test_result": has_test_result,
            "test_result_count": test_result_count,
            "compatible_full_source_windows": compatible,
            "common_full_source_shifts": common_shifts,
            "histograms": (compatibility or {}).get("histograms"),
        },
        "held_out_test_policy": {
            "test_eligible": test_eligible,
            "test_result_count": test_result_count,
            "test_allowed_next": bool(test_eligible and not has_test_result),
            "leaked_test_result": leaked_test_result,
            "exactly_one_test_after_validation": bool(test_eligible and test_result_count == 1),
        },
        "data_schema": data_schema,
        "recommendation": (
            "Rebuild a benchmark split with shared transport-rate support, or retire the constant-shift "
            "target and pursue a learned/state-conditioned transport mechanism."
        ),
    }


def exit_code_for_status(status: str, mode: str) -> int:
    allowed = PASS_STATUSES_BY_MODE[mode]
    if allowed is None:
        return 0
    return 0 if status in allowed else 2


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit benchmark-clean transport shift goal evidence")
    parser.add_argument(
        "--official-gate-json",
        default="reports/research/sota_loop/transport_shift_gate.json",
        help="Result from scripts/run_transport_shift_gate.py",
    )
    parser.add_argument(
        "--compatible-window-selection-json",
        default="reports/research/sota_loop/remote_transport_shift_candidate_all_splits/compatible_window_selection.json",
        help="Result from scripts/select_transport_compatible_windows.py",
    )
    parser.add_argument("--data-root", default="data/pdebench", help="Directory containing light-v1 HDF5 files")
    parser.add_argument("--task", default="advection1d")
    parser.add_argument("--schema-splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument(
        "--require-status",
        choices=tuple(PASS_STATUSES_BY_MODE),
        default="report",
        help=(
            "Exit policy: report always returns 0; test-ready returns 0 only when a held-out test is allowed "
            "or already recorded; achieved returns 0 only after a passed gate and recorded test."
        ),
    )
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    record = audit_goal(args)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))
    raise SystemExit(exit_code_for_status(str(record["status"]), args.require_status))


if __name__ == "__main__":
    main()
