#!/usr/bin/env python
from __future__ import annotations

"""Build a guarded val/test root for the scoped model-side beta-head pretest."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.build_p2_parameter_full_task_root import (
    TASKS,
    _h5_summary,
    _link_or_copy,
    _source_root_for_task,
    _validate_advection_beta_provenance,
)
from scripts.validate_p2_model_side_beta_head_pretest_contract import (
    DEFAULT_CONTRACT_JSON,
    load_json,
    validate_contract,
)

MEASUREMENT_TYPE = "p2_model_side_beta_head_pretest_root_manifest"
SPLITS = ("val", "test")


def _validate_contract_boundary(
    *,
    contract_json: Path,
    repo_root: Path,
    measurement_key: str | None,
) -> dict[str, Any]:
    contract = load_json(contract_json)
    errors = validate_contract(contract, repo_root=repo_root)
    if errors:
        raise ValueError(f"pretest contract {contract_json} is invalid: " + "; ".join(errors))
    intended = contract.get("intended_held_out") or {}
    expected_key = str(intended.get("measurement_key") or "")
    if measurement_key and measurement_key != expected_key:
        raise ValueError(
            f"measurement key mismatch: expected {expected_key}, got {measurement_key}"
        )
    return {
        "contract_json": str(contract_json),
        "measurement_key": expected_key,
        "claim_contract_label": contract.get("protocol_decision", {}).get("claim_contract_label"),
    }


def build_pretest_root(
    *,
    base_root: Path,
    advection_root: Path,
    out_root: Path,
    manifest_json: Path,
    contract_json: Path,
    repo_root: Path = REPO_ROOT,
    measurement_key: str | None = None,
    allow_heldout_pretest_root: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    if not allow_heldout_pretest_root:
        raise ValueError(
            "Refusing to build a held-out pretest root without " "--allow-heldout-pretest-root"
        )

    contract_record = _validate_contract_boundary(
        contract_json=contract_json,
        repo_root=repo_root,
        measurement_key=measurement_key,
    )

    out_root.mkdir(parents=True, exist_ok=True)
    expected_names = {f"{task}_{split}.h5" for task in TASKS for split in SPLITS}
    unexpected_h5 = sorted(
        path for path in out_root.glob("*.h5") if path.name not in expected_names
    )
    if unexpected_h5 and not overwrite:
        names = ", ".join(path.name for path in unexpected_h5)
        raise ValueError(f"{out_root} contains unexpected HDF5 files: {names}")
    for path in unexpected_h5:
        path.unlink()

    sources: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        split_sources: dict[str, Any] = {}
        for task in TASKS:
            source_root_kind, source_root = _source_root_for_task(
                task, base_root=base_root, advection_root=advection_root
            )
            source_path = source_root / f"{task}_{split}.h5"
            if not source_path.exists():
                raise FileNotFoundError(source_path)
            beta_provenance = (
                _validate_advection_beta_provenance(source_path)
                if task == "advection1d"
                else {
                    "required": False,
                    "has_source_file_index": False,
                    "has_source_paths": False,
                }
            )
            destination = out_root / source_path.name
            transfer = _link_or_copy(source_path, destination, overwrite=overwrite)
            source_summary = _h5_summary(source_path)
            destination_summary = _h5_summary(destination)
            if source_summary["sha256"] != destination_summary["sha256"]:
                raise ValueError(f"{destination} hash differs from source {source_path}")
            split_sources[task] = {
                "task": task,
                "split": split,
                "source_root_kind": source_root_kind,
                "source_root": str(source_root),
                "source_path": str(source_path),
                "output_path": str(destination),
                "transfer": transfer,
                "bytes": destination_summary["bytes"],
                "sha256": destination_summary["sha256"],
                "source_sha256": source_summary["sha256"],
                "datasets": destination_summary["datasets"],
                "file_attrs": destination_summary["file_attrs"],
                "beta_provenance": beta_provenance,
            }
        sources[split] = split_sources

    manifest = {
        "measurement_type": MEASUREMENT_TYPE,
        "version": 1,
        "contract": contract_record,
        "tasks": list(TASKS),
        "splits": list(SPLITS),
        "base_root": str(base_root),
        "advection_root": str(advection_root),
        "out_root": str(out_root),
        "source_policy": (
            "Burgers and Darcy from base_root; Advection from official "
            "beta-provenance root; copies val and test only for the scoped "
            "pre-registered held-out workflow."
        ),
        "held_out_test_data_materialized": True,
        "held_out_test_used": False,
        "test_ledger_writes": [],
        "sources": sources,
    }
    manifest_json.parent.mkdir(parents=True, exist_ok=True)
    manifest_json.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, default=Path("data/pdebench"))
    parser.add_argument(
        "--advection-root",
        type=Path,
        default=Path("data/pdebench_official_advection_light"),
    )
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--contract-json", type=Path, default=Path(DEFAULT_CONTRACT_JSON))
    parser.add_argument("--measurement-key")
    parser.add_argument("--allow-heldout-pretest-root", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest = build_pretest_root(
        base_root=args.base_root,
        advection_root=args.advection_root,
        out_root=args.out_root,
        manifest_json=args.manifest_json,
        contract_json=args.contract_json,
        measurement_key=args.measurement_key,
        allow_heldout_pretest_root=args.allow_heldout_pretest_root,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
