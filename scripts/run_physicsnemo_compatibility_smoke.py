#!/usr/bin/env python
from __future__ import annotations

"""Write a dry PhysicsNeMo recipe-compatibility smoke manifest.

This is not a PhysicsNeMo performance measurement. The default path avoids a
mandatory PhysicsNeMo install and records the recipe contract that must be
satisfied before any live validation metric or held-out test comparison.
"""

import argparse
import importlib
import importlib.util
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PHYSICSNEMO_PACKAGE = "nvidia-physicsnemo"
PHYSICSNEMO_IMPORT = "physicsnemo"
PHYSICSNEMO_SOURCE_URL = "https://github.com/NVIDIA/physicsnemo"
PHYSICSNEMO_DOCS_URL = "https://docs.nvidia.com/physicsnemo/latest/"
PHYSICSNEMO_EXAMPLES_URL = "https://docs.nvidia.com/physicsnemo/latest/examples_catalog.html"
PHYSICSNEMO_INSTALL_URL = (
    "https://docs.nvidia.com/physicsnemo/latest/getting-started/installation.html"
)
SMOKE_MEASUREMENT_TYPE = "physicsnemo_compatibility_smoke"
SMOKE_STATUS = "compatibility_smoke_ready"


def _command_record(args: argparse.Namespace) -> list[str]:
    command = [
        "python",
        "scripts/run_physicsnemo_compatibility_smoke.py",
        "--name",
        args.name,
        "--output-root",
        args.output_root,
        "--evidence-json",
        args.evidence_json,
        "--train-split",
        args.train_split,
        "--eval-split",
        args.eval_split,
        "--tasks",
        *list(args.tasks),
    ]
    if args.live_import:
        command.append("--live-import")
    if args.require_live_import:
        command.append("--require-live-import")
    return command


def _package_probe(*, live_import: bool, require_live_import: bool) -> dict[str, Any]:
    base: dict[str, Any] = {
        "pip_name": PHYSICSNEMO_PACKAGE,
        "import_name": PHYSICSNEMO_IMPORT,
        "declared_python_requires": ">=3.11,<=3.14",
        "install_url": PHYSICSNEMO_INSTALL_URL,
        "source_url": PHYSICSNEMO_SOURCE_URL,
        "docs_url": PHYSICSNEMO_DOCS_URL,
        "live_import_requested": bool(live_import),
        "live_import_required": bool(require_live_import),
    }
    if not live_import:
        base["live_import_status"] = "not_requested"
        base["module_spec_checked"] = False
        return base
    spec = importlib.util.find_spec(PHYSICSNEMO_IMPORT)
    base["module_spec_checked"] = True
    base["module_spec_available"] = spec is not None
    try:
        module = importlib.import_module(PHYSICSNEMO_IMPORT)
    except Exception as exc:
        if require_live_import:
            raise RuntimeError(
                f"{PHYSICSNEMO_IMPORT} import is required but failed: {type(exc).__name__}: {exc}"
            ) from exc
        base.update(
            {
                "live_import_status": "failed_optional",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        return base
    base.update(
        {
            "live_import_status": "available",
            "version": str(getattr(module, "__version__", "")),
        }
    )
    return base


def _recipe_contract(args: argparse.Namespace) -> dict[str, Any]:
    tasks = list(dict.fromkeys(str(task) for task in args.tasks))
    inspected_splits = list(dict.fromkeys([str(args.train_split), str(args.eval_split)]))
    return {
        "scope": "dry ecosystem compatibility recipe manifest",
        "tasks": tasks,
        "inspected_splits": inspected_splits,
        "data_interface": {
            "source": "repo light-v1 PDEBench-shaped HDF5 shards",
            "expected_shard_pattern": "{task}_{split}.h5",
            "candidate_adapter": (
                "repo PDEBench tensors to a PhysicsNeMo data-driven recipe/datapipe"
            ),
            "held_out_test_data_read": False,
        },
        "candidate_recipe": {
            "first_target": (
                "PhysicsNeMo data-driven neural-operator recipe adapted to light-v1 "
                "train/validation tensors"
            ),
            "examples_catalog_url": PHYSICSNEMO_EXAMPLES_URL,
            "why_first": (
                "Recipe compatibility proves framework interop before reporting a "
                "framework metric."
            ),
        },
        "live_metric_allowed": False,
        "model_training_performed": False,
        "held_out_test_policy": "No test split or held-out ledger write in this smoke gate.",
        "next_gate": (
            "Run a live PhysicsNeMo recipe adapter on train/val in a Python 3.11+ "
            "or PhysicsNeMo container environment, record validation-only provenance, "
            "then decide whether any held-out test budget is justified."
        ),
    }


def build_physicsnemo_smoke_summary(args: argparse.Namespace) -> dict[str, Any]:
    inspected_splits = list(dict.fromkeys([str(args.train_split), str(args.eval_split)]))
    if "test" in inspected_splits:
        raise RuntimeError("PhysicsNeMo compatibility smoke must not inspect split=test")
    package = _package_probe(
        live_import=bool(args.live_import),
        require_live_import=bool(args.require_live_import),
    )
    output_root = Path(args.output_root)
    summary_path = output_root / args.name / "summary.json"
    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": SMOKE_STATUS,
        "measurement_type": SMOKE_MEASUREMENT_TYPE,
        "run_name": args.name,
        "summary_json": str(summary_path),
        "evidence_json": str(args.evidence_json),
        "split": args.eval_split,
        "inspected_splits": inspected_splits,
        "metrics": {},
        "claim_comparable": False,
        "published_numbers_directly_comparable": False,
        "held_out_test_used": False,
        "held_out_test_data_read": False,
        "stages": ["external_physicsnemo_compatibility_smoke"],
        "source_refs": ["physicsnemo_official_repo", "physicsnemo_docs"],
        "extra": {
            "baseline": "external_physicsnemo_compatibility",
            "implementation": PHYSICSNEMO_IMPORT,
            "source_url": PHYSICSNEMO_SOURCE_URL,
            "docs_url": PHYSICSNEMO_DOCS_URL,
            "examples_catalog_url": PHYSICSNEMO_EXAMPLES_URL,
            "command": _command_record(args),
        },
        "details": {
            "package": package,
            "recipe_contract": _recipe_contract(args),
            "claim_boundary": (
                "Compatibility smoke only; no model training, no PhysicsNeMo metric, "
                "and no held-out test access."
            ),
        },
    }
    errors = validate_physicsnemo_smoke_summary(summary)
    if errors:
        summary["status"] = "invalid"
        summary["validation_errors"] = errors
    return summary


def validate_physicsnemo_smoke_summary(summary: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if summary.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if summary.get("status") not in {SMOKE_STATUS, "invalid"}:
        errors.append(f"status must be one of {[SMOKE_STATUS, 'invalid']}")
    if summary.get("measurement_type") != SMOKE_MEASUREMENT_TYPE:
        errors.append(f"measurement_type must be {SMOKE_MEASUREMENT_TYPE}")
    if summary.get("claim_comparable") is not False:
        errors.append("claim_comparable must be false")
    if summary.get("published_numbers_directly_comparable") is not False:
        errors.append("published_numbers_directly_comparable must be false")
    if summary.get("held_out_test_used") is not False:
        errors.append("held_out_test_used must be false")
    if summary.get("held_out_test_data_read") is not False:
        errors.append("held_out_test_data_read must be false")

    inspected_splits = summary.get("inspected_splits")
    if not isinstance(inspected_splits, list) or not inspected_splits:
        errors.append("inspected_splits must be a non-empty list")
        inspected_splits = []
    if "test" in inspected_splits:
        errors.append("compatibility smoke must not inspect split=test")

    metrics = summary.get("metrics")
    if not isinstance(metrics, Mapping):
        errors.append("metrics must be an object")
        metrics = {}
    if metrics:
        errors.append("compatibility smoke must not report metrics")
    if "decoded_rollout_nrmse" in metrics:
        errors.append("compatibility smoke must not report decoded_rollout_nrmse")

    details = summary.get("details")
    if not isinstance(details, Mapping):
        errors.append("details must be an object")
        details = {}
    package = details.get("package")
    if not isinstance(package, Mapping):
        errors.append("details.package is required")
        package = {}
    if package.get("pip_name") != PHYSICSNEMO_PACKAGE:
        errors.append(f"details.package.pip_name must be {PHYSICSNEMO_PACKAGE}")
    if package.get("import_name") != PHYSICSNEMO_IMPORT:
        errors.append(f"details.package.import_name must be {PHYSICSNEMO_IMPORT}")

    contract = details.get("recipe_contract")
    if not isinstance(contract, Mapping):
        errors.append("details.recipe_contract is required")
        contract = {}
    tasks = contract.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        errors.append("details.recipe_contract.tasks must be a non-empty list")
    if contract.get("live_metric_allowed") is not False:
        errors.append("details.recipe_contract.live_metric_allowed must be false")
    if not contract.get("next_gate"):
        errors.append("details.recipe_contract.next_gate is required")
    return errors


def run_compatibility_smoke(args: argparse.Namespace) -> Path:
    summary = build_physicsnemo_smoke_summary(args)
    output_root = Path(args.output_root)
    run_dir = output_root / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    evidence_path = Path(args.evidence_json)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": summary["status"],
                "summary": str(summary_path),
                "evidence_json": str(evidence_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return summary_path


def check_compatibility_smoke(args: argparse.Namespace) -> bool:
    evidence_path = Path(args.evidence_json)
    if not evidence_path.exists():
        raise FileNotFoundError(f"PhysicsNeMo compatibility evidence not found: {evidence_path}")
    expected = build_physicsnemo_smoke_summary(args)
    actual = json.loads(evidence_path.read_text(encoding="utf-8"))
    if actual != expected:
        print(
            json.dumps(
                {
                    "status": "out_of_date",
                    "evidence_json": str(evidence_path),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return False
    print(
        json.dumps(
            {
                "status": "up_to_date",
                "evidence_json": str(evidence_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="physicsnemo_compatibility_smoke_light_v1")
    parser.add_argument("--output-root", default="reports/research/sota_loop/external_baselines")
    parser.add_argument(
        "--evidence-json",
        default="docs/claim_evidence/physicsnemo_compatibility_smoke_light_v1.json",
    )
    parser.add_argument("--tasks", nargs="+", default=["advection1d", "burgers1d", "darcy2d"])
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="val")
    parser.add_argument(
        "--live-import",
        action="store_true",
        help="Optionally import physicsnemo and record local import status.",
    )
    parser.add_argument(
        "--require-live-import",
        action="store_true",
        help="Fail if --live-import cannot import physicsnemo.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare the deterministic dry smoke manifest with --evidence-json.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.require_live_import and not args.live_import:
        raise RuntimeError("--require-live-import requires --live-import")
    if args.check:
        if not check_compatibility_smoke(args):
            raise SystemExit(1)
        return
    run_compatibility_smoke(args)


if __name__ == "__main__":
    main()
