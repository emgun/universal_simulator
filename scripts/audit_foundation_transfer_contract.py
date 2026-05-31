#!/usr/bin/env python
from __future__ import annotations

"""Write a machine-checkable foundation-model transfer readiness contract.

This audit intentionally does not evaluate held-out test data. Its job is to
define what must be true before Poseidon/CNO-FM transfer can be compared against
the current light-v1 claim.
"""

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts import run_external_neuraloperator_fno_baseline as fno_runner
from ups.data.latent_pairs import infer_channel_count, infer_grid_shape
from ups.data.pdebench import PDEBenchConfig, PDEBenchDataset, get_pdebench_spec

POSEIDON_SOURCE_URL = "https://github.com/camlab-ethz/poseidon"
POSEIDON_MODEL_IMPORT = "scOT.model.ScOT"
POSEIDON_PRETRAINED_MODEL_TEMPLATE = "camlab-ethz/Poseidon-{T,B,L}"
POSEIDON_PRETRAINING_COLLECTION = (
    "https://huggingface.co/collections/camlab-ethz/poseidon-664fa125729c53d8607e209a"
)
POSEIDON_DOWNSTREAM_COLLECTION = (
    "https://huggingface.co/collections/camlab-ethz/"
    "poseidon-downstream-tasks-664fa237cd6b0c097971ef14"
)

CNO_FM_SOURCE_URL = "https://github.com/camlab-ethz/ConvolutionalNeuralOperator"
CNO_FM_SOURCE_PATH = "CNO2d_temporal"
CNO_FM_WEIGHTS_URL = "https://zenodo.org/records/11401801"

ALLOWED_STATUSES = {
    "contract_defined_measurement_pending",
    "invalid",
}
REQUIRED_CHECKS = {
    "held_out_budget_preserved",
    "poseidon_source_available",
    "poseidon_pretrained_entrypoint_declared",
    "poseidon_dataset_adapter_required",
    "poseidon_shape_adapter_required",
    "cno_fm_source_available",
    "cno_fm_channel_adapter_required",
    "foundation_measurement_ready",
}


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _git_commit(path: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=True,
        )
    except Exception:
        return "unknown"
    return proc.stdout.strip()


def _source_snapshot(
    *,
    source_id: str,
    repo_path: Path | None,
    source_url: str,
    required_files: Sequence[str],
    evidence_patterns: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    exists = repo_path is not None and repo_path.exists()
    files: dict[str, dict[str, Any]] = {}
    evidence: dict[str, bool] = {}
    for relative in required_files:
        path = repo_path / relative if exists and repo_path is not None else Path(relative)
        text = _read_text(path)
        files[relative] = {
            "exists": bool(text),
            "bytes": path.stat().st_size if path.exists() else 0,
        }
    for key, patterns in evidence_patterns.items():
        evidence[key] = False
        for relative in required_files:
            path = repo_path / relative if exists and repo_path is not None else Path(relative)
            text = _read_text(path)
            if all(pattern in text for pattern in patterns):
                evidence[key] = True
                break
    return {
        "source_id": source_id,
        "source_url": source_url,
        "repo_path": str(repo_path) if repo_path is not None else "",
        "available": exists,
        "commit": _git_commit(repo_path) if exists and repo_path is not None else "missing",
        "required_files": files,
        "evidence": evidence,
    }


def _field_step_count(fields: torch.Tensor) -> int:
    if fields.dim() >= 3 and fields.shape[0] > 1:
        return int(fields.shape[0])
    return 1


def inspect_light_protocol_tasks(
    *,
    cfg: Mapping[str, Any],
    tasks: Sequence[str],
    data_root: str | None,
    splits: Sequence[str],
    max_samples: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for task in tasks:
        spec = get_pdebench_spec(task)
        for split in splits:
            dataset = PDEBenchDataset(
                PDEBenchConfig(
                    task=task,
                    split=split,
                    root=data_root or cfg.get("data", {}).get("root"),
                    max_samples=max_samples,
                )
            )
            fields = dataset[0]["fields"].float()
            grid_shape = infer_grid_shape(fields)
            channels = infer_channel_count(fields, grid_shape)
            records.append(
                {
                    "task": task,
                    "split": split,
                    "sample_count_inspected": min(max_samples, len(dataset)),
                    "raw_sample_shape": list(fields.shape),
                    "repo_inferred_grid_shape": list(grid_shape),
                    "repo_inferred_channels": channels,
                    "repo_inferred_step_count": _field_step_count(fields),
                    "family": spec.family,
                    "traits": list(spec.traits),
                    "poseidon_direct_dataset_identifier": _poseidon_dataset_identifier(task),
                }
            )
    return records


def _poseidon_dataset_identifier(task: str) -> str | None:
    # Keep this conservative. A near-sounding PDE is not a protocol match unless
    # the official Poseidon code names the same dataset semantics.
    mapping = {
        "darcy2d": None,
        "advection1d": None,
        "burgers1d": None,
        "navier_stokes2d": None,
    }
    return mapping.get(task)


def _has_non_square_or_1d_grid(records: Sequence[Mapping[str, Any]]) -> bool:
    for record in records:
        height, width = record.get("repo_inferred_grid_shape", [0, 0])
        if int(height) != int(width):
            return True
    return False


def _has_unmapped_poseidon_dataset(records: Sequence[Mapping[str, Any]]) -> bool:
    return any(record.get("poseidon_direct_dataset_identifier") is None for record in records)


def _has_non_fm_channel_count(records: Sequence[Mapping[str, Any]], *, required: int) -> bool:
    return any(int(record.get("repo_inferred_channels", -1)) != required for record in records)


def _check(key: str, status: str, detail: str, **extra: Any) -> dict[str, Any]:
    return {"key": key, "status": status, "detail": detail, **extra}


def build_foundation_transfer_contract(
    *,
    cfg: Mapping[str, Any],
    config_path: str,
    tasks: Sequence[str],
    data_root: str | None,
    inspected_splits: Sequence[str],
    max_samples: int,
    poseidon_repo: Path | None,
    cno_repo: Path | None,
    run_name: str,
) -> dict[str, Any]:
    task_records = inspect_light_protocol_tasks(
        cfg=cfg,
        tasks=tasks,
        data_root=data_root,
        splits=inspected_splits,
        max_samples=max_samples,
    )
    poseidon = _source_snapshot(
        source_id="poseidon_official_repo",
        repo_path=poseidon_repo,
        source_url=POSEIDON_SOURCE_URL,
        required_files=[
            "README.md",
            "pyproject.toml",
            "scOT/model.py",
            "scOT/train.py",
            "scOT/inference.py",
            "scOT/problems/base.py",
        ],
        evidence_patterns={
            "from_pretrained": ("ScOT.from_pretrained",),
            "pretrained_hf_collection": ("huggingface.co/collections/camlab-ethz/poseidon",),
            "finetune_mismatched_sizes": ("ignore_mismatched_sizes=True",),
            "square_image_assumption": ("assumes square images",),
        },
    )
    cno_fm = _source_snapshot(
        source_id="cno_fm_official_source",
        repo_path=cno_repo,
        source_url=CNO_FM_SOURCE_URL,
        required_files=[
            "CNO2d_temporal/readme.md",
            "CNO2d_temporal/CNO_FineTune.py",
            "CNO2d_temporal/CNO_timeModule_CIN.py",
            "CNO2d_temporal/DataLoaders/all_experiments.json",
        ],
        evidence_patterns={
            "foundation_weights": ("CNO-Foundation Model", "zenodo.org/records/11401801"),
            "input_dim_5_output_dim_4": ("Input dimension", "5", "Output dimension", "4"),
            "finetune_entrypoint": ("CNO_FineTune.py",),
            "custom_loader_required": ("CNO_TimeLoaders.py",),
        },
    )

    poseidon_needs_dataset_adapter = _has_unmapped_poseidon_dataset(task_records)
    poseidon_needs_shape_adapter = _has_non_square_or_1d_grid(task_records)
    cno_fm_needs_channel_adapter = _has_non_fm_channel_count(task_records, required=5)

    checks = [
        _check(
            "held_out_budget_preserved",
            "pass",
            "Only train/val metadata is inspected; no held-out test split is opened or evaluated.",
            inspected_splits=list(inspected_splits),
        ),
        _check(
            "poseidon_source_available",
            "pass" if poseidon["available"] else "fail",
            (
                "Official Poseidon source checkout is available."
                if poseidon["available"]
                else "Official Poseidon source checkout is missing."
            ),
            commit=poseidon["commit"],
        ),
        _check(
            "poseidon_pretrained_entrypoint_declared",
            "pass" if poseidon["evidence"].get("from_pretrained") else "fail",
            (
                "Poseidon declares ScOT.from_pretrained and Hugging Face pretrained model loading."
                if poseidon["evidence"].get("from_pretrained")
                else "Poseidon pretrained model loading entrypoint was not found."
            ),
            model_import=POSEIDON_MODEL_IMPORT,
            model_template=POSEIDON_PRETRAINED_MODEL_TEMPLATE,
        ),
        _check(
            "poseidon_dataset_adapter_required",
            "blocker" if poseidon_needs_dataset_adapter else "pass",
            (
                "At least one light-v1 task lacks a direct official Poseidon dataset identifier."
                if poseidon_needs_dataset_adapter
                else "All light-v1 tasks have direct Poseidon dataset identifiers."
            ),
        ),
        _check(
            "poseidon_shape_adapter_required",
            "blocker" if poseidon_needs_shape_adapter else "pass",
            (
                "Current repo-inferred light-v1 grids are not uniformly square image tensors."
                if poseidon_needs_shape_adapter
                else "Current repo-inferred light-v1 grids are square image tensors."
            ),
        ),
        _check(
            "cno_fm_source_available",
            "pass" if cno_fm["available"] else "fail",
            (
                "Official CNO-FM source checkout is available."
                if cno_fm["available"]
                else "Official CNO-FM source checkout is missing."
            ),
            commit=cno_fm["commit"],
            source_path=CNO_FM_SOURCE_PATH,
        ),
        _check(
            "cno_fm_channel_adapter_required",
            "blocker" if cno_fm_needs_channel_adapter else "pass",
            (
                "CNO-FM declares a 5-channel foundation input, while light-v1 records are scalar under the current protocol."
                if cno_fm_needs_channel_adapter
                else "Light-v1 channel count matches the CNO-FM foundation input contract."
            ),
            foundation_input_channels=5,
            foundation_output_channels=4,
        ),
    ]
    measurement_blockers = [
        check["key"] for check in checks if check["status"] in {"blocker", "fail"}
    ]
    checks.append(
        _check(
            "foundation_measurement_ready",
            "pending" if measurement_blockers else "pass",
            (
                "Foundation transfer measurement remains pending until source, dataset, shape, and channel adapters are implemented and validated."
                if measurement_blockers
                else "Foundation transfer measurement can proceed to validation-only evaluation."
            ),
            blockers=measurement_blockers,
        )
    )

    status = "contract_defined_measurement_pending"
    contract = {
        "schema_version": 1,
        "status": status,
        "run_name": run_name,
        "generated_unix": int(time.time()),
        "claim_comparable": False,
        "published_numbers_directly_comparable": False,
        "held_out_test_used": False,
        "held_out_test_data_read": False,
        "measurement_type": "foundation_transfer_readiness_contract",
        "inspected_splits": list(inspected_splits),
        "claim_protocol_snapshot": {
            "config": config_path,
            "data_root": data_root or cfg.get("data", {}).get("root"),
            "task_set": list(tasks),
            "claim_split": "test",
            "inspected_splits": list(inspected_splits),
            "metric_name": "decoded_rollout_nrmse",
            "rollout_steps": 16,
            "max_eval_samples": 32,
        },
        "source_snapshots": {
            "poseidon": poseidon,
            "cno_fm": cno_fm,
        },
        "local_protocol_observations": {
            "tasks": task_records,
        },
        "readiness_checks": checks,
        "measurement_blockers": measurement_blockers,
        "next_validation_gate": {
            "split": "val",
            "requires_pretrained_weight_provenance": True,
            "requires_train_only_adapter_selection": True,
            "requires_no_held_out_test_before_validation": True,
            "minimum_outputs": [
                "adapter source and commit",
                "pretrained checkpoint handle and hash",
                "train/val-only dataset adapter manifest",
                "decoded_rollout_nrmse on validation",
                "no held-out test ledger write",
            ],
        },
        "not_claimable_yet": [
            "UPS beats Poseidon published paper/table values.",
            "UPS beats CNO-FM published or Zenodo checkpoint results.",
            "The current light-v1 scalar/height-1 protocol is directly comparable to Poseidon or CNO-FM without dataset and shape/channel adapters.",
        ],
        "recommendation": (
            "Implement a validation-only Poseidon ScOT adapter first, because Poseidon exposes "
            "a Hugging Face from_pretrained path and an embedding/recovery replacement mode. "
            "Treat CNO-FM as a separate 2D/channel-rich transfer track."
        ),
    }
    errors = validate_foundation_transfer_contract(contract)
    if errors:
        contract["status"] = "invalid"
        contract["validation_errors"] = errors
    return contract


def validate_foundation_transfer_contract(contract: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if contract.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if contract.get("status") not in ALLOWED_STATUSES:
        errors.append(f"status must be one of {sorted(ALLOWED_STATUSES)}")
    if contract.get("held_out_test_used") is not False:
        errors.append("held_out_test_used must be false for readiness contracts")
    if contract.get("held_out_test_data_read") is not False:
        errors.append("held_out_test_data_read must be false for readiness contracts")
    if contract.get("claim_comparable") is not False:
        errors.append("claim_comparable must be false until a held-out transfer measurement exists")
    if contract.get("published_numbers_directly_comparable") is not False:
        errors.append("published_numbers_directly_comparable must be false")

    protocol = contract.get("claim_protocol_snapshot")
    if not isinstance(protocol, Mapping):
        errors.append("claim_protocol_snapshot must be an object")
        protocol = {}
    inspected_splits = protocol.get("inspected_splits", [])
    if not isinstance(inspected_splits, list):
        errors.append("claim_protocol_snapshot.inspected_splits must be a list")
        inspected_splits = []
    top_level_inspected_splits = contract.get("inspected_splits", [])
    if top_level_inspected_splits != inspected_splits:
        errors.append("inspected_splits must match claim_protocol_snapshot.inspected_splits")
    if "test" in inspected_splits:
        errors.append("readiness contracts must not inspect split=test")

    checks = contract.get("readiness_checks")
    if not isinstance(checks, list):
        errors.append("readiness_checks must be a list")
        checks = []
    check_keys = {
        str(check.get("key"))
        for check in checks
        if isinstance(check, Mapping) and check.get("key") is not None
    }
    missing = REQUIRED_CHECKS - check_keys
    for key in sorted(missing):
        errors.append(f"readiness_checks missing {key}")
    for index, check in enumerate(checks):
        if not isinstance(check, Mapping):
            errors.append(f"readiness_checks[{index}] must be an object")
            continue
        if check.get("status") not in {"pass", "pending", "blocker", "fail"}:
            errors.append(f"readiness_checks[{index}].status is invalid")
        if not check.get("detail"):
            errors.append(f"readiness_checks[{index}].detail is required")

    blockers = contract.get("measurement_blockers")
    if not isinstance(blockers, list):
        errors.append("measurement_blockers must be a list")
    elif "foundation_measurement_ready" in blockers:
        errors.append("foundation_measurement_ready must summarize blockers, not be a blocker")
    return errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_multitask_heterogeneous_light_best.yaml")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--name", default="foundation_transfer_readiness_light_v1")
    parser.add_argument("--data-root")
    parser.add_argument("--tasks", nargs="+", default=[])
    parser.add_argument("--inspect-splits", nargs="+", default=["train", "val"])
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--poseidon-repo")
    parser.add_argument("--cno-repo")
    parser.add_argument("--print-json", action="store_true")
    return parser


def run_audit(args: argparse.Namespace) -> Path:
    cfg = fno_runner._load_cfg(args.config)
    tasks = fno_runner._as_task_names(cfg, args.tasks)
    inspected_splits = [str(split) for split in args.inspect_splits]
    if "test" in inspected_splits:
        raise RuntimeError("Foundation transfer readiness audit must not inspect split=test")
    contract = build_foundation_transfer_contract(
        cfg=cfg,
        config_path=args.config,
        tasks=tasks,
        data_root=args.data_root,
        inspected_splits=inspected_splits,
        max_samples=args.max_samples,
        poseidon_repo=Path(args.poseidon_repo) if args.poseidon_repo else None,
        cno_repo=Path(args.cno_repo) if args.cno_repo else None,
        run_name=args.name,
    )
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(contract, indent=2, sort_keys=True), encoding="utf-8")
    if args.print_json:
        print(json.dumps(contract, indent=2, sort_keys=True))
    else:
        result = {
            "status": contract["status"],
            "output_json": str(output_path),
            "measurement_blockers": contract.get("measurement_blockers", []),
        }
        print(json.dumps(result, indent=2, sort_keys=True))
    return output_path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_path = run_audit(args)
    contract = json.loads(output_path.read_text(encoding="utf-8"))
    return 1 if contract.get("status") == "invalid" else 0


if __name__ == "__main__":
    raise SystemExit(main())
