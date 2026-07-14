#!/usr/bin/env python
from __future__ import annotations

"""Run a physical-space persistence baseline and write light-experiment summaries."""

import argparse
import copy
import csv
import hashlib
import json
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ups.data.manifests import canonical_sha256, load_data_lock
from ups.eval.persistence_baselines import evaluate_persistence_decoded
from ups.eval.promotion import evaluate_promotion_rules, parse_promotion_rule
from ups.utils.config_loader import load_config_with_includes

STRAT_V1_TASKS = ("advection1d", "burgers1d", "darcy2d")


def _parse_override(text: str) -> tuple[str, Any]:
    if "=" not in text:
        raise ValueError(f"Invalid override '{text}'. Expected key=value")
    key, raw = text.split("=", 1)
    return key.strip(), yaml.safe_load(raw)


def _set_dotpath(cfg: dict[str, Any], path: str, value: Any) -> None:
    cursor = cfg
    parts = [part for part in path.split(".") if part]
    if not parts:
        raise ValueError(f"Invalid override path '{path}'")
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def _apply_overrides(cfg: dict[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    updated = copy.deepcopy(cfg)
    for text in overrides:
        key, value = _parse_override(text)
        _set_dotpath(updated, key, value)
    return updated


def _main_metric(metrics: dict[str, float]) -> tuple[str, float]:
    for key in ("macro_primary_nrmse", "decoded_rollout_nrmse", "decoded_step1_nrmse", "mse"):
        if key in metrics:
            return key, float(metrics[key])
    first_key = next(iter(metrics))
    return first_key, float(metrics[first_key])


def _append_results_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base_fieldnames = [
        "run_name",
        "timestamp",
        "stages",
        "decoded",
        "train_split",
        "eval_split",
        "transfer_tasks",
        "promotion_passed",
        "main_metric_name",
        "main_metric_value",
        "summary_json",
    ]
    fieldnames = list(base_fieldnames)
    row_map: dict[str, dict[str, Any]] = {}
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            fieldnames = list(reader.fieldnames or base_fieldnames)
            for name in base_fieldnames:
                if name not in fieldnames:
                    fieldnames.append(name)
            for existing in reader:
                run_name = existing.get("run_name")
                if run_name:
                    row_map[run_name] = dict(existing)
    row_map[str(row["run_name"])] = row
    for name in row:
        if name not in fieldnames:
            fieldnames.append(name)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for name in sorted(row_map):
            result_row = row_map[name]
            writer.writerow({field: result_row.get(field, "") for field in fieldnames})


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _working_tree_dirty() -> bool:
    try:
        return bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=REPO_ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return True


def _code_files_sha256(config_path: Path) -> tuple[str, list[str]]:
    """Hash the exact Python/config source surface used by the official baseline."""

    paths = sorted((REPO_ROOT / "src").rglob("*.py"))
    paths.extend([Path(__file__).resolve(), config_path.resolve()])

    def label(path: Path) -> str:
        try:
            return str(path.relative_to(REPO_ROOT))
        except ValueError:
            return f"external-config/{path.name}"

    unique = sorted(set(paths), key=label)
    digest = hashlib.sha256()
    names = []
    for path in unique:
        relative = label(path)
        content = path.read_bytes()
        encoded_name = relative.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
        names.append(relative)
    return digest.hexdigest(), names


def _validate_official_inputs(
    *,
    data_cfg: dict[str, Any],
    data_lock_path: Path,
    expected_lock_sha256: str | None,
    rollout_steps: int | None,
) -> tuple[Any, str]:
    tasks = data_cfg.get("task")
    task_names = [tasks] if isinstance(tasks, str) else list(tasks or [])
    if tuple(task_names) != STRAT_V1_TASKS:
        raise ValueError(
            "Official strat-v1 persistence requires tasks in canonical order: "
            + ", ".join(STRAT_V1_TASKS)
        )
    if data_cfg.get("split") not in {"val", "valid", "validation"}:
        raise ValueError("Official strat-v1 persistence is validation-only")
    if data_cfg.get("max_samples") is not None:
        raise ValueError(
            "Official strat-v1 persistence must evaluate the complete validation split"
        )
    if rollout_steps != 16:
        raise ValueError("Official strat-v1 persistence requires exactly 16 rollout steps")

    lock = load_data_lock(data_lock_path)
    if lock.purpose != "training" or "valid" not in lock.requested_roles:
        raise PermissionError("Persistence requires a training lock authorizing validation")
    if "test" in lock.requested_roles or any(item.role == "test" for item in lock.objects):
        raise PermissionError("Persistence training lock must not contain test objects")
    if expected_lock_sha256 and lock.lock_sha256 != expected_lock_sha256:
        raise ValueError("Data lock does not match the frozen expected lock SHA-256")

    locked_valid_names = {Path(item.path).name for item in lock.objects if item.role == "valid"}
    expected_valid_names = {f"{task}_val.h5" for task in STRAT_V1_TASKS}
    if locked_valid_names != expected_valid_names:
        raise ValueError(
            "Training lock validation objects do not match the complete strat-v1 task set"
        )

    root = Path(str(data_cfg.get("root", ""))).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Staged training run view does not exist: {root}")
    test_files = sorted(path for path in root.rglob("*.h5") if "test" in path.name.lower())
    if test_files:
        raise PermissionError("Staged training run view contains test HDF5 files")
    missing_valid = sorted(name for name in expected_valid_names if not (root / name).is_file())
    if missing_valid:
        raise FileNotFoundError(
            "Staged training run view is missing validation objects: " + ", ".join(missing_valid)
        )
    return lock, canonical_sha256(lock.selection)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run decoded persistence baseline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-root", default="reports/light_experiments")
    parser.add_argument(
        "--override", action="append", default=[], help="Config override like data.root=..."
    )
    parser.add_argument("--data-root", help="Override data.root")
    parser.add_argument("--split", help="Override data.split")
    parser.add_argument(
        "--task", action="append", default=[], help="Override data.task; repeat for multitask"
    )
    parser.add_argument("--max-samples", type=int, help="Override data.max_samples")
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--data-lock", required=True, help="Frozen training data lock")
    parser.add_argument(
        "--expected-data-lock-sha256",
        help="Optional frozen lock identity; defaults to data.data_lock_sha256",
    )
    parser.add_argument("--promotion-rule", action="append", default=[])
    args = parser.parse_args()

    cfg = _apply_overrides(load_config_with_includes(args.config), args.override)
    data_cfg = cfg.setdefault("data", {})
    if args.data_root:
        data_cfg["root"] = args.data_root
    if args.split:
        data_cfg["split"] = args.split
    if args.task:
        data_cfg["task"] = args.task if len(args.task) > 1 else args.task[0]
    if args.max_samples is not None:
        data_cfg["max_samples"] = int(args.max_samples)

    data_lock_path = Path(args.data_lock).resolve()
    expected_lock_sha256 = args.expected_data_lock_sha256 or data_cfg.get("data_lock_sha256")
    lock, selection_sha256 = _validate_official_inputs(
        data_cfg=data_cfg,
        data_lock_path=data_lock_path,
        expected_lock_sha256=expected_lock_sha256,
        rollout_steps=args.rollout_steps,
    )
    data_cfg["data_lock_path"] = str(data_lock_path)
    data_cfg["data_lock_sha256"] = lock.lock_sha256
    data_cfg["selection_sha256"] = selection_sha256

    output_root = Path(args.output_root)
    run_dir = output_root / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_cfg_path = run_dir / "resolved_eval.yaml"
    resolved_cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    started = time.time()
    report = evaluate_persistence_decoded(
        cfg, rollout_steps=args.rollout_steps, strict_contract=True
    )
    finished = time.time()
    if args.promotion_rule:
        result = evaluate_promotion_rules(
            report.metrics,
            [parse_promotion_rule(rule) for rule in args.promotion_rule],
        )
        report.extra["promotion_passed"] = result.passed
        report.extra["promotion_failed_rules"] = result.failed_rules
        report.extra["promotion_missing_metrics"] = result.missing_metrics

    code_sha256, code_files = _code_files_sha256(Path(args.config))
    protocol_details = {
        "data_lock_path": str(data_lock_path),
        "data_lock_sha256": lock.lock_sha256,
        "source_manifest_sha256": lock.source_manifest_sha256,
        "protocol_manifest_sha256": lock.protocol_manifest_sha256,
        "selection_sha256": selection_sha256,
        "code_files_sha256": code_sha256,
        "code_files": code_files,
        "resolved_config_sha256": hashlib.sha256(resolved_cfg_path.read_bytes()).hexdigest(),
        "git_commit": _git_commit(),
        "working_tree_dirty": _working_tree_dirty(),
        "checkpoint": None,
        "normalization": "none_physical_space",
        "normalization_sha256": None,
        "physical_parameter_conditioning": False,
        "inferred_parameter_context": False,
        "regime_metadata_reporting_only": True,
    }
    report_extra = dict(report.extra or {})
    evaluation_details = report_extra.pop("details", {})
    summary = {
        "metrics": report.metrics,
        "extra": report_extra,
        "details": {**evaluation_details, "protocol": protocol_details},
        "checkpoints": {"operator": None, "encoder": None, "decoder": None},
        "run_name": args.name,
        "split": data_cfg["split"],
        "stages": ["persistence"],
        "config": str(resolved_cfg_path),
        "eval_config": str(resolved_cfg_path),
        "duration_sec": finished - started,
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    main_metric_name, main_metric_value = _main_metric(report.metrics)
    _append_results_row(
        output_root / "results.tsv",
        {
            "run_name": args.name,
            "timestamp": int(finished),
            "stages": "persistence",
            "decoded": True,
            "train_split": "",
            "eval_split": data_cfg.get("split", ""),
            "transfer_tasks": "",
            "promotion_passed": report.extra.get("promotion_passed"),
            "main_metric_name": main_metric_name,
            "main_metric_value": main_metric_value,
            "summary_json": str(summary_path),
        },
    )
    print(
        json.dumps(
            {"summary": str(summary_path), "main_metric": {main_metric_name: main_metric_value}},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
