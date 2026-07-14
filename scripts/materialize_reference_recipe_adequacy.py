#!/usr/bin/env python
from __future__ import annotations

"""Select a validation-only FNO/UNO reference recipe from frozen rung evidence."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from ups.data.manifests import canonical_sha256
from ups.eval.regime_metrics import regime_spread_ratio

TASKS = ("advection1d", "burgers1d", "darcy2d")
ARCHITECTURES = ("fno", "uno")
REGIME_COUNTS = {"advection1d": 8, "burgers1d": 12, "darcy2d": 5}


class InvalidMetricEvidenceError(ValueError):
    """A structurally valid discovery summary contains a non-finite metric."""


def _parsed_command(command: Any, *, architecture: str, runner_path: str) -> dict[str, Any]:
    if not isinstance(command, list) or len(command) < 2 or command[1] != runner_path:
        raise ValueError(f"{architecture} command does not name the frozen runner")
    if architecture == "fno":
        from scripts.run_external_neuraloperator_fno_baseline import build_parser
    else:
        from scripts.run_external_neuraloperator_uno_baseline import build_parser
    try:
        return vars(build_parser().parse_args(command[2:]))
    except SystemExit as exc:
        raise ValueError(f"{architecture} command cannot be parsed by its frozen runner") from exc


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise InvalidMetricEvidenceError(f"{name} is non-finite")
    if positive and result <= 0:
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _validate_plan(plan: dict[str, Any]) -> None:
    recorded = plan.get("plan_sha256")
    payload = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if recorded != canonical_sha256(payload):
        raise ValueError("recipe-adequacy plan SHA-256 does not match its canonical payload")
    if (
        plan.get("mode") != "validation_only"
        or plan.get("heldout_access") != "forbidden"
        or plan.get("measurement_lock_access") != "forbidden"
    ):
        raise ValueError("recipe-adequacy selection requires a validation-only plan")
    discovery = plan.get("discovery", {})
    if discovery.get("architectures") != list(ARCHITECTURES):
        raise ValueError("plan must contain exactly FNO and UNO in canonical order")
    if discovery.get("tasks") != list(TASKS) or discovery.get("seed") != 17:
        raise ValueError("discovery must cover all three tasks at seed 17")
    rungs = discovery.get("epoch_rungs")
    if rungs != [3, 6, 12, 24, 48] or discovery.get("maximum_epochs") != 48:
        raise ValueError("discovery epoch rungs are not the frozen adequacy ladder")
    runs = discovery.get("runs")
    if not isinstance(runs, list) or len(runs) != 2:
        raise ValueError("plan must contain exactly two discovery runs")
    plateau = plan.get("plateau_criterion", {})
    if (
        plateau.get("consecutive_transitions_required") != 2
        or plateau.get("relative_improvement_threshold") != 0.01
        or plateau.get("operator") != "strictly_less_than"
    ):
        raise ValueError("plan plateau rule is not the frozen two-transition rule")
    gate = plan.get("selection", {}).get("secondary_eligibility_gate", {})
    if gate.get("operator") != "less_than_or_equal" or gate.get("maximum") != 1.5:
        raise ValueError("plan corrected-regime gate is not the frozen <=1.5 rule")


def _relative(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _primary_and_spread(metrics: dict[str, Any]) -> tuple[float, dict[str, float]]:
    task_metrics: dict[str, float] = {}
    max_spread: dict[str, float] = {}
    for task in TASKS:
        primary_suffix = "decoded_solution_nrmse" if task == "darcy2d" else "decoded_rollout_nrmse"
        primary_key = f"task_{task}_{primary_suffix}"
        primary = _finite(metrics.get(primary_key), primary_key, positive=True)
        task_metrics[task] = primary
        prefix = f"task_{task}_regime_"
        suffix = f"_{primary_suffix.replace('_nrmse', '_global_scale_nrmse')}"
        values = [
            _finite(value, key)
            for key, value in metrics.items()
            if key.startswith(prefix) and key.endswith(suffix)
        ]
        if len(values) != REGIME_COUNTS[task]:
            raise ValueError(
                f"{task} rung has {len(values)} corrected regime metrics; "
                f"expected {REGIME_COUNTS[task]}"
            )
        max_spread[task] = max(regime_spread_ratio(value, primary) for value in values)
    return sum(task_metrics.values()) / len(TASKS), max_spread


def _validate_checkpoint(record: Any, *, epoch: int, repo_root: Path) -> dict[str, Any]:
    if not isinstance(record, dict) or record.get("epoch") != epoch:
        raise ValueError(f"checkpoint record for epoch {epoch} is missing or ambiguous")
    path_value = record.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"checkpoint path for epoch {epoch} is missing")
    path = Path(path_value)
    path = path if path.is_absolute() else repo_root / path
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = _file_sha256(path)
    if record.get("sha256") != observed:
        raise ValueError(f"checkpoint SHA-256 mismatch at epoch {epoch}")
    return {"epoch": epoch, "path": _relative(path, repo_root), "sha256": observed}


def _validate_summary(
    summary: dict[str, Any],
    *,
    summary_path: Path,
    architecture: str,
    run: dict[str, Any],
    plan: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    if summary.get("status") != "complete" or summary.get("split") != "val":
        raise ValueError(f"{architecture} discovery summary is incomplete or non-validation")
    extra = summary.get("extra")
    if not isinstance(extra, dict):
        raise ValueError(f"{architecture} summary lacks runner provenance")
    heldout = summary.get("held_out_test_policy", {})
    if heldout.get("enabled") or heldout.get("recorded") or extra.get("allow_held_out_test_eval"):
        raise ValueError(f"{architecture} summary contains held-out access")
    if summary.get("run_name") != run.get("run_id"):
        raise ValueError(f"{architecture} summary run identity does not match the plan")
    if extra.get("task") != list(TASKS) or extra.get("seed") != 17 or extra.get("epochs") != 48:
        raise ValueError(f"{architecture} summary is not the all-task seed-17 discovery run")
    expected_baseline = f"external_neuraloperator_{architecture}"
    if extra.get("baseline") != expected_baseline:
        raise ValueError(f"{architecture} summary baseline identity is ambiguous")
    command = extra.get("command")
    planned_command = run.get("command")
    runner_spec = plan["runner_identity"][architecture]
    if _parsed_command(
        command, architecture=architecture, runner_path=runner_spec["path"]
    ) != _parsed_command(
        planned_command, architecture=architecture, runner_path=runner_spec["path"]
    ):
        raise ValueError(f"{architecture} summary command does not match the frozen plan")
    if run.get("command_sha256") != canonical_sha256(planned_command):
        raise ValueError(f"{architecture} plan command SHA-256 is invalid")

    binding = plan["bindings"]["training_lock"]
    provenance = summary.get("data_provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"{architecture} summary lacks data provenance")
    expected_provenance = {
        "lock_sha256": binding["lock_sha256"],
        "source_revision": binding["source_revision"],
        "source_manifest_sha256": binding["source_manifest_sha256"],
        "protocol_manifest_sha256": binding["protocol_manifest_sha256"],
        "selection_sha256": binding["selection_sha256"],
    }
    for key, expected in expected_provenance.items():
        if provenance.get(key) != expected:
            raise ValueError(f"{architecture} summary {key} does not match the frozen lock")
    if provenance.get("purpose") != "training" or set(provenance.get("requested_roles", [])) != {
        "train",
        "valid",
    }:
        raise ValueError(f"{architecture} summary provenance permits an invalid data role")
    if any(item.get("role") == "test" for item in provenance.get("objects", [])):
        raise ValueError(f"{architecture} summary provenance contains a test object")

    runner = plan["runner_identity"][architecture]
    runner_path = repo_root / runner["path"]
    if not runner_path.is_file() or _file_sha256(runner_path) != runner.get("file_sha256"):
        raise ValueError(f"{architecture} runner content hash no longer matches the plan")

    history = summary.get("details", {}).get("validation_history")
    expected_rungs = plan["discovery"]["epoch_rungs"]
    if not isinstance(history, list) or [row.get("epoch") for row in history] != expected_rungs:
        raise ValueError(f"{architecture} validation history is missing or not continuous")
    checkpoint_records = summary.get("checkpoints", {}).get("rungs")
    if not isinstance(checkpoint_records, dict) or set(checkpoint_records) != {
        str(epoch) for epoch in expected_rungs
    }:
        raise ValueError(f"{architecture} rung checkpoint coverage is incomplete")
    checkpoints = {
        epoch: _validate_checkpoint(
            checkpoint_records[str(epoch)], epoch=epoch, repo_root=repo_root
        )
        for epoch in expected_rungs
    }

    rung_rows: list[dict[str, Any]] = []
    best = math.inf
    stale = 0
    plateau_epoch: int | None = None
    for row in history:
        epoch = row["epoch"]
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError(f"{architecture} epoch {epoch} lacks full metrics")
        macro, spreads = _primary_and_spread(metrics)
        declared = _finite(row.get("metric_value"), f"{architecture} epoch {epoch} metric")
        if not math.isclose(declared, macro, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"{architecture} epoch {epoch} declared macro disagrees with metrics")
        previous_best = best
        best = min(best, macro)
        improvement = None if math.isinf(previous_best) else (previous_best - best) / previous_best
        if improvement is not None:
            stale = stale + 1 if improvement < 0.01 else 0
            if stale == 2 and plateau_epoch is None:
                plateau_epoch = epoch
        rung_rows.append(
            {
                "epoch": epoch,
                "macro_primary_nrmse": macro,
                "best_so_far_relative_improvement": improvement,
                "max_regime_spread_by_task": spreads,
                "regime_gate_passed": all(value <= 1.5 for value in spreads.values()),
                "checkpoint": checkpoints[epoch],
            }
        )

    if plateau_epoch is None:
        label = plan["plateau_criterion"]["maximum_rung_without_plateau_label"]
        chosen = min(rung_rows, key=lambda row: (row["macro_primary_nrmse"], row["epoch"]))
    else:
        label = plan["plateau_criterion"]["adequate_label"]
        eligible_rungs = [row for row in rung_rows if row["epoch"] <= plateau_epoch]
        chosen = min(eligible_rungs, key=lambda row: (row["macro_primary_nrmse"], row["epoch"]))
    gate_passed = chosen["regime_gate_passed"]
    runner_choice = min(rung_rows, key=lambda row: (row["macro_primary_nrmse"], row["epoch"]))
    recipe = summary.get("recipe_adequacy")
    if not isinstance(recipe, dict):
        raise ValueError(f"{architecture} summary lacks recipe-adequacy evidence")
    selected_checkpoint = _validate_checkpoint(
        summary.get("checkpoints", {}).get("selected"),
        epoch=recipe.get("selected_epoch"),
        repo_root=repo_root,
    )
    if (
        recipe.get("validation_rungs") != expected_rungs
        or recipe.get("selection_metric") != "decoded_rollout_nrmse"
        or recipe.get("selected_epoch") != runner_choice["epoch"]
        or not math.isclose(
            _finite(recipe.get("selected_metric_value"), "runner selected metric"),
            runner_choice["macro_primary_nrmse"],
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or recipe.get("selection_rule") != "minimum_finite_validation_metric_earliest_tie"
        or selected_checkpoint != runner_choice["checkpoint"]
        or summary.get("metrics")
        != history[expected_rungs.index(runner_choice["epoch"])]["metrics"]
    ):
        raise ValueError(f"{architecture} runner-selected rung evidence is inconsistent")

    compute = summary.get("compute")
    required_compute = (
        "total_parameter_count",
        "trainable_parameter_count",
        "optimizer_steps",
        "examples_seen",
        "duration_sec",
        "device",
    )
    if not isinstance(compute, dict) or any(key not in compute for key in required_compute):
        raise ValueError(f"{architecture} summary lacks complete compute evidence")
    for key in (
        "total_parameter_count",
        "trainable_parameter_count",
        "optimizer_steps",
        "examples_seen",
    ):
        if isinstance(compute[key], bool) or not isinstance(compute[key], int) or compute[key] <= 0:
            raise ValueError(f"{architecture} compute.{key} must be a positive integer")
    _finite(compute["duration_sec"], f"{architecture} compute.duration_sec", positive=True)
    if not isinstance(compute["device"], str) or not compute["device"]:
        raise ValueError(f"{architecture} compute.device must be a non-empty string")
    if "peak_cuda_memory_bytes" in compute and (
        isinstance(compute["peak_cuda_memory_bytes"], bool)
        or not isinstance(compute["peak_cuda_memory_bytes"], int)
        or compute["peak_cuda_memory_bytes"] < 0
    ):
        raise ValueError(f"{architecture} CUDA peak memory must be non-negative")
    return {
        "architecture": architecture,
        "label": label,
        "plateau_epoch": plateau_epoch,
        "eligible": label == "adequate" and gate_passed,
        "chosen_epoch": chosen["epoch"],
        "chosen_macro_primary_nrmse": chosen["macro_primary_nrmse"],
        "chosen_regime_gate_passed": gate_passed,
        "chosen_checkpoint": chosen["checkpoint"],
        "compute": compute,
        "rungs": rung_rows,
        "summary": {
            "path": _relative(summary_path, repo_root),
            "sha256": _file_sha256(summary_path),
        },
    }


def build_selection(
    plan: dict[str, Any], *, summary_paths: list[Path], repo_root: Path
) -> dict[str, Any]:
    _validate_plan(plan)
    if len(summary_paths) != 2 or len({path.resolve() for path in summary_paths}) != 2:
        raise ValueError("exactly two distinct discovery summaries are required")
    runs = {run["architecture"]: run for run in plan["discovery"]["runs"]}
    if set(runs) != set(ARCHITECTURES):
        raise ValueError("discovery run architecture mapping is ambiguous")
    supplied = {_relative(path, repo_root): path for path in summary_paths}
    expected = {run["expected_summary"]: architecture for architecture, run in runs.items()}
    if set(supplied) != set(expected):
        raise ValueError(
            "supplied summaries do not exactly match the two planned discovery outputs"
        )

    architecture_rows = {}
    for relative, architecture in expected.items():
        path = supplied[relative]
        summary = json.loads(path.read_text(encoding="utf-8"))
        try:
            architecture_rows[architecture] = _validate_summary(
                summary,
                summary_path=path,
                architecture=architecture,
                run=runs[architecture],
                plan=plan,
                repo_root=repo_root,
            )
        except InvalidMetricEvidenceError as exc:
            architecture_rows[architecture] = {
                "architecture": architecture,
                "label": plan["plateau_criterion"]["non_finite_label"],
                "invalid_reason": str(exc),
                "plateau_epoch": None,
                "eligible": False,
                "chosen_epoch": None,
                "chosen_macro_primary_nrmse": None,
                "chosen_regime_gate_passed": False,
                "chosen_checkpoint": None,
                "rungs": [],
                "summary": {"path": relative, "sha256": _file_sha256(path)},
            }
    eligible = [row for row in architecture_rows.values() if row["eligible"]]
    selected = min(
        eligible,
        key=lambda row: (row["chosen_macro_primary_nrmse"], row["architecture"]),
        default=None,
    )
    payload = {
        "schema_version": 1,
        "selection_id": "strat-v1.1-reference-recipe-adequacy-selection-v1",
        "status": "complete_validation_only",
        "plan_sha256": plan["plan_sha256"],
        "architectures": architecture_rows,
        "selection": (
            None
            if selected is None
            else {
                "architecture": selected["architecture"],
                "epoch": selected["chosen_epoch"],
                "macro_primary_nrmse": selected["chosen_macro_primary_nrmse"],
                "checkpoint": selected["chosen_checkpoint"],
            }
        ),
        "no_eligible_architecture": selected is None,
        "held_out_measurements": 0,
    }
    return {**payload, "selection_sha256": canonical_sha256(payload)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--summary", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    plan_path = Path(args.plan)
    plan_path = plan_path if plan_path.is_absolute() else repo_root / plan_path
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    summary_paths = [
        Path(value) if Path(value).is_absolute() else repo_root / value for value in args.summary
    ]
    artifact = build_selection(plan, summary_paths=summary_paths, repo_root=repo_root)
    output = Path(args.output)
    output = output if output.is_absolute() else repo_root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"output": str(output), "selection_sha256": artifact["selection_sha256"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
