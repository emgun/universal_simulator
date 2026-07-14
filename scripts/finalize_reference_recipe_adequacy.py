#!/usr/bin/env python
from __future__ import annotations

"""Finalize a frozen reference recipe from three validation-only seeds."""

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from scripts.materialize_reference_recipe_adequacy import (
    REGIME_COUNTS,
    TASKS,
    _file_sha256,
    _parsed_command,
    _validate_checkpoint,
)
from ups.data.manifests import canonical_sha256
from ups.eval.regime_metrics import regime_spread_ratio

CONFIRMATION_SEEDS = (29, 43)


def _validate_self_hash(value: dict[str, Any], field: str, label: str) -> str:
    recorded = value.get(field)
    payload = {key: item for key, item in value.items() if key != field}
    if recorded != canonical_sha256(payload):
        raise ValueError(f"{label} canonical SHA-256 is invalid")
    return recorded


def _finite(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0):
        raise ValueError(f"{label} must be finite" + (" and positive" if positive else ""))
    return number


def _validate_compute(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} lacks compute evidence")
    for key in (
        "total_parameter_count",
        "trainable_parameter_count",
        "optimizer_steps",
        "examples_seen",
    ):
        if (
            isinstance(value.get(key), bool)
            or not isinstance(value.get(key), int)
            or value[key] <= 0
        ):
            raise ValueError(f"{label} compute.{key} must be a positive integer")
    _finite(value.get("duration_sec"), f"{label} compute.duration_sec", positive=True)
    if not isinstance(value.get("device"), str) or not value["device"]:
        raise ValueError(f"{label} compute.device must be a non-empty string")
    if "peak_cuda_memory_bytes" in value and (
        isinstance(value["peak_cuda_memory_bytes"], bool)
        or not isinstance(value["peak_cuda_memory_bytes"], int)
        or value["peak_cuda_memory_bytes"] < 0
    ):
        raise ValueError(f"{label} peak CUDA memory must be non-negative")
    return value


def _metric_evidence(metrics: Any) -> dict[str, Any]:
    if not isinstance(metrics, dict):
        raise ValueError("seed summary lacks full metrics")
    task_values: dict[str, float] = {}
    spreads: dict[str, float] = {}
    for task in TASKS:
        primary = "decoded_solution_nrmse" if task == "darcy2d" else "decoded_rollout_nrmse"
        key = f"task_{task}_{primary}"
        value = _finite(metrics.get(key), key, positive=True)
        task_values[task] = value
        global_suffix = f"_{primary.replace('_nrmse', '_global_scale_nrmse')}"
        regime_values = [
            _finite(metric_value, metric_key)
            for metric_key, metric_value in metrics.items()
            if metric_key.startswith(f"task_{task}_regime_") and metric_key.endswith(global_suffix)
        ]
        if len(regime_values) != REGIME_COUNTS[task]:
            raise ValueError(f"{task} corrected regime coverage is incomplete")
        spreads[task] = max(regime_spread_ratio(item, value) for item in regime_values)
    return {
        "task_primary_nrmse": task_values,
        "macro_primary_nrmse": sum(task_values.values()) / len(TASKS),
        "max_regime_spread_by_task": spreads,
        "regime_gate_passed": all(value <= 1.5 for value in spreads.values()),
    }


def _validate_access_and_provenance(
    summary: dict[str, Any], *, plan: dict[str, Any], label: str
) -> None:
    extra = summary.get("extra")
    heldout = summary.get("held_out_test_policy", {})
    if (
        summary.get("status") != "complete"
        or summary.get("split") != "val"
        or not isinstance(extra, dict)
        or extra.get("split") != "val"
        or extra.get("allow_held_out_test_eval")
        or heldout.get("enabled")
        or heldout.get("recorded")
    ):
        raise ValueError(f"{label} is incomplete or not validation-only")
    provenance = summary.get("data_provenance")
    binding = plan.get("bindings", {}).get("training_lock", {})
    if not isinstance(provenance, dict):
        raise ValueError(f"{label} lacks data provenance")
    for key in (
        "lock_sha256",
        "source_revision",
        "source_manifest_sha256",
        "protocol_manifest_sha256",
        "selection_sha256",
    ):
        if provenance.get(key) != binding.get(key):
            raise ValueError(f"{label} {key} differs from the confirmation plan")
    if provenance.get("purpose") != "training" or set(provenance.get("requested_roles", [])) != {
        "train",
        "valid",
    }:
        raise ValueError(f"{label} data roles are not exactly train and valid")
    if any(item.get("role") == "test" for item in provenance.get("objects", [])):
        raise ValueError(f"{label} provenance contains a test object")


def _validate_confirmation_summary(
    summary: dict[str, Any],
    *,
    path: Path,
    run: dict[str, Any],
    architecture: str,
    epoch: int,
    plan: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    seed = run["seed"]
    label = f"confirmation seed {seed}"
    _validate_access_and_provenance(summary, plan=plan, label=label)
    extra = summary["extra"]
    if (
        summary.get("run_name") != run.get("run_id")
        or extra.get("baseline") != f"external_neuraloperator_{architecture}"
        or extra.get("task") != list(TASKS)
        or extra.get("seed") != seed
        or extra.get("epochs") != epoch
    ):
        raise ValueError(f"{label} identity differs from the frozen selection")
    runner = plan["runner_identity"][architecture]
    runner_path = repo_root / runner["path"]
    if not runner_path.is_file() or _file_sha256(runner_path) != runner["file_sha256"]:
        raise ValueError("selected runner changed after confirmation planning")
    if run.get("command_sha256") != canonical_sha256(run.get("command")):
        raise ValueError(f"{label} planned command hash is invalid")
    if _parsed_command(
        extra.get("command"), architecture=architecture, runner_path=runner["path"]
    ) != _parsed_command(run.get("command"), architecture=architecture, runner_path=runner["path"]):
        raise ValueError(f"{label} command differs from the confirmation plan")

    history = summary.get("details", {}).get("validation_history")
    if not isinstance(history, list) or len(history) != 1 or history[0].get("epoch") != epoch:
        raise ValueError(f"{label} must contain exactly the selected validation rung")
    metrics = history[0].get("metrics")
    evidence = _metric_evidence(metrics)
    if not math.isclose(
        _finite(history[0].get("metric_value"), f"{label} macro"),
        evidence["macro_primary_nrmse"],
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{label} declared macro disagrees with full metrics")
    recipe = summary.get("recipe_adequacy", {})
    if (
        recipe.get("validation_rungs") != [epoch]
        or recipe.get("selected_epoch") != epoch
        or summary.get("metrics") != metrics
    ):
        raise ValueError(f"{label} selected-rung evidence is inconsistent")
    rungs = summary.get("checkpoints", {}).get("rungs")
    if not isinstance(rungs, dict) or set(rungs) != {str(epoch)}:
        raise ValueError(f"{label} checkpoint coverage is not exactly one rung")
    checkpoint = _validate_checkpoint(rungs[str(epoch)], epoch=epoch, repo_root=repo_root)
    selected = _validate_checkpoint(
        summary.get("checkpoints", {}).get("selected"), epoch=epoch, repo_root=repo_root
    )
    if checkpoint != selected:
        raise ValueError(f"{label} selected checkpoint differs from its rung")
    return {
        "seed": seed,
        **evidence,
        "checkpoint": checkpoint,
        "summary": {
            "path": str(path.resolve().relative_to(repo_root.resolve())),
            "sha256": _file_sha256(path),
        },
        "compute": _validate_compute(summary.get("compute"), label),
    }


def _validate_discovery_seed(
    selection: dict[str, Any], *, summary_path: Path, repo_root: Path
) -> dict[str, Any]:
    selected = selection["selection"]
    architecture = selected["architecture"]
    epoch = selected["epoch"]
    row = selection["architectures"][architecture]
    expected_summary = Path(row["summary"]["path"])
    expected_summary = (
        expected_summary if expected_summary.is_absolute() else repo_root / expected_summary
    )
    if summary_path.resolve() != expected_summary.resolve():
        raise ValueError("selected discovery summary path does not match the selection artifact")
    if _file_sha256(summary_path) != row["summary"]["sha256"]:
        raise ValueError("selected discovery summary hash does not match the selection artifact")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    history = summary.get("details", {}).get("validation_history", [])
    matches = [item for item in history if item.get("epoch") == epoch]
    if len(matches) != 1:
        raise ValueError("selected discovery rung is absent or ambiguous")
    evidence = _metric_evidence(matches[0].get("metrics"))
    checkpoint = _validate_checkpoint(
        summary.get("checkpoints", {}).get("rungs", {}).get(str(epoch)),
        epoch=epoch,
        repo_root=repo_root,
    )
    if checkpoint != selected["checkpoint"] or checkpoint != row["chosen_checkpoint"]:
        raise ValueError("selected discovery checkpoint identity is inconsistent")
    return {
        "seed": 17,
        **evidence,
        "checkpoint": checkpoint,
        "summary": row["summary"],
        "compute": _validate_compute(summary.get("compute"), "discovery seed 17"),
    }


def build_confirmed_recipe(
    plan: dict[str, Any],
    *,
    selection: dict[str, Any],
    selection_path: Path,
    discovery_summary_path: Path,
    confirmation_summary_paths: list[Path],
    repo_root: Path,
) -> dict[str, Any]:
    plan_sha = _validate_self_hash(plan, "plan_sha256", "confirmation plan")
    selection_sha = _validate_self_hash(selection, "selection_sha256", "selection artifact")
    if (
        plan.get("mode") != "validation_only"
        or plan.get("heldout_access") != "forbidden"
        or plan.get("measurement_lock_access") != "forbidden"
    ):
        raise ValueError("confirmation plan is not validation-only")
    if (
        selection.get("selection_id") != "strat-v1.1-reference-recipe-adequacy-selection-v1"
        or selection.get("status") != "complete_validation_only"
        or selection.get("no_eligible_architecture") is not False
        or selection.get("held_out_measurements") != 0
    ):
        raise ValueError("selection artifact is not eligible validation-only evidence")
    confirmation = plan.get("confirmation", {})
    binding = confirmation.get("evidence_binding", {})
    selection_binding = binding.get("selection_artifact", {})
    discovery_binding = binding.get("discovery_plan", {})
    selection_binding_path = Path(selection_binding.get("path", ""))
    selection_binding_path = (
        selection_binding_path
        if selection_binding_path.is_absolute()
        else repo_root / selection_binding_path
    )
    if (
        selection_binding.get("selection_sha256") != selection_sha
        or selection_binding.get("file_sha256") != _file_sha256(selection_path)
        or selection_binding_path.resolve() != selection_path.resolve()
        or discovery_binding.get("plan_sha256") != selection.get("plan_sha256")
    ):
        raise ValueError("confirmation plan evidence binding is inconsistent")
    discovery_plan_path = Path(discovery_binding.get("path", ""))
    discovery_plan_path = (
        discovery_plan_path
        if discovery_plan_path.is_absolute()
        else repo_root / discovery_plan_path
    )
    if not discovery_plan_path.is_file() or discovery_binding.get("file_sha256") != _file_sha256(
        discovery_plan_path
    ):
        raise ValueError("bound discovery plan file hash is invalid")
    discovery_plan = json.loads(discovery_plan_path.read_text(encoding="utf-8"))
    if (
        _validate_self_hash(discovery_plan, "plan_sha256", "discovery plan")
        != selection["plan_sha256"]
    ):
        raise ValueError("selection does not bind the discovery plan")

    selected = selection.get("selection")
    if not isinstance(selected, dict):
        raise ValueError("selection artifact has no eligible architecture")
    architecture, epoch = selected.get("architecture"), selected.get("epoch")
    selected_row = selection.get("architectures", {}).get(architecture)
    if (
        not isinstance(selected_row, dict)
        or selected_row.get("eligible") is not True
        or selected_row.get("chosen_epoch") != epoch
        or selected_row.get("chosen_macro_primary_nrmse") != selected.get("macro_primary_nrmse")
        or selected_row.get("chosen_checkpoint") != selected.get("checkpoint")
    ):
        raise ValueError("selected architecture row does not support the frozen selection")
    if (
        confirmation.get("selected_architecture") != architecture
        or confirmation.get("selected_epochs") != epoch
    ):
        raise ValueError("confirmation plan changed the selected architecture or epoch")
    runs = confirmation.get("runs")
    if (
        not isinstance(runs, list)
        or [(run.get("architecture"), run.get("seed"), run.get("epochs")) for run in runs]
        != [(architecture, 29, epoch), (architecture, 43, epoch)]
        or len(confirmation_summary_paths) != 2
    ):
        raise ValueError("confirmation requires exactly seeds 29 and 43 at the selected recipe")
    supplied = {
        str(path.resolve().relative_to(repo_root.resolve())): path
        for path in confirmation_summary_paths
    }
    expected = {run["expected_summary"]: run for run in runs}
    if set(supplied) != set(expected):
        raise ValueError("confirmation summaries do not match the two planned outputs")

    seed_rows = [
        _validate_discovery_seed(
            selection, summary_path=discovery_summary_path, repo_root=repo_root
        )
    ]
    for relative, run in expected.items():
        path = supplied[relative]
        seed_rows.append(
            _validate_confirmation_summary(
                json.loads(path.read_text(encoding="utf-8")),
                path=path,
                run=run,
                architecture=architecture,
                epoch=epoch,
                plan=plan,
                repo_root=repo_root,
            )
        )
    seed_rows.sort(key=lambda row: row["seed"])
    task_aggregate = {}
    for task in TASKS:
        values = [row["task_primary_nrmse"][task] for row in seed_rows]
        task_aggregate[task] = {"mean": statistics.fmean(values), "std": statistics.pstdev(values)}
    macros = [row["macro_primary_nrmse"] for row in seed_rows]
    all_eligible = all(row["regime_gate_passed"] for row in seed_rows)
    payload = {
        "schema_version": 1,
        "recipe_id": "strat-v1.1-claim-grade-reference-recipe-v1",
        "status": "confirmed_validation_only" if all_eligible else "stopped_regime_ineligible",
        "confirmation_plan_sha256": plan_sha,
        "selection_sha256": selection_sha,
        "architecture": architecture,
        "epoch": epoch,
        "seeds": seed_rows,
        "aggregate": {
            "task_primary_nrmse": task_aggregate,
            "macro_primary_nrmse": {
                "mean": statistics.fmean(macros),
                "std": statistics.pstdev(macros),
            },
        },
        "all_seeds_regime_eligible": all_eligible,
        "held_out_measurements": 0,
    }
    return {**payload, "recipe_sha256": canonical_sha256(payload)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--selection-artifact", required=True)
    parser.add_argument("--discovery-summary", required=True)
    parser.add_argument("--confirmation-summary", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else repo_root / path

    plan_path = resolve(args.plan)
    selection_path = resolve(args.selection_artifact)
    artifact = build_confirmed_recipe(
        json.loads(plan_path.read_text(encoding="utf-8")),
        selection=json.loads(selection_path.read_text(encoding="utf-8")),
        selection_path=selection_path,
        discovery_summary_path=resolve(args.discovery_summary),
        confirmation_summary_paths=[resolve(value) for value in args.confirmation_summary],
        repo_root=repo_root,
    )
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "recipe_sha256": artifact["recipe_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
