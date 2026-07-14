#!/usr/bin/env python
from __future__ import annotations

"""Validate the paired Darcy ablation and materialize its frozen gate result."""

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ups.data.manifests import canonical_sha256  # noqa: E402


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def plateau_epoch(history: list[dict[str, Any]], threshold: float) -> int | None:
    best = math.inf
    stale = 0
    for row in history:
        value = finite(row.get("primary_value"), "rung primary")
        previous = best
        best = min(best, value)
        if not math.isinf(previous):
            improvement = (previous - best) / previous
            stale = stale + 1 if improvement < threshold else 0
            if stale == 2:
                return int(row["epoch"])
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    plan_sha = plan.get("plan_sha256")
    if plan_sha != canonical_sha256({k: v for k, v in plan.items() if k != "plan_sha256"}):
        raise ValueError("plan self hash is invalid")
    if plan.get("mode") != "validation_only" or plan.get("heldout_access") != "forbidden":
        raise PermissionError("plan is not validation-only")
    runner_binding = plan["bindings"]["runner"]
    runner = REPO_ROOT / runner_binding["path"]
    if file_sha256(runner) != runner_binding["file_sha256"]:
        raise ValueError("runner bytes changed after pre-registration")
    materializer_binding = plan["bindings"]["materializer"]
    if (
        materializer_binding["path"]
        != "scripts/materialize_darcy_fno_conditioning_ablation.py"
        or file_sha256(Path(__file__)) != materializer_binding["file_sha256"]
    ):
        raise ValueError("materializer bytes changed after pre-registration")
    source_binding = plan["bindings"]["source"]
    for relative, expected in source_binding["files"].items():
        if file_sha256(REPO_ROOT / relative) != expected:
            raise ValueError(f"source bytes changed after pre-registration: {relative}")
    current_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_binding["implementation_commit"], current_commit],
        cwd=REPO_ROOT,
        check=False,
    )
    if ancestor.returncode != 0:
        raise ValueError("implementation commit is not an ancestor of the run commit")

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    summary_sha = summary.get("artifact_sha256")
    if summary_sha != canonical_sha256({k: v for k, v in summary.items() if k != "artifact_sha256"}):
        raise ValueError("summary self hash is invalid")
    if (
        summary.get("status") != "complete_validation_only"
        or summary.get("held_out_reads") != 0
        or summary.get("task") != "darcy2d"
    ):
        raise PermissionError("summary is not eligible validation-only Darcy evidence")
    if summary.get("source", {}).get("git_commit") != current_commit:
        raise ValueError("summary git commit differs from the materialization checkout")
    dependency = summary.get("architecture", {}).get("dependency", {})
    if (
        dependency.get("available") is not True
        or dependency.get("version") != plan["dependencies"]["neuraloperator"]
    ):
        raise ValueError("summary neuraloperator dependency differs from the plan")
    lock = summary.get("training_lock", {})
    binding = plan["bindings"]["training_lock"]
    if lock.get("lock_sha256") != binding["lock_sha256"]:
        raise ValueError("summary lock differs from the plan")
    observed_objects = {
        key: value.get("sha256") for key, value in lock.get("darcy_objects", {}).items()
    }
    if observed_objects != binding["darcy_objects"]:
        raise ValueError("summary Darcy objects differ from the plan")

    design = plan["design"]
    matched = summary.get("matched_design", {})
    if (
        matched.get("arms") != design["arms"]
        or matched.get("seed") != design["seed"]
        or matched.get("rungs") != design["epoch_rungs"]
        or not matched.get("same_data_order_updates")
    ):
        raise ValueError("summary matched design differs from the plan")

    arm_rows: dict[str, Any] = {}
    plateau_rows: dict[str, int | None] = {}
    for arm in design["arms"]:
        evidence = summary.get("arms", {}).get(arm, {})
        history = evidence.get("validation_history")
        if not isinstance(history, list) or [row.get("epoch") for row in history] != design["epoch_rungs"]:
            raise ValueError(f"arm {arm} has incomplete rung evidence")
        for row in history:
            finite(row.get("primary_value"), f"arm {arm} primary")
            per_beta = row.get("per_beta")
            if not isinstance(per_beta, list) or len(per_beta) != 5:
                raise ValueError(f"arm {arm} lacks five regime rows")
            for regime in per_beta:
                finite(regime.get("global_scale_nrmse"), f"arm {arm} regime")
                finite(regime.get("spread_ratio_to_primary"), f"arm {arm} spread")
        selected = evidence.get("selection", {})
        winner = min(history, key=lambda row: (row["primary_value"], row["epoch"]))
        if selected.get("selected_epoch") != winner["epoch"] or not math.isclose(
            finite(selected.get("selected_value"), f"arm {arm} selection"),
            finite(winner["primary_value"], f"arm {arm} winner"),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(f"arm {arm} selection is inconsistent")
        checkpoints = evidence.get("checkpoints", {}).get("rungs", {})
        if set(checkpoints) != {str(epoch) for epoch in design["epoch_rungs"]}:
            raise ValueError(f"arm {arm} checkpoint coverage is incomplete")
        for record in checkpoints.values():
            path = Path(record["path"])
            if not path.is_absolute():
                path = REPO_ROOT / path
            if not path.is_file() or file_sha256(path) != record.get("sha256"):
                raise ValueError(f"arm {arm} checkpoint hash mismatch")
        compute = evidence.get("compute", {})
        for key in ("total_parameter_count", "trainable_parameter_count", "optimizer_steps", "examples_seen"):
            if isinstance(compute.get(key), bool) or not isinstance(compute.get(key), int) or compute[key] <= 0:
                raise ValueError(f"arm {arm} compute.{key} is invalid")
        finite(compute.get("duration_sec"), f"arm {arm} duration")
        plateau_rows[arm] = plateau_epoch(history, plan["gates"]["plateau"]["best_so_far_relative_improvement_threshold"])
        arm_rows[arm] = {
            "selected_epoch": winner["epoch"],
            "primary_value": winner["primary_value"],
            "maximum_corrected_spread_ratio": winner["maximum_corrected_spread_ratio"],
            "plateau_epoch": plateau_rows[arm],
            "compute": compute,
        }

    u = finite(arm_rows["U"]["primary_value"], "U primary")
    k = finite(arm_rows["K"]["primary_value"], "K primary")
    improvement = (u - k) / u
    shuffled = finite(summary["diagnostics"]["deterministic_shuffled_beta"]["primary_value"], "shuffled primary")
    shuffle_degradation = (shuffled - k) / k
    sensitivity = finite(
        summary["diagnostics"]["counterfactual_beta_sensitivity"]["K"]["relative_prediction_rms_from_first_beta"],
        "conditioned sensitivity",
    )
    gates = plan["gates"]
    checks = {
        "conditioned_primary_improvement": improvement >= gates["conditioned_primary_relative_improvement_minimum"],
        "conditioned_regime_spread": finite(arm_rows["K"]["maximum_corrected_spread_ratio"], "K spread") <= gates["conditioned_max_corrected_regime_spread_maximum"],
        "conditioned_beta_sensitivity": sensitivity > gates["conditioned_counterfactual_relative_prediction_rms_minimum_exclusive"],
        "shuffled_beta_degradation": shuffle_degradation >= gates["shuffled_beta_primary_relative_degradation_minimum"],
        "plateau_by_cap": all(value is not None and value <= gates["plateau"]["must_occur_by_epoch"] for value in plateau_rows.values()),
    }
    payload = {
        "schema_version": 1,
        "artifact_id": "strat-v1.1-darcy-fno-conditioning-ablation-result",
        "status": "complete_validation_only",
        "heldout_reads": 0,
        "plan_sha256": plan_sha,
        "source_summary": {"path": str(args.summary), "file_sha256": file_sha256(args.summary), "artifact_sha256": summary_sha},
        "arms": arm_rows,
        "effect": {
            "conditioned_primary_relative_improvement": improvement,
            "shuffled_beta_primary_value": shuffled,
            "shuffled_beta_relative_degradation": shuffle_degradation,
            "conditioned_counterfactual_relative_prediction_rms": sensitivity,
        },
        "gate_checks": checks,
        "all_gates_passed": all(checks.values()),
        "interpretation": "conditioning_mechanism_validated" if all(checks.values()) else "conditioning_mechanism_not_yet_validated",
    }
    payload["artifact_sha256"] = canonical_sha256(payload)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite result: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "artifact_sha256": payload["artifact_sha256"], "all_gates_passed": payload["all_gates_passed"]}))


if __name__ == "__main__":
    main()
