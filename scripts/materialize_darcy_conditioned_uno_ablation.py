#!/usr/bin/env python
from __future__ import annotations

"""Independently verify D4 UNO evidence and materialize its frozen gates."""

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from ups.data.manifests import canonical_sha256  # noqa: E402
from ups.training.resumable_checkpoint import FORMAT_VERSION, verify_checkpoint_record  # noqa: E402

EXPECTED_BETAS = (0.01, 0.1, 1.0, 10.0, 100.0)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite numeric evidence")
    return float(value)


def regime_values_match(values: list[Any]) -> bool:
    if len(values) != len(EXPECTED_BETAS):
        return False
    try:
        observed = [float(value) for value in values]
    except (TypeError, ValueError):
        return False
    return all(
        math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-9)
        for actual, expected in zip(observed, EXPECTED_BETAS, strict=True)
    )


def plateau_epoch(history: list[dict[str, Any]], threshold: float, required: int) -> int | None:
    best, stale = math.inf, 0
    for row in history:
        value = finite(row.get("primary_value"), "rung primary")
        previous, best = best, min(best, value)
        if not math.isinf(previous):
            stale = stale + 1 if (previous - best) / previous < threshold else 0
            if stale >= required:
                return int(row["epoch"])
    return None


def load_and_validate_plan(path: Path) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != 2:
        raise ValueError("D4 plan schema version must be 2")
    if (
        plan.get("mode") != "validation_only"
        or plan.get("heldout_access") != "forbidden"
        or plan.get("measurement_lock_access") != "forbidden"
    ):
        raise PermissionError("plan does not enforce validation-only access")
    if plan.get("plan_sha256") != canonical_sha256(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    ):
        raise ValueError("plan self hash is invalid")
    command = plan.get("command")
    if not isinstance(command, list) or plan.get("command_sha256") != canonical_sha256(command):
        raise ValueError("plan command hash is invalid")
    if any("test" in str(item).lower() for item in command):
        raise PermissionError("plan command is test-capable")
    return plan


def _checkpoint_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("checkpoint payload has the wrong format")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = load_and_validate_plan(args.plan)
    plan_sha = plan["plan_sha256"]
    for relative, expected in plan["bindings"]["source"]["files"].items():
        if sha(REPO_ROOT / relative) != expected:
            raise ValueError(f"source bytes changed after pre-registration: {relative}")
    implementation_commit = plan["bindings"]["source"]["implementation_commit"]
    current = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", implementation_commit, current], cwd=REPO_ROOT
    ).returncode:
        raise ValueError("implementation commit is not an ancestor of materialization HEAD")

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    summary_sha = summary.get("artifact_sha256")
    if summary_sha != canonical_sha256(
        {k: v for k, v in summary.items() if k != "artifact_sha256"}
    ):
        raise ValueError("summary self hash is invalid")
    if (summary.get("status"), summary.get("task"), summary.get("held_out_reads")) != (
        "complete_validation_only",
        "darcy2d",
        0,
    ):
        raise PermissionError("summary is not eligible validation-only Darcy evidence")
    if summary.get("source", {}).get("git_commit") != current:
        raise ValueError("summary commit differs from materialization checkout")
    lock_binding = plan["bindings"]["training_lock"]
    lock = summary.get("training_lock", {})
    if (
        lock.get("lock_sha256") != lock_binding["lock_sha256"]
        or lock.get("lock_file_sha256") != lock_binding["file_sha256"]
        or {k: v.get("sha256") for k, v in lock.get("darcy_objects", {}).items()}
        != lock_binding["darcy_objects"]
    ):
        raise ValueError("summary training lock differs from plan")
    identities = summary.get("integrity_bindings", {})
    if identities.get("plan_fingerprint") != plan_sha:
        raise ValueError("summary plan fingerprint differs from plan")
    design = plan["design"]
    frozen_control = plan["bindings"]["d3_result"]["R-mean"]
    historical = summary.get("historical_control", {})
    if (
        historical.get("arm") != "R-mean"
        or historical.get("selected_epoch") != frozen_control["selected_epoch"]
        or historical.get("plateau_epoch") != frozen_control["plateau_epoch"]
        or historical.get("primary_value") != frozen_control["primary_value"]
        or historical.get("beta100_global_scale_nrmse")
        != frozen_control["beta100_global_scale_nrmse"]
        or historical.get("maximum_corrected_spread_ratio")
        != frozen_control["maximum_corrected_spread_ratio"]
    ):
        raise ValueError("summary historical control differs from frozen D3")
    matched = summary.get("matched_design", {})
    if (
        matched.get("live_arms"),
        matched.get("seed"),
        matched.get("rungs"),
        matched.get("regime_complete_batch_size"),
        matched.get("samples_per_regime_per_batch"),
    ) != (design["live_arms"], design["seed"], design["epoch_rungs"], 10, 2):
        raise ValueError("summary design differs from plan")
    architecture = summary.get("architecture", {})
    dependency = architecture.get("dependency", {})
    if (
        dependency.get("available") is not True
        or dependency.get("version") != plan["dependencies"]["neuraloperator"]
    ):
        raise ValueError("dependency evidence differs from plan")
    observed_architecture = {k: v for k, v in architecture.items() if k != "dependency"}
    if observed_architecture != design["architecture"]:
        raise ValueError("UNO architecture differs from plan")
    optimizer = summary.get("optimizer", {})
    if optimizer.get("batch_size") != 10 or optimizer.get("objective") != design["objective"]:
        raise ValueError("summary objective differs from plan")

    arm = "U-conditioned"
    evidence = summary.get("arms", {}).get(arm, {})
    history = evidence.get("validation_history")
    if (
        not isinstance(history, list)
        or [row.get("epoch") for row in history] != design["epoch_rungs"]
    ):
        raise ValueError("UNO rung evidence is incomplete")
    for row in history:
        finite(row.get("primary_value"), "UNO primary")
        regimes = row.get("per_beta")
        if not isinstance(regimes, list) or not regime_values_match(
            [x.get("beta") for x in regimes]
        ):
            raise ValueError("UNO regime evidence is incomplete")
        for regime in regimes:
            finite(regime.get("slice_normalized_nrmse"), "UNO slice regime")
            finite(regime.get("global_scale_nrmse"), "UNO regime")
            finite(regime.get("spread_ratio_to_primary"), "UNO spread")
            if (
                isinstance(regime.get("element_count"), bool)
                or not isinstance(regime.get("element_count"), int)
                or regime["element_count"] <= 0
            ):
                raise ValueError("UNO regime element_count is invalid")
        maximum = max(float(x["spread_ratio_to_primary"]) for x in regimes)
        if not math.isclose(
            finite(row.get("maximum_corrected_spread_ratio"), "UNO max spread"),
            maximum,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("UNO maximum spread is inconsistent")
    winner = min(history, key=lambda row: (row["primary_value"], row["epoch"]))
    selection = evidence.get("selection", {})
    if selection.get("selected_epoch") != winner["epoch"] or not math.isclose(
        finite(selection.get("selected_value"), "UNO selection"),
        winner["primary_value"],
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("UNO selection is inconsistent")
    checkpoints = evidence.get("checkpoints", {}).get("rungs", {})
    if set(checkpoints) != {str(epoch) for epoch in design["epoch_rungs"]}:
        raise ValueError("UNO checkpoint coverage is incomplete")
    parent = None
    for epoch in design["epoch_rungs"]:
        row = checkpoints[str(epoch)]
        path = Path(row["path"])
        if not path.is_absolute():
            path = REPO_ROOT / path
        record = verify_checkpoint_record(path, expected_checkpoint_sha256=row.get("sha256"))
        payload = _checkpoint_payload(path)
        if (
            record.parent_checkpoint_sha256 != parent
            or row.get("parent_checkpoint_sha256") != parent
            or payload.get("parent_checkpoint_sha256") != parent
            or payload.get("progress", {}).get("completed_epoch") != epoch
        ):
            raise ValueError(f"UNO checkpoint lineage is invalid at epoch {epoch}")
        bindings = payload.get("bindings", {})
        for key in (
            "plan_fingerprint",
            "data_fingerprint",
            "source_fingerprint",
            "runtime_fingerprint",
        ):
            if bindings.get(key) != identities.get(key):
                raise ValueError(f"UNO checkpoint {key} differs from summary")
        parent = record.checkpoint_sha256

    beta100 = next(x for x in winner["per_beta"] if math.isclose(float(x["beta"]), 100.0))
    plateau = plateau_epoch(
        history,
        plan["gates"]["plateau"]["best_so_far_relative_improvement_threshold"],
        plan["gates"]["plateau"]["consecutive_transitions_required"],
    )
    diagnostics = summary.get("diagnostics", {})
    shuffled = finite(
        diagnostics["deterministic_shuffled_beta"]["arms"][arm][
            "relative_degradation_vs_true_beta"
        ],
        "UNO shuffled degradation",
    )
    sensitivity = finite(
        diagnostics["counterfactual_beta_sensitivity"][arm][
            "relative_prediction_rms_from_first_beta"
        ],
        "UNO beta sensitivity",
    )
    baseline = plan["bindings"]["d3_result"]["R-mean"]
    gates = plan["gates"]
    candidate = {
        "selected_epoch": winner["epoch"],
        "primary_value": winner["primary_value"],
        "beta100_global_scale_nrmse": beta100["global_scale_nrmse"],
        "maximum_corrected_spread_ratio": winner["maximum_corrected_spread_ratio"],
        "plateau_epoch": plateau,
    }
    checks = {
        "heldout_zero": summary["held_out_reads"] == gates["heldout_reads"],
        "candidate_primary_better_than_d3_control": candidate["primary_value"]
        < gates["candidate_primary_strictly_below_d3_control"],
        "candidate_beta100_better_than_d3_control": candidate["beta100_global_scale_nrmse"]
        < gates["candidate_beta100_strictly_below_d3_control"],
        "candidate_regime_spread": candidate["maximum_corrected_spread_ratio"]
        <= gates["candidate_max_corrected_regime_spread_maximum"],
        "candidate_shuffled_beta_degradation": shuffled
        >= gates["candidate_shuffled_beta_relative_degradation_minimum"],
        "candidate_beta_sensitivity": sensitivity
        > gates["candidate_counterfactual_relative_prediction_rms_minimum_exclusive"],
        "candidate_plateau_by_cap": plateau is not None
        and plateau <= gates["plateau"]["must_occur_by_epoch"],
    }
    result = {
        "schema_version": 2,
        "artifact_id": "strat-v1-darcy-conditioned-uno-d4-result",
        "status": "complete_validation_only",
        "heldout_reads": 0,
        "plan_sha256": plan_sha,
        "historical_baseline": baseline,
        "source_summary": {
            "path": str(args.summary),
            "file_sha256": sha(args.summary),
            "artifact_sha256": summary_sha,
        },
        "arms": {arm: candidate},
        "effect": {
            "candidate_primary_improvement_vs_d3_control": (
                baseline["primary_value"] - candidate["primary_value"]
            )
            / baseline["primary_value"],
            "candidate_shuffled_beta_relative_degradation": shuffled,
            "candidate_counterfactual_relative_prediction_rms": sensitivity,
        },
        "gate_checks": checks,
        "all_gates_passed": all(checks.values()),
        "interpretation": (
            "conditioned_uno_validated" if all(checks.values()) else "conditioned_uno_not_validated"
        ),
    }
    result["artifact_sha256"] = canonical_sha256(result)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite result: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "all_gates_passed": result["all_gates_passed"]}))


if __name__ == "__main__":
    main()
