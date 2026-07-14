#!/usr/bin/env python
from __future__ import annotations

"""Independently verify D2 evidence and materialize its frozen gate result."""

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


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite numeric evidence")
    return float(value)


EXPECTED_BETA_REGIMES = (0.01, 0.1, 1.0, 10.0, 100.0)


def regime_values_match(values: list[Any]) -> bool:
    """Accept the frozen regimes after their lossless float32 HDF5 decode."""

    if len(values) != len(EXPECTED_BETA_REGIMES):
        return False
    try:
        observed = [float(value) for value in values]
    except (TypeError, ValueError):
        return False
    return all(
        math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-9)
        for actual, expected in zip(observed, EXPECTED_BETA_REGIMES, strict=True)
    )


def load_repair_manifest(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("materialization repair manifest schema must be 1")
    manifest_sha = manifest.get("manifest_sha256")
    if manifest_sha != canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    ):
        raise ValueError("materialization repair manifest self hash is invalid")
    return manifest


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


def _checkpoint_payload(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # torch < 2.6
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("checkpoint payload has the wrong format")
    return payload


def load_and_validate_plan(path: Path) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != 2:
        raise ValueError("D2 plan schema version must be 2")
    if (
        plan.get("mode") != "validation_only"
        or plan.get("heldout_access") != "forbidden"
        or plan.get("measurement_lock_access") != "forbidden"
    ):
        raise PermissionError("plan does not enforce validation-only access")
    plan_sha = plan.get("plan_sha256")
    if plan_sha != canonical_sha256(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    ):
        raise ValueError("plan self hash is invalid")
    command = plan.get("command")
    if not isinstance(command, list) or plan.get("command_sha256") != canonical_sha256(command):
        raise ValueError("plan command hash is invalid")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repair-manifest", type=Path)
    args = parser.parse_args()

    plan = load_and_validate_plan(args.plan)
    plan_sha = plan.get("plan_sha256")
    repair = load_repair_manifest(args.repair_manifest)
    materializer_relative = "scripts/materialize_darcy_fno_affine_head_ablation.py"
    for relative, expected in plan["bindings"]["source"]["files"].items():
        observed = sha(REPO_ROOT / relative)
        repaired_materializer = (
            repair is not None
            and relative == materializer_relative
            and repair.get("original_plan_sha256") == plan_sha
            and repair.get("original_materializer_sha256") == expected
            and repair.get("repaired_materializer_sha256") == observed
        )
        if observed != expected and not repaired_materializer:
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
    summary_commit = summary.get("source", {}).get("git_commit")
    if repair is None:
        if summary_commit != current:
            raise ValueError("summary commit differs from materialization checkout")
    elif (
        summary_commit != repair.get("measurement_commit")
        or subprocess.run(
            ["git", "merge-base", "--is-ancestor", str(summary_commit), current],
            cwd=REPO_ROOT,
        ).returncode
    ):
        raise ValueError("repair manifest does not bind an ancestor measurement commit")
    if repair is not None and (
        repair.get("summary_file_sha256") != sha(args.summary)
        or repair.get("summary_artifact_sha256") != summary_sha
    ):
        raise ValueError("repair manifest does not bind the exact summary")
    lock_binding = plan["bindings"]["training_lock"]
    lock = summary.get("training_lock", {})
    if (
        lock.get("lock_sha256") != lock_binding["lock_sha256"]
        or lock.get("lock_file_sha256") != lock_binding["file_sha256"]
    ):
        raise ValueError("summary training lock differs from plan")
    if {k: v.get("sha256") for k, v in lock.get("darcy_objects", {}).items()} != lock_binding[
        "darcy_objects"
    ]:
        raise ValueError("summary Darcy objects differ from plan")
    design = plan["design"]
    matched = summary.get("matched_design", {})
    if (
        matched.get("arms"),
        matched.get("seed"),
        matched.get("rungs"),
        matched.get("same_optimizer_and_raw_solution_mse"),
    ) != (design["arms"], design["seed"], design["epoch_rungs"], True):
        raise ValueError("summary design differs from plan")
    dependency = summary.get("architecture", {}).get("dependency", {})
    if (
        dependency.get("available") is not True
        or dependency.get("version") != plan["dependencies"]["neuraloperator"]
    ):
        raise ValueError("dependency evidence differs from plan")

    identities = summary.get("integrity_bindings", {})
    if identities.get("plan_fingerprint") != plan_sha:
        raise ValueError("summary plan fingerprint differs from the pre-registered plan")
    arm_rows: dict[str, Any] = {}
    for arm in design["arms"]:
        evidence = summary.get("arms", {}).get(arm, {})
        history = evidence.get("validation_history")
        if (
            not isinstance(history, list)
            or [x.get("epoch") for x in history] != design["epoch_rungs"]
        ):
            raise ValueError(f"arm {arm} has incomplete rung evidence")
        for row in history:
            finite(row.get("primary_value"), f"{arm} primary")
            regimes = row.get("per_beta")
            if not isinstance(regimes, list) or not regime_values_match(
                [item.get("beta") for item in regimes]
            ):
                raise ValueError(f"arm {arm} has incomplete regime evidence")
            for regime in regimes:
                finite(regime.get("slice_normalized_nrmse"), f"{arm} slice regime")
                finite(regime.get("global_scale_nrmse"), f"{arm} regime")
                finite(regime.get("spread_ratio_to_primary"), f"{arm} spread")
                count = regime.get("element_count")
                if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                    raise ValueError(f"{arm} regime element_count must be a positive integer")
            maximum_spread = max(float(regime["spread_ratio_to_primary"]) for regime in regimes)
            if not math.isclose(
                finite(row.get("maximum_corrected_spread_ratio"), f"{arm} max spread"),
                maximum_spread,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(f"arm {arm} maximum spread does not match regime rows")
        winner = min(history, key=lambda x: (x["primary_value"], x["epoch"]))
        selection = evidence.get("selection", {})
        if selection.get("selected_epoch") != winner["epoch"] or not math.isclose(
            finite(selection.get("selected_value"), f"{arm} selection"),
            winner["primary_value"],
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(f"arm {arm} selection is inconsistent")
        checkpoints = evidence.get("checkpoints", {}).get("rungs", {})
        if set(checkpoints) != {str(x) for x in design["epoch_rungs"]}:
            raise ValueError(f"arm {arm} checkpoint coverage is incomplete")
        parent = None
        for epoch in design["epoch_rungs"]:
            row = checkpoints[str(epoch)]
            path = Path(row["path"])
            if not path.is_absolute():
                path = REPO_ROOT / path
            record = verify_checkpoint_record(path, expected_checkpoint_sha256=row.get("sha256"))
            if (
                record.parent_checkpoint_sha256 != parent
                or row.get("parent_checkpoint_sha256") != parent
            ):
                raise ValueError(f"arm {arm} checkpoint lineage is invalid at epoch {epoch}")
            payload = _checkpoint_payload(path)
            if (
                payload.get("parent_checkpoint_sha256") != parent
                or payload.get("progress", {}).get("completed_epoch") != epoch
            ):
                raise ValueError(f"arm {arm} checkpoint payload progress/lineage mismatch")
            bindings = payload.get("bindings", {})
            for key in (
                "plan_fingerprint",
                "data_fingerprint",
                "source_fingerprint",
                "runtime_fingerprint",
            ):
                if bindings.get(key) != identities.get(key):
                    raise ValueError(f"arm {arm} checkpoint {key} differs from summary")
            parent = record.checkpoint_sha256
        threshold = plan["gates"]["plateau"]["best_so_far_relative_improvement_threshold"]
        required = plan["gates"]["plateau"]["consecutive_transitions_required"]
        beta100 = next(x for x in winner["per_beta"] if float(x["beta"]) == 100.0)
        arm_rows[arm] = {
            "selected_epoch": winner["epoch"],
            "primary_value": winner["primary_value"],
            "maximum_corrected_spread_ratio": winner["maximum_corrected_spread_ratio"],
            "beta100_global_scale_nrmse": beta100["global_scale_nrmse"],
            "plateau_epoch": plateau_epoch(history, threshold, required),
        }

    control, candidate = arm_rows["K-long"], arm_rows["A-affine"]
    diagnostics = summary.get("diagnostics", {})
    shuffled = finite(
        diagnostics["deterministic_shuffled_beta"]["arms"]["A-affine"][
            "relative_degradation_vs_true_beta"
        ],
        "candidate shuffled degradation",
    )
    sensitivity = finite(
        diagnostics["counterfactual_beta_sensitivity"]["A-affine"][
            "relative_prediction_rms_from_first_beta"
        ],
        "candidate sensitivity",
    )
    gates = plan["gates"]
    checks = {
        "heldout_zero": summary["held_out_reads"] == gates["heldout_reads"],
        "candidate_primary_below_frozen_d1": candidate["primary_value"]
        <= gates["candidate_primary_maximum"],
        "candidate_primary_better_than_control": candidate["primary_value"]
        < control["primary_value"],
        "candidate_beta100_better_than_control": candidate["beta100_global_scale_nrmse"]
        < control["beta100_global_scale_nrmse"],
        "candidate_regime_spread": candidate["maximum_corrected_spread_ratio"]
        <= gates["candidate_max_corrected_regime_spread_maximum"],
        "candidate_shuffled_beta_degradation": shuffled
        >= gates["candidate_shuffled_beta_relative_degradation_minimum"],
        "candidate_beta_sensitivity": sensitivity
        > gates["candidate_counterfactual_relative_prediction_rms_minimum_exclusive"],
        "plateau_by_cap": all(
            x["plateau_epoch"] is not None
            and x["plateau_epoch"] <= gates["plateau"]["must_occur_by_epoch"]
            for x in arm_rows.values()
        ),
    }
    result = {
        "schema_version": 2,
        "artifact_id": "strat-v1-darcy-fno-affine-head-ablation-d2-result",
        "status": "complete_validation_only",
        "heldout_reads": 0,
        "plan_sha256": plan_sha,
        "source_summary": {
            "path": str(args.summary),
            "file_sha256": sha(args.summary),
            "artifact_sha256": summary_sha,
        },
        "materialization_repair": (
            None
            if repair is None
            else {
                "manifest_path": str(args.repair_manifest),
                "manifest_sha256": repair["manifest_sha256"],
                "reason": repair["reason"],
                "original_materializer_sha256": repair["original_materializer_sha256"],
                "repaired_materializer_sha256": repair["repaired_materializer_sha256"],
            }
        ),
        "arms": arm_rows,
        "effect": {
            "candidate_primary_improvement_vs_control": (
                control["primary_value"] - candidate["primary_value"]
            )
            / control["primary_value"],
            "candidate_shuffled_beta_relative_degradation": shuffled,
            "candidate_counterfactual_relative_prediction_rms": sensitivity,
        },
        "gate_checks": checks,
        "all_gates_passed": all(checks.values()),
        "interpretation": (
            "affine_head_validated" if all(checks.values()) else "affine_head_not_validated"
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
