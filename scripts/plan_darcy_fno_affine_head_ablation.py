#!/usr/bin/env python
from __future__ import annotations

"""Pre-register the validation-only Darcy K-long/A-affine D2 experiment."""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ups.data.manifests import canonical_sha256, load_data_lock  # noqa: E402

LOCK_SHA256 = "5666fa8bb646b270d23077d1936c8ee80fa1cfeaf8f5870c55103fc58dd80afd"
DARCY_OBJECTS = {
    "darcy2d-train": "47945f27fa1f56f856733d3bc1aa1b0b5f498669a73cdb7352940292d71d09fe",
    "darcy2d-valid": "2b345a587f6f95a9ff4a12f6cce80ac4c8c83540a03c2a11f87ffdc91be1b595",
}
D1_PRIMARY = 0.18947497984876743
SOURCE_FILES = (
    "scripts/run_darcy_fno_affine_head_ablation.py",
    "scripts/run_darcy_fno_conditioning_ablation.py",
    "scripts/run_external_neuraloperator_fno_baseline.py",
    "src/ups/training/resumable_checkpoint.py",
    "scripts/plan_darcy_fno_affine_head_ablation.py",
    "scripts/materialize_darcy_fno_affine_head_ablation.py",
    "scripts/run_remote_darcy_fno_affine_head_ablation.sh",
    "scripts/launch_darcy_fno_affine_head_ablation_vast.sh",
    "scripts/vast_remote_bootstrap.sh",
    "scripts/vast_launch.py",
    "scripts/vast_watchdog.py",
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_paths() -> tuple[str, ...]:
    shared = tuple(
        str(path.relative_to(REPO_ROOT)) for path in sorted((REPO_ROOT / "src/ups").rglob("*.py"))
    )
    return tuple(dict.fromkeys((*shared, *SOURCE_FILES)))


def source_manifest(implementation_commit: str) -> dict[str, str]:
    """Prove that every registered source is tracked and equals the commit bytes."""

    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *source_paths()], cwd=REPO_ROOT, text=True
    )
    if status.strip():
        raise ValueError(f"D2 source paths must be clean and tracked:\n{status.rstrip()}")
    manifest: dict[str, str] = {}
    for relative in source_paths():
        live = (REPO_ROOT / relative).read_bytes()
        try:
            committed = subprocess.check_output(
                ["git", "show", f"{implementation_commit}:{relative}"], cwd=REPO_ROOT
            )
        except subprocess.CalledProcessError as exc:
            raise ValueError(
                f"D2 source is not tracked by implementation commit: {relative}"
            ) from exc
        if live != committed:
            raise ValueError(f"D2 source bytes differ from implementation commit: {relative}")
        manifest[relative] = hashlib.sha256(live).hexdigest()
    return manifest


def checked_artifact(path: Path, *, label: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.get("artifact_sha256")
    if recorded != canonical_sha256({k: v for k, v in payload.items() if k != "artifact_sha256"}):
        raise ValueError(f"{label} self hash is invalid")
    if payload.get("status") != "complete_validation_only":
        raise PermissionError(f"{label} is not validation-only evidence")
    heldout = payload.get("heldout_reads", payload.get("access", {}).get("heldout_reads"))
    if heldout != 0:
        raise PermissionError(f"{label} used held-out data")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-lock", type=Path, required=True)
    parser.add_argument("--d1-result", type=Path, required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--output-plan", type=Path, required=True)
    args = parser.parse_args()

    if not re.fullmatch(r"[0-9a-f]{40}", args.implementation_commit):
        raise ValueError("implementation commit must be a full lowercase 40-character commit")
    resolved = subprocess.check_output(
        ["git", "rev-parse", f"{args.implementation_commit}^{{commit}}"], cwd=REPO_ROOT, text=True
    ).strip()
    if resolved != args.implementation_commit:
        raise ValueError("implementation commit did not resolve exactly")

    lock = load_data_lock(args.training_lock)
    if lock.lock_sha256 != LOCK_SHA256 or lock.purpose != "training":
        raise ValueError("D2 requires the frozen universal training lock")
    if set(lock.requested_roles) != {"train", "valid"} or any(
        x.role == "test" for x in lock.objects
    ):
        raise PermissionError("D2 must remain train/validation-only")
    observed = {
        x.object_id: x.checksums.get("sha256") for x in lock.objects if x.object_id in DARCY_OBJECTS
    }
    if observed != DARCY_OBJECTS:
        raise ValueError("Darcy object identities differ from the D2 contract")

    d1 = checked_artifact(args.d1_result, label="D1 result")
    d1_primary = float(d1.get("arms", {}).get("K", {}).get("primary_value", float("nan")))
    if d1_primary != D1_PRIMARY:
        raise ValueError("D1 conditioned baseline differs from the frozen value")
    missing = [relative for relative in source_paths() if not (REPO_ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"D2 source files are missing: {missing}")
    sources = source_manifest(resolved)

    command = [
        "python",
        "scripts/run_darcy_fno_affine_head_ablation.py",
        "--training-lock",
        str(args.training_lock),
        "--data-root",
        args.data_root,
        "--output-dir",
        args.output_dir,
        "--plan-path",
        str(args.output_plan),
        "--hidden-channels",
        "16",
        "--fourier-modes",
        "16",
        "--n-layers",
        "4",
        "--learning-rate",
        "0.001",
        "--weight-decay",
        "0.0001",
        "--batch-size",
        "8",
        "--device",
        args.device,
    ]
    plan = {
        "schema_version": 2,
        "plan_id": "strat-v1-darcy-fno-affine-head-ablation-d2",
        "mode": "validation_only",
        "heldout_access": "forbidden",
        "measurement_lock_access": "forbidden",
        "question": "Does a beta-affine output basis remove the remaining beta=100 Darcy error concentration?",
        "bindings": {
            "source": {
                "implementation_commit": resolved,
                "files": sources,
            },
            "training_lock": {
                "path": str(args.training_lock),
                "lock_sha256": lock.lock_sha256,
                "file_sha256": file_sha256(args.training_lock),
                "darcy_objects": DARCY_OBJECTS,
            },
            "d1_result": {
                "path": str(args.d1_result),
                "file_sha256": file_sha256(args.d1_result),
                "artifact_sha256": d1["artifact_sha256"],
                "conditioned_primary": D1_PRIMARY,
            },
        },
        "dependencies": {"neuraloperator": "2.0.0"},
        "design": {
            "arms": ["K-long", "A-affine"],
            "seed": 17,
            "epoch_rungs": [3, 6, 12, 24, 48, 96, 192],
            "objective": "raw_solution_mse",
            "single_continuous_trajectory_per_arm": True,
            "same_data_order_optimizer_and_updates": True,
            "K-long": "D1 conditioned FNO control trained to the longer cap",
            "A-affine": "coefficient plus presence to two basis fields; h0 + train-zscored raw beta * h1",
        },
        "gates": {
            "finite_all_primary_and_regime_metrics": True,
            "candidate_max_corrected_regime_spread_maximum": 1.5,
            "candidate_primary_maximum": D1_PRIMARY,
            "candidate_primary_strictly_better_than_control": True,
            "candidate_beta100_global_scale_nrmse_strictly_better_than_control": True,
            "candidate_shuffled_beta_relative_degradation_minimum": 0.05,
            "candidate_counterfactual_relative_prediction_rms_minimum_exclusive": 1e-6,
            "plateau": {
                "best_so_far_relative_improvement_threshold": 0.01,
                "consecutive_transitions_required": 2,
                "must_occur_by_epoch": 192,
            },
            "heldout_reads": 0,
        },
        "command": command,
        "command_sha256": canonical_sha256(command),
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    if args.output_plan.exists():
        raise FileExistsError(f"refusing to overwrite plan: {args.output_plan}")
    args.output_plan.parent.mkdir(parents=True, exist_ok=True)
    args.output_plan.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output_plan), "plan_sha256": plan["plan_sha256"]}))


if __name__ == "__main__":
    main()
