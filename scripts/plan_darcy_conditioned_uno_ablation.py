#!/usr/bin/env python
from __future__ import annotations

"""Pre-register the validation-only Darcy conditioned-UNO D4 experiment."""

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
D3_ARTIFACT_SHA256 = "25d29a00d69233acfd2e05789f4dd6bd5488a8011cc584a7a11ba8d384d1ee1a"
D3_BASELINE = {
    "selected_epoch": 384,
    "primary_value": 0.11694165553982801,
    "beta100_global_scale_nrmse": 0.25762147974587746,
    "maximum_corrected_spread_ratio": 2.2029915564017024,
    "plateau_epoch": 384,
}
SOURCE_FILES = (
    "scripts/run_darcy_conditioned_uno_ablation.py",
    "scripts/run_darcy_fno_regime_balanced_objective.py",
    "scripts/run_darcy_fno_affine_head_ablation.py",
    "scripts/run_darcy_fno_conditioning_ablation.py",
    "scripts/run_external_neuraloperator_uno_baseline.py",
    "scripts/run_external_neuraloperator_fno_baseline.py",
    "src/ups/training/resumable_checkpoint.py",
    "scripts/plan_darcy_conditioned_uno_ablation.py",
    "scripts/materialize_darcy_conditioned_uno_ablation.py",
    "scripts/run_remote_darcy_conditioned_uno_ablation.sh",
    "scripts/launch_darcy_conditioned_uno_ablation_vast.sh",
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
    paths = source_paths()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *paths], cwd=REPO_ROOT, text=True
    )
    if status.strip():
        raise ValueError(f"D4 source paths must be clean and tracked:\n{status.rstrip()}")
    manifest: dict[str, str] = {}
    for relative in paths:
        live = (REPO_ROOT / relative).read_bytes()
        try:
            committed = subprocess.check_output(
                ["git", "show", f"{implementation_commit}:{relative}"], cwd=REPO_ROOT
            )
        except subprocess.CalledProcessError as exc:
            raise ValueError(f"D4 source is not tracked: {relative}") from exc
        if live != committed:
            raise ValueError(f"D4 source differs from implementation commit: {relative}")
        manifest[relative] = hashlib.sha256(live).hexdigest()
    return manifest


def checked_d3(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.get("artifact_sha256")
    if recorded != canonical_sha256({k: v for k, v in payload.items() if k != "artifact_sha256"}):
        raise ValueError("D3 result self hash is invalid")
    if recorded != D3_ARTIFACT_SHA256:
        raise ValueError("D3 result differs from the frozen artifact")
    if payload.get("status") != "complete_validation_only" or payload.get("heldout_reads") != 0:
        raise PermissionError("D3 result is not validation-only evidence")
    if payload.get("arms", {}).get("R-mean") != D3_BASELINE:
        raise ValueError("D3 R-mean baseline differs from the frozen values")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-lock", type=Path, required=True)
    parser.add_argument("--d3-result", type=Path, required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--output-plan", type=Path, required=True)
    args = parser.parse_args()

    if not re.fullmatch(r"[0-9a-f]{40}", args.implementation_commit):
        raise ValueError("implementation commit must be a full lowercase commit")
    resolved = subprocess.check_output(
        ["git", "rev-parse", f"{args.implementation_commit}^{{commit}}"], cwd=REPO_ROOT, text=True
    ).strip()
    if resolved != args.implementation_commit:
        raise ValueError("implementation commit did not resolve exactly")
    lock = load_data_lock(args.training_lock)
    if lock.lock_sha256 != LOCK_SHA256 or lock.purpose != "training":
        raise ValueError("D4 requires the frozen universal training lock")
    if set(lock.requested_roles) != {"train", "valid"} or any(
        x.role == "test" for x in lock.objects
    ):
        raise PermissionError("D4 must remain train/validation-only")
    observed = {
        x.object_id: x.checksums.get("sha256") for x in lock.objects if x.object_id in DARCY_OBJECTS
    }
    if observed != DARCY_OBJECTS:
        raise ValueError("Darcy object identities differ from the D4 contract")
    d3 = checked_d3(args.d3_result)
    sources = source_manifest(resolved)
    command = [
        "python",
        "scripts/run_darcy_conditioned_uno_ablation.py",
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
        "--lifting-channels",
        "32",
        "--projection-channels",
        "32",
        "--channel-mlp-skip",
        "linear",
        "--learning-rate",
        "0.001",
        "--weight-decay",
        "0.0001",
        "--batch-size",
        "10",
        "--device",
        args.device,
    ]
    plan = {
        "schema_version": 2,
        "plan_id": "strat-v1-darcy-conditioned-uno-d4",
        "mode": "validation_only",
        "heldout_access": "forbidden",
        "measurement_lock_access": "forbidden",
        "question": "Does parameter-conditioned UNO remove the residual Darcy regime imbalance?",
        "bindings": {
            "source": {"implementation_commit": resolved, "files": sources},
            "training_lock": {
                "path": str(args.training_lock),
                "lock_sha256": lock.lock_sha256,
                "file_sha256": file_sha256(args.training_lock),
                "darcy_objects": DARCY_OBJECTS,
            },
            "d3_result": {
                "path": str(args.d3_result),
                "file_sha256": file_sha256(args.d3_result),
                "artifact_sha256": d3["artifact_sha256"],
                "R-mean": D3_BASELINE,
            },
        },
        "dependencies": {"neuraloperator": "2.0.0"},
        "design": {
            "live_arms": ["U-conditioned"],
            "historical_baseline": "D3/R-mean",
            "seed": 17,
            "epoch_rungs": [3, 6, 12, 24, 48, 96, 192, 384],
            "batch_size": 10,
            "batch_composition": "two_samples_per_each_of_five_beta_regimes",
            "objective": "mean_per_regime_raw_mse",
            "single_continuous_trajectory": True,
            "conditioning": "coefficient_plus_train_zscore_log10_beta_plus_presence_direct_solution",
            "architecture": {
                "implementation": "neuralop.models.UNO",
                "hidden_channels": 16,
                "fourier_modes": 16,
                "n_layers": 4,
                "lifting_channels": 32,
                "projection_channels": 32,
                "channel_mlp_skip": "linear",
                "identity_scaling": False,
                "residual": False,
                "in_channels": 3,
                "out_channels": 1,
            },
        },
        "gates": {
            "candidate_primary_strictly_below_d3_control": D3_BASELINE["primary_value"],
            "candidate_beta100_strictly_below_d3_control": D3_BASELINE[
                "beta100_global_scale_nrmse"
            ],
            "candidate_max_corrected_regime_spread_maximum": 1.5,
            "candidate_shuffled_beta_relative_degradation_minimum": 0.05,
            "candidate_counterfactual_relative_prediction_rms_minimum_exclusive": 1e-6,
            "plateau": {
                "best_so_far_relative_improvement_threshold": 0.01,
                "consecutive_transitions_required": 2,
                "must_occur_by_epoch": 384,
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
