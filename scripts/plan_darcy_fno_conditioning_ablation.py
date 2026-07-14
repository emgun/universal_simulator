#!/usr/bin/env python
from __future__ import annotations

"""Freeze the validation-only Darcy FNO conditioning ablation contract."""

import argparse
import hashlib
import json
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
SOURCE_ROOTS = (
    "src/ups",
    "scripts/run_darcy_fno_conditioning_ablation.py",
    "scripts/run_external_neuraloperator_fno_baseline.py",
    "scripts/run_physical_conv_baseline.py",
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_manifest() -> dict[str, str]:
    paths: list[Path] = []
    for value in SOURCE_ROOTS:
        path = REPO_ROOT / value
        paths.extend(sorted(path.rglob("*.py")) if path.is_dir() else [path])
    return {str(path.relative_to(REPO_ROOT)): file_sha256(path) for path in paths}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-lock", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--output-plan", type=Path, required=True)
    args = parser.parse_args()

    resolved_commit = subprocess.check_output(
        ["git", "rev-parse", f"{args.implementation_commit}^{{commit}}"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    if resolved_commit != args.implementation_commit:
        raise ValueError("implementation commit must be a full 40-character commit")

    lock = load_data_lock(args.training_lock)
    if lock.lock_sha256 != LOCK_SHA256 or lock.purpose != "training":
        raise ValueError("ablation requires the frozen universal training lock")
    if set(lock.requested_roles) != {"train", "valid"} or any(
        item.role == "test" for item in lock.objects
    ):
        raise PermissionError("ablation plan must remain train/validation-only")
    observed = {
        item.object_id: item.checksums.get("sha256")
        for item in lock.objects
        if item.object_id in DARCY_OBJECTS
    }
    if observed != DARCY_OBJECTS:
        raise ValueError("Darcy object identities differ from the frozen contract")

    diagnostic = json.loads(args.diagnostic.read_text(encoding="utf-8"))
    recorded = diagnostic.get("artifact_sha256")
    unhashed = {k: v for k, v in diagnostic.items() if k != "artifact_sha256"}
    if recorded != canonical_sha256(unhashed):
        raise ValueError("D0 diagnostic self hash is invalid")
    if (
        diagnostic.get("status") != "complete_validation_only"
        or diagnostic.get("access", {}).get("heldout_reads") != 0
        or diagnostic.get("bindings", {}).get("training_lock_sha256") != LOCK_SHA256
    ):
        raise ValueError("D0 diagnostic is not eligible validation evidence")

    runner = REPO_ROOT / "scripts/run_darcy_fno_conditioning_ablation.py"
    materializer = REPO_ROOT / "scripts/materialize_darcy_fno_conditioning_ablation.py"
    command = [
        "python",
        "scripts/run_darcy_fno_conditioning_ablation.py",
        "--training-lock", str(args.training_lock),
        "--data-root", args.data_root,
        "--output-dir", args.output_dir,
        "--hidden-channels", "16",
        "--fourier-modes", "16",
        "--n-layers", "4",
        "--learning-rate", "0.001",
        "--weight-decay", "0.0001",
        "--batch-size", "8",
        "--device", args.device,
    ]
    payload = {
        "schema_version": 1,
        "plan_id": "strat-v1.1-darcy-fno-conditioning-ablation-d1",
        "mode": "validation_only",
        "heldout_access": "forbidden",
        "measurement_lock_access": "forbidden",
        "question": "Does explicit beta conditioning remove the Darcy identifiability ceiling under a matched FNO recipe?",
        "bindings": {
            "source": {
                "implementation_commit": resolved_commit,
                "files": source_manifest(),
            },
            "training_lock": {
                "path": str(args.training_lock),
                "lock_sha256": lock.lock_sha256,
                "file_sha256": file_sha256(args.training_lock),
                "darcy_objects": DARCY_OBJECTS,
            },
            "d0_diagnostic": {
                "path": str(args.diagnostic),
                "file_sha256": file_sha256(args.diagnostic),
                "artifact_sha256": recorded,
            },
            "runner": {
                "path": str(runner.relative_to(REPO_ROOT)),
                "file_sha256": file_sha256(runner),
            },
            "materializer": {
                "path": str(materializer.relative_to(REPO_ROOT)),
                "file_sha256": file_sha256(materializer),
            },
        },
        "dependencies": {"neuraloperator": "2.0.0"},
        "design": {
            "arms": ["U", "K"],
            "U": "coefficient only",
            "K": "coefficient plus train-normalized log10(beta) constant channel plus presence channel",
            "seed": 17,
            "epoch_rungs": [3, 6, 12, 24],
            "same_data_order_optimizer_updates_and_selection_rule": True,
            "single_continuous_trajectory_per_arm": True,
        },
        "gates": {
            "conditioned_primary_relative_improvement_minimum": 0.10,
            "conditioned_max_corrected_regime_spread_maximum": 1.5,
            "conditioned_counterfactual_relative_prediction_rms_minimum_exclusive": 1e-6,
            "shuffled_beta_primary_relative_degradation_minimum": 0.05,
            "finite_all_primary_and_regime_metrics": True,
            "plateau": {
                "best_so_far_relative_improvement_threshold": 0.01,
                "consecutive_transitions_required": 2,
                "must_occur_by_epoch": 24,
            },
        },
        "command": command,
        "command_sha256": canonical_sha256(command),
    }
    payload["plan_sha256"] = canonical_sha256(payload)
    args.output_plan.parent.mkdir(parents=True, exist_ok=True)
    if args.output_plan.exists():
        raise FileExistsError(f"refusing to overwrite plan: {args.output_plan}")
    args.output_plan.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output_plan), "plan_sha256": payload["plan_sha256"]}))


if __name__ == "__main__":
    main()
