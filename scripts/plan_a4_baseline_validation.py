#!/usr/bin/env python
"""Plan or explicitly launch the A4 strat-v1 validation baseline wall.

The default mode is validation-only planning: it verifies the immutable training
control plane and writes commands, but never imports model packages, opens HDF5
data, or starts training.  ``--execute`` remains validation-only (train + valid)
and requires a separate acknowledgement; test objects are never accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ups.data.manifests import (  # noqa: E402
    canonical_sha256,
    load_data_lock,
    load_protocol_manifest,
    load_source_manifest,
)

TASKS = ("advection1d", "burgers1d", "darcy2d")
METRIC = "decoded_rollout_nrmse"
A4_TRAINING_LOCK_SHA256 = "5666fa8bb646b270d23077d1936c8ee80fa1cfeaf8f5870c55103fc58dd80afd"
A4_SOURCE_REVISION = "sha256:9d43d283f04f5b8d17cf6126ad189075c53307e715d7d4f61af440c2fed155c1"
RUNNERS = {
    "persistence": "scripts/run_persistence_baseline.py",
    "fno": "scripts/run_external_neuraloperator_fno_baseline.py",
    "uno": "scripts/run_external_neuraloperator_uno_baseline.py",
    "unet": "scripts/run_external_pdebench_unet_baseline.py",
    "cno": "scripts/run_external_cno_baseline.py",
}
MODEL_IDENTITIES = {
    "persistence": {
        "implementation": "decoded physical-space persistence",
        "source": "repository",
    },
    "fno": {
        "implementation": "neuralop.models.FNO",
        "source": "https://github.com/neuraloperator/neuraloperator",
    },
    "uno": {
        "implementation": "neuralop.models.UNO",
        "source": "https://github.com/neuraloperator/neuraloperator",
    },
    "unet": {
        "implementation": "pdebench.models.unet.unet.UNet1d/UNet2d",
        "source": "https://github.com/pdebench/PDEBench",
        "source_revision": "4ff3e3a4aa1561721b5571fa3a048a0a463e0568",
    },
    "cno": {
        "implementation": "camlab-ethz.ConvolutionalNeuralOperator.CNO1d_simplified.CNO1d",
        "source": "https://github.com/camlab-ethz/ConvolutionalNeuralOperator",
        "source_revision": "6e765198aa02b56352e0a3437104b9d9e337176e",
        "scope": "height=1 grids only",
    },
}
COMMON_FIT = {
    # Burgers is the largest temporal training shard: 288 samples times four
    # stride-4 transitions through horizon 16.  Covering all 1,152 pairs avoids
    # the legacy first-N cap, which would select only early regime blocks.
    "max_pairs_per_task": 1152,
    "train_stride": 4,
    "epochs": 3,
    "learning_rate": 1.0e-3,
    "weight_decay": 1.0e-4,
    "batch_size": 8,
    "seed": 17,
}
MODEL_FIT = {
    "fno": {"hidden_channels": 16, "fourier_modes": 16, "n_layers": 4},
    "uno": {
        "hidden_channels": 16,
        "fourier_modes": 16,
        "n_layers": 4,
        "lifting_channels": 32,
        "projection_channels": 32,
        "channel_mlp_skip": "linear",
    },
    "unet": {"init_features": 32},
    "cno": {
        "n_layers": 3,
        "n_res": 1,
        "n_res_neck": 1,
        "channel_multiplier": 8,
        "lift_latent_dim": 64,
    },
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _task_from_object_id(object_id: str) -> str:
    for task in TASKS:
        if object_id == f"{task}-train" or object_id == f"{task}-valid":
            return task
    raise ValueError(f"A4 training lock contains an unexpected object: {object_id}")


def _validate_control_plane(training_lock_path: Path, source_path: Path, protocol_path: Path):
    lock = load_data_lock(training_lock_path)
    source = load_source_manifest(source_path)
    protocol = load_protocol_manifest(protocol_path)
    if lock.purpose != "training" or set(lock.requested_roles) != {"train", "valid"}:
        raise ValueError("A4 requires one training lock containing exactly train and valid roles")
    if lock.lock_sha256 != A4_TRAINING_LOCK_SHA256 or lock.source_revision != A4_SOURCE_REVISION:
        raise ValueError("A4 requires the frozen universal strat-v1 training lock identity")
    if lock.measurement_contract_id is not None or any(
        item.role == "test" for item in lock.objects
    ):
        raise ValueError("A4 validation planning forbids test access")
    if lock.source_manifest_sha256 != source.manifest_sha256:
        raise ValueError("Training lock source manifest hash does not match")
    if lock.protocol_manifest_sha256 != protocol.manifest_sha256:
        raise ValueError("Training lock protocol manifest hash does not match")
    if lock.source_revision != source.revision or protocol.source_revision != source.revision:
        raise ValueError("Training lock and manifests do not share one source revision")

    roles_by_task = {task: set() for task in TASKS}
    for item in lock.objects:
        roles_by_task[_task_from_object_id(item.object_id)].add(item.role)
    incomplete = [task for task, roles in roles_by_task.items() if roles != {"train", "valid"}]
    if incomplete:
        raise ValueError(f"A4 training lock lacks train/valid objects for: {', '.join(incomplete)}")
    return lock, source, protocol


def _command(
    *,
    model: str,
    task: str,
    config: Path,
    data_root: Path,
    output_root: Path,
    max_train_samples: int,
    max_eval_samples: int,
    rollout_steps: int,
    device: str,
    training_lock: Path,
    training_lock_sha256: str,
) -> list[str]:
    run_name = (
        "a4_strat_v1_universal_persistence_val"
        if model == "persistence"
        else f"a4_strat_v1_{task}_{model}_val"
    )
    base = [sys.executable, RUNNERS[model], "--config", str(config), "--name", run_name]
    base += ["--output-root", str(output_root), "--data-root", str(data_root)]
    if model == "persistence":
        for task_name in TASKS:
            base.extend(["--task", task_name])
    else:
        base.extend(["--task", task])
    base += [
        "--data-lock",
        str(training_lock),
        "--expected-data-lock-sha256",
        training_lock_sha256,
    ]
    if model == "persistence":
        return base + [
            "--split",
            "val",
            "--rollout-steps",
            str(rollout_steps),
        ]
    command = base + [
        "--train-split",
        "train",
        "--eval-split",
        "val",
        "--max-train-samples",
        str(max_train_samples),
        "--max-eval-samples",
        str(max_eval_samples),
        "--rollout-steps",
        str(rollout_steps),
        "--max-pairs-per-task",
        str(COMMON_FIT["max_pairs_per_task"]),
        "--train-stride",
        str(COMMON_FIT["train_stride"]),
        "--epochs",
        str(COMMON_FIT["epochs"]),
        "--learning-rate",
        str(COMMON_FIT["learning_rate"]),
        "--weight-decay",
        str(COMMON_FIT["weight_decay"]),
        "--batch-size",
        str(COMMON_FIT["batch_size"]),
        "--seed",
        str(COMMON_FIT["seed"]),
        "--device",
        device,
        "--metric",
        METRIC,
        "--strict-contract",
    ]
    for name, value in MODEL_FIT[model].items():
        command.extend([f"--{name.replace('_', '-')}", str(value)])
    return command


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    training_lock_path = Path(args.training_lock)
    source_path = Path(args.source_manifest)
    protocol_path = Path(args.protocol_manifest)
    config = Path(args.config)
    for path in (training_lock_path, source_path, protocol_path, config):
        if not path.is_file():
            raise FileNotFoundError(path)
    lock, source, protocol = _validate_control_plane(training_lock_path, source_path, protocol_path)
    source_by_id = {item.object_id: item for item in source.objects}
    sample_counts = {
        task: {
            role: int(source_by_id[f"{task}-{role}"].metadata["sample_count"])
            for role in ("train", "valid")
        }
        for task in TASKS
    }
    required_train_samples = max(counts["train"] for counts in sample_counts.values())
    required_eval_samples = max(counts["valid"] for counts in sample_counts.values())
    temporal_pairs_per_sample = len(range(0, args.rollout_steps, COMMON_FIT["train_stride"]))
    required_pair_cap = max(
        sample_counts[task]["train"] * temporal_pairs_per_sample
        for task in ("advection1d", "burgers1d")
    )
    if args.max_train_samples < required_train_samples:
        raise ValueError(
            "A4 forbids first-N training truncation; "
            f"max_train_samples must be at least {required_train_samples}"
        )
    if args.max_eval_samples < required_eval_samples:
        raise ValueError(
            "A4 forbids first-N validation truncation; "
            f"max_eval_samples must be at least {required_eval_samples}"
        )
    if COMMON_FIT["max_pairs_per_task"] < required_pair_cap:
        raise ValueError("A4 pair cap does not cover every strat-v1 temporal training sample")

    model_identities = json.loads(json.dumps(MODEL_IDENTITIES))
    for model, runner in RUNNERS.items():
        runner_path = REPO_ROOT / runner
        model_identities[model]["runner"] = runner
        model_identities[model]["runner_sha256"] = _sha256_file(runner_path)
    for model in ("fno", "uno"):
        model_identities[model][
            "source_revision"
        ] = f"neuraloperator=={args.neuraloperator_version}"

    persistence_command = _command(
        model="persistence",
        task="universal",
        config=config,
        data_root=Path(args.data_root),
        output_root=Path(args.output_root),
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        rollout_steps=args.rollout_steps,
        device=args.device,
        training_lock=training_lock_path,
        training_lock_sha256=lock.lock_sha256,
    )
    persistence_id = "a4_strat_v1_universal_persistence_val"
    runs = [
        {
            "run_id": persistence_id,
            "task": list(TASKS),
            "model": "persistence",
            "train_role": None,
            "evaluation_role": "valid",
            "command": persistence_command,
            "command_display": shlex.join(persistence_command),
            "expected_summary": str(Path(args.output_root) / persistence_id / "summary.json"),
            "model_identity": model_identities["persistence"],
            "conditioning": {
                "task_specialist": False,
                "physical_parameter": False,
                "inferred_parameter_context": False,
            },
            "fit_contract": None,
        }
    ]
    for task in TASKS:
        models = ("fno", "uno", "unet")
        if task != "darcy2d":
            models += ("cno",)
        for model in models:
            command = _command(
                model=model,
                task=task,
                config=config,
                data_root=Path(args.data_root),
                output_root=Path(args.output_root),
                max_train_samples=args.max_train_samples,
                max_eval_samples=args.max_eval_samples,
                rollout_steps=args.rollout_steps,
                device=args.device,
                training_lock=training_lock_path,
                training_lock_sha256=lock.lock_sha256,
            )
            if "test" in command or "--allow-held-out-test-eval" in command:
                raise AssertionError("A4 validation command attempted to request held-out access")
            run_name = f"a4_strat_v1_{task}_{model}_val"
            runs.append(
                {
                    "run_id": run_name,
                    "task": task,
                    "model": model,
                    "train_role": "train",
                    "evaluation_role": "valid",
                    "command": command,
                    "command_display": shlex.join(command),
                    "expected_summary": str(Path(args.output_root) / run_name / "summary.json"),
                    "model_identity": model_identities[model],
                    "conditioning": {
                        "task_specialist": True,
                        "physical_parameter": False,
                        "inferred_parameter_context": False,
                    },
                    "fit_contract": {**COMMON_FIT, **MODEL_FIT[model], "device": args.device},
                }
            )

    scorecard_rows = []
    by_id = {item["run_id"]: item for item in runs}
    for task in TASKS:
        primary = (
            "task_darcy2d_decoded_solution_nrmse"
            if task == "darcy2d"
            else f"task_{task}_decoded_rollout_nrmse"
        )
        models = ("persistence", "fno", "uno", "unet")
        if task != "darcy2d":
            models += ("cno",)
        for model in models:
            source_run = (
                persistence_id if model == "persistence" else f"a4_strat_v1_{task}_{model}_val"
            )
            scorecard_rows.append(
                {
                    "row_id": f"a4_strat_v1_{task}_{model}_val",
                    "task": task,
                    "model": model,
                    "source_summary": by_id[source_run]["expected_summary"],
                    "metric_key": primary,
                }
            )

    payload = {
        "schema_version": 1,
        "plan_id": "a4-strat-v1-baseline-validation-v1",
        "mode": "validation_only",
        "execution_policy": "dry_run_first_explicit_execute",
        "test_access": "forbidden",
        "training_lock": {
            "path": str(training_lock_path),
            "lock_sha256": lock.lock_sha256,
            "source_revision": lock.source_revision,
            "source_manifest_sha256": source.manifest_sha256,
            "protocol_id": protocol.protocol_id,
            "protocol_manifest_sha256": protocol.manifest_sha256,
            "objects": [
                {
                    "object_id": item.object_id,
                    "role": item.role,
                    "path": item.path,
                    "size_bytes": item.size_bytes,
                    "sha256": item.checksums["sha256"],
                    "uris": list(item.uris),
                }
                for item in lock.objects
            ],
            "selection": dict(lock.selection),
            "normalization": dict(lock.normalization),
        },
        "config": {"path": str(config), "sha256": _sha256_file(config)},
        "metric_contract": {
            "primary": METRIC,
            "temporal_rollout_steps": args.rollout_steps,
            "steady_task": "darcy2d",
            "aggregation": "macro_average_by_task",
            "sample_selection": {
                "policy": "full_shards_no_first_n_truncation",
                "sample_counts": sample_counts,
                "temporal_train_stride": COMMON_FIT["train_stride"],
                "temporal_pairs_per_sample": temporal_pairs_per_sample,
                "pair_cap": COMMON_FIT["max_pairs_per_task"],
                "regime_balance_authority": "training lock source objects",
            },
        },
        "runs": runs,
        "scorecard_plan": {
            "row_order": [item["row_id"] for item in scorecard_rows],
            "rows": scorecard_rows,
            "summary_inputs": [item["expected_summary"] for item in runs],
            "primary_metric": METRIC,
            "task_wall": "minimum_validation_nrmse_by_task_among_applicable_models",
            "overall_wall": "macro_task_average_for_models_applicable_to_all_three_tasks",
            "cno_exclusion": "Darcy excluded because the audited runner is CNO1d-only",
            "required_report_contract": [
                "macro overall and per-task primary NRMSE",
                "per-regime NRMSE for every task",
                "per-horizon NRMSE through horizon 16 for temporal tasks",
                "training lock, code, checkpoint, normalization, and selection hashes",
                "physical-parameter and inferred-context conditioning disclosure",
            ],
            "output_json": str(Path(args.output_root) / "a4_strat_v1_baseline_scorecard.json"),
        },
    }
    return {**payload, "plan_sha256": canonical_sha256(payload)}


def _verify_local_training_cache(plan: dict[str, Any], data_root: Path) -> None:
    test_files = sorted(data_root.glob("*_test.h5"))
    if test_files:
        raise ValueError("A4 execution refuses a data root containing test HDF5 objects")
    for item in plan["training_lock"]["objects"]:
        path = data_root / item["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != item["size_bytes"] or _sha256_file(path) != item["sha256"]:
            raise ValueError(f"Training cache object does not match lock: {path}")


def _verify_neuraloperator_version(expected: str) -> None:
    for package in ("neuraloperator", "neuralop"):
        try:
            observed = importlib.metadata.version(package)
            break
        except importlib.metadata.PackageNotFoundError:
            continue
    else:
        raise RuntimeError("A4 execution requires the pinned neuraloperator package")
    if observed != expected:
        raise RuntimeError(f"neuraloperator version mismatch: expected {expected}, got {observed}")


def main() -> None:
    release = (
        "docs/data/releases/strat-v1/universal/"
        "9d43d283f04f5b8d17cf6126ad189075c53307e715d7d4f61af440c2fed155c1"
    )
    parser = argparse.ArgumentParser(description="Plan the A4 strat-v1 validation baseline wall")
    parser.add_argument("--training-lock", default=f"{release}/training.lock.json")
    parser.add_argument("--source-manifest", default=f"{release}/source.manifest.yaml")
    parser.add_argument("--protocol-manifest", default=f"{release}/protocol.manifest.yaml")
    parser.add_argument("--config", default="configs/a4_strat_v1_baselines.yaml")
    parser.add_argument("--data-root", default="data/pdebench_strat_v1_training")
    parser.add_argument("--output-root", default="reports/a4_strat_v1_baselines")
    parser.add_argument("--output-plan", default="reports/a4_strat_v1_baseline_plan.json")
    parser.add_argument("--neuraloperator-version", required=True)
    parser.add_argument("--max-train-samples", type=int, default=288)
    parser.add_argument("--max-eval-samples", type=int, default=72)
    parser.add_argument("--rollout-steps", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-validation-only", action="store_true")
    args = parser.parse_args()
    if args.max_train_samples <= 0 or args.max_eval_samples <= 0 or args.rollout_steps <= 0:
        parser.error("sample counts and rollout steps must be positive")
    if args.rollout_steps != 16:
        parser.error("A4 strat-v1 fixes temporal rollout_steps at 16")
    if args.execute and not args.confirm_validation_only:
        parser.error("--execute requires --confirm-validation-only")

    plan = build_plan(args)
    output = Path(args.output_plan)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"A4 validation plan: {output}")
    print(f"plan_sha256={plan['plan_sha256']}")
    for run in plan["runs"]:
        print(f"DRY_RUN: {run['command_display']}")
    if not args.execute:
        return

    _verify_local_training_cache(plan, Path(args.data_root))
    _verify_neuraloperator_version(args.neuraloperator_version)
    for run in plan["runs"]:
        subprocess.run(run["command"], cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
