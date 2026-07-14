#!/usr/bin/env python
from __future__ import annotations

"""Generate a bounded remote experiment queue for the UPS demo loop."""

import argparse
import csv
import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TIER_CAPS: dict[str, dict[str, Any]] = {
    "smoke": {
        "train_max_samples": 8,
        "eval_max_samples": 4,
        "decoded_rollout_steps": 4,
        "remote_b2_prefix": "strat-v1-smoke",
        "required_gb": 4,
    },
}
RESERVED_LEGACY_PREFIXES = {"smoke-v1", "light-v1", "medium-v1"}


@dataclass(frozen=True)
class Variant:
    name: str
    description: str
    overrides: tuple[str, ...] = ()
    priority: int = 100
    stages: str | None = None
    requires_checkpoint_source: bool = False


VARIANTS: tuple[Variant, ...] = (
    Variant(
        name="current_best",
        description="Current heterogeneous task-signature config.",
        priority=10,
    ),
    Variant(
        name="no_conditioning",
        description="Matched control with semantic conditioning disabled.",
        overrides=(
            "training.auto_conditioning=false",
            "operator.conditioning.sources={}",
        ),
        priority=20,
    ),
    Variant(
        name="task_signature_only",
        description="Reduced flat semantic conditioning: task id plus equation signature only.",
        overrides=('operator.conditioning.sources={"task_id":3,"equation_signature":15}',),
        priority=30,
    ),
    Variant(
        name="task_signature_semigroup0",
        description="Task-signature conditioning with semigroup loss disabled.",
        overrides=(
            'operator.conditioning.sources={"task_id":3,"equation_signature":15}',
            "training.lambda_semigroup=0.0",
        ),
        priority=31,
    ),
    Variant(
        name="task_signature_joint16",
        description="Task-signature conditioning with a shorter joint codec/operator stage.",
        overrides=(
            'operator.conditioning.sources={"task_id":3,"equation_signature":15}',
            "stages.joint_codec_operator.epochs=16",
        ),
        priority=32,
    ),
    Variant(
        name="task_signature_joint48",
        description="Task-signature conditioning with a longer joint codec/operator stage.",
        overrides=(
            'operator.conditioning.sources={"task_id":3,"equation_signature":15}',
            "stages.joint_codec_operator.epochs=48",
        ),
        priority=33,
    ),
    Variant(
        name="task_signature_opdecoded4",
        description="Task-signature conditioning with more frozen-codec decoded operator fine-tuning.",
        overrides=(
            'operator.conditioning.sources={"task_id":3,"equation_signature":15}',
            "stages.operator_decoded.epochs=4",
        ),
        priority=34,
    ),
    Variant(
        name="task_signature_opdecoded4_joint16",
        description="Task-signature conditioning with more decoded fine-tuning and a shorter joint stage.",
        overrides=(
            'operator.conditioning.sources={"task_id":3,"equation_signature":15}',
            "stages.operator_decoded.epochs=4",
            "stages.joint_codec_operator.epochs=16",
        ),
        priority=35,
    ),
    Variant(
        name="task_signature_recon0",
        description="Task-signature conditioning with joint reconstruction loss disabled.",
        overrides=(
            'operator.conditioning.sources={"task_id":3,"equation_signature":15}',
            "stages.joint_codec_operator.lambda_reconstruction=0.0",
        ),
        priority=36,
    ),
    Variant(
        name="task_signature_rollout4",
        description="Task-signature conditioning with longer decoded rollout loss.",
        overrides=(
            'operator.conditioning.sources={"task_id":3,"equation_signature":15}',
            "stages.joint_codec_operator.rollout_steps=4",
        ),
        priority=37,
    ),
    Variant(
        name="task_signature_residual_alpha25",
        description="Task-signature conditioning evaluated as a 25% UPS residual over physical persistence.",
        overrides=(
            'operator.conditioning.sources={"task_id":3,"equation_signature":15}',
            "evaluation.decoded_persistence_residual_alpha=0.25",
        ),
        priority=38,
    ),
    Variant(
        name="task_signature_residual_alpha50",
        description="Task-signature conditioning evaluated as a 50% UPS residual over physical persistence.",
        overrides=(
            'operator.conditioning.sources={"task_id":3,"equation_signature":15}',
            "evaluation.decoded_persistence_residual_alpha=0.5",
        ),
        priority=39,
    ),
    Variant(
        name="task_signature_trained_residual",
        description="Task-signature conditioning with decoded persistence-residual and spectral training losses.",
        overrides=(
            'operator.conditioning.sources={"task_id":3,"equation_signature":15}',
            "stages.operator_decoded.lambda_persistence_residual=0.5",
            "stages.operator_decoded.lambda_persistence_residual_spectral=0.05",
            "stages.joint_codec_operator.lambda_persistence_residual=0.5",
            "stages.joint_codec_operator.lambda_persistence_residual_spectral=0.05",
            "evaluation.decoded_persistence_residual_alpha=0.25",
        ),
        priority=40,
    ),
    Variant(
        name="task_signature_rollout4_residual_ft2",
        description=(
            "Checkpoint fine-tune using two joint rollout-4 epochs, decoded residual losses, "
            "and validation-selected transport alpha 0.18."
        ),
        overrides=(
            'operator.conditioning.sources={"task_id":3,"equation_signature":15}',
            "stages.joint_codec_operator.epochs=2",
            "stages.joint_codec_operator.rollout_steps=4",
            "stages.joint_codec_operator.lambda_rollout=1.0",
            "stages.joint_codec_operator.lambda_persistence_residual=0.5",
            "stages.joint_codec_operator.lambda_persistence_residual_spectral=0.05",
            "evaluation.decoded_persistence_residual_alpha=0.0",
            'evaluation.decoded_persistence_residual_alpha_by_family={"transport":0.18}',
        ),
        priority=41,
        stages="joint_codec_operator",
        requires_checkpoint_source=True,
    ),
    Variant(
        name="task_signature_transport_residual_gate",
        description="Eval-only blend that uses validation-calibrated UPS residual only for the transport/advection family.",
        overrides=(
            'operator.conditioning.sources={"task_id":3,"equation_signature":15}',
            'evaluation.decoded_persistence_residual_alpha_by_family={"transport":0.2}',
            "evaluation.decoded_persistence_residual_alpha=0.0",
        ),
        priority=41,
    ),
    Variant(
        name="task_signature_advection_roll_shift40",
        description="Eval-only transport correction using validation-selected periodic advection shift +40.",
        overrides=(
            'operator.conditioning.sources={"task_id":3,"equation_signature":15}',
            "evaluation.decoded_persistence_residual_alpha=0.0",
            'evaluation.decoded_roll_shift_by_task={"advection1d":40}',
        ),
        priority=42,
    ),
    Variant(
        name="task_signature_joint48_rollout4",
        description="Task-signature conditioning with longer joint training and rollout loss.",
        overrides=(
            'operator.conditioning.sources={"task_id":3,"equation_signature":15}',
            "stages.joint_codec_operator.epochs=48",
            "stages.joint_codec_operator.rollout_steps=4",
        ),
        priority=43,
    ),
    Variant(
        name="semigroup0",
        description="Disable semigroup loss to test whether it helps real held-out rollouts.",
        overrides=("training.lambda_semigroup=0.0",),
        priority=50,
    ),
    Variant(
        name="semigroup10",
        description="Increase semigroup loss modestly without changing architecture.",
        overrides=("training.lambda_semigroup=0.1",),
        priority=60,
    ),
    Variant(
        name="joint16",
        description="Cheaper joint codec/operator stage.",
        overrides=("stages.joint_codec_operator.epochs=16",),
        priority=70,
    ),
    Variant(
        name="joint48",
        description="Longer joint codec/operator stage for decoded training depth.",
        overrides=("stages.joint_codec_operator.epochs=48",),
        priority=80,
    ),
    Variant(
        name="rollout4",
        description="Train joint stage against longer decoded rollout loss.",
        overrides=("stages.joint_codec_operator.rollout_steps=4",),
        priority=90,
    ),
)


def _shell_assignment(key: str, value: str | int) -> str:
    return f"{key}={shlex.quote(str(value))}"


def _light_extra_args(
    variant: Variant, *, train_max_samples: int, eval_max_samples: int, rollout_steps: int
) -> str:
    args = [
        "--override",
        f"data.max_samples={train_max_samples}",
        "--eval-override",
        f"data.max_samples={eval_max_samples}",
        "--decoded-rollout-steps",
        str(rollout_steps),
        "--promotion-rule",
        "decoded_rollout_nrmse<=1.0",
    ]
    for override in variant.overrides:
        args.extend(["--override", override])
    return " ".join(args)


def _command(row: dict[str, Any], *, wrapper: str, env_file: str, dry_run: int) -> str:
    assignments = [
        _shell_assignment("ENV_FILE", env_file),
        _shell_assignment("DRY_RUN", dry_run),
        _shell_assignment("TASKS", row["tasks"]),
        _shell_assignment("TRAIN_CONFIG", row["train_config"]),
        _shell_assignment("REMOTE_B2_PREFIX", row["remote_b2_prefix"]),
        _shell_assignment("EVAL_SPLIT", row["eval_split"]),
        _shell_assignment("REQUIRED_GB", row["required_gb"]),
        _shell_assignment("STAGES", row["stages"]),
        _shell_assignment("RUN_NAME", row["run_name"]),
        _shell_assignment("OUTPUT_ROOT", row["output_root"]),
        _shell_assignment("LIGHT_EXTRA_ARGS", row["light_extra_args"]),
    ]
    if row.get("checkpoint_source"):
        assignments.append(_shell_assignment("CHECKPOINT_SOURCE", row["checkpoint_source"]))
    return " ".join(assignments + ["bash", shlex.quote(wrapper)])


def build_rows(
    *,
    tier: str,
    variants: list[str] | None,
    train_config: str,
    tasks: str,
    output_root: str,
    eval_split: str,
    stages: str,
    run_prefix: str,
    remote_b2_prefix: str | None,
    required_gb: int | None,
    checkpoint_source: str | None = None,
) -> list[dict[str, Any]]:
    if tier not in TIER_CAPS:
        raise SystemExit(
            f"Tier {tier!r} is not available for new queues; legacy light/medium tiers are frozen"
        )
    caps = TIER_CAPS[tier]
    resolved_prefix = remote_b2_prefix or str(caps["remote_b2_prefix"])
    if resolved_prefix in RESERVED_LEGACY_PREFIXES:
        raise SystemExit(
            f"Remote prefix {resolved_prefix!r} is frozen historical evidence and cannot be queued"
        )
    selected = sorted(VARIANTS, key=lambda item: item.priority)
    if variants:
        wanted = set(variants)
        selected = [variant for variant in selected if variant.name in wanted]
        missing = wanted.difference(variant.name for variant in selected)
        if missing:
            raise SystemExit(f"Unknown variants: {', '.join(sorted(missing))}")

    rows: list[dict[str, Any]] = []
    for variant in selected:
        if variant.requires_checkpoint_source and not checkpoint_source:
            raise SystemExit(f"Variant {variant.name} requires --checkpoint-source")
        run_name = f"{run_prefix}_{tier}_{variant.name}"
        light_extra_args = _light_extra_args(
            variant,
            train_max_samples=int(caps["train_max_samples"]),
            eval_max_samples=int(caps["eval_max_samples"]),
            rollout_steps=int(caps["decoded_rollout_steps"]),
        )
        rows.append(
            {
                "run_name": run_name,
                "tier": tier,
                "variant": variant.name,
                "priority": variant.priority,
                "description": variant.description,
                "train_config": train_config,
                "tasks": tasks,
                "output_root": output_root,
                "eval_split": eval_split,
                "stages": variant.stages or stages,
                "checkpoint_source": checkpoint_source or "",
                "remote_b2_prefix": resolved_prefix,
                "required_gb": required_gb if required_gb is not None else int(caps["required_gb"]),
                "train_max_samples": caps["train_max_samples"],
                "eval_max_samples": caps["eval_max_samples"],
                "decoded_rollout_steps": caps["decoded_rollout_steps"],
                "variant_overrides": " ".join(variant.overrides),
                "light_extra_args": light_extra_args,
            }
        )
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_tsv(rows: list[dict[str, Any]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_shell(
    rows: list[dict[str, Any]], path: str | Path, *, wrapper: str, env_file: str, dry_run: int
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Generated by scripts/plan_demo_experiments.py.",
        "# Commands default to DRY_RUN=1 unless --dry-run-value 0 was used.",
        "",
    ]
    for row in rows:
        lines.append(f"# {row['variant']}: {row['description']}")
        lines.append(_command(row, wrapper=wrapper, env_file=env_file, dry_run=dry_run))
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    output_path.chmod(0o755)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan bounded UPS demo experiment queues")
    parser.add_argument("--tier", choices=sorted(TIER_CAPS), default="smoke")
    parser.add_argument(
        "--variant", action="append", default=None, help="Variant to include; repeat to subset"
    )
    parser.add_argument(
        "--train-config", default="configs/train_multitask_heterogeneous_light_best.yaml"
    )
    parser.add_argument("--tasks", default="burgers1d,advection1d,darcy2d")
    parser.add_argument("--output-root", default="reports/light_experiments_remote")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument(
        "--stages", default="operator,decoder,operator_decoded,joint_codec_operator"
    )
    parser.add_argument(
        "--checkpoint-source",
        default=None,
        help="Optional checkpoint source passed through to remote runs as CHECKPOINT_SOURCE",
    )
    parser.add_argument("--run-prefix", default="ups")
    parser.add_argument("--remote-b2-prefix", default=None)
    parser.add_argument("--required-gb", type=int, default=None)
    parser.add_argument("--remote-wrapper", default="scripts/run_remote_light_promotion.sh")
    parser.add_argument("--env-file", default="/workspace/.env")
    parser.add_argument("--dry-run-value", type=int, choices=(0, 1), default=1)
    parser.add_argument("--output-jsonl", default="reports/demo/experiment_queue.jsonl")
    parser.add_argument("--output-tsv", default="reports/demo/experiment_queue.tsv")
    parser.add_argument("--output-sh", default="reports/demo/run_experiment_queue.sh")
    args = parser.parse_args()

    rows = build_rows(
        tier=args.tier,
        variants=args.variant,
        train_config=args.train_config,
        tasks=args.tasks,
        output_root=args.output_root,
        eval_split=args.eval_split,
        stages=args.stages,
        run_prefix=args.run_prefix,
        remote_b2_prefix=args.remote_b2_prefix,
        required_gb=args.required_gb,
        checkpoint_source=args.checkpoint_source,
    )
    write_jsonl(rows, args.output_jsonl)
    write_tsv(rows, args.output_tsv)
    write_shell(
        rows,
        args.output_sh,
        wrapper=args.remote_wrapper,
        env_file=args.env_file,
        dry_run=args.dry_run_value,
    )
    print(args.output_jsonl)
    print(args.output_tsv)
    print(args.output_sh)


if __name__ == "__main__":
    main()
