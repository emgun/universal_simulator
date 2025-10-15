#!/usr/bin/env python
from __future__ import annotations

"""Helper utilities for launching Vast.ai training runs."""

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

# Add parent directory to path for imports
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.onstart_template import OnstartConfig, generate_onstart_script

ONSTART_DIR = REPO_ROOT / ".vast"


def run(cmd: list[str], *, check: bool = True) -> int:
    print("$", " ".join(shlex.quote(part) for part in cmd))
    result = subprocess.run(cmd)
    if check and result.returncode != 0:
        raise SystemExit(result.returncode)
    return result.returncode


def git_remote_url() -> str:
    try:
        out = subprocess.check_output(["git", "config", "--get", "remote.origin.url"], cwd=REPO_ROOT)
        return out.decode().strip()
    except subprocess.CalledProcessError:
        raise SystemExit("Could not determine git remote URL. Configure remote.origin first.")


def ensure_onstart(
    datasets: str | None,
    overrides: str | None,
    workdir: str,
    repo_url: str,
    auto_shutdown: bool,
    wandb_project: str | None,
    wandb_entity: str | None,
    wandb_api_key: str | None,
    b2_key_id: str | None,
    b2_app_key: str | None,
    b2_bucket: str | None,
    b2_prefix: str | None,
    b2_s3_endpoint: str | None,
    b2_s3_region: str | None,
) -> Path:
    """Generate onstart script using the centralized template."""
    ONSTART_DIR.mkdir(exist_ok=True)
    script_path = ONSTART_DIR / "onstart.sh"
    
    # Parse overrides to extract training config
    train_config = None
    train_stage = "all"
    reset_cache = True
    
    if overrides:
        # Parse KEY=VALUE pairs from overrides
        for pair in overrides.replace(",", " ").split():
            if "=" in pair:
                key, value = pair.split("=", 1)
                if key == "TRAIN_CONFIG":
                    train_config = value
                elif key == "TRAIN_STAGE":
                    train_stage = value
                elif key == "RESET_CACHE":
                    reset_cache = value == "1"
    
    # Build config using template
    config = OnstartConfig(
        repo_url=repo_url,
        workdir=workdir,
        train_config=train_config,
        train_stage=train_stage,
        reset_cache=reset_cache,
        datasets=datasets,
        b2_key_id=b2_key_id,
        b2_app_key=b2_app_key,
        b2_bucket=b2_bucket,
        b2_prefix=b2_prefix,
        b2_s3_endpoint=b2_s3_endpoint,
        b2_s3_region=b2_s3_region,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_api_key=wandb_api_key,
        auto_shutdown=auto_shutdown,
    )
    
    # Generate script from template
    script_content = generate_onstart_script(config)
    script_path.write_text(script_content)
    script_path.chmod(0o755)
    
    return script_path


def cmd_set_key(args: argparse.Namespace) -> None:
    key = args.key or os.environ.get("VAST_KEY") or os.environ.get("VAST_API_KEY")
    if not key:
        raise SystemExit("Provide --key or set VAST_KEY / VAST_API_KEY in the environment.")
    run(["vastai", "set", "api-key", key])


def cmd_search(args: argparse.Namespace) -> None:
    cmd = ["vastai", "search", "offers"]
    cmd.extend(args.filters)
    run(cmd)


def cmd_launch(args: argparse.Namespace) -> None:
    repo_url = args.repo_url or git_remote_url()
    onstart = ensure_onstart(
        args.datasets,
        args.overrides,
        args.workdir,
        repo_url,
        args.auto_shutdown,
        args.wandb_project,
        args.wandb_entity,
        args.wandb_api_key,
        args.b2_key_id,
        args.b2_app_key,
        args.b2_bucket,
        args.b2_prefix,
        args.b2_s3_endpoint,
        args.b2_s3_region,
    )

    env_parts = []
    if args.wandb_project:
        env_parts.append(f"WANDB_PROJECT={args.wandb_project}")
    if args.wandb_entity:
        env_parts.append(f"WANDB_ENTITY={args.wandb_entity}")
    if args.wandb_api_key:
        env_parts.append(f"WANDB_API_KEY={args.wandb_api_key}")
    if args.b2_key_id:
        env_parts.append(f"B2_KEY_ID={args.b2_key_id}")
    if args.b2_app_key:
        env_parts.append(f"B2_APP_KEY={args.b2_app_key}")
    if args.b2_bucket:
        env_parts.append(f"B2_BUCKET={args.b2_bucket}")
    if args.b2_prefix:
        env_parts.append(f"B2_PREFIX={args.b2_prefix}")
    if args.b2_s3_endpoint:
        env_parts.append(f"B2_S3_ENDPOINT={args.b2_s3_endpoint}")
    if args.b2_s3_region:
        env_parts.append(f"B2_S3_REGION={args.b2_s3_region}")
    env_str = ",".join(env_parts) if env_parts else None

    # If an offer-id is provided, use it directly; otherwise filter by GPU/image/disk
    if args.offer_id:
        cmd = [
            "vastai",
            "create",
            "instance",
            str(args.offer_id),
            "--image",
            args.image,
            "--disk",
            str(args.disk),
        ]
    else:
        cmd = [
            "vastai",
            "launch",
            "instance",
            "-g",
            args.gpu,
            "-n",
            str(args.num_gpus),
            "-i",
            args.image,
            "-d",
            str(args.disk),
        ]
        if args.region:
            cmd.extend(["-r", args.region])
    if env_str:
        cmd.extend(["--env", env_str])
    cmd.extend(["--ssh", "--onstart", str(onstart)])
    if args.dry_run:
        print("DRY RUN: would execute ->", " ".join(cmd))
        print("\nGenerated onstart script:\n" + onstart.read_text())
        return
    run(cmd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Helpers for Vast.ai training launches")
    sub = parser.add_subparsers(dest="command", required=True)

    p_key = sub.add_parser("set-key", help="Set Vast API key (reads VAST_KEY if not provided)")
    p_key.add_argument("--key", help="API key literal")
    p_key.set_defaults(func=cmd_set_key)

    p_search = sub.add_parser("search", help="Wrapper around 'vastai search offers'")
    p_search.add_argument("filters", nargs=argparse.REMAINDER, help="Filters and flags to append")
    p_search.set_defaults(func=cmd_search)

    p_launch = sub.add_parser("launch", help="Launch instance and run training on Vast")
    p_launch.add_argument("--offer-id", help="Explicit Vast offer id to create instance from")
    p_launch.add_argument("--gpu", default="RTX_4090", help="GPU model (default RTX_4090)")
    p_launch.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs")
    p_launch.add_argument("--image", default="pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime", help="Docker image")
    p_launch.add_argument("--disk", type=int, default=64, help="Disk in GB")
    p_launch.add_argument("--region", help="Region filter for launch instance")
    p_launch.add_argument("--datasets", default=os.environ.get("WANDB_DATASETS"), help="WANDB_DATASETS value to pass to training")
    p_launch.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT"), help="WANDB project name")
    p_launch.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"), help="WANDB entity name")
    p_launch.add_argument("--wandb-api-key", default=os.environ.get("WANDB_API_KEY"), help="WANDB API key")
    p_launch.add_argument("--overrides", help="Additional Hydra overrides for run_remote_scale.sh")
    p_launch.add_argument("--repo-url", help="Git remote URL (defaults to origin)")
    p_launch.add_argument("--workdir", default="/workspace", help="Remote working directory")
    p_launch.add_argument("--auto-shutdown", action="store_true", help="Power off instance after training completes")
    p_launch.add_argument("--b2-key-id", default=os.environ.get("B2_KEY_ID"), help="B2 application key ID for dataset fetch")
    p_launch.add_argument("--b2-app-key", default=os.environ.get("B2_APP_KEY"), help="B2 application key secret for dataset fetch")
    p_launch.add_argument("--b2-bucket", default=os.environ.get("B2_BUCKET"), help="Override B2 bucket for dataset fetch")
    p_launch.add_argument("--b2-prefix", default=os.environ.get("B2_PREFIX"), help="Override B2 prefix for dataset fetch")
    p_launch.add_argument("--b2-s3-endpoint", default=os.environ.get("B2_S3_ENDPOINT"), help="Override B2 S3 endpoint for dataset fetch")
    p_launch.add_argument("--b2-s3-region", default=os.environ.get("B2_S3_REGION"), help="Override B2 S3 region for dataset fetch")
    p_launch.add_argument("--dry-run", action="store_true", help="Print commands without launching")
    p_launch.set_defaults(func=cmd_launch)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
