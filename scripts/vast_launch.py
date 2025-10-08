#!/usr/bin/env python
from __future__ import annotations

"""Helper utilities for launching Vast.ai training runs."""

import argparse
import os
import shlex
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
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
    b2_key_id: str | None,
    b2_app_key: str | None,
    b2_bucket: str | None,
    b2_prefix: str | None,
) -> Path:
    ONSTART_DIR.mkdir(exist_ok=True)
    script_path = ONSTART_DIR / "onstart.sh"
    datasets_export = f"export WANDB_DATASETS=\"{datasets}\"" if datasets else "# WANDB_DATASETS optional"
    fetch_cmd = "if [ -n \"$WANDB_DATASETS\" ]; then\n  bash scripts/fetch_datasets_b2.sh $WANDB_DATASETS\nfi"
    overrides_cmd = f"bash scripts/run_remote_scale.sh {overrides}" if overrides else "bash scripts/run_remote_scale.sh"
    shutdown_cmd = "\nif command -v poweroff >/dev/null 2>&1; then\n  sync\n  poweroff\nfi" if auto_shutdown else ""
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        "export DEBIAN_FRONTEND=noninteractive",
        "command -v git >/dev/null 2>&1 || (apt-get update && apt-get install -y git)",
        "command -v pip >/dev/null 2>&1 || (apt-get update && apt-get install -y python3-pip)",
        "",
        f"mkdir -p {workdir}",
        f"cd {workdir}",
        "",
        "if [ ! -d universal_simulator ]; then",
        f"  git clone {repo_url} universal_simulator",
        "fi",
        "cd universal_simulator",
        "git pull",
        "",
        "python3 -m pip install --upgrade pip",
        "python3 -m pip install -e .[dev]",
        "",
        datasets_export,
        (f"export B2_KEY_ID=\"{b2_key_id}\"" if b2_key_id else "# B2_KEY_ID optional"),
        (f"export B2_APP_KEY=\"{b2_app_key}\"" if b2_app_key else "# B2_APP_KEY optional"),
        (f"export B2_BUCKET=\"{b2_bucket}\"" if b2_bucket else "# B2_BUCKET optional"),
        (f"export B2_PREFIX=\"{b2_prefix}\"" if b2_prefix else "# B2_PREFIX optional"),
        "if [ -f scripts/load_env.sh ]; then",
        "  bash scripts/load_env.sh || true",
        "fi",
        fetch_cmd,
        f"{overrides_cmd}{shutdown_cmd}",
    ]
    script_path.write_text("\n".join(lines))
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
        args.b2_key_id,
        args.b2_app_key,
        args.b2_bucket,
        args.b2_prefix,
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
    env_str = ",".join(env_parts) if env_parts else None

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
    p_launch.add_argument("--gpu", default="RTX_4090", help="GPU model (default RTX_4090)")
    p_launch.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs")
    p_launch.add_argument("--image", default="pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime", help="Docker image")
    p_launch.add_argument("--disk", type=int, default=64, help="Disk in GB")
    p_launch.add_argument("--region", help="Region filter for launch instance")
    p_launch.add_argument("--datasets", help="WANDB_DATASETS value to pass to training")
    p_launch.add_argument("--wandb-project", help="WANDB project name")
    p_launch.add_argument("--wandb-entity", help="WANDB entity name")
    p_launch.add_argument("--wandb-api-key", help="WANDB API key")
    p_launch.add_argument("--overrides", help="Additional Hydra overrides for run_remote_scale.sh")
    p_launch.add_argument("--repo-url", help="Git remote URL (defaults to origin)")
    p_launch.add_argument("--workdir", default="/workspace", help="Remote working directory")
    p_launch.add_argument("--auto-shutdown", action="store_true", help="Power off instance after training completes")
    p_launch.add_argument("--b2-key-id", help="B2 application key ID for dataset fetch")
    p_launch.add_argument("--b2-app-key", help="B2 application key secret for dataset fetch")
    p_launch.add_argument("--b2-bucket", help="Override B2 bucket for dataset fetch")
    p_launch.add_argument("--b2-prefix", help="Override B2 prefix for dataset fetch")
    p_launch.add_argument("--dry-run", action="store_true", help="Print commands without launching")
    p_launch.set_defaults(func=cmd_launch)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
