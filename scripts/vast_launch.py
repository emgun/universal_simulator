#!/usr/bin/env python3
"""
Streamlined VastAI launcher for training runs.

Commands:
  setup-env  - One-time: Configure VastAI environment variables
  launch     - Launch training instance
  search     - Search for available instances

Usage:
  # One-time setup
  python scripts/vast_launch.py setup-env

  # Launch training
  python scripts/vast_launch.py launch \\
    --config configs/train_burgers_32dim.yaml \\
    --auto-shutdown
"""

import argparse
import os
import time
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ONSTART_DIR = REPO_ROOT / ".vast"


def run(cmd: list[str], *, check: bool = True) -> int:
    """Execute command and print it."""
    print("$", " ".join(shlex.quote(part) for part in cmd))
    result = subprocess.run(cmd)
    if check and result.returncode != 0:
        raise SystemExit(result.returncode)
    return result.returncode


def git_remote_url() -> str:
    """Get git remote URL."""
    try:
        out = subprocess.check_output(["git", "config", "--get", "remote.origin.url"], cwd=REPO_ROOT)
        return out.decode().strip()
    except subprocess.CalledProcessError:
        return "https://github.com/emgun/universal_simulator.git"


def generate_onstart_script(
    config: str,
    stage: str = "all",
    repo_url: str = None,
    branch: str = "feature/sota_burgers_upgrades",
    workdir: str = "/workspace",
    auto_shutdown: bool = False,
) -> str:
    """
    Generate a simple onstart script.
    
    Since VastAI env-vars are globally configured, the script is minimal:
    - Install dependencies
    - Clone repo
    - Run training
    """
    repo_url = repo_url or git_remote_url()
    
    script = f"""#!/bin/bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

# Install core dependencies
apt-get update && apt-get install -y git rclone build-essential

# Clone and setup repository
mkdir -p {workdir}
cd {workdir}

if [ ! -d universal_simulator ]; then
  git clone {repo_url} universal_simulator
fi
cd universal_simulator
git fetch origin
git checkout {branch}
git pull origin {branch}

# Use VastAI's preinstalled PyTorch venv (has proper CUDA/Triton setup)
source /venv/main/bin/activate

# Install our additional dependencies (PyTorch already installed in venv)
pip install --upgrade pip
pip install -e .[dev]

# Download training data using VastAI-injected B2 credentials
echo "📥 Downloading training data..."
export RCLONE_CONFIG_B2TRAIN_TYPE=s3
export RCLONE_CONFIG_B2TRAIN_PROVIDER=B2
export RCLONE_CONFIG_B2TRAIN_ACCESS_KEY_ID="$B2_KEY_ID"
export RCLONE_CONFIG_B2TRAIN_SECRET_ACCESS_KEY="$B2_APP_KEY"
export RCLONE_CONFIG_B2TRAIN_ENDPOINT="$B2_S3_ENDPOINT"
export RCLONE_CONFIG_B2TRAIN_REGION="$B2_S3_REGION"

mkdir -p data/pdebench
if [ ! -f data/pdebench/burgers1d_train_000.h5 ]; then
  rclone copy B2TRAIN:pdebench/full/burgers1d/burgers1d_train_000.h5 data/pdebench/ --progress || exit 1
  echo "✅ Training data downloaded"
else
  echo "✅ Training data already exists"
fi

# Download validation data
if [ ! -f data/pdebench/burgers1d_val.h5 ]; then
  echo "📥 Downloading validation data..."
  rclone copy B2TRAIN:PDEbench/pdebench/burgers1d_full_v1/burgers1d_val.h5 data/pdebench/ --progress || echo "⚠️  Val data download failed"
  echo "✅ Validation data downloaded"
else
  echo "✅ Validation data already exists"
fi

# Download test data for evaluation
if [ ! -f data/pdebench/burgers1d_test.h5 ]; then
  echo "📥 Downloading test data..."
  rclone copy B2TRAIN:PDEbench/pdebench/burgers1d_full_v1/burgers1d_test.h5 data/pdebench/ --progress || echo "⚠️  Test data download failed, eval may not work"
  echo "✅ Test data downloaded"
else
  echo "✅ Test data already exists"
fi

# Ensure expected filenames are present for validators
ln -sf burgers1d_train_000.h5 data/pdebench/burgers1d_train.h5 || true

# Ensure clean caches for each run
rm -rf data/latent_cache checkpoints/scale || true
mkdir -p data/latent_cache checkpoints/scale

# Run fast-to-SOTA automation (VastAI env-vars already injected)
export WANDB_MODE=online
python scripts/run_fast_to_sota.py \
  --train-config {config} \
  --small-eval-config configs/small_eval_burgers.yaml \
  --full-eval-config configs/full_eval_burgers.yaml \
  --eval-device cuda \
  --run-dir artifacts/runs \
  --leaderboard-csv reports/leaderboard.csv \
  --leaderboard-html reports/leaderboard.html \
  --wandb-mode online \
  --wandb-sync \
  --wandb-project "${{WANDB_PROJECT:-universal-simulator}}" \
  --wandb-entity "${{WANDB_ENTITY:-}}" \
  --wandb-group fast-to-sota \
  --wandb-tags vast \
  --wandb-tags production \
  --skip-dry-run \
  --leaderboard-wandb \
  --leaderboard-wandb-project "${{WANDB_PROJECT:-universal-simulator}}" \
  --leaderboard-wandb-entity "${{WANDB_ENTITY:-}}" \
  --copy-checkpoints \
  --strict-exit \
  --tag environment=vast \
  --tag launch_mode=auto
"""

    if auto_shutdown:
        script += """
# Auto-shutdown after completion
if command -v poweroff >/dev/null 2>&1; then
  sync
  poweroff
fi
"""

    return script


def cmd_setup_env(args: argparse.Namespace) -> None:
    """Set up VastAI environment variables (one-time)."""
    # Load from .env if present
    env_file = REPO_ROOT / ".env"
    if not env_file.exists():
        print("❌ ERROR: .env file not found")
        print("   Please create .env with B2 and WandB credentials")
        sys.exit(1)
    
    # Parse .env file
    env_vars = {}
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            env_vars[key.strip()] = value.strip().strip('"').strip("'")
    
    print("═══════════════════════════════════════════════════════════")
    print("Setting up VastAI Environment Variables")
    print("═══════════════════════════════════════════════════════════")
    print()
    
    # B2 credentials
    print("📦 Setting B2 credentials...")
    for key in ["B2_KEY_ID", "B2_APP_KEY", "B2_S3_ENDPOINT", "B2_S3_REGION", "B2_BUCKET"]:
        if key in env_vars:
            value = env_vars[key]
            run(["vastai", "create", "env-var", key, value], check=False)
    
    # WandB credentials
    print("🔬 Setting WandB credentials...")
    for key in ["WANDB_API_KEY", "WANDB_PROJECT", "WANDB_ENTITY"]:
        if key in env_vars:
            value = env_vars[key]
            run(["vastai", "create", "env-var", key, value], check=False)
    
    print()
    print("✅ All environment variables configured!")
    print()
    print("Verification:")
    run(["vastai", "show", "env-vars"])
    print()
    print("✅ Setup complete! You can now launch instances without passing credentials.")


def cmd_search(args: argparse.Namespace) -> None:
    """Search for available instances."""
    cmd = ["vastai", "search", "offers"]
    cmd.extend(args.filters)
    run(cmd)


def cmd_launch(args: argparse.Namespace) -> None:
    """Launch a training instance."""
    # Generate onstart script
    ONSTART_DIR.mkdir(exist_ok=True)
    onstart_path = ONSTART_DIR / "onstart.sh"
    
    script_content = generate_onstart_script(
        config=args.config,
        stage=args.stage,
        repo_url=args.repo_url,
        branch=args.branch,
        workdir=args.workdir,
        auto_shutdown=args.auto_shutdown,
    )
    
    onstart_path.write_text(script_content)
    onstart_path.chmod(0o755)
    
    print(f"✅ Generated onstart script: {onstart_path}")
    
    # Build launch command
    if args.offer_id:
        cmd = [
            "vastai", "create", "instance",
            str(args.offer_id),
            "--image", args.image,
            "--disk", str(args.disk),
        ]
    else:
        cmd = [
            "vastai", "launch", "instance",
            "-g", args.gpu,
            "-n", str(args.num_gpus),
            "-i", args.image,
            "-d", str(args.disk),
        ]
        if args.region:
            cmd.extend(["-r", args.region])
    
    cmd.extend(["--ssh", "--onstart", str(onstart_path)])
    
    if args.dry_run:
        print("DRY RUN: would execute ->", " ".join(cmd))
        print("\nGenerated onstart script:\n" + onstart_path.read_text())
        return
    
    attempts = max(1, int(getattr(args, "retries", 0)) + 1)
    wait_s = max(0, int(getattr(args, "retry_wait", 20)))
    for i in range(attempts):
        rc = run(cmd, check=False)
        if rc == 0:
            return
        if i < attempts - 1:
            print(f"Launch failed (rc={rc}). Retrying in {wait_s}s… [{i+1}/{attempts-1}]")
            time.sleep(wait_s)
    raise SystemExit(rc)


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    parser = argparse.ArgumentParser(
        description="VastAI launcher for training runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # setup-env command
    p_setup = sub.add_parser("setup-env", help="One-time setup of VastAI environment variables")
    p_setup.set_defaults(func=cmd_setup_env)

    # search command
    p_search = sub.add_parser("search", help="Search for available instances")
    p_search.add_argument("filters", nargs=argparse.REMAINDER, help="Filters and flags")
    p_search.set_defaults(func=cmd_search)

    # launch command
    p_launch = sub.add_parser("launch", help="Launch training instance")
    p_launch.add_argument("--offer-id", help="Explicit offer ID to create from")
    p_launch.add_argument("--gpu", default="RTX_4090", help="GPU model (default: RTX_4090)")
    p_launch.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs")
    p_launch.add_argument("--image", default="vastai/pytorch", 
                         help="Docker image (vastai/pytorch has PyTorch preinstalled with proper CUDA/Triton setup)")
    p_launch.add_argument("--disk", type=int, default=64, help="Disk in GB")
    p_launch.add_argument("--region", help="Region filter")
    p_launch.add_argument("--config", required=True, help="Training config (e.g., configs/train_burgers_32dim.yaml)")
    p_launch.add_argument("--stage", default="all", choices=["all", "operator", "diffusion", "distill"],
                         help="Training stage (default: all)")
    p_launch.add_argument("--repo-url", help="Git remote URL (default: auto-detect)")
    p_launch.add_argument("--branch", default="feature/sota_burgers_upgrades", help="Git branch")
    p_launch.add_argument("--workdir", default="/workspace", help="Remote working directory")
    p_launch.add_argument("--auto-shutdown", action="store_true", help="Auto-shutdown after completion")
    p_launch.add_argument("--dry-run", action="store_true", help="Print commands without launching")
    p_launch.add_argument("--retries", type=int, default=0, help="Retries for failed launch attempts")
    p_launch.add_argument("--retry-wait", type=int, default=20, help="Seconds to wait between retries")
    p_launch.set_defaults(func=cmd_launch)

    return parser


def main() -> None:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
