#!/usr/bin/env python3
"""
Streamlined VastAI launcher for training runs.

Commands:
  setup-env  - One-time: Configure VastAI environment variables
  launch     - Launch training instance
  resume     - Resume training on existing instance with checkpoint
  search     - Search for available instances

Usage:
  # One-time setup
  python scripts/vast_launch.py setup-env

  # Launch training
  python scripts/vast_launch.py launch \\
    --config configs/train_burgers_32dim.yaml \\
    --auto-shutdown

  # Resume from checkpoint on existing instance
  python scripts/vast_launch.py resume \\
    --instance-id 12345 \\
    --config configs/train_burgers_32dim.yaml \\
    --resume-from-wandb train-20251027_193043
"""

import argparse
import os
import time
import shlex
import subprocess
import sys
from pathlib import Path
import yaml

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


def git_current_branch() -> str:
    """Get current git branch name."""
    try:
        out = subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO_ROOT)
        return out.decode().strip()
    except subprocess.CalledProcessError:
        return "feature/sota_burgers_upgrades"  # fallback


def generate_onstart_script(
    config: str,
    stage: str = "all",
    repo_url: str = None,
    branch: str = "feature/sota_burgers_upgrades",
    workdir: str = "/workspace",
    auto_shutdown: bool = False,
    run_args: list[str] | None = None,
    precompute: bool = True,
) -> str:
    """
    Generate a simple onstart script.
    
    Since VastAI env-vars are globally configured, the script is minimal:
    - Install dependencies
    - Clone repo
    - Run training
    """
    repo_url = repo_url or git_remote_url()
    run_args = run_args or []
    extra_args = ""
    launch_mode_line = "--tag launch_mode=auto"
    if run_args:
        extra_args = "\n  " + " \\\n  ".join(run_args)
        launch_mode_line += " \\"

    # Convert absolute paths to relative paths for remote execution
    train_cfg_path = Path(config)
    if train_cfg_path.is_absolute():
        try:
            train_cfg_path = train_cfg_path.relative_to(REPO_ROOT)
        except ValueError:
            # Non-repo path, use as-is without smart matching
            train_cfg_path = Path(config)

    # Use relative path for the actual config parameter in the script
    config_for_script = train_cfg_path.as_posix()

    # Try to inline the training config contents so remote runs do not depend
    # on the Git branch containing local, unpushed changes.
    config_inline = None
    try:
        local_cfg_path = (REPO_ROOT / train_cfg_path) if not train_cfg_path.is_absolute() else train_cfg_path
        if local_cfg_path.exists():
            # Only inline very small configs to keep VastAI args under limits
            max_inline_bytes = 800
            size = local_cfg_path.stat().st_size
            if size <= max_inline_bytes:
                config_inline = local_cfg_path.read_text(encoding="utf-8")
            else:
                config_inline = None
        else:
            config_inline = None
    except Exception:
        config_inline = None
    
    # Build optional inline block for config
    inline_block = ""
    if config_inline is not None:
        inline_block = f"""
# Inline training config to ensure availability on remote
mkdir -p "$(dirname {config_for_script})"
cat > {config_for_script} << 'EOF_CFG'
{config_inline}
EOF_CFG
"""

    script = f"""#!/bin/bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

mkdir -p {workdir}
cd {workdir}

if [ ! -d universal_simulator ]; then
  git clone {repo_url} universal_simulator
fi
cd universal_simulator
git fetch origin
git checkout {branch}
git pull origin {branch}
apt-get update && apt-get install -y git rclone build-essential

# Activate venv if it exists (vastai/pytorch), otherwise use system Python
if [ -f /venv/main/bin/activate ]; then
  source /venv/main/bin/activate
fi

pip install -e .[dev]

export RCLONE_CONFIG_B2TRAIN_TYPE=s3
export RCLONE_CONFIG_B2TRAIN_PROVIDER=Other
export RCLONE_CONFIG_B2TRAIN_ACCESS_KEY_ID="$B2_KEY_ID"
export RCLONE_CONFIG_B2TRAIN_SECRET_ACCESS_KEY="$B2_APP_KEY"
export RCLONE_CONFIG_B2TRAIN_ENDPOINT="$B2_S3_ENDPOINT"
export RCLONE_CONFIG_B2TRAIN_REGION="$B2_S3_REGION"
export RCLONE_CONFIG_B2TRAIN_ACL=private
export RCLONE_CONFIG_B2TRAIN_NO_CHECK_BUCKET=true

mkdir -p data/pdebench
# Download all files in parallel (backgrounded with &)
if [ ! -f data/pdebench/burgers1d_train_000.h5 ]; then
  rclone copy B2TRAIN:pdebench/full/burgers1d/burgers1d_train_000.h5 data/pdebench/ --progress &
fi
if [ ! -f data/pdebench/burgers1d_val.h5 ] && [ ! -f data/pdebench/burgers1d_valid.h5 ]; then
  rclone copy B2TRAIN:PDEbench/pdebench/burgers1d_full_v1/burgers1d_val.h5 data/pdebench/ --progress &
fi
if [ ! -f data/pdebench/burgers1d_test.h5 ]; then
  rclone copy B2TRAIN:PDEbench/pdebench/burgers1d_full_v1/burgers1d_test.h5 data/pdebench/ --progress &
fi

# Wait for all downloads to complete
wait

# Continue with file management after downloads finish
if [ -f data/pdebench/burgers1d_val.h5 ] && [ ! -f data/pdebench/burgers1d_valid.h5 ]; then
  mv -f data/pdebench/burgers1d_val.h5 data/pdebench/burgers1d_valid.h5 || true
fi
ln -sf burgers1d_train_000.h5 data/pdebench/burgers1d_train.h5 || true
ln -sf burgers1d_valid.h5 data/pdebench/burgers1d_val.h5 || true

# Only clear checkpoints, preserve cache (will be validated by precompute script)
rm -rf checkpoints || true
mkdir -p data/latent_cache checkpoints
mkdir -p artifacts/runs reports

{inline_block}

{('echo "Precomputing latent caches…"\nPYTHONPATH=src python scripts/precompute_latent_cache.py --config ' + config_for_script + ' --tasks burgers1d --splits train val --root data/pdebench --cache-dir data/latent_cache --cache-dtype float16 --device cuda --batch-size 16 --num-workers 4 --pin-memory --parallel || echo "⚠️  Latent cache precompute failed (continuing)"') if precompute else 'echo "Skipping latent cache precompute (quick-run)"'}

export WANDB_MODE=online

python scripts/run_fast_to_sota.py --train-config {config_for_script} --train-stage {stage} --skip-small-eval --eval-device cuda --run-dir artifacts/runs --leaderboard-csv reports/leaderboard.csv --wandb-mode online --wandb-sync --wandb-project "${{WANDB_PROJECT:-universal-simulator}}" --wandb-entity "${{WANDB_ENTITY:-}}" --wandb-group fast-to-sota --wandb-tags vast --strict-exit --tag environment=vast {launch_mode_line}{extra_args} || echo "⚠️  Training exited with code $?"

echo "✓ Training pipeline completed"
"""

    if auto_shutdown:
        script += """
# Auto-stop instance
pip install -q vastai 2>&1 || true
sleep 10
[ -n "${CONTAINER_ID:-}" ] && vastai stop instance $CONTAINER_ID || true
exit 0
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
    # Auto-detect branch if not specified
    branch = args.branch if args.branch else git_current_branch()

    # Generate onstart script
    ONSTART_DIR.mkdir(exist_ok=True)
    onstart_path = ONSTART_DIR / "onstart.sh"

    script_content = generate_onstart_script(
        config=args.config,
        stage=args.stage,
        repo_url=args.repo_url,
        branch=branch,
        workdir=args.workdir,
        auto_shutdown=args.auto_shutdown,
        run_args=args.run_arg,
        precompute=not getattr(args, "no_precompute", False),
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


def cmd_resume(args: argparse.Namespace) -> None:
    """Resume training on existing instance with checkpoint resumption."""
    # Get SSH connection details for the instance
    import json
    result = subprocess.run(
        ["vastai", "show", "instance", str(args.instance_id), "--raw"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"❌ Failed to get instance info for {args.instance_id}")
        sys.exit(1)

    instance_info = json.loads(result.stdout)
    ssh_host = instance_info.get("ssh_host")
    ssh_port = instance_info.get("ssh_port")

    if not ssh_host or not ssh_port:
        print(f"❌ Could not get SSH details for instance {args.instance_id}")
        sys.exit(1)

    print(f"✓ Found instance {args.instance_id} at {ssh_host}:{ssh_port}")

    # Generate resume script
    branch = args.branch if args.branch else git_current_branch()
    workdir = args.workdir
    config_path = args.config

    # Convert to relative path for remote
    train_cfg_path = Path(config_path)
    if train_cfg_path.is_absolute():
        try:
            train_cfg_path = train_cfg_path.relative_to(REPO_ROOT)
        except ValueError:
            train_cfg_path = Path(config_path)
    config_for_script = train_cfg_path.as_posix()

    # Build training command with checkpoint resumption
    resume_args = [
        f"--resume-from-wandb {args.resume_from_wandb}",
        f"--resume-mode {args.resume_mode}"
    ]

    script = f"""#!/bin/bash
set -euo pipefail

cd {workdir}/universal_simulator

# Pull latest code
git fetch origin
git checkout {branch}
git pull origin {branch}

# Activate venv if it exists
if [ -f /venv/main/bin/activate ]; then
  source /venv/main/bin/activate
fi

# Ensure dependencies are up to date
pip install -e .[dev] --quiet

export WANDB_MODE=online

echo "=== Resuming training from WandB run {args.resume_from_wandb} ==="

# Resume training using checkpoint manager
PYTHONPATH=src python scripts/train.py \\
  --config {config_for_script} \\
  --stage {args.stage} \\
  {resume_args[0]} \\
  {resume_args[1]} || echo "⚠️  Training exited with code $?"

echo "✓ Training resumed and running"
"""

    if args.auto_shutdown:
        script += """
# Auto-stop instance after completion
pip install -q vastai 2>&1 || true
sleep 10
[ -n "${CONTAINER_ID:-}" ] && vastai stop instance $CONTAINER_ID || true
"""

    # Write script to temp file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write(script)
        script_path = f.name

    try:
        # Upload script to instance
        print(f"📤 Uploading resume script to instance...")
        scp_cmd = [
            "scp",
            "-o", "StrictHostKeyChecking=no",
            "-P", str(ssh_port),
            script_path,
            f"root@{ssh_host}:/tmp/resume_training.sh"
        ]
        run(scp_cmd)

        # Execute script on instance
        print(f"🚀 Starting training resumption on instance {args.instance_id}...")
        ssh_cmd = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-p", str(ssh_port),
            f"root@{ssh_host}",
            "chmod +x /tmp/resume_training.sh && nohup /tmp/resume_training.sh > /tmp/resume_training.log 2>&1 &"
        ]
        run(ssh_cmd)

        print(f"\n✅ Training resumption started on instance {args.instance_id}")
        print(f"   WandB run: {args.resume_from_wandb}")
        print(f"   Config: {config_path}")
        print(f"   Stage: {args.stage}")
        print(f"\nMonitor with:")
        print(f"   vastai logs {args.instance_id}")
        print(f"   ssh -p {ssh_port} root@{ssh_host} 'tail -f /tmp/resume_training.log'")

    finally:
        # Clean up temp file
        os.unlink(script_path)


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
    p_launch.add_argument("--gpu", default="A100_PCIE", help="GPU model (default: A100_PCIE)")
    p_launch.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs")
    p_launch.add_argument("--image", default="pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel",
                         help="Docker image (default: PyTorch 2.7 CUDA 12.8 for Blackwell support)")
    p_launch.add_argument("--disk", type=int, default=64, help="Disk in GB")
    p_launch.add_argument("--region", help="Region filter")
    p_launch.add_argument("--config", required=True, help="Training config (e.g., configs/train_burgers_32dim.yaml)")
    p_launch.add_argument("--stage", default="all", choices=["all", "operator", "diffusion", "distill"],
                         help="Training stage (default: all)")
    p_launch.add_argument("--repo-url", help="Git remote URL (default: auto-detect)")
    p_launch.add_argument("--branch", default=None, help="Git branch (default: auto-detect from current branch)")
    p_launch.add_argument("--workdir", default="/workspace", help="Remote working directory")
    p_launch.add_argument("--auto-shutdown", action="store_true", help="Auto-shutdown after completion")
    p_launch.add_argument("--dry-run", action="store_true", help="Print commands without launching")
    p_launch.add_argument("--retries", type=int, default=0, help="Retries for failed launch attempts")
    p_launch.add_argument("--retry-wait", type=int, default=20, help="Seconds to wait between retries")
    p_launch.add_argument(
        "--run-arg",
        action="append",
        default=[],
        help="Additional argument to append to run_fast_to_sota.py command (may repeat)",
    )
    p_launch.add_argument("--no-precompute", action="store_true", help="Skip latent cache precompute in onstart (faster startup)")
    p_launch.set_defaults(func=cmd_launch)

    # resume command
    p_resume = sub.add_parser("resume", help="Resume training on existing instance with checkpoint")
    p_resume.add_argument("--instance-id", required=True, type=int, help="VastAI instance ID")
    p_resume.add_argument("--config", required=True, help="Training config (e.g., configs/train_burgers_32dim.yaml)")
    p_resume.add_argument("--resume-from-wandb", required=True, help="WandB run ID to resume from (e.g., train-20251027_193043)")
    p_resume.add_argument("--resume-mode", default="allow", choices=["allow", "must", "never"],
                         help="WandB resume mode (default: allow)")
    p_resume.add_argument("--stage", default="all", choices=["all", "operator", "diffusion", "distill"],
                         help="Training stage (default: all)")
    p_resume.add_argument("--branch", default=None, help="Git branch (default: auto-detect from current branch)")
    p_resume.add_argument("--workdir", default="/workspace", help="Remote working directory")
    p_resume.add_argument("--auto-shutdown", action="store_true", help="Auto-shutdown after completion")
    p_resume.set_defaults(func=cmd_resume)

    return parser


def main() -> None:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
