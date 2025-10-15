#!/usr/bin/env python3
"""
CLI tool for manually generating VastAI onstart scripts.

Usage:
    python scripts/generate_onstart.py \\
      --train-config configs/train_burgers_512dim_v2_pru2jxc4.yaml \\
      --datasets burgers1d_full_v1 \\
      --auto-shutdown

This is useful for:
- Generating custom onstart scripts
- Testing script generation locally
- Creating scripts for manual deployment
"""

import argparse
import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.onstart_template import generate_onstart_script_simple


def main():
    parser = argparse.ArgumentParser(
        description="Generate VastAI onstart scripts using the centralized template",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate standard training script
  python scripts/generate_onstart.py \\
    --train-config configs/train_burgers_512dim_v2_pru2jxc4.yaml \\
    --datasets burgers1d_full_v1 \\
    --auto-shutdown

  # Resume training without cache reset
  python scripts/generate_onstart.py \\
    --train-config configs/train_burgers_512dim_v2_pru2jxc4.yaml \\
    --datasets burgers1d_full_v1 \\
    --no-reset-cache

  # Train only operator stage
  python scripts/generate_onstart.py \\
    --train-config configs/train_burgers_512dim_v2_pru2jxc4.yaml \\
    --datasets burgers1d_full_v1 \\
    --train-stage operator

  # Output to custom file
  python scripts/generate_onstart.py \\
    --train-config configs/train_burgers_32dim_pru2jxc4.yaml \\
    --datasets burgers1d_full_v1 \\
    --output my_custom_onstart.sh
        """
    )
    
    # Training args
    parser.add_argument(
        "--train-config",
        help="Path to training config (e.g., configs/train_burgers_512dim_v2_pru2jxc4.yaml)"
    )
    parser.add_argument(
        "--train-stage",
        default="all",
        choices=["all", "operator", "diffusion", "distill"],
        help="Training stage to run (default: all)"
    )
    parser.add_argument(
        "--reset-cache",
        dest="reset_cache",
        action="store_true",
        default=True,
        help="Reset latent cache and checkpoints (default: True)"
    )
    parser.add_argument(
        "--no-reset-cache",
        dest="reset_cache",
        action="store_false",
        help="Keep existing cache and checkpoints"
    )
    
    # Data args
    parser.add_argument(
        "--datasets",
        help="WandB datasets to download (e.g., burgers1d_full_v1)"
    )
    
    # Repository args
    parser.add_argument(
        "--repo-url",
        default="https://github.com/emgun/universal_simulator.git",
        help="Git repository URL"
    )
    parser.add_argument(
        "--branch",
        default="feature/sota_burgers_upgrades",
        help="Git branch to checkout"
    )
    parser.add_argument(
        "--workdir",
        default="/workspace",
        help="Working directory on instance"
    )
    
    # B2 credentials (optional)
    parser.add_argument("--b2-key-id", help="Backblaze B2 key ID")
    parser.add_argument("--b2-app-key", help="Backblaze B2 application key")
    parser.add_argument("--b2-bucket", help="Backblaze B2 bucket name")
    parser.add_argument("--b2-prefix", help="Backblaze B2 prefix/path")
    parser.add_argument("--b2-s3-endpoint", help="Backblaze B2 S3 endpoint")
    parser.add_argument("--b2-s3-region", help="Backblaze B2 S3 region")
    
    # WandB credentials (optional)
    parser.add_argument("--wandb-project", help="WandB project name")
    parser.add_argument("--wandb-entity", help="WandB entity/username")
    parser.add_argument("--wandb-api-key", help="WandB API key")
    
    # System args
    parser.add_argument(
        "--auto-shutdown",
        action="store_true",
        help="Auto-shutdown instance after training completes"
    )
    parser.add_argument(
        "--no-install-deps",
        dest="install_deps",
        action="store_false",
        default=True,
        help="Skip dependency installation (for pre-configured instances)"
    )
    
    # Output args
    parser.add_argument(
        "--output",
        "-o",
        default=".vast/onstart.sh",
        help="Output file path (default: .vast/onstart.sh)"
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Print to stdout instead of writing to file"
    )
    
    args = parser.parse_args()
    
    # Validate
    if not args.train_config and not args.datasets:
        parser.error("At least one of --train-config or --datasets must be provided")
    
    # Generate script
    script = generate_onstart_script_simple(
        train_config=args.train_config,
        datasets=args.datasets,
        auto_shutdown=args.auto_shutdown,
        # Optional overrides
        repo_url=args.repo_url,
        branch=args.branch,
        workdir=args.workdir,
        train_stage=args.train_stage,
        reset_cache=args.reset_cache,
        b2_key_id=args.b2_key_id,
        b2_app_key=args.b2_app_key,
        b2_bucket=args.b2_bucket,
        b2_prefix=args.b2_prefix,
        b2_s3_endpoint=args.b2_s3_endpoint,
        b2_s3_region=args.b2_s3_region,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_api_key=args.wandb_api_key,
        install_deps=args.install_deps,
    )
    
    # Output
    if args.print:
        print(script)
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(script)
        output_path.chmod(0o755)
        print(f"✅ Generated onstart script: {output_path}")
        print(f"   - Training config: {args.train_config or 'None'}")
        print(f"   - Training stage: {args.train_stage}")
        print(f"   - Reset cache: {args.reset_cache}")
        print(f"   - Datasets: {args.datasets or 'None'}")
        print(f"   - Auto-shutdown: {args.auto_shutdown}")
        print()
        print("Next steps:")
        print("  1. Review the script: cat", output_path)
        print("  2. Upload to instance: vastai copy <id>", output_path, "/workspace/onstart.sh")
        print("  3. Execute: vastai ssh <id> 'bash /workspace/onstart.sh'")


if __name__ == "__main__":
    main()

