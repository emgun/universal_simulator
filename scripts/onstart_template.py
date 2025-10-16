"""
Single source of truth for VastAI onstart script generation.

This module generates onstart scripts for VastAI instances with:
- Consistent dependency installation (git, rclone, gcc, etc.)
- Proper environment setup
- Data downloading
- Training launch
- Optional auto-shutdown

Usage:
    from scripts.onstart_template import generate_onstart_script
    
    script = generate_onstart_script(
        train_config="configs/train_burgers_512dim_v2_pru2jxc4.yaml",
        datasets="burgers1d_full_v1",
        auto_shutdown=True
    )
"""

from typing import Optional
from dataclasses import dataclass


@dataclass
class OnstartConfig:
    """Configuration for onstart script generation."""
    
    # Repository settings
    repo_url: str = "https://github.com/emgun/universal_simulator.git"
    branch: str = "feature/sota_burgers_upgrades"
    workdir: str = "/workspace"
    
    # Training settings
    train_config: Optional[str] = None  # e.g., "configs/train_burgers_512dim_v2_pru2jxc4.yaml"
    train_stage: str = "all"  # or "operator", "diffusion", etc.
    reset_cache: bool = True
    
    # Data settings
    datasets: Optional[str] = None  # WandB datasets to download (test/val)
    
    # B2 credentials (optional, can be loaded from .env)
    b2_key_id: Optional[str] = None
    b2_app_key: Optional[str] = None
    b2_bucket: Optional[str] = None
    b2_prefix: Optional[str] = None
    b2_s3_endpoint: Optional[str] = None
    b2_s3_region: Optional[str] = None
    
    # WandB settings (optional, can be loaded from .env)
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_api_key: Optional[str] = None
    
    # System settings
    auto_shutdown: bool = False
    install_deps: bool = True


def generate_onstart_script(config: OnstartConfig) -> str:
    """
    Generate a VastAI onstart script from configuration.
    
    Args:
        config: OnstartConfig with all settings
        
    Returns:
        Complete bash script as string
    """
    lines = []
    
    # Shebang and error handling
    lines.extend([
        "#!/bin/bash",
        "set -euo pipefail",
        "export DEBIAN_FRONTEND=noninteractive",
        "",
    ])
    
    # Install dependencies
    if config.install_deps:
        lines.extend([
            "# Install core dependencies",
            "command -v git >/dev/null 2>&1 || (apt-get update && apt-get install -y git)",
            "command -v pip >/dev/null 2>&1 || (apt-get update && apt-get install -y python3-pip)",
            "command -v rclone >/dev/null 2>&1 || (apt-get update && apt-get install -y rclone)",
            "command -v gcc >/dev/null 2>&1 || (apt-get update && apt-get install -y build-essential)",
            "",
        ])
    
    # Clone repository
    lines.extend([
        "# Clone and setup repository",
        f"mkdir -p {config.workdir}",
        f"cd {config.workdir}",
        "",
        "if [ ! -d universal_simulator ]; then",
        f"  git clone {config.repo_url} universal_simulator",
        "fi",
        "cd universal_simulator",
        "git fetch origin",
        f"git checkout {config.branch}",
        f"git pull origin {config.branch}",
        "",
    ])
    
    # Install Python dependencies
    lines.extend([
        "# Install Python dependencies",
        "python3 -m pip install --upgrade pip",
        "python3 -m pip install -e .[dev]",
        "",
    ])
    
    # Export environment variables
    lines.append("# Export environment variables")
    
    if config.datasets:
        lines.append(f'export WANDB_DATASETS="{config.datasets}"')
    
    if config.b2_key_id:
        lines.append(f'export B2_KEY_ID="{config.b2_key_id}"')
    if config.b2_app_key:
        lines.append(f'export B2_APP_KEY="{config.b2_app_key}"')
    if config.b2_bucket:
        lines.append(f'export B2_BUCKET="{config.b2_bucket}"')
    if config.b2_prefix:
        lines.append(f'export B2_PREFIX="{config.b2_prefix}"')
    if config.b2_s3_endpoint:
        lines.append(f'export B2_S3_ENDPOINT="{config.b2_s3_endpoint}"')
    if config.b2_s3_region:
        lines.append(f'export B2_S3_REGION="{config.b2_s3_region}"')
    
    if config.wandb_project:
        lines.append(f'export WANDB_PROJECT="{config.wandb_project}"')
    if config.wandb_entity:
        lines.append(f'export WANDB_ENTITY="{config.wandb_entity}"')
    if config.wandb_api_key:
        lines.append(f'export WANDB_API_KEY="{config.wandb_api_key}"')
    
    lines.extend([
        "",
        "# Load local .env if present (export all keys)",
        "if [ -f .env ]; then",
        "  set -a",
        "  source .env",
        "  set +a",
        "fi",
        "",
    ])
    
    # Download data
    if config.datasets:
        lines.extend([
            "# Download test/val datasets",
            "if [ -n \"$WANDB_DATASETS\" ]; then",
            "  bash scripts/fetch_datasets_b2.sh || echo '⚠️ Test/val download failed'",
            "fi",
            "",
        ])
    
    # Download training data (if config specifies it)
    if config.train_config:
        lines.extend([
            "# Download training data",
            f'TRAIN_CONFIG="{config.train_config}"',
            "echo '📥 Downloading training data...'",
            "",
            "# Configure rclone for training data",
            "export RCLONE_CONFIG_B2TRAIN_TYPE=s3",
            "export RCLONE_CONFIG_B2TRAIN_PROVIDER=B2",
            'export RCLONE_CONFIG_B2TRAIN_ACCESS_KEY_ID="$B2_KEY_ID"',
            'export RCLONE_CONFIG_B2TRAIN_SECRET_ACCESS_KEY="$B2_APP_KEY"',
            'export RCLONE_CONFIG_B2TRAIN_ENDPOINT="$B2_S3_ENDPOINT"',
            'export RCLONE_CONFIG_B2TRAIN_REGION="$B2_S3_REGION"',
            "",
            "# Download full training data",
            "mkdir -p data/pdebench",
            "if [ ! -f data/pdebench/burgers1d_train_000.h5 ]; then",
            "  echo 'Downloading burgers1d_train_000.h5...'",
            f'  rclone copy B2TRAIN:{config.b2_bucket or "ups-datasets"}/full/burgers1d/burgers1d_train_000.h5 data/pdebench/ || {{',
            "    echo '⚠️ Training data download failed!'",
            "  }",
            "fi",
            "",
            "# Create symlink",
            "if [ -f data/pdebench/burgers1d_train_000.h5 ] && [ ! -e data/pdebench/burgers1d_train.h5 ]; then",
            "  cd data/pdebench && ln -sf burgers1d_train_000.h5 burgers1d_train.h5 && cd ../..",
            "fi",
            "",
        ])
    
    # Run training
    if config.train_config:
        reset_flag = "1" if config.reset_cache else "0"
        lines.extend([
            "# Run training",
            f'export TRAIN_CONFIG="{config.train_config}"',
            f'export TRAIN_STAGE="{config.train_stage}"',
            f'export RESET_CACHE={reset_flag}',
            "bash scripts/run_remote_scale.sh",
            "",
        ])
    
    # Auto-shutdown
    if config.auto_shutdown:
        lines.extend([
            "# Auto-shutdown after completion",
            "if command -v poweroff >/dev/null 2>&1; then",
            "  sync",
            "  poweroff",
            "fi",
        ])
    
    return "\n".join(lines)


def generate_onstart_script_simple(
    train_config: Optional[str] = None,
    datasets: Optional[str] = None,
    auto_shutdown: bool = False,
    **kwargs
) -> str:
    """
    Simplified interface for common use cases.
    
    Args:
        train_config: Path to training config (e.g., "configs/train_burgers_512dim_v2_pru2jxc4.yaml")
        datasets: WandB datasets to download (e.g., "burgers1d_full_v1")
        auto_shutdown: Whether to auto-shutdown after completion
        **kwargs: Additional OnstartConfig fields to override
        
    Returns:
        Complete bash script as string
        
    Example:
        script = generate_onstart_script_simple(
            train_config="configs/train_burgers_512dim_v2_pru2jxc4.yaml",
            datasets="burgers1d_full_v1",
            auto_shutdown=True
        )
    """
    config = OnstartConfig(
        train_config=train_config,
        datasets=datasets,
        auto_shutdown=auto_shutdown,
        **kwargs
    )
    return generate_onstart_script(config)


if __name__ == "__main__":
    # Example usage
    script = generate_onstart_script_simple(
        train_config="configs/train_burgers_512dim_v2_pru2jxc4.yaml",
        datasets="burgers1d_full_v1",
        auto_shutdown=True
    )
    print(script)

