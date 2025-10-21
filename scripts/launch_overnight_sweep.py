#!/usr/bin/env python3
"""
Launch overnight SOTA sweep using vast_launch.py with proper tracking.

Uses the exact syntax requested:
  python scripts/vast_launch.py launch \
    --gpu <GPU> \
    --config configs/<config> \
    --auto-shutdown \
    --run-arg=--wandb-run-name=<run_name> \
    --run-arg=--tag=config=<config>
"""

import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs/overnight_sota"
MANIFEST = CONFIGS_DIR / "manifest.txt"

def run_cmd(cmd: list[str]) -> int:
    """Run command and return exit code."""
    print("$", " ".join(cmd))
    result = subprocess.run(cmd)
    return result.returncode

def launch_config(config_path: Path, gpu: str = "RTX_5880Ada", delay: int = 5) -> bool:
    """Launch a single config with proper tracking."""
    # Extract run name from config filename (without .yaml)
    run_name = config_path.stem

    # Build command using exact requested syntax
    cmd = [
        "python", "scripts/vast_launch.py", "launch",
        "--gpu", gpu,
        "--config", str(config_path),
        "--auto-shutdown",
        f"--run-arg=--wandb-run-name={run_name}",
        f"--run-arg=--tag=config={run_name}",
    ]

    print(f"\n🚀 Launching: {run_name}")
    print(f"   Config: {config_path.relative_to(REPO_ROOT)}")

    rc = run_cmd(cmd)

    if rc == 0:
        print(f"✅ {run_name} launched successfully")
        time.sleep(delay)  # Delay between launches
        return True
    else:
        print(f"❌ {run_name} failed to launch (exit code {rc})")
        return False

def main():
    """Launch all configs from manifest."""
    if not MANIFEST.exists():
        print(f"❌ ERROR: Manifest not found: {MANIFEST}")
        print("   Run: python scripts/generate_overnight_sweep.py")
        sys.exit(1)

    # Read configs from manifest
    configs = []
    with open(MANIFEST) as f:
        for line in f:
            line = line.strip()
            if line:
                configs.append(REPO_ROOT / line)

    print("="*60)
    print("Overnight SOTA Sweep Launcher")
    print("="*60)
    print(f"Total configs: {len(configs)}")
    print(f"Delay between launches: 5 seconds")
    print(f"Expected total time: ~{len(configs) * 5 / 60:.1f} minutes for launches")
    print("="*60)

    # Launch all configs
    launched = 0
    failed = 0

    for i, config_path in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] " + "="*50)

        if not config_path.exists():
            print(f"⚠️  Config not found: {config_path}")
            failed += 1
            continue

        success = launch_config(config_path)
        if success:
            launched += 1
        else:
            failed += 1

    # Summary
    print("\n" + "="*60)
    print("🎉 Launch Summary")
    print("="*60)
    print(f"✅ Launched: {launched}/{len(configs)}")
    if failed > 0:
        print(f"❌ Failed: {failed}/{len(configs)}")
    print("\n📊 Next steps:")
    print("  1. Monitor instances: vastai show instances")
    print("  2. Check W&B: https://wandb.ai/emgun-morpheus-space/universal-simulator")
    print(f"  3. Expected completion: ~{len(configs) * 0.5:.1f} hours (30 min/run)")
    print("="*60)

if __name__ == "__main__":
    main()
