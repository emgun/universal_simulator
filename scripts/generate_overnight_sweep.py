#!/usr/bin/env python3
"""
Generate 15 overnight SOTA sweep configurations based on playbooks.

Strategy:
- Round A (9 runs): LR × Warmup grid
- Round B (3 runs): Capacity scaling
- Round C (3 runs): Hybrid best combinations
"""

import yaml
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO_ROOT / "configs/train_burgers_32dim.yaml"
OUTPUT_DIR = REPO_ROOT / "configs/overnight_sota"

def load_base():
    """Load base configuration."""
    with open(BASE_CONFIG) as f:
        return yaml.safe_load(f)

def write_config(name: str, config: dict, tags: list[str]):
    """Write configuration file."""
    # Update metadata
    config["logging"]["wandb"]["run_name"] = name
    config["logging"]["wandb"]["tags"] = tags
    config["logging"]["wandb"]["group"] = "overnight-sota"

    output_path = OUTPUT_DIR / f"{name}.yaml"
    with open(output_path, "w") as f:
        yaml.dump(config, f, sort_keys=False, default_flow_style=False)

    print(f"✅ Generated: {output_path.name}")
    return output_path

def generate_round_a():
    """Round A: Optimizer Grid (9 runs)

    LR × Warmup: [2e-4, 3e-4, 4.5e-4] × [3%, 5%, 6%]
    Fixed: EMA=0.9995, everything else from base
    """
    print("\n" + "="*60)
    print("Round A: Optimizer Grid (9 runs)")
    print("="*60)

    lrs = [2e-4, 3e-4, 4.5e-4]
    warmups = [0.03, 0.05, 0.06]

    configs = []
    for lr in lrs:
        for warmup in warmups:
            base = load_base()

            # Update optimizer
            base["stages"]["operator"]["optimizer"]["lr"] = lr

            # Calculate warmup epochs (apply to operator total epochs)
            op_epochs = base["stages"]["operator"]["epochs"]
            warmup_epochs = int(op_epochs * warmup)

            # Add warmup to scheduler (if not present, add it)
            if "scheduler" not in base["stages"]["operator"]:
                base["stages"]["operator"]["scheduler"] = {}

            # For cosine scheduler, we use a linear warmup prefix
            # PyTorch doesn't have built-in warmup, so we document it
            # The actual implementation would need to be in train.py
            # For now, we'll encode it in the config as a flag
            base["stages"]["operator"]["warmup_epochs"] = warmup_epochs

            # Update EMA
            base["training"]["ema_decay"] = 0.9995

            # Generate name
            lr_str = f"lr{int(lr*1e5):d}e5"  # e.g., lr20e5 for 2e-4
            w_str = f"w{int(warmup*100):d}pct"  # e.g., w3pct for 3%
            name = f"round_a_{lr_str}_{w_str}"

            tags = [
                "overnight-sota",
                "round-a",
                "optimizer-grid",
                f"lr={lr:.1e}",
                f"warmup={warmup:.0%}"
            ]

            configs.append(write_config(name, base, tags))

    print(f"\n✅ Round A: {len(configs)} configs generated")
    return configs

def generate_round_b():
    """Round B: Capacity Scaling (3 runs)

    hidden_dim: [64, 96, 128]
    Adjust num_heads and group_size to maintain divisibility
    Keep best LR from Round A (use 3e-4 as baseline expectation)
    """
    print("\n" + "="*60)
    print("Round B: Capacity Scaling (3 runs)")
    print("="*60)

    capacities = [
        {"hidden_dim": 64, "num_heads": 4, "group_size": 8, "name": "cap64"},
        {"hidden_dim": 96, "num_heads": 6, "group_size": 12, "name": "cap96"},
        {"hidden_dim": 128, "num_heads": 8, "group_size": 16, "name": "cap128"},
    ]

    configs = []
    for cap in capacities:
        base = load_base()

        # Update capacity
        base["operator"]["pdet"]["hidden_dim"] = cap["hidden_dim"]
        base["operator"]["pdet"]["num_heads"] = cap["num_heads"]
        base["operator"]["pdet"]["group_size"] = cap["group_size"]
        base["diffusion"]["hidden_dim"] = cap["hidden_dim"]
        base["ttc"]["decoder"]["hidden_dim"] = cap["hidden_dim"]
        base["ttc"]["decoder"]["num_heads"] = cap["num_heads"]

        # Use expected best LR from Round A (middle value)
        base["stages"]["operator"]["optimizer"]["lr"] = 3e-4
        base["stages"]["operator"]["warmup_epochs"] = int(base["stages"]["operator"]["epochs"] * 0.05)
        base["training"]["ema_decay"] = 0.9995

        name = f"round_b_{cap['name']}"
        tags = [
            "overnight-sota",
            "round-b",
            "capacity-scaling",
            f"hidden_dim={cap['hidden_dim']}",
            f"num_heads={cap['num_heads']}"
        ]

        configs.append(write_config(name, base, tags))

    print(f"\n✅ Round B: {len(configs)} configs generated")
    return configs

def generate_round_c():
    """Round C: Hybrid Best (3 runs)

    Combine best practices with targeted improvements:
    1. Best LR + larger capacity + enhanced TTC
    2. Best LR + more training epochs
    3. Best LR + more latent tokens
    """
    print("\n" + "="*60)
    print("Round C: Hybrid Best (3 runs)")
    print("="*60)

    configs = []

    # C1: Enhanced capacity + TTC
    base = load_base()
    base["operator"]["pdet"]["hidden_dim"] = 128
    base["operator"]["pdet"]["num_heads"] = 8
    base["operator"]["pdet"]["group_size"] = 16
    base["diffusion"]["hidden_dim"] = 128
    base["ttc"]["decoder"]["hidden_dim"] = 128
    base["ttc"]["decoder"]["num_heads"] = 8
    base["ttc"]["candidates"] = 12  # Enhanced from 8
    base["ttc"]["max_evaluations"] = 200  # Enhanced from 150
    base["stages"]["operator"]["optimizer"]["lr"] = 3e-4
    base["stages"]["operator"]["warmup_epochs"] = int(base["stages"]["operator"]["epochs"] * 0.05)
    base["training"]["ema_decay"] = 0.9995

    configs.append(write_config(
        "round_c_cap128_ttc",
        base,
        ["overnight-sota", "round-c", "hybrid", "enhanced-capacity", "enhanced-ttc"]
    ))

    # C2: Extended training
    base = load_base()
    base["stages"]["operator"]["epochs"] = 35  # +10 from baseline
    base["stages"]["diff_residual"]["epochs"] = 10  # +2
    base["stages"]["consistency_distill"]["epochs"] = 10  # +2
    base["stages"]["operator"]["optimizer"]["lr"] = 3e-4
    base["stages"]["operator"]["warmup_epochs"] = int(base["stages"]["operator"]["epochs"] * 0.05)
    base["training"]["ema_decay"] = 0.9995

    configs.append(write_config(
        "round_c_extended",
        base,
        ["overnight-sota", "round-c", "hybrid", "extended-training"]
    ))

    # C3: More latent tokens
    base = load_base()
    base["latent"]["tokens"] = 48  # +16 from baseline (32 → 48)
    base["stages"]["operator"]["optimizer"]["lr"] = 3e-4
    base["stages"]["operator"]["warmup_epochs"] = int(base["stages"]["operator"]["epochs"] * 0.05)
    base["training"]["ema_decay"] = 0.9995
    # Reduce batch size slightly to fit in memory
    base["training"]["batch_size"] = 10

    configs.append(write_config(
        "round_c_tokens48",
        base,
        ["overnight-sota", "round-c", "hybrid", "more-tokens"]
    ))

    print(f"\n✅ Round C: {len(configs)} configs generated")
    return configs

def main():
    """Generate all 15 configs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Overnight SOTA Sweep Configuration Generator")
    print("="*60)
    print(f"Base config: {BASE_CONFIG.relative_to(REPO_ROOT)}")
    print(f"Output dir: {OUTPUT_DIR.relative_to(REPO_ROOT)}")

    all_configs = []
    all_configs.extend(generate_round_a())
    all_configs.extend(generate_round_b())
    all_configs.extend(generate_round_c())

    print("\n" + "="*60)
    print(f"✅ Total: {len(all_configs)} configs generated")
    print("="*60)

    # Write manifest
    manifest = OUTPUT_DIR / "manifest.txt"
    with open(manifest, "w") as f:
        for cfg in all_configs:
            f.write(f"{cfg.relative_to(REPO_ROOT)}\n")

    print(f"\n📋 Manifest: {manifest.relative_to(REPO_ROOT)}")
    print("\nNext steps:")
    print("  1. Review configs in configs/overnight_sota/")
    print("  2. Launch with: python scripts/launch_overnight_sweep.py")

if __name__ == "__main__":
    main()
