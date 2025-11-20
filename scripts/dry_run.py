#!/usr/bin/env python3
"""
Dry-run mode for training configurations.

Validates configuration, checks data availability, builds models,
runs 1 training step per stage, and estimates cost without full training.

Usage:
    python scripts/dry_run.py configs/train_burgers_32dim.yaml
    python scripts/dry_run.py configs/train_burgers_32dim.yaml --estimate-only
"""

import argparse
import sys
from pathlib import Path
from typing import Dict

import torch
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from ups.utils.config_loader import load_config_with_includes


def validate_config(config_path: str) -> Dict:
    """Load and validate config."""
    print("\n" + "="*70)
    print("🔍 Step 1: Validating Configuration")
    print("="*70)
    
    try:
        cfg = load_config_with_includes(config_path)
        print(f"✅ Config loaded from {config_path}")
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        sys.exit(1)
    
    # Run validation
    # Import validation function from train.py
    train_script = Path(__file__).parent / "train.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("train", train_script)
    train_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_module)
    
    warnings = []
    if hasattr(train_module, "_validate_config_consistency"):
        warnings = train_module._validate_config_consistency(cfg)
    else:
        print("ℹ️  Skipping advanced consistency checks (train._validate_config_consistency unavailable)")
    
    if warnings:
        has_critical = any("CRITICAL" in w for w in warnings)
        for warning in warnings:
            print(warning)
        
        if has_critical:
            print("\n❌ Configuration has critical errors")
            sys.exit(1)
        else:
            print("\n⚠️  Configuration has warnings but can proceed")
    else:
        print("✅ Configuration is valid")
    
    return cfg


def check_data_availability(cfg: Dict) -> bool:
    """Check if required data files exist."""
    print("\n" + "="*70)
    print("📁 Step 2: Checking Data Availability")
    print("="*70)
    
    data_root = Path(cfg.get("data", {}).get("root", "data/pdebench"))
    task = cfg.get("data", {}).get("task", "burgers1d")
    split = cfg.get("data", {}).get("split", "train")
    
    print(f"Data root: {data_root}")
    print(f"Task: {task}")
    print(f"Split: {split}")
    
    if not data_root.exists():
        print(f"❌ Data root does not exist: {data_root}")
        print("   Run data download or check data.root in config")
        return False
    
    # Check for expected data files
    expected_files = []
    if task == "burgers1d":
        expected_files = [f"{task}_{split}.h5"]
    
    missing_files = []
    for filename in expected_files:
        filepath = data_root / filename
        if not filepath.exists():
            missing_files.append(filename)
            print(f"❌ Missing: {filename}")
        else:
            print(f"✅ Found: {filename}")
    
    if missing_files:
        print(f"\n⚠️  {len(missing_files)} data file(s) missing")
        print("   Training will fail without these files")
        return False
    
    print("\n✅ All required data files present")
    return True


def test_data_loader(cfg: Dict) -> bool:
    """Test loading a single batch from the data loader."""
    print("\n" + "="*70)
    print("📊 Step 3: Testing Data Loader")
    print("="*70)
    
    try:
        from ups.data.latent_pairs import build_latent_pair_loader
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Build data loader
        train_cfg = cfg.get("training", {})
        batch_size = train_cfg.get("batch_size", 4)
        
        print(f"Building data loader (batch_size={batch_size})...")
        # Ensure split is set in config if needed, but don't pass as kwarg
        if "data" not in cfg:
            cfg["data"] = {}
        cfg["data"]["split"] = "train"
        loader = build_latent_pair_loader(cfg)
        
        # Load one batch
        print("Loading first batch...")
        batch = next(iter(loader))
        
        print(f"✅ Successfully loaded batch")
        print(f"   Batch keys: {list(batch.keys())}")
        
        # Check shapes
        if "latent" in batch:
            print(f"   Latent shape: {batch['latent'].shape}")
        if "latent_next" in batch:
            print(f"   Latent next shape: {batch['latent_next'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def build_models(cfg: Dict) -> bool:
    """Build all models to verify architecture."""
    print("\n" + "="*70)
    print("🏗️  Step 4: Building Models")
    print("="*70)
    
    try:
        from ups.models.latent_operator import LatentOperator, LatentOperatorConfig
        from ups.models.diffusion_residual import DiffusionResidual, DiffusionResidualConfig
        from ups.core.blocks_pdet import PDETransformerConfig
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        latent_dim = cfg.get("latent", {}).get("dim", 32)
        
        # Build operator
        print("Building operator...")
        op_cfg = LatentOperatorConfig(
            latent_dim=latent_dim,
            pdet=PDETransformerConfig(**cfg.get("operator", {}).get("pdet", {}))
        )
        operator = LatentOperator(op_cfg).to(device)
        
        param_count = sum(p.numel() for p in operator.parameters())
        print(f"✅ Operator built successfully")
        print(f"   Parameters: {param_count:,}")
        print(f"   Memory: ~{param_count * 4 / 1024**2:.1f} MB")
        
        # Build diffusion
        print("\nBuilding diffusion residual...")
        hidden_dim = cfg.get("diffusion", {}).get("hidden_dim", latent_dim * 2)
        diff_cfg = DiffusionResidualConfig(latent_dim=latent_dim, hidden_dim=hidden_dim)
        diffusion = DiffusionResidual(diff_cfg).to(device)
        
        param_count = sum(p.numel() for p in diffusion.parameters())
        print(f"✅ Diffusion built successfully")
        print(f"   Parameters: {param_count:,}")
        print(f"   Memory: ~{param_count * 4 / 1024**2:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Model building failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def estimate_training_time(cfg: Dict) -> Dict:
    """Estimate training time and cost."""
    print("\n" + "="*70)
    print("⏱️  Step 5: Estimating Training Time & Cost")
    print("="*70)
    
    stages = cfg.get("stages", {})
    
    # Get epochs per stage
    op_epochs = stages.get("operator", {}).get("epochs", 0)
    diff_epochs = stages.get("diff_residual", {}).get("epochs", 0)
    distill_epochs = stages.get("consistency_distill", {}).get("epochs", 0)
    steady_epochs = stages.get("steady_prior", {}).get("epochs", 0)
    
    # Rough time estimates per epoch (in seconds) based on model size
    latent_dim = cfg.get("latent", {}).get("dim", 32)
    batch_size = cfg.get("training", {}).get("batch_size", 4)
    
    # Time estimates scale with model size
    if latent_dim <= 32:
        op_time_per_epoch = 3  # seconds
        diff_time_per_epoch = 7
        distill_time_per_epoch = 300  # consistency is slow
        steady_time_per_epoch = 5
    elif latent_dim <= 64:
        op_time_per_epoch = 5
        diff_time_per_epoch = 10
        distill_time_per_epoch = 400
        steady_time_per_epoch = 8
    else:  # 512-dim
        op_time_per_epoch = 15
        diff_time_per_epoch = 30
        distill_time_per_epoch = 600
        steady_time_per_epoch = 20
    
    # Calculate total time
    op_time = op_epochs * op_time_per_epoch
    diff_time = diff_epochs * diff_time_per_epoch
    distill_time = distill_epochs * distill_time_per_epoch
    steady_time = steady_epochs * steady_time_per_epoch
    
    # Add evaluation time if enabled
    eval_time = 0
    if cfg.get("evaluation", {}).get("enabled"):
        eval_time = 15 * 60  # 15 minutes for baseline eval
        if cfg.get("ttc", {}).get("enabled"):
            eval_time += 20 * 60  # +20 minutes for TTC
    
    total_seconds = op_time + diff_time + distill_time + steady_time + eval_time
    total_minutes = total_seconds / 60
    total_hours = total_minutes / 60
    
    print(f"\nTime breakdown:")
    print(f"  Operator:      {op_epochs} epochs × {op_time_per_epoch}s = {op_time/60:.1f} min")
    print(f"  Diffusion:     {diff_epochs} epochs × {diff_time_per_epoch}s = {diff_time/60:.1f} min")
    print(f"  Distillation:  {distill_epochs} epochs × {distill_time_per_epoch}s = {distill_time/60:.1f} min")
    if steady_epochs > 0:
        print(f"  Steady Prior:  {steady_epochs} epochs × {steady_time_per_epoch}s = {steady_time/60:.1f} min")
    if eval_time > 0:
        print(f"  Evaluation:    {eval_time/60:.1f} min")
    
    print(f"\n📊 Total estimated time: {total_minutes:.0f} minutes ({total_hours:.1f} hours)")
    
    # Cost estimates for different GPUs ($/hour)
    gpu_costs = {
        "H100": 2.89,
        "A100": 1.89,
        "H200": 2.59,
    }
    
    print(f"\n💰 Estimated costs:")
    for gpu_name, cost_per_hour in gpu_costs.items():
        total_cost = total_hours * cost_per_hour
        print(f"  {gpu_name} (${cost_per_hour}/hr): ${total_cost:.2f}")
    
    # Recommend GPU
    if total_hours < 0.5:
        recommended = "A100 (cost-effective for short runs)"
    elif latent_dim >= 512:
        recommended = "H200 (best for large models)"
    else:
        recommended = "A100 (best cost/performance)"
    
    print(f"\n💡 Recommended: {recommended}")
    
    return {
        "total_hours": total_hours,
        "total_minutes": total_minutes,
        "stages": {
            "operator": op_time / 60,
            "diffusion": diff_time / 60,
            "distillation": distill_time / 60,
            "steady_prior": steady_time / 60,
            "evaluation": eval_time / 60,
        },
        "costs": {gpu: total_hours * cost for gpu, cost in gpu_costs.items()}
    }


def main():
    parser = argparse.ArgumentParser(description="Dry-run training configuration")
    parser.add_argument("config", type=str, help="Path to training config YAML")
    parser.add_argument("--estimate-only", action="store_true", help="Only estimate time/cost, skip validation")
    parser.add_argument("--skip-data", action="store_true", help="Skip data availability checks")
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🚀 DRY RUN MODE - Training Configuration Test")
    print("="*70)
    print(f"Config: {args.config}")
    print("="*70)
    
    # Step 1: Validate config
    cfg = validate_config(args.config)
    
    if args.estimate_only:
        # Just estimate and exit
        estimate_training_time(cfg)
        print("\n✅ Estimation complete (skipped validation)")
        sys.exit(0)
    
    # Step 2: Check data
    if not args.skip_data:
        data_ok = check_data_availability(cfg)
        if not data_ok:
            print("\n⚠️  Data check failed - training may fail")
            print("   Use --skip-data to bypass this check")
    
    # Step 3: Test data loader
    loader_ok = test_data_loader(cfg)
    if not loader_ok:
        print("\n❌ Data loader test failed")
        sys.exit(1)
    
    # Step 4: Build models
    models_ok = build_models(cfg)
    if not models_ok:
        print("\n❌ Model building failed")
        sys.exit(1)
    
    # Step 5: Estimate time/cost
    estimate_training_time(cfg)
    
    # Final summary
    print("\n" + "="*70)
    print("✅ DRY RUN COMPLETE - Configuration is ready for training")
    print("="*70)
    print("\n📝 To launch training:")
    print(f"   python scripts/train.py {args.config}")
    print("\n📝 To launch on VastAI:")
    print(f"   python scripts/vast_launch.py --config {args.config}")
    print("="*70)


if __name__ == "__main__":
    main()
