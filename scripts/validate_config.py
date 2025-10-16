#!/usr/bin/env python3
"""
Validate training config before launching expensive GPU runs.

Usage:
    python scripts/validate_config.py configs/train_burgers_32dim_v2_fixed.yaml
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

def load_config(config_path: str) -> Dict:
    """Load config using the same loader as training script."""
    try:
        from ups.utils.config_loader import load_config_with_includes
        return load_config_with_includes(config_path)
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        sys.exit(1)

def validate_architecture(cfg: Dict) -> List[Tuple[str, bool, str]]:
    """Validate architecture consistency."""
    checks = []
    
    latent_dim = cfg.get("latent", {}).get("dim")
    op_input = cfg.get("operator", {}).get("pdet", {}).get("input_dim")
    op_hidden = cfg.get("operator", {}).get("pdet", {}).get("hidden_dim")
    diff_latent = cfg.get("diffusion", {}).get("latent_dim")
    diff_hidden = cfg.get("diffusion", {}).get("hidden_dim")
    ttc_decoder_latent = cfg.get("ttc", {}).get("decoder", {}).get("latent_dim")
    ttc_decoder_hidden = cfg.get("ttc", {}).get("decoder", {}).get("hidden_dim")
    
    # Check 1: latent.dim is set
    checks.append((
        "latent.dim defined",
        latent_dim is not None,
        f"latent.dim = {latent_dim}" if latent_dim else "Missing latent.dim"
    ))
    
    # Check 2: operator.pdet.input_dim matches latent.dim
    checks.append((
        "operator.pdet.input_dim == latent.dim",
        op_input == latent_dim if op_input and latent_dim else False,
        f"{op_input} == {latent_dim}" if op_input else "Missing operator.pdet.input_dim"
    ))
    
    # Check 3: operator.pdet.hidden_dim is defined
    checks.append((
        "operator.pdet.hidden_dim defined",
        op_hidden is not None,
        f"hidden_dim = {op_hidden}" if op_hidden else "Missing operator.pdet.hidden_dim"
    ))
    
    # Check 4: diffusion.latent_dim matches latent.dim
    checks.append((
        "diffusion.latent_dim == latent.dim",
        diff_latent == latent_dim if diff_latent and latent_dim else False,
        f"{diff_latent} == {latent_dim}" if diff_latent else "Missing diffusion.latent_dim"
    ))
    
    # Check 5: diffusion.hidden_dim matches operator.pdet.hidden_dim
    checks.append((
        "diffusion.hidden_dim == operator.pdet.hidden_dim",
        diff_hidden == op_hidden if diff_hidden and op_hidden else False,
        f"{diff_hidden} == {op_hidden}" if diff_hidden else "Missing diffusion.hidden_dim"
    ))
    
    # Check 6: TTC decoder latent_dim matches latent.dim
    if cfg.get("ttc", {}).get("enabled"):
        checks.append((
            "ttc.decoder.latent_dim == latent.dim",
            ttc_decoder_latent == latent_dim if ttc_decoder_latent and latent_dim else False,
            f"{ttc_decoder_latent} == {latent_dim}" if ttc_decoder_latent else "Missing ttc.decoder.latent_dim"
        ))
        
        # Check 7: TTC decoder hidden_dim matches operator.pdet.hidden_dim
        checks.append((
            "ttc.decoder.hidden_dim == operator.pdet.hidden_dim",
            ttc_decoder_hidden == op_hidden if ttc_decoder_hidden and op_hidden else False,
            f"{ttc_decoder_hidden} == {op_hidden}" if ttc_decoder_hidden else "Missing ttc.decoder.hidden_dim"
        ))
    
    return checks

def validate_wandb(cfg: Dict) -> List[Tuple[str, bool, str]]:
    """Validate WandB configuration."""
    checks = []
    
    wandb_cfg = cfg.get("logging", {}).get("wandb", {})
    
    checks.append((
        "wandb.enabled is true",
        wandb_cfg.get("enabled") is True,
        f"enabled = {wandb_cfg.get('enabled')}"
    ))
    
    checks.append((
        "wandb.project defined",
        wandb_cfg.get("project") is not None,
        f"project = {wandb_cfg.get('project', 'MISSING')}"
    ))
    
    checks.append((
        "wandb.entity defined",
        wandb_cfg.get("entity") is not None,
        f"entity = {wandb_cfg.get('entity', 'MISSING')}"
    ))
    
    return checks

def validate_stages(cfg: Dict) -> List[Tuple[str, bool, str]]:
    """Validate training stages configuration."""
    checks = []
    
    stages = cfg.get("stages", {})
    
    # Check operator stage
    op_epochs = stages.get("operator", {}).get("epochs", 0)
    checks.append((
        "operator.epochs > 0",
        op_epochs > 0,
        f"epochs = {op_epochs}"
    ))
    
    # Check diffusion stage
    diff_epochs = stages.get("diff_residual", {}).get("epochs", 0)
    checks.append((
        "diff_residual.epochs > 0",
        diff_epochs > 0,
        f"epochs = {diff_epochs}"
    ))
    
    # Check optimizer for operator
    op_opt = stages.get("operator", {}).get("optimizer", {})
    checks.append((
        "operator.optimizer.lr defined",
        op_opt.get("lr") is not None,
        f"lr = {op_opt.get('lr', 'MISSING')}"
    ))
    
    return checks

def validate_evaluation(cfg: Dict) -> List[Tuple[str, bool, str]]:
    """Validate evaluation configuration."""
    checks = []
    
    eval_enabled = cfg.get("evaluation", {}).get("enabled")
    checks.append((
        "evaluation.enabled is true",
        eval_enabled is True,
        f"enabled = {eval_enabled}"
    ))
    
    return checks

def print_results(category: str, checks: List[Tuple[str, bool, str]]):
    """Print validation results for a category."""
    print(f"\n{category}:")
    print("─" * 70)
    
    for name, passed, details in checks:
        status = "✅" if passed else "❌"
        print(f"{status} {name}")
        if not passed or "--verbose" in sys.argv:
            print(f"   {details}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/validate_config.py <config_path>")
        print("\nExample:")
        print("  python scripts/validate_config.py configs/train_burgers_32dim_v2_fixed.yaml")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    if not Path(config_path).exists():
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)
    
    print(f"Validating: {config_path}")
    print("=" * 70)
    
    # Load config
    cfg = load_config(config_path)
    print("✅ Config loaded successfully")
    
    # Run validation checks
    all_checks = []
    
    arch_checks = validate_architecture(cfg)
    print_results("Architecture", arch_checks)
    all_checks.extend(arch_checks)
    
    wandb_checks = validate_wandb(cfg)
    print_results("WandB", wandb_checks)
    all_checks.extend(wandb_checks)
    
    stage_checks = validate_stages(cfg)
    print_results("Training Stages", stage_checks)
    all_checks.extend(stage_checks)
    
    eval_checks = validate_evaluation(cfg)
    print_results("Evaluation", eval_checks)
    all_checks.extend(eval_checks)
    
    # Summary
    passed = sum(1 for _, p, _ in all_checks if p)
    total = len(all_checks)
    
    print("\n" + "=" * 70)
    print(f"Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n✅ Config is valid and ready for training!")
        sys.exit(0)
    else:
        print(f"\n❌ {total - passed} issues found. Please fix before training.")
        sys.exit(1)

if __name__ == "__main__":
    main()

