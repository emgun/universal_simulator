#!/usr/bin/env python3
"""
Validate training config before launching expensive GPU runs.

Usage:
    python scripts/validate_config.py configs/train_burgers_32dim_v2_fixed.yaml
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

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
    
    # Check diffusion stage (allow 0 for ablation tests)
    diff_epochs = stages.get("diff_residual", {}).get("epochs", 0)
    checks.append((
        "diff_residual.epochs >= 0",
        diff_epochs >= 0,
        f"epochs = {diff_epochs} (0 = disabled for ablation test)"
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


def validate_data(cfg: Dict) -> List[Tuple[str, bool, str]]:
    """Validate data configuration and availability."""
    checks = []
    
    data_cfg = cfg.get("data", {})
    
    # Check data.task is defined
    task = data_cfg.get("task")
    checks.append((
        "data.task defined",
        task is not None,
        f"task = {task}" if task else "Missing data.task"
    ))
    
    # Check data.split is valid
    split = data_cfg.get("split", "train")
    valid_splits = ["train", "val", "test"]
    checks.append((
        "data.split is valid",
        split in valid_splits,
        f"split = {split} (valid: {valid_splits})"
    ))
    
    # Check data.root exists
    root = data_cfg.get("root")
    if root:
        root_path = Path(root)
        checks.append((
            "data.root exists",
            root_path.exists(),
            f"path = {root} ({'exists' if root_path.exists() else 'NOT FOUND'})"
        ))
    else:
        checks.append((
            "data.root defined",
            False,
            "Missing data.root"
        ))
    
    # Check num_workers compatibility with cache
    num_workers = cfg.get("training", {}).get("num_workers", 0)
    latent_cache_dir = cfg.get("training", {}).get("latent_cache_dir")
    
    if num_workers > 0 and not latent_cache_dir:
        checks.append((
            "num_workers > 0 requires latent_cache_dir or use_parallel_encoding",
            cfg.get("training", {}).get("use_parallel_encoding", False),
            f"num_workers={num_workers} but no cache/parallel encoding configured"
        ))
    
    return checks


def validate_hardware(cfg: Dict) -> List[Tuple[str, bool, str]]:
    """Validate hardware-related settings."""
    checks = []
    
    training_cfg = cfg.get("training", {})
    latent_dim = cfg.get("latent", {}).get("dim", 32)
    batch_size = training_cfg.get("batch_size", 4)
    
    # Estimate GPU memory requirement (rough heuristic)
    # Model size: latent_dim^2 * hidden_dim_multiplier
    # Batch memory: batch_size * latent_dim * sequence_length
    hidden_dim = cfg.get("operator", {}).get("pdet", {}).get("hidden_dim", latent_dim * 2)
    
    # Rough memory estimate in GB
    model_params = (latent_dim * hidden_dim * 10) / 1e6  # ~10 layers worth
    batch_memory = (batch_size * latent_dim * 32 * 4) / 1e9  # 32 timesteps, float32
    total_memory_gb = (model_params * 4 + batch_memory) * 2  # 2x for gradients/optimizer
    
    # Check if batch size is reasonable for dimension
    if latent_dim <= 32:
        max_safe_batch = 16
    elif latent_dim <= 64:
        max_safe_batch = 12
    elif latent_dim <= 128:
        max_safe_batch = 8
    else:  # 512-dim
        max_safe_batch = 4
    
    checks.append((
        f"batch_size appropriate for {latent_dim}-dim model",
        batch_size <= max_safe_batch,
        f"batch_size={batch_size}, recommended ≤{max_safe_batch} for {latent_dim}-dim (~{total_memory_gb:.1f}GB GPU RAM)"
    ))
    
    # Check compilation settings
    compile_enabled = training_cfg.get("compile", False)
    rollout_horizon = training_cfg.get("rollout_horizon", 0)
    lambda_rollout = training_cfg.get("lambda_rollout", 0.0)
    
    if compile_enabled and rollout_horizon > 0 and lambda_rollout > 0:
        checks.append((
            "compile + rollout_loss compatibility",
            True,  # This is now handled in train.py with mode="default"
            f"compile={compile_enabled}, rollout={rollout_horizon} (using mode='default' for compatibility)"
        ))
    
    return checks


def validate_hyperparameters(cfg: Dict) -> List[Tuple[str, bool, str]]:
    """Validate hyperparameters for suspicious values."""
    checks = []
    
    stages = cfg.get("stages", {})
    
    # Check operator learning rate
    op_lr = stages.get("operator", {}).get("optimizer", {}).get("lr")
    if op_lr:
        checks.append((
            "operator LR in reasonable range",
            1e-5 <= op_lr <= 1e-2,
            f"lr = {op_lr} (recommended: 1e-4 to 1e-3)"
        ))
    
    # Check diffusion learning rate
    diff_lr = stages.get("diff_residual", {}).get("optimizer", {}).get("lr")
    if diff_lr:
        checks.append((
            "diffusion LR in reasonable range",
            1e-6 <= diff_lr <= 1e-3,
            f"lr = {diff_lr} (recommended: 3e-5 to 1e-4)"
        ))
    
    # Check weight decay
    weight_decay = stages.get("operator", {}).get("optimizer", {}).get("weight_decay", 0)
    checks.append((
        "weight_decay in reasonable range",
        0 <= weight_decay <= 0.1,
        f"weight_decay = {weight_decay} (recommended: 0.01 to 0.03)"
    ))
    
    # Check epochs aren't accidentally 0 or too high
    op_epochs = stages.get("operator", {}).get("epochs", 0)
    if op_epochs > 0:
        checks.append((
            "operator epochs reasonable",
            1 <= op_epochs <= 100,
            f"epochs = {op_epochs} (typical: 5-50)"
        ))
    
    # Check gradient clipping if present
    grad_clip = cfg.get("training", {}).get("grad_clip", 1.0)
    checks.append((
        "grad_clip in reasonable range",
        0.1 <= grad_clip <= 10.0,
        f"grad_clip = {grad_clip} (typical: 0.5-2.0)"
    ))
    
    # Check time_stride
    time_stride = cfg.get("training", {}).get("time_stride", 1)
    checks.append((
        "time_stride reasonable",
        1 <= time_stride <= 4,
        f"time_stride = {time_stride} (typical: 1-2)"
    ))
    
    return checks


def validate_checkpoints(cfg: Dict) -> List[Tuple[str, bool, str]]:
    """Validate checkpoint paths if resuming."""
    checks = []
    
    checkpoint_cfg = cfg.get("checkpoint", {})
    checkpoint_dir = Path(checkpoint_cfg.get("dir", "checkpoints"))
    
    checks.append((
        "checkpoint.dir exists",
        checkpoint_dir.exists(),
        f"path = {checkpoint_dir} ({'exists' if checkpoint_dir.exists() else 'will be created'})"
    ))
    
    # Check if checkpoints exist (for resume scenarios)
    if checkpoint_dir.exists():
        op_ckpt = checkpoint_dir / "operator.pt"
        diff_ckpt = checkpoint_dir / "diffusion_residual.pt"
        
        if op_ckpt.exists():
            checks.append((
                "operator checkpoint found",
                True,
                f"found at {op_ckpt} ({op_ckpt.stat().st_size / 1e6:.1f} MB)"
            ))
        
        if diff_ckpt.exists():
            checks.append((
                "diffusion checkpoint found",
                True,
                f"found at {diff_ckpt} ({diff_ckpt.stat().st_size / 1e6:.1f} MB)"
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
    
    data_checks = validate_data(cfg)
    print_results("Data Configuration", data_checks)
    all_checks.extend(data_checks)
    
    hardware_checks = validate_hardware(cfg)
    print_results("Hardware & Performance", hardware_checks)
    all_checks.extend(hardware_checks)
    
    hyperparam_checks = validate_hyperparameters(cfg)
    print_results("Hyperparameters", hyperparam_checks)
    all_checks.extend(hyperparam_checks)
    
    checkpoint_checks = validate_checkpoints(cfg)
    print_results("Checkpoints", checkpoint_checks)
    all_checks.extend(checkpoint_checks)
    
    wandb_checks = validate_wandb(cfg)
    print_results("WandB Logging", wandb_checks)
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

