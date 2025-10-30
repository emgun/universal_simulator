# Multi-Stage Training Pipeline Documentation - Index

## Document Files

This directory contains comprehensive documentation of the UPS multi-stage training pipeline:

### 1. **MULTI_STAGE_TRAINING_SUMMARY.md** (346 lines) - START HERE
   - **Purpose**: Executive summary for quick reference
   - **Best for**: Getting oriented, understanding architecture at a glance
   - **Contents**:
     - Quick reference for all 4 stages
     - Loss function reference tables
     - Gradient norm interpretation guide
     - Checkpoint flow diagram
     - Essential configuration template
     - Common troubleshooting

### 2. **MULTI_STAGE_TRAINING.md** (1,106 lines) - DETAILED REFERENCE
   - **Purpose**: Comprehensive technical documentation
   - **Best for**: Implementation details, debugging, understanding loss functions
   - **Contents**:
     - Stage 1: Operator Training (294 lines)
     - Stage 2: Diffusion Residual (186 lines)
     - Stage 3: Consistency Distillation (263 lines)
     - Stage 4: Steady Prior (104 lines)
     - Checkpoint management (83 lines)
     - Logging & WandB integration (93 lines)
     - Configuration structure (200+ lines)
     - Gradient norm analysis (52 lines)
     - References & file locations (53 lines)

### 3. **MULTI_STAGE_TRAINING_INDEX.md** (THIS FILE)
   - Navigation guide for the documentation

---

## Document Structure (Main Document)

### Stage 1: Operator Training
- **Implementation**: `scripts/train.py` lines 400-693
- **Function**: `train_operator()`
- **Key Topics**:
  - Architecture: `LatentOperator` with PDETransformer
  - Losses: Forward (primary) + 4 optional auxiliary losses
  - Curriculum learning for inverse losses
  - Gradient accumulation, AMP, EMA, early stopping
  - Checkpointing: `operator.pt`, `operator_ema.pt`

### Stage 2: Diffusion Residual Training  
- **Implementation**: `scripts/train.py` lines 695-880
- **Function**: `train_diffusion()`
- **Key Topics**:
  - Architecture: `DiffusionResidual` (3-layer MLP)
  - Teacher: Frozen operator from Stage 1
  - Loss: MSE of residual correction
  - Tau sampling (uniform/beta distribution)
  - Periodic checkpoints (optional)
  - Checkpointing: `diffusion_residual.pt`, `diffusion_residual_ema.pt`

### Stage 3: Consistency Distillation
- **Implementation**: `scripts/train.py` lines 942-1204
- **Function**: `train_consistency()`
- **Key Topics**:
  - Purpose: Few-step distillation of diffusion model
  - Teacher: Operator + Diffusion
  - Student: Same diffusion model
  - Optimizations: Teacher caching, micro-batching, async transfers, torch.compile
  - Configuration: tau schedule, target loss, micro-batch size
  - Checkpointing: Overwrites `diffusion_residual.pt`

### Stage 4: Steady Prior (Optional)
- **Implementation**: `scripts/train.py` lines 1206-1309
- **Function**: `train_steady_prior()`
- **Key Topics**:
  - Architecture: `SteadyPrior` (iterative refinement)
  - Loss: Simple MSE
  - Configuration: Set `epochs: 0` to skip
  - Checkpointing: `steady_prior.pt`

### Supporting Topics
- **Loss Functions** (comprehensive reference)
- **Checkpoint Management** (loading, stripping compiled prefix, device placement)
- **Global Step & Logging** (WandB context, TrainingLogger, gradient norm tracking)
- **Configuration Structure** (complete YAML template with all parameters)
- **Gradient Norm Analysis** (interpretation, debugging)
- **Fast-to-SOTA Integration** (standalone vs orchestrator)

---

## Quick Navigation

### By Question

**"How do I run training?"**
- See: SUMMARY.md → Usage Examples
- Full details: DETAILED.md → Integration with Fast-to-SOTA Orchestrator

**"What are the loss functions?"**
- See: SUMMARY.md → Loss Functions Reference
- Full details: DETAILED.md → Stage 1/2/3/4 Loss Functions sections

**"How does the pipeline connect stages?"**
- See: SUMMARY.md → Checkpoint Flow
- Full details: DETAILED.md → Checkpoint Passing & State Management

**"My gradient norm is exploding/vanishing!"**
- See: SUMMARY.md → Troubleshooting
- Full details: DETAILED.md → Gradient Norm Analysis & Debugging

**"What configuration parameters do I need?"**
- See: SUMMARY.md → Configuration Template
- Full details: DETAILED.md → Configuration Structure section (~200 lines)

**"How does Stage 3 optimize training?"**
- See: SUMMARY.md → Key Optimizations → Stage 3
- Full details: DETAILED.md → Stage 3: Consistency Distillation → Optimizations

**"Which checkpoints are saved where?"**
- See: SUMMARY.md → Checkpoint Flow
- Full details: DETAILED.md → Files & Locations → Checkpoints table

---

## Implementation File Locations

### Training Scripts
- **Main**: `/Users/emerygunselman/Code/universal_simulator/scripts/train.py` (1,812 lines)
  - Stages 1-4 training functions
  - All utility functions (seed, logging, optimization)
  
- **Orchestrator**: `/Users/emerygunselman/Code/universal_simulator/scripts/run_fast_to_sota.py` (1,225 lines)
  - Pipeline orchestration
  - Validation, evaluation, gating, leaderboard

### Core Models
- `src/ups/models/latent_operator.py` → Stage 1
- `src/ups/models/diffusion_residual.py` → Stage 2 & 3
- `src/ups/models/steady_prior.py` → Stage 4

### Training Utilities
- `src/ups/training/losses.py` → Loss functions
- `src/ups/training/loop_train.py` → Legacy curriculum training
- `src/ups/training/consistency_distill.py` → Consistency utilities
- `src/ups/data/latent_pairs.py` → Data loading

### Checkpoint Locations
```
checkpoints/
├── operator.pt                          (Stage 1)
├── operator_ema.pt                      (Stage 1, optional)
├── diffusion_residual.pt                (Stages 2, 3)
├── diffusion_residual_ema.pt            (Stages 2, 3, optional)
├── diffusion_residual_epoch_*.pt        (Stage 2, optional periodic)
└── steady_prior.pt                      (Stage 4, optional)
```

---

## Key Concepts Summary

### The Four Stages (Sequential Pipeline)

```
Stage 1: Operator
  Input: z(t)
  Output: z(t+dt)
  Model: PDE-Transformer (deterministic)
  Loss: Forward prediction + optional inverse/rollout/spectral
  ↓
Stage 2: Diffusion Residual
  Input: z + tau (diffusion time)
  Output: residual correction
  Model: 3-layer MLP
  Teacher: Frozen operator from Stage 1
  Loss: MSE of residual
  ↓
Stage 3: Consistency Distillation
  Input: Same as Stage 2
  Output: Same as Stage 2 (distilled)
  Purpose: Train teacher forcing (few-step inference)
  Overwrites: diffusion_residual.pt
  ↓
Stage 4: Steady Prior (Optional)
  Input: z
  Output: refined z
  Model: Iterative refinement
  Loss: MSE to target
  Independent: No checkpoint dependencies
```

### Loss Function Hierarchy

**Stage 1 (Operator)**
- L_forward (always) + L_inv_enc + L_inv_dec + L_rollout + L_spec
- Inverse losses follow curriculum: warmup 15 epochs → ramp 15 epochs → steady

**Stage 2 (Diffusion)**
- L_base (always) + L_spec + L_rel
- Learns residual: z_target - operator(z)

**Stage 3 (Consistency)**
- L_distillation (implicit via teacher forcing)
- MSE across multiple tau samples

**Stage 4 (Steady Prior)**
- L_mse (simple MSE)

### Checkpoint Passing
1. Stage 1 saves → `operator.pt`
2. Stage 2 loads operator (frozen), saves → `diffusion_residual.pt`
3. Stage 3 loads both, distills, OVERWRITES → `diffusion_residual.pt`
4. Stage 4 trains independently, saves → `steady_prior.pt`

### Key Optimizations
- **Stage 1**: Gradient accumulation, AMP, torch.compile, EMA
- **Stage 2**: Frozen teacher, periodic checkpoints, tau sampling
- **Stage 3**: Teacher caching (~2x), micro-batching, async transfers, torch.compile (~1.3-1.5x)
- **Stage 4**: Minimal (epoch-based)

---

## Configuration Quick Reference

### Essential Dimension Consistency
```yaml
latent:
  dim: 32                          # MUST match below
operator:
  pdet:
    input_dim: 32                  # MUST match latent.dim
diffusion:
  latent_dim: 32                   # MUST match latent.dim (if used)
```

### Per-Stage Config Example
```yaml
stages:
  operator:
    epochs: 25                     # Training epochs
    patience: 5                    # Early stopping
    optimizer: {name: adamw, lr: 0.001, weight_decay: 1e-5}
    scheduler: {name: cosineannealinglr, t_max: 25, eta_min: 1e-6}
  
  diff_residual:
    epochs: 15
    optimizer: {name: adamw, lr: 0.0005}
    checkpoint_interval: 0         # Set > 0 for periodic saves
  
  consistency_distill:
    epochs: 5
    batch_size: 8                  # Micro-batch size
    distill_num_taus: 3            # Tau samples per batch
    distill_micro_batch: 8         # Gradient accumulation chunk
    tau_schedule: [3,3,3,3,3]      # Optional per-epoch taus
    target_loss: 0.0               # Early stop if reached (0=disabled)
  
  steady_prior:
    epochs: 0                      # Set to 0 to skip
```

### Training Hyperparameters
```yaml
training:
  dt: 0.1                          # Time step
  batch_size: 32                   # Batch size (stages 1,2,4)
  num_workers: 4                   # DataLoader workers
  amp: true                        # Automatic mixed precision
  compile: true                    # torch.compile
  grad_clip: 1.0                   # Gradient clipping
  ema_decay: 0.999                 # Exponential moving average decay
  
  # Loss weights
  lambda_spectral: 0.0
  lambda_relative: 0.0
  lambda_rollout: 0.0
  lambda_inv_enc: 0.0              # Inverse encoding loss
  lambda_inv_dec: 0.0              # Inverse decoding loss
  
  # Curriculum
  inverse_loss_warmup_epochs: 15
  inverse_loss_max_weight: 0.05
```

---

## WandB Integration

### One Run Per Pipeline
All 4 stages log to the same WandB run with stage prefixes:
- `operator/loss`, `operator/grad_norm`, `operator/lr`, ...
- `diff_residual/loss`, `diff_residual/lr`, ...
- `consistency_distill/loss`, ...
- `steady_prior/loss`, ...

### Checkpoints & Artifacts
All stage checkpoints uploaded automatically:
- `operator.pt`
- `diffusion_residual.pt`
- `steady_prior.pt`

### Integration with Orchestrator
- Training subprocess creates WandB run
- Context saved to `WANDB_CONTEXT_FILE`
- Evaluation subprocesses load context, log to same run
- No run proliferation (one run per pipeline)

---

## Troubleshooting Guide

| Issue | Cause | Solution |
|-------|-------|----------|
| Dimension mismatch | Config inconsistency | `latent.dim == operator.pdet.input_dim` |
| OOM Stage 1 | Large batch/model | Reduce batch_size or latent.dim |
| OOM Stage 3 | Too many taus | Reduce distill_num_taus |
| Grad norm < 0.001 | Poor training | Increase lr, check data |
| Grad norm spikes | High lr, instability | Enable grad_clip, reduce lr |
| Grad norm explodes @ epoch 15 | Inverse loss curriculum | Increase inverse_loss_warmup_epochs |
| Loss doesn't decrease | Learning rate/data issue | Check lr, validate data loading |

---

## Reading Recommendations

### For Users (First Time)
1. Read: **SUMMARY.md** (10 min)
2. Look up: Specific stage section in SUMMARY.md
3. Run: Quick example from Usage Examples section
4. Refer to: DETAILED.md for specifics during implementation

### For Developers
1. Understand: Full pipeline from DETAILED.md (~1 hour)
2. Reference: Loss function mathematics (DETAILED.md)
3. Implement: Changes to specific stage functions
4. Validate: With gradient norm tracking and early stopping

### For Debugging
1. Check: Troubleshooting table in SUMMARY.md
2. Read: Gradient norm analysis section (DETAILED.md)
3. Verify: Configuration consistency (Config Structure section)
4. Monitor: WandB charts for loss/grad_norm trends

---

## File Structure Summary

```
Documentation:
  ├── MULTI_STAGE_TRAINING_SUMMARY.md      (Quick ref, 346 lines)
  ├── MULTI_STAGE_TRAINING.md              (Full ref, 1,106 lines)
  └── MULTI_STAGE_TRAINING_INDEX.md        (This file)

Source Code:
  scripts/
    ├── train.py                           (Main entry, 1,812 lines)
    └── run_fast_to_sota.py               (Orchestrator, 1,225 lines)
  
  src/ups/training/
    ├── losses.py                          (Loss functions)
    ├── loop_train.py                      (Legacy curriculum)
    └── consistency_distill.py             (Distillation utils)
  
  src/ups/models/
    ├── latent_operator.py                 (Stage 1)
    ├── diffusion_residual.py              (Stages 2,3)
    └── steady_prior.py                    (Stage 4)

Checkpoints:
  checkpoints/
    ├── operator.pt
    ├── operator_ema.pt
    ├── diffusion_residual.pt
    ├── diffusion_residual_ema.pt
    └── steady_prior.pt
```

---

## Contact & Questions

For detailed questions about:
- **Loss functions**: See DETAILED.md "Loss Functions" sections
- **Optimizations**: See SUMMARY.md "Key Optimizations" or DETAILED.md Stage 3
- **Configuration**: See DETAILED.md "Configuration Structure"
- **Debugging**: See SUMMARY.md "Troubleshooting" or DETAILED.md "Gradient Norm Analysis"

