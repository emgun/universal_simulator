# Multi-Stage Training Pipeline - Executive Summary

## Document Overview

**Main Document**: `/Users/emerygunselman/Code/universal_simulator/MULTI_STAGE_TRAINING.md` (1,106 lines)

Comprehensive documentation of the UPS multi-stage training pipeline, covering all four sequential training stages with implementation details, loss functions, optimizations, and configuration guidance.

---

## Quick Reference: The Four Stages

### Stage 1: Operator Training (Deterministic)
**Purpose**: Train latent-space PDE-Transformer to predict one-step ahead  
**Input**: z(t) at time t  
**Output**: z(t+dt) at time t+dt  
**Model**: `LatentOperator` with PDETransformer backbone  
**Losses**:
- Primary: Forward prediction loss (MSE)
- Optional: Inverse encoding/decoding, rollout, spectral losses
- Curriculum: Inverse losses ramp up gradually (15 epoch warmup)

**Checkpoints**: `operator.pt`, `operator_ema.pt`  
**Duration**: ~25 minutes (A100)  
**File**: `scripts/train.py` lines 400-693

---

### Stage 2: Diffusion Residual Training
**Purpose**: Learn residual correction to operator via diffusion  
**Teacher**: Frozen operator from Stage 1  
**Key Mechanism**: Predicts z_target - operator(z) at sampled diffusion times (tau)  
**Model**: `DiffusionResidual` (3-layer MLP)  
**Input**: Latent state + tau scalar (diffusion time)  
**Tau Sampling**: Uniform or Beta distribution (broadens supervision)  
**Losses**:
- Primary: MSE of residual prediction
- Optional: Spectral, NRMSE losses

**Checkpoints**: `diffusion_residual.pt`, `diffusion_residual_ema.pt`  
**Duration**: ~15 minutes (A100)  
**File**: `scripts/train.py` lines 695-880

---

### Stage 3: Consistency Distillation
**Purpose**: Distill diffusion model to few-step sampler  
**Teacher**: Operator + Diffusion model  
**Student**: Same diffusion model being trained  
**Key Innovations**:
1. Teacher caching (compute once per batch): ~2x speedup
2. Micro-batching with gradient accumulation
3. Vectorized tau sampling
4. Optional torch.compile: ~1.3-1.5x speedup

**Configuration**:
- `distill_num_taus`: Number of tau values per batch (default: 3)
- `distill_micro_batch`: Micro-batch size (default: 8)
- `tau_schedule`: Optional per-epoch tau adjustment
- `target_loss`: Early stopping threshold

**Checkpoints**: Updates `diffusion_residual.pt` (overwrites Stage 2)  
**Duration**: ~10 minutes (A100)  
**File**: `scripts/train.py` lines 942-1204

---

### Stage 4: Steady Prior (Optional)
**Purpose**: Learn steady-state latent prior (optional refinement)  
**Model**: `SteadyPrior` (iterative refinement)  
**Design**: Runs fixed num_steps (typically 4-6) with learnable drift  
**Loss**: Simple MSE to target latents  
**Configuration**: Set `epochs: 0` to skip  

**Checkpoints**: `steady_prior.pt`  
**Duration**: Negligible if skipped  
**File**: `scripts/train.py` lines 1206-1309

---

## Loss Functions Reference

### Stage 1: Operator

| Loss | Formula | Weight | Default | Purpose |
|------|---------|--------|---------|---------|
| Forward | MSE(pred, target) | lambda_forward | 1.0 | One-step prediction |
| Inverse Enc | MSE(decoder(latent), input_fields) | lambda_inv_enc | 0.0 | Latent decodability |
| Inverse Dec | MSE(encoder(decoder(latent)), latent) | lambda_inv_dec | 0.0 | Decoder-encoder invertibility |
| Rollout | MSE(rollout_pred, rollout_tgt) | lambda_rollout | 0.0 | Multi-step trajectory |
| Spectral | Energy diff in Fourier space | lambda_spectral | 0.0 | Frequency preservation |

**Curriculum Learning** (Inverse losses only):
```
Epochs 0-15: weight = 0 (pure forward training)
Epochs 15-30: weight ramps linearly from 0 to base_weight
Epochs 30+: weight = min(base_weight, 0.05)
```

### Stage 2: Diffusion

| Loss | Formula | Weight | Default |
|------|---------|--------|---------|
| Base | MSE(drift, residual_target) | 1.0 | - |
| Spectral | Energy diff (drift vs residual) | lambda_spectral | 0.0 |
| NRMSE | Relative NRMSE (drift vs residual) | lambda_relative | 0.0 |

### Stage 3: Consistency

| Loss | Formula | Weight |
|------|---------|--------|
| Distillation | MSE(student_z, base_z) across tau values | 1.0 |

### Stage 4: Steady Prior

| Loss | Formula |
|------|---------|
| MSE | MSE(prior(z), target) |

---

## Gradient Norm Tracking

**What It Means**:
```
< 0.01      → Vanishing gradients (poor optimization)
0.01 - 1.0  → Healthy range (typical)
> 1.0       → Apply clipping or reduce learning rate
> 10.0      → Exploding gradients (instability)
```

**How It Works**:
1. After each gradient accumulation step
2. Computed via `torch.nn.utils.clip_grad_norm_()`
3. Value BEFORE clipping (for diagnostics)
4. Clipping applied if `grad_clip` is set
5. Per-epoch mean logged to WandB

**Common Issues**:
- **Sudden spikes**: High learning rate → reduce or enable clipping
- **Always < 0.001**: Not training → check data/loss computation
- **Inverse loss explosion**: Enable curriculum learning (default)

---

## Checkpoint Flow

```
Operator (S1)
    ↓ saves operator.pt
    ↓
Diffusion (S2)
    ↓ loads operator.pt (frozen)
    ↓ saves diffusion_residual.pt
    ↓
Consistency (S3)
    ↓ loads operator.pt + diffusion_residual.pt
    ↓ distills diffusion, OVERWRITES diffusion_residual.pt
    ↓
Steady Prior (S4)
    ↓ independent training
    ↓ saves steady_prior.pt
```

**Key Detail**: Compiled models have `_orig_mod.` prefix removed before loading via `_strip_compiled_prefix()`.

---

## Configuration Template

### Essential Settings
```yaml
latent:
  dim: 32              # MUST match operator.pdet.input_dim
  tokens: 48

operator:
  pdet:
    input_dim: 32      # MUST match latent.dim
    hidden_dim: 64     # 2x latent_dim typical
    depths: [1, 1, 1]

training:
  dt: 0.1              # Time step
  amp: true            # Automatic mixed precision
  grad_clip: 1.0       # Gradient clipping

stages:
  operator:
    epochs: 25
    optimizer: {name: adamw, lr: 0.001}
    scheduler: {name: cosineannealinglr, t_max: 25}
  
  diff_residual:
    epochs: 15
    optimizer: {name: adamw, lr: 0.0005}
  
  consistency_distill:
    epochs: 5
    distill_num_taus: 3
    distill_micro_batch: 8
  
  steady_prior:
    epochs: 0  # Set to skip
```

---

## Key Optimizations

### Stage 1: Operator
- **Gradient Accumulation**: Enables larger effective batch size
- **AMP**: ~20% speedup with bf16 (enabled by default)
- **torch.compile**: ~10-15% speedup (safe mode, no CUDA graphs)
- **EMA**: Stabilizes training, separate checkpoint

### Stage 2: Diffusion
- **Frozen Teacher**: Operator as frozen teacher for efficiency
- **Periodic Checkpoints**: Optional epoch-wise saves via `checkpoint_interval`
- **Tau Sampling**: Broadens supervision across diffusion space

### Stage 3: Consistency Distillation
- **Teacher Caching** (OPTIMIZATION #1): Compute once per batch → ~2x speedup
- **AMP for Teacher** (OPTIMIZATION #6): ~20% teacher speedup → ~8% overall
- **Async GPU Transfers**: Non-blocking .to() calls
- **Persistent Workers**: Avoid DataLoader respawning
- **Compiled Distillation**: Optional torch.compile (~1.3-1.5x)
- **Micro-batching**: Process large batches in chunks

### Stage 4: Steady Prior
- Simple epoch-based training, minimal optimizations needed

---

## WandB Logging Architecture

**One Run Per Pipeline** (not one run per stage):
```python
wandb_ctx = create_wandb_context(cfg, run_id="run_timestamp")
# All stages log to same run with stage prefix:
wandb_ctx.log_training_metric("operator", "loss", 0.001, step=1)
wandb_ctx.log_training_metric("diff_residual", "loss", 0.0005, step=26)
```

**Metrics Logged** (all stages):
- `{stage}/loss`: Mean epoch loss
- `{stage}/lr`: Learning rate
- `{stage}/grad_norm`: Mean gradient norm
- `{stage}/epochs_since_improve`: Patience counter
- `{stage}/epoch_time_sec`: Epoch duration
- Individual loss components (operator only)

**Integration with Orchestrator** (`run_fast_to_sota.py`):
- Training creates WandB run
- Context saved to `WANDB_CONTEXT_FILE` (JSON)
- Evaluation subprocess loads context, logs to same run
- No additional runs created

---

## Files & Locations

### Source Code
```
scripts/train.py                   # Main entry point (1812 lines)
  ├── train_operator()            # Lines 400-693
  ├── train_diffusion()           # Lines 695-880
  ├── train_consistency()         # Lines 942-1204
  ├── train_steady_prior()        # Lines 1206-1309
  └── train_all_stages()          # Lines 1516-1733

scripts/run_fast_to_sota.py        # Orchestrator (1225 lines)

src/ups/training/
  ├── losses.py                   # Loss functions
  ├── loop_train.py               # Legacy CurriculumConfig
  └── consistency_distill.py      # Consistency utilities

src/ups/models/
  ├── latent_operator.py          # Stage 1 model
  ├── diffusion_residual.py       # Stage 2 model
  └── steady_prior.py             # Stage 4 model
```

### Checkpoints
```
checkpoints/
  ├── operator.pt                 # Stage 1 best
  ├── operator_ema.pt             # Stage 1 EMA
  ├── diffusion_residual.pt       # Stages 2,3
  ├── diffusion_residual_ema.pt   # Stages 2,3 EMA
  ├── diffusion_residual_epoch_*.pt  # Stage 2 periodic (optional)
  └── steady_prior.pt             # Stage 4

reports/
  └── training_log.jsonl          # Per-epoch metrics (JSONL)
```

---

## Usage Examples

### Run All Stages Standalone
```bash
python scripts/train.py --config config.yaml --stage all
```

### Run Single Stage
```bash
python scripts/train.py --config config.yaml --stage operator
python scripts/train.py --config config.yaml --stage diff_residual
python scripts/train.py --config config.yaml --stage consistency_distill
python scripts/train.py --config config.yaml --stage steady_prior
```

### Via Fast-to-SOTA Orchestrator
```bash
python scripts/run_fast_to_sota.py \
  --train-config config.yaml \
  --small-eval-config small_eval.yaml \
  --full-eval-config full_eval.yaml
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| OOM during operator | Large batch/model | Reduce batch_size or latent.dim |
| OOM during consistency | Too many taus | Reduce distill_num_taus or distill_micro_batch |
| Grad norm explodes at epoch 15 | Inverse loss curriculum kicking in | Increase inverse_loss_warmup_epochs |
| Loss doesn't decrease | Learning rate too low | Increase lr or check data loading |
| Dimension mismatch error | Config inconsistency | Verify latent.dim == operator.pdet.input_dim |

---

## References

See main document (`MULTI_STAGE_TRAINING.md`) for:
- Complete loss function mathematics
- Full code examples for each stage
- Configuration options (all ~80+ parameters)
- Detailed optimization explanations
- Gradient norm analysis & debugging guide
- Integration with orchestrator details
