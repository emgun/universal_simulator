# Decoder & Regularization - Exact Code Locations

## Decoder Implementation

### Decoder Output Heads
**File**: `/Users/emerygunselman/Code/universal_simulator/src/ups/io/decoder_anypoint.py`

| Component | Lines | Code | Notes |
|-----------|-------|------|-------|
| Output head definition | 81-87 | `self.heads = nn.ModuleDict()` for each field | 2-layer MLP, GELU activation, NO output constraint |
| Decoder output return | 131-134 | `outputs[name] = head(queries)` | Direct assignment, no clamping |
| Query embedding | 122-123 | Fourier encoding + linear projection | Uses configurable frequencies |
| Cross-attention loop | 125-129 | Attention + residual + FFN + residual | 3 configurable layers default |

### Where to Add Log-Clamping
```
File: src/ups/io/decoder_anypoint.py
Lines: 131-134 (in forward() method)

Pattern to copy from:
src/ups/models/physics_guards.py:31-32
```

---

## Loss Functions

### Loss Function Implementations
**File**: `/Users/emerygunselman/Code/universal_simulator/src/ups/training/losses.py`

| Loss Name | Lines | Formula | Status |
|-----------|-------|---------|--------|
| `mse()` | 21-22 | F.mse_loss | Base utility |
| `inverse_encoding_loss()` | 25-60 | MSE(decoder(latent), input_fields) | UPT-specific |
| `inverse_decoding_loss()` | 63-99 | MSE(encoder(decoder(latent)), latent) | UPT-specific |
| `one_step_loss()` | 102-103 | weight * MSE | Used always |
| `rollout_loss()` | 106-109 | weight * MSE over sequence | Optional |
| `spectral_loss()` | 112-116 | weight * MSE(FFT magnitudes) | Optional, active=0.05 |
| `consistency_loss()` | 119-122 | MSE of spatial means | Defined but unused |
| `edge_total_variation()` | 125-131 | Mean absolute diffs on edges | Graph regularization |

### Loss Bundle Computation
**File**: `/Users/emerygunselman/Code/universal_simulator/src/ups/training/losses.py`

| Component | Lines | Purpose |
|-----------|-------|---------|
| `compute_operator_loss_bundle()` | 168-251 | Main loss aggregator (handles all 5 loss types + curriculum) |
| Inverse loss curriculum | 134-165 | `compute_inverse_loss_curriculum_weight()` |
| Phase 1 (epochs 0-15) | 156-158 | weight = 0 |
| Phase 2 (epochs 15-30) | 159-162 | Linear ramp |
| Phase 3 (epochs 30+) | 163-165 | Capped at max_weight (0.05) |

### Where to Add Latent Norm Penalty
```
File: scripts/train.py
Current loss weight construction: lines 620-629
Add loss computation: after line 676 (after loss_bundle creation)

Pattern:
loss_weights["lambda_latent_norm"] = float(train_cfg.get("lambda_latent_norm", 0.0))
if loss_weights.get("lambda_latent_norm", 0.0) > 0:
    latent_norm_loss = torch.norm(state.z) * loss_weights["lambda_latent_norm"]
    loss_bundle.components["L_latent_norm"] = latent_norm_loss
```

---

## Regularization Terms

### Weight Decay (L2)
**File**: `/Users/emerygunselman/Code/universal_simulator/configs/train_burgers_golden.yaml`

| Stage | Lines | Value | Context |
|-------|-------|-------|---------|
| Operator | 111 | 0.03 | Strongest regularization |
| Diffusion | 122 | 0.015 | Medium regularization |
| Distillation | 139 | 0.015 | Medium regularization |

**Implementation**: Passed to AdamW optimizer in training loop

### Gradient Clipping
**File**: `/Users/emerygunselman/Code/universal_simulator/scripts/train.py`

| Location | Lines | Code | Value |
|----------|-------|------|-------|
| Implementation | 699-705 | `torch.nn.utils.clip_grad_norm_()` | Default: 1.0 |
| Config source | Config `grad_clip` | Passed from YAML | Per-stage configurable |

**Reference**: `src/ups/training/loop_train.py:70-71` (alternate implementation)

### EMA (Exponential Moving Average)
**File**: `/Users/emerygunselman/Code/universal_simulator/src/ups/training/loop_train.py`

| Component | Lines | Details |
|-----------|-------|---------|
| EMA initialization | 43-47 | Creates EMA model copy |
| EMA update | 49-54 | `p_ema.mul_(decay).add_(p, alpha=1-decay)` |
| Config parameter | line 86 (config) | `ema_decay: 0.999` |

### Spectral Loss
**File**: `/Users/emerygunselman/Code/universal_simulator/src/ups/training/losses.py:112-116`

```python
def spectral_loss(pred: Tensor, target: Tensor, weight: float = 1.0) -> Tensor:
    pred_fft = torch.fft.rfft(pred, dim=-1)
    target_fft = torch.fft.rfft(target, dim=-1)
    loss = mse(pred_fft.abs(), target_fft.abs())
    return weight * loss
```

**Weight**: `lambda_spectral: 0.05` (from `train_burgers_golden.yaml:92`)

---

## Transformer Block Regularization

### PDETransformerBlock (U-Shaped - Current Default)
**File**: `/Users/emerygunselman/Code/universal_simulator/src/ups/core/blocks_pdet.py`

| Component | Lines | Status | Issue |
|-----------|-------|--------|-------|
| PDETransformerBlock | 168-234 | Complete | No drop-path |
| TransformerLayer | 124-135 | Complete | No drop-path in residuals |
| Down/up paths | 214-230 | Complete | Uses skip connections |
| Bottleneck | 222-223 | Complete | 3 layers (config depths[-1]) |

**Where Drop-Path Missing**: Lines 133-134
```python
# Current:
x = x + self.attn(self.attn_norm(x))
x = x + self.ff(x)

# Should be (reference pattern below):
x = x + self.drop_path_attn(self.attn(self.attn_norm(x)))
x = x + self.drop_path_ff(self.ff(x))
```

### PureTransformer (Stacked - UPT New)
**File**: `/Users/emerygunselman/Code/universal_simulator/src/ups/models/pure_transformer.py`

| Component | Lines | Status | Notes |
|-----------|-------|--------|-------|
| TransformerBlock | 64-142 | Complete | Has drop-path implemented |
| Drop-path initialization | 103-104 | Working | Linear scaling across depth |
| Drop-path in attention | 130-131 | Working | `drop_path_attn` applied |
| Drop-path in FFN | 141 | Working | `drop_path_ff` applied |

**Configuration**: `configs/train_burgers_upt_256tokens_pure.yaml:53`
```yaml
pdet:
  drop_path: 0.1    # Linear scale from 0 to 0.1 across 8 layers
```

### Drop-Path Implementation
**File**: `/Users/emerygunselman/Code/universal_simulator/src/ups/core/drop_path.py`

| Component | Lines | Details |
|-----------|-------|---------|
| DropPath class | 12-64 | Stochastic depth implementation |
| Initialization | 29-32 | Takes drop_prob and scale_by_keep |
| Forward pass | 34-61 | Binary masking, scaling by 1/keep_prob |
| Docstring | 13-27 | Reference: arXiv:1603.09382 |

**Usage Pattern**:
```python
self.drop_path = DropPath(drop_prob)
x = x + self.drop_path(residual)  # Randomly drops 10% of time
```

---

## Training Loop Integration

### Operator Training (Main Stage)
**File**: `/Users/emerygunselman/Code/universal_simulator/scripts/train.py`

| Step | Lines | Action |
|------|-------|--------|
| Loss weight construction | 620-629 | Build loss_weights dict from config |
| Forward prediction | 617 | operator(state, dt) |
| Loss computation | 652-676 | compute_operator_loss_bundle() |
| Backward pass | 695-697 | Loss backprop (with/without AMP) |
| Gradient clipping | 699-705 | clip_grad_norm_ |
| Optimizer step | 706-709 | optimizer.step() |

### Loss Bundle Assembly
**Lines**: 652-676 in `scripts/train.py`

```python
loss_bundle = compute_operator_loss_bundle(
    # Inverse encoding (optional, curriculum)
    input_fields=...,
    encoded_latent=...,
    decoder=...,
    input_positions=...,
    # Inverse decoding (optional, curriculum)
    encoder=...,
    query_positions=...,
    coords=...,
    meta=...,
    # Forward (always)
    pred_next=next_state.z,
    target_next=target,
    # Rollout (optional)
    pred_rollout=...,
    target_rollout=...,
    # Spectral (optional)
    spectral_pred=...,
    spectral_target=...,
    weights=loss_weights,
    current_epoch=epoch,  # For curriculum learning
)
```

---

## Configuration Files

### Golden Config (Production)
**File**: `/Users/emerygunselman/Code/universal_simulator/configs/train_burgers_golden.yaml`

**Key Regularization Parameters**:
- Line 85: `grad_clip: 1.0`
- Line 86: `ema_decay: 0.999`
- Line 92: `lambda_spectral: 0.05`
- Line 111: `weight_decay: 0.03` (operator)
- Line 122: `weight_decay: 0.015` (diffusion)
- Line 139: `weight_decay: 0.015` (distillation)

### UPT Pure Transformer Config
**File**: `/Users/emerygunselman/Code/universal_simulator/configs/train_burgers_upt_256tokens_pure.yaml`

**Key Regularization Parameters**:
- Line 53: `drop_path: 0.1` (linear scale across depth)
- Line 54: `dropout: 0.0`

---

## Summary Table: What's Implemented vs. Missing

| Feature | Implemented? | Files | Status |
|---------|--------------|-------|--------|
| Weight Decay | YES | Config + adamw factory | Production-ready |
| Gradient Clipping | YES | scripts/train.py:699-705 | Production-ready |
| EMA | YES | loop_train.py:49-54 | Optional feature |
| Spectral Loss | YES | losses.py:112-116 | Active in configs |
| Drop-Path (PureTransformer) | YES | pure_transformer.py:82-142 | UPT only |
| Drop-Path (PDETBlock) | NO | blocks_pdet.py:124-135 | MISSING |
| Latent Norm Penalty | NO | scripts/train.py:650-680 | NOT IMPLEMENTED |
| Output Clamping | NO | decoder_anypoint.py:131-134 | NOT IMPLEMENTED |
| Inverse Loss Curriculum | YES | losses.py:134-165 | Production-ready |

---

## Code Location Index

### Decoder & Output Constraints
- **Decoder heads**: `src/ups/io/decoder_anypoint.py:81-87`
- **Decoder output**: `src/ups/io/decoder_anypoint.py:131-134` (← Add clamping here)
- **Physics guard pattern**: `src/ups/models/physics_guards.py:31-32` (reference)

### Loss Functions
- **Inverse encoding**: `src/ups/training/losses.py:25-60`
- **Inverse decoding**: `src/ups/training/losses.py:63-99`
- **Spectral loss**: `src/ups/training/losses.py:112-116`
- **Curriculum schedule**: `src/ups/training/losses.py:134-165`
- **Loss bundle**: `src/ups/training/losses.py:168-251`

### Regularization
- **Weight decay**: `configs/train_burgers_golden.yaml:111,122,139`
- **Gradient clipping**: `scripts/train.py:699-705`
- **EMA**: `src/ups/training/loop_train.py:49-54`
- **Latent norm penalty**: `scripts/train.py:620-680` (← Add here)

### Transformer Blocks
- **PDETBlock (no drop-path)**: `src/ups/core/blocks_pdet.py:124-135`
- **PureTransformer (with drop-path)**: `src/ups/models/pure_transformer.py:64-142`
- **DropPath module**: `src/ups/core/drop_path.py:12-64`

### Configurations
- **Production config**: `configs/train_burgers_golden.yaml`
- **UPT config**: `configs/train_burgers_upt_256tokens_pure.yaml`
