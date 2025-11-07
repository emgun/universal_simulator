# Decoder Implementation & Regularization Research Report

## Executive Summary

This report documents the current decoder architecture, regularization mechanisms, and loss functions in the UPS training pipeline. The decoder is implemented as a Perceiver-style cross-attention decoder without output clamping or constraints. Multiple regularization strategies are present through weight decay, drop-path, and loss weights.

---

## 1. DECODER ARCHITECTURE

### File: `/Users/emerygunselman/Code/universal_simulator/src/ups/io/decoder_anypoint.py`

#### 1.1 Decoder Output Heads (Lines 81-87)

```python
self.heads = nn.ModuleDict()
for name, out_ch in cfg.output_channels.items():
    self.heads[name] = nn.Sequential(
        nn.Linear(cfg.hidden_dim, cfg.mlp_hidden_dim),
        nn.GELU(),
        nn.Linear(cfg.mlp_hidden_dim, out_ch),
    )
```

**Key Findings:**
- **Output Layer Implementation**: Each field has a 2-layer MLP head
  - First layer: `Linear(hidden_dim → mlp_hidden_dim)` with GELU activation
  - Second layer: `Linear(mlp_hidden_dim → out_ch)` **with NO activation or clamping**
- **Activation Functions**: Only GELU is used in hidden layers
- **Output Constraints**: **NONE** - outputs are unconstrained and unbounded

#### 1.2 Decoder Architecture Components

**Query Processing (Lines 122-123):**
```python
enriched_points = _fourier_encode(points, self.cfg.frequencies)
queries = self.query_embed(enriched_points)
```
- Fourier encoding of coordinates with configurable frequencies
- Single linear projection to hidden_dim

**Cross-Attention Layers (Lines 125-129):**
```python
for attn, ln_q, ff, ln_ff in self.layers:
    attn_out, _ = attn(queries, latents, latents)
    queries = ln_q(queries + attn_out)
    ff_out = ff(queries)
    queries = ln_ff(queries + ff_out)
```
- Pre-norm cross-attention (queries vs latent keys/values)
- Residual connections around both attention and FFN
- LayerNorm (not RMSNorm)

---

## 2. OUTPUT CONSTRAINTS & CLAMPING

### Current State: NO OUTPUT CONSTRAINTS

**Finding**: The decoder has **zero output constraints**:
- No clamping (`.clamp()`)
- No log-space constraints (`.log()` / `.exp()`)
- No softplus/sigmoid gating
- No physics-guided bounds

### Where Log-Clamping Could Be Added

**Recommended Location**: `/Users/emerygunselman/Code/universal_simulator/src/ups/io/decoder_anypoint.py:131-134`

```python
# CURRENT (unconstrained):
outputs: Dict[str, torch.Tensor] = {}
for name, head in self.heads.items():
    outputs[name] = head(queries)
return outputs
```

**Could be modified to:**
```python
# PROPOSED (with field-specific constraints):
outputs: Dict[str, torch.Tensor] = {}
for name, head in self.heads.items():
    output = head(queries)
    
    # Log-clamping for density/positive fields:
    if name in ['rho', 'density', 'mass']:
        output = torch.log(torch.clamp(output, min=1e-6)) 
        # Then exp back if needed for physical space
    
    # Clamp energy fields:
    elif name in ['e', 'energy']:
        output = torch.clamp(output, min=1e-8)
    
    outputs[name] = output
return outputs
```

### Existing Physics Guards (Different Module)

**File**: `/Users/emerygunselman/Code/universal_simulator/src/ups/models/physics_guards.py:31-32`

Only one positivity guard exists (not in decoder):
```python
def positify(values: torch.Tensor, min_value: float = 1e-6) -> torch.Tensor:
    return torch.exp(torch.clamp(torch.log(values.clamp_min(min_value)), min=-20.0))
```

This shows the **log-clamp-exp pattern** is already established in codebase but NOT used in decoder.

---

## 3. LOSS FUNCTIONS IN TRAINING PIPELINE

### File: `/Users/emerygunselman/Code/universal_simulator/src/ups/training/losses.py`

#### 3.1 Core Loss Functions

| Function | Lines | Purpose | Details |
|----------|-------|---------|---------|
| `mse()` | 21-22 | Base MSE loss | `F.mse_loss(pred, target, reduction)` |
| `inverse_encoding_loss()` | 25-60 | UPT: Decoder reconstruction | MSE(decoder(latent), input_fields) |
| `inverse_decoding_loss()` | 63-99 | UPT: Encoder invertibility | MSE(encoder(decoder(latent)), latent) |
| `one_step_loss()` | 102-103 | Forward prediction | Weighted MSE for single step |
| `rollout_loss()` | 106-109 | Multi-step prediction | Weighted MSE for trajectory |
| `spectral_loss()` | 112-116 | Frequency domain matching | MSE of FFT magnitudes |
| `consistency_loss()` | 119-122 | Mean field constraint | MSE of spatial means |
| `edge_total_variation()` | 125-131 | Graph smoothness | Mean absolute differences on edges |

#### 3.2 Loss Bundle Computation

**Main Function**: `compute_operator_loss_bundle()` (Lines 168-251)

**Loss Components Computed:**
```python
comp["L_inv_enc"] = inverse_encoding_loss(...)      # UPT inverse encoding
comp["L_inv_dec"] = inverse_decoding_loss(...)      # UPT inverse decoding
comp["L_forward"] = one_step_loss(...)              # Always computed
comp["L_rollout"] = rollout_loss(...)               # Optional (lambda_rollout)
comp["L_spec"] = spectral_loss(...)                 # Optional (lambda_spectral)
```

**Curriculum Learning for Inverse Losses** (Lines 134-165):
```python
def compute_inverse_loss_curriculum_weight(epoch, base_weight, 
                                           warmup_epochs=15, 
                                           max_weight=0.05):
    # Phase 1 (0-15 epochs): weight = 0 (forward only)
    # Phase 2 (15-30 epochs): linear ramp from 0 to base_weight
    # Phase 3 (30+ epochs): weight = min(base_weight, max_weight)
```

---

## 4. REGULARIZATION TERMS

### 4.1 Weight Decay (AdamW L2 Regularization)

**File**: `/Users/emerygunselman/Code/universal_simulator/configs/train_burgers_golden.yaml`

**Current Configuration:**
```yaml
stages:
  operator:
    optimizer:
      name: adamw
      lr: 1.0e-3
      weight_decay: 0.03        # Line 111 - Operator stage

  diff_residual:
    optimizer:
      name: adamw
      lr: 5.0e-5
      weight_decay: 0.015       # Line 122 - Diffusion stage
      
  consistency_distill:
    optimizer:
      name: adamw
      lr: 3.0e-5
      weight_decay: 0.015       # Line 139 - Distillation stage
```

**Implementation** in training loop (`scripts/train.py:620-629`):
- Weight decay is passed to optimizer factory via config
- Decoupled weight decay (AdamW default)
- No per-layer weight decay scaling

### 4.2 Latent Norm Penalty

**Status**: **NO LATENT NORM PENALTY CURRENTLY IMPLEMENTED**

**Where It Could Be Added**: `scripts/train.py:650-680` (loss computation block)

Current weights dict (line 620-629):
```python
loss_weights = {
    "lambda_forward": 1.0,
    "lambda_inv_enc": float(train_cfg.get("lambda_inv_enc", 0.0)),
    "lambda_inv_dec": float(train_cfg.get("lambda_inv_dec", 0.0)),
    "lambda_spectral": lam_spec,
    "lambda_rollout": lam_rollout,
    "inverse_loss_warmup_epochs": 15,
    "inverse_loss_max_weight": 0.05,
}
```

**Could be extended to:**
```python
# Add latent norm penalty weight
"lambda_latent_norm": float(train_cfg.get("lambda_latent_norm", 0.0)),
```

Then in loss bundle (after line 676):
```python
# Add latent norm penalty
if loss_weights.get("lambda_latent_norm", 0.0) > 0:
    latent_norm_loss = torch.norm(state.z) * loss_weights["lambda_latent_norm"]
    loss_bundle.components["L_latent_norm"] = latent_norm_loss
    loss = loss_bundle.total + latent_norm_loss
```

### 4.3 Spectral Loss

**Lines**: `losses.py:112-116`

```python
def spectral_loss(pred: Tensor, target: Tensor, weight: float = 1.0) -> Tensor:
    pred_fft = torch.fft.rfft(pred, dim=-1)
    target_fft = torch.fft.rfft(target, dim=-1)
    loss = mse(pred_fft.abs(), target_fft.abs())
    return weight * loss
```

**Current Weight**: Configured per-stage (e.g., `lambda_spectral: 0.05` in operator)

### 4.4 Other Regularization

**Gradient Clipping** (`loop_train.py:70-71`, `scripts/train.py:699-705`):
```python
if self.curriculum.grad_clip is not None:
    torch.nn.utils.clip_grad_norm_(self.operator.parameters(), self.curriculum.grad_clip)
```
- Global gradient norm clipping
- Default: `grad_clip: 1.0` (in configs)

**EMA (Exponential Moving Average)** (`loop_train.py:49-54`):
```python
for p_ema, p in zip(self.ema_model.parameters(), self.operator.parameters()):
    p_ema.mul_(self.ema_decay).add_(p.data, alpha=1.0 - self.ema_decay)
```
- Optional EMA model tracking
- Default decay: `ema_decay: 0.999`

---

## 5. TRANSFORMER BLOCK REGULARIZATION

### File: `/Users/emerygunselman/Code/universal_simulator/src/ups/core/blocks_pdet.py`

#### 5.1 U-Shaped PDETransformer (Current Default)

**Status**: **NO DROP-PATH REGULARIZATION IN PDETransformerBlock**

**Architecture** (Lines 168-234):
```python
class PDETransformerBlock(nn.Module):
    # Downsampling path (with skip connections)
    for layer_pack in self.down_layers:
        blocks, proj = layer_pack
        for layer in blocks:
            x = layer(x)  # TransformerLayer - has NO DropPath
        skips.append(x)
        x = _downsample_tokens(x)
        x = proj(x)
    
    # Bottleneck
    for layer in self.bottleneck:
        x = layer(x)  # TransformerLayer - has NO DropPath
    
    # Upsampling path
    for layer_pack, skip in zip(self.up_layers, reversed(skips)):
        proj, blocks = layer_pack
        x = _upsample_tokens(x, skip.shape[1])
        x = x + proj(skip)  # Skip connections
        for layer in blocks:
            x = layer(x)  # TransformerLayer - has NO DropPath
```

**TransformerLayer Structure** (Lines 124-135):
```python
class TransformerLayer(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))          # No DropPath
        x = x + self.ff(x)                             # No DropPath
        return x
```

**Where Drop-Path Could Be Added:**
- Lines 133 & 134 in TransformerLayer:
```python
# Current:
x = x + self.attn(self.attn_norm(x))
x = x + self.ff(x)

# Could add:
x = x + self.drop_path_attn(self.attn(self.attn_norm(x)))
x = x + self.drop_path_ff(self.ff(x))
```

### File: `/Users/emerygunselman/Code/universal_simulator/src/ups/models/pure_transformer.py`

#### 5.2 Pure Stacked Transformer (UPT - New)

**Status**: **DROP-PATH IS FULLY IMPLEMENTED**

**Drop-Path in PureTransformer** (Lines 64-142):
```python
class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm, attention, FFN, and drop-path."""
    
    def __init__(self, ..., drop_path: float, ...):
        self.drop_path_attn = DropPath(drop_path)
        self.drop_path_ff = DropPath(drop_path)
    
    def forward(self, x):
        x = x + self.drop_path_attn(self.attn(self.norm1(x)))
        x = x + self.drop_path_ff(self.ff(self.norm2(x)))
        return x
```

**Configuration Example** (`configs/train_burgers_upt_256tokens_pure.yaml`):
```yaml
operator:
  architecture_type: pdet_stack
  pdet:
    input_dim: 384
    hidden_dim: 384
    depth: 8
    drop_path: 0.1            # Line 53 - Stochastic depth enabled
    dropout: 0.0
```

**Recommended Drop-Path Values** (from `pure_transformer.py:41-49`):
- Small (4 layers): `drop_path: 0.0`
- Medium (8 layers): `drop_path: 0.1`
- Large (12 layers): `drop_path: 0.15`

### File: `/Users/emerygunselman/Code/universal_simulator/src/ups/core/drop_path.py`

#### 5.3 Drop-Path Implementation

**Drop-Path Module** (Lines 12-64):

```python
class DropPath(nn.Module):
    """Drop paths (stochastic depth) per sample.
    
    When drop_prob > 0, randomly drops entire paths (residual branches)
    during training. At test time, scales by (1 - drop_prob) for expected value.
    """
    
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Binarize: 0 or 1
        
        if self.scale_by_keep:
            output = x.div(keep_prob) * random_tensor  # Scale by 1/keep_prob
        else:
            output = x * random_tensor
        
        return output
```

**Reference**: Deep Networks with Stochastic Depth (arXiv:1603.09382)

---

## 6. REGULARIZATION SUMMARY TABLE

| Regularization Method | Implementation | Current Status | Where |
|----------------------|-----------------|-----------------|-------|
| **Weight Decay (L2)** | AdamW decoupled | ✓ Active | Config (adamw.weight_decay) |
| **Drop-Path** | Stochastic depth | ✓ In PureTransformer only | `pure_transformer.py:82-142` |
| **Drop-Path** | Stochastic depth | ✗ Missing in PDETBlock | `blocks_pdet.py:124-135` |
| **Gradient Clipping** | Global norm | ✓ Active | `scripts/train.py:699-705` |
| **EMA** | Exponential avg | ✓ Optional | `loop_train.py:49-54` |
| **Latent Norm Penalty** | L2 latent norm | ✗ Not implemented | Could be in `train.py:650-680` |
| **Output Clamping** | Min/max bounds | ✗ Not implemented | Decoder: `decoder_anypoint.py:131-134` |
| **Log-Clamping** | Log-domain bounds | ✗ Not implemented | Physics guard example exists (physics_guards.py:31-32) |
| **Spectral Loss** | FFT matching | ✓ Active | `losses.py:112-116` |
| **Inverse Loss Curriculum** | Warm-up schedule | ✓ Active | `losses.py:134-165` |

---

## 7. SPECIFIC FILE:LINE REFERENCES FOR IMPLEMENTATION

### For Log-Clamping
- **Add location**: `/Users/emerygunselman/Code/universal_simulator/src/ups/io/decoder_anypoint.py:131-134`
- **Reference pattern**: `/Users/emerygunselman/Code/universal_simulator/src/ups/models/physics_guards.py:31-32`

### For Latent Norm Penalty  
- **Add location**: `/Users/emerygunselman/Code/universal_simulator/scripts/train.py:650-680` (loss computation)
- **Add to config**: `configs/train_burgers_golden.yaml` (new weight parameter)

### For Drop-Path in U-Shaped Transformer
- **Current missing**: `/Users/emerygunselman/Code/universal_simulator/src/ups/core/blocks_pdet.py:124-135` (TransformerLayer)
- **Reference implementation**: `/Users/emerygunselman/Code/universal_simulator/src/ups/core/drop_path.py:12-64`
- **Working example**: `/Users/emerygunselman/Code/universal_simulator/src/ups/models/pure_transformer.py:82-142`

### For Current Regularization Parameters
- **Weight decay config**: `/Users/emerygunselman/Code/universal_simulator/configs/train_burgers_golden.yaml:111, 122, 139`
- **Gradient clipping**: `/Users/emerygunselman/Code/universal_simulator/scripts/train.py:699-705`
- **Spectral loss weight**: `/Users/emerygunselman/Code/universal_simulator/scripts/train.py:624`
- **Inverse loss curriculum**: `/Users/emerygunselman/Code/universal_simulator/src/ups/training/losses.py:134-165`

---

## 8. KEY INSIGHTS

1. **Decoder is Unconstrained**: The decoder outputs are completely unconstrained and unbounded. This could be problematic for physical fields that must be positive (density, energy).

2. **Log-Clamp Pattern Exists**: The codebase already has the `positify()` function that implements log-clamping, but it's not used in the decoder.

3. **Two Architectures, Different Regularization**:
   - **PDETransformerBlock** (U-shaped): No drop-path despite ~5 layers
   - **PureTransformer** (UPT): Full drop-path support with configurable linearly-scaled rates

4. **Weight Decay is Strong**: Currently 0.03 for operator, 0.015 for diffusion/distillation - higher than typical (usually 0.01-0.02)

5. **No Latent Norm Penalty**: Unlike modern diffusion models, no explicit latent norm regularization

6. **Inverse Loss Curriculum**: Smart warm-up strategy prevents early instability but could benefit from latent norm constraint

---

## 9. RECOMMENDATIONS FOR REGULARIZATION

**Priority 1 - Low Hanging Fruit:**
- Add drop-path to PDETransformerBlock (Lines 133-134)
- Add output clamping to decoder heads for positive fields (Line 133-134)

**Priority 2 - Moderate Complexity:**
- Add configurable latent norm penalty in loss computation
- Make weight decay per-stage configurable (currently fixed)

**Priority 3 - Advanced:**
- Implement field-specific constraints (density bounds, energy bounds)
- Add spectral regularization for high-frequency stability
- Implement adaptive weight scheduling for regularization terms
