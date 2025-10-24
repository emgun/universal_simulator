# Latent Operator Quick Reference

## Core Answer Sheet

### 1. What is the Latent Operator?
A learned neural operator that evolves PDE solutions in a compressed latent space:
- **Input**: LatentState(z ∈ R^(batch×tokens×dim), t, conditioning)
- **Process**: Single time step evolution via residual connection
- **Output**: LatentState(z_new = z + residual(z, dt), t_new = t+dt, cond)

### 2. Latent Operator Implementation
**File**: `/Users/emerygunselman/Code/universal_simulator/src/ups/models/latent_operator.py`

**Key Class**: `LatentOperator(nn.Module)`

**Forward Pass**:
```python
def forward(state: LatentState, dt: torch.Tensor) -> LatentState:
    residual = self.step(state, dt)     # Compute drift
    new_z = state.z + residual          # Residual connection
    new_t = state.t + dt if state.t else dt
    return LatentState(z=new_z, t=new_t, cond=state.cond)
```

### 3. Architecture: PDE-Transformer Backbone
**File**: `/Users/emerygunselman/Code/universal_simulator/src/ups/core/blocks_pdet.py`

**U-Shaped Structure**:
```
Input (batch, tokens, latent_dim)
    ↓
input_proj: latent_dim → hidden_dim
    ↓
DOWN: downsample tokens by 2× (2-3 layers)
    ↓ (save skip connections)
    ↓
BOTTLENECK: intensive processing at coarsest scale
    ↓
UP: upsample tokens by 2× (2-3 layers)
    ↓ (add skip connections)
    ↓
output_proj: hidden_dim → latent_dim
    ↓
Output (batch, tokens, latent_dim)
```

### 4. Core Transformer Components
**Channel-Separated Self-Attention**:
- Split latent_dim into groups (e.g., dim=96, group_size=12 → 8 groups)
- Apply multi-head attention independently per group
- Efficiency: O(tokens² × group_size) vs O(tokens² × dim)

**TransformerLayer**:
- Attention: Q,K,V projections + RMSNorm + scaled-dot-product
- MLP: 2-layer feedforward with activation
- Both with residual connections

### 5. Token Representation
**Configuration**:
```yaml
latent:
  dim: 16        # Features per token
  tokens: 32     # Number of spatial tokens
```

**Total Shape**: (batch=12, tokens=32, dim=16)

**Origin**: Grid encoding with patch embedding + optional pooling

### 6. Temporal Evolution (Δt Handling)
**Time Embedding Pipeline**:
```
Δt (scalar)
    ↓
TimeEmbedding: Linear(1 → 64) → SiLU → Linear(64 → 64)
    ↓
dt_embed (batch, 64)
    ↓
time_to_latent: Linear(64 → 16)
    ↓
time_feat (batch, 1, 16) - broadcast to all tokens
    ↓
z_conditioned = z + time_feat
```

**Multi-Step Rollout** (autoregressive):
```python
for t in range(num_steps):
    state = operator(state, dt_tensor)  # Iteratively apply
```

### 7. Efficiency: Shifted Windows (Currently Unused)
**File**: `/Users/emerygunselman/Code/universal_simulator/src/ups/core/shifted_window.py`

**Purpose**: Local window-based attention (Swin Transformer pattern)

**Current Status**: 
- ✓ Implemented and available
- ✗ Not used in PDETransformerBlock
- Alternative: Token pooling + channel-separated attention

**Functions**:
- `partition_windows()`: Split (B,H,W,C) → (B*windows, window_area, C)
- `merge_windows()`: Reverse operation

### 8. Configuration Consistency
**Critical constraints** - all must match:
```yaml
latent.dim = operator.pdet.input_dim = diffusion.latent_dim = ttc.decoder.latent_dim

operator.pdet.hidden_dim % operator.pdet.group_size = 0
operator.pdet.group_size % operator.pdet.num_heads = 0
```

**Golden Config Example**:
```yaml
latent.dim: 16
operator.pdet.input_dim: 16           # MATCH
operator.pdet.hidden_dim: 96          # 6× latent_dim
operator.pdet.group_size: 12          # 96/12=8 groups
operator.pdet.num_heads: 6            # 12/6=2 head_dim
```

---

## File Map

| File | Purpose | Lines |
|------|---------|-------|
| `src/ups/models/latent_operator.py` | Main operator, time embedding | 92 |
| `src/ups/core/blocks_pdet.py` | PDE-Transformer U-net | 235 |
| `src/ups/core/latent_state.py` | LatentState data class | 71 |
| `src/ups/core/conditioning.py` | AdaLNConditioner | 84 |
| `src/ups/core/shifted_window.py` | Window utilities | 172 |
| `src/ups/io/enc_grid.py` | Grid encoder | 232 |
| `src/ups/inference/rollout_transient.py` | Multi-step rollout | 75 |

---

## Performance Numbers (Golden Config)

- **Latent Dim**: 16
- **Tokens**: 32
- **Hidden Dim**: 96
- **Operator Final Loss**: ~0.00023
- **Baseline NRMSE**: ~0.78
- **TTC NRMSE**: ~0.09 (88% improvement)
- **Training Time**: ~14.5 min (A100)

---

## Key Design Decisions

1. **Residual Connection**: Enables learning of small perturbations (typical PDE residuals)
2. **U-Shape with Token Pooling**: Long-range dependencies without full attention cost
3. **Channel-Separated Attention**: Efficient per-group computation, SIMD-friendly
4. **Time as Continuous Embedding**: Learns sensitivity to different Δt values
5. **RMSNorm**: Preserves mean signal (important for physics)
6. **AdaLN Conditioning**: Physics-aware modulation via scale/shift/gate

---

## Data Flow Diagram

```
SINGLE STEP:
─────────────
(z, t, cond) + dt
    ↓
[TimeEmbedding] dt → continuous embedding
    ↓
[Broadcast] Inject time across tokens
    ↓
[Conditioning] Optional: apply scale/shift/gate
    ↓
[PDETransformerBlock]
  ├─ input_proj (16→96)
  ├─ Down (tokens→tokens/2→tokens/4)
  ├─ Bottleneck (tokens/4)
  ├─ Up (tokens/4→tokens/2→tokens)
  └─ output_proj (96→16)
    ↓
[Output] residual
    ↓
z_new = z + residual
(z_new, t+dt, cond)


MULTI-STEP (e.g., 10 steps):
────────────────────────────
state₀ → [Op] → state₁ → [Op] → state₂ → ... → state₁₀
```

