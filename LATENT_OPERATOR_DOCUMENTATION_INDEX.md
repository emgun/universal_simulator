# Latent Operator Architecture Documentation Index

## Overview

This documentation comprehensively describes the **Latent Operator** architecture - the deterministic neural operator core of the Universal Physics Stack (UPS) that evolves PDE solutions forward in time in a learned latent space.

**Key Principle**: Physics evolves as `z_new = z_old + residual_step(z_old, Δt)` using a U-shaped PDE-Transformer backbone.

---

## Documents

### 1. LATENT_OPERATOR_ARCHITECTURE.md (606 lines, 20KB)

**Comprehensive technical deep-dive** - Read this for complete understanding.

**Contents**:
- Architecture hierarchy (tree structure of all components)
- LatentOperator implementation (lines 40-92 of latent_operator.py)
- Time evolution mechanisms (single-step and multi-step autoregressive)
- PDE-Transformer core design (U-shaped with hierarchical token processing)
- Component breakdown:
  - TimeEmbedding (scalar Δt → continuous embedding)
  - RMSNorm (magnitude-preserving normalization)
  - ChannelSeparatedSelfAttention (efficient per-group attention)
  - TransformerLayer (attention + MLP with residuals)
  - PDETransformerBlock (full U-net architecture)
- Token representation (spatial latent encoding)
- Temporal evolution with Δt handling
- Shifted window mechanisms (currently unused)
- Configuration consistency requirements
- Performance reference numbers
- Complete data flow diagrams

**Best for**: Understanding the full architecture, implementation details, design decisions

**Location**: `/Users/emerygunselman/Code/universal_simulator/LATENT_OPERATOR_ARCHITECTURE.md`

---

### 2. LATENT_OPERATOR_QUICK_REFERENCE.md (193 lines, 8KB)

**Quick reference guide** - Read this for fast lookup.

**Contents**:
- Core answer sheet (8 key questions answered)
- LatentOperator implementation summary
- Architecture at a glance
- Core transformer components
- Token representation
- Temporal evolution (Δt handling)
- Shifted windows status
- Configuration constraints
- File map with line counts
- Performance numbers
- Key design decisions
- Data flow diagrams

**Best for**: Quick lookup, design decisions, configuration examples, high-level overview

**Location**: `/Users/emerygunselman/Code/universal_simulator/LATENT_OPERATOR_QUICK_REFERENCE.md`

---

## Key Questions Answered

### 1. LatentOperator Implementation
**File**: `src/ups/models/latent_operator.py` (92 lines)

Core class that evolves latent states forward in time using:
- TimeEmbedding: Converts scalar Δt → (batch, time_embed_dim)
- time_to_latent: Projects embedding → (batch, 1, latent_dim)
- PDETransformerBlock: U-shaped transformer core
- AdaLNConditioner: Optional physics conditioning
- Residual connection: z_new = z + core(z, dt)

### 2. Architecture: Time Evolution
**Single step**: Embed Δt → broadcast → condition → process → residual → add

**Multi-step**: Iterate operator application (state₀ → state₁ → ... → stateₙ)

**File**: `src/ups/inference/rollout_transient.py` (75 lines) for multi-step rollout

### 3. Core Components: Transformer Mechanisms
**File**: `src/ups/core/blocks_pdet.py` (235 lines)

1. **RMSNorm**: Root mean square normalization (preserves magnitude)
2. **ChannelSeparatedSelfAttention**: Per-group multi-head attention (efficient)
3. **TransformerLayer**: Attention + MLP with residuals
4. **PDETransformerBlock**: Full U-shaped transformer

### 4. PDE-Transformer Architecture
**U-shaped design**:
- Input proj: latent_dim → hidden_dim
- DOWN: Token pooling (×2) with attention (2-3 layers)
- BOTTLENECK: Intensive processing at coarsest scale
- UP: Token upsampling (×2) with skip connections (2-3 layers)
- Output proj: hidden_dim → latent_dim

### 5. Token Handling
**Configuration** (from golden config):
```yaml
latent:
  dim: 16        # 16 features per token
  tokens: 32     # 32 spatial tokens
```

**Total shape**: (batch=12, tokens=32, dim=16)

**Origin**: Grid encoding → patch embedding → Fourier features → projection → pooling

**File**: `src/ups/io/enc_grid.py` (232 lines) for grid encoder

### 6. Temporal Evolution: Δt Handling
**Pipeline**:
1. Scalar Δt → TimeEmbedding (MLP: 1→64→64)
2. Embedding → time_to_latent (64→16)
3. Broadcast: z_cond = z + time_feat[:, None, :]
4. Process through PDE-Transformer
5. Add residual: z_new = z + core_output

**Supports**: Arbitrary-length rollouts via autoregressive iteration

### 7. Shifted Window Efficiency
**File**: `src/ups/core/shifted_window.py` (172 lines)

**Purpose**: Local window-based attention (Swin Transformer pattern)

**Functions**:
- `partition_windows()`: (B,H,W,C) → (B*windows, window_area, C)
- `merge_windows()`: Reverse operation

**Status**: Implemented but NOT used in PDETransformerBlock
- Currently uses: Token pooling + channel-separated attention
- Alternative efficiency mechanism

---

## Critical Configuration

All dimensions must match exactly:
```yaml
latent.dim = operator.pdet.input_dim = diffusion.latent_dim = ttc.decoder.latent_dim
```

Divisibility constraints:
```yaml
operator.pdet.hidden_dim % operator.pdet.group_size = 0
operator.pdet.group_size % operator.pdet.num_heads = 0
```

**Golden config example**:
- latent.dim: 16
- operator.pdet.input_dim: 16 (MUST match)
- operator.pdet.hidden_dim: 96 (6× latent_dim)
- operator.pdet.group_size: 12 (divides 96)
- operator.pdet.num_heads: 6 (divides 12)

---

## File Reference

| File | Purpose | Lines | Key Classes |
|------|---------|-------|------------|
| src/ups/models/latent_operator.py | Main operator | 92 | LatentOperator, TimeEmbedding |
| src/ups/core/blocks_pdet.py | PDE-Transformer core | 235 | PDETransformerBlock, ChannelSeparatedSelfAttention, RMSNorm |
| src/ups/core/latent_state.py | State container | 71 | LatentState |
| src/ups/core/conditioning.py | Physics conditioning | 84 | AdaLNConditioner |
| src/ups/core/shifted_window.py | Window utilities | 172 | partition_windows, merge_windows |
| src/ups/io/enc_grid.py | Grid encoder | 232 | GridEncoder |
| src/ups/inference/rollout_transient.py | Multi-step rollout | 75 | rollout_transient |
| configs/train_burgers_golden.yaml | Reference config | 221 | All dimensions |

---

## Performance Reference

**Golden Config** (`configs/train_burgers_golden.yaml`):
- Latent Dimension: 16
- Tokens: 32
- Hidden Dimension: 96
- Operator Final Loss: ~0.00023
- Baseline NRMSE: ~0.78
- TTC NRMSE: ~0.09 (88% improvement)
- Training Time: ~14.5 min (A100)
- Operator Epochs: 25

---

## Design Philosophy

1. **Residual Connection**: Learning small perturbations (typical PDE behavior)
2. **U-Shaped Architecture**: Hierarchical multi-scale processing
3. **Token Pooling**: Efficient coarsening without full attention cost
4. **Channel-Separated Attention**: Per-group SIMD-friendly computation
5. **Continuous Time Embedding**: Learned sensitivity to different Δt values
6. **RMSNorm**: Preserves signal magnitude (important for physics)
7. **AdaLN Conditioning**: Physics-aware scale/shift/gate modulation

---

## Data Flow Summary

```
Single Step:
  LatentState(z, t, cond) + dt
    ↓ [TimeEmbedding] dt → embedding
    ↓ [Broadcast] Inject across tokens
    ↓ [Conditioning] Optional: apply AdaLN
    ↓ [PDETransformerBlock] Process:
      - input_proj: latent_dim → hidden_dim
      - down: pool tokens (×2 iterations)
      - bottleneck: coarsest scale
      - up: upsample tokens (×2 iterations)
      - output_proj: hidden_dim → latent_dim
    ↓ [Residual] Add to original
    ↓ LatentState(z_new=z+residual, t_new=t+dt, cond)

Multi-Step:
  state₀ → [Operator] → state₁ → [Operator] → ... → stateₙ
```

---

## How to Use This Documentation

**For architecture understanding**: Read LATENT_OPERATOR_ARCHITECTURE.md (sections 1-7)

**For quick reference**: Check LATENT_OPERATOR_QUICK_REFERENCE.md (sections 1-8)

**For configuration**: Review golden config at `configs/train_burgers_golden.yaml`

**For implementation**: Check source files in `src/ups/models/` and `src/ups/core/`

**For inference**: See `src/ups/inference/rollout_transient.py` and `src/ups/inference/rollout_ttc.py`

---

## Document Generation

Both documents were generated on 2025-10-23 through systematic analysis of:
- Source code in src/ups/models/ and src/ups/core/
- Configuration examples in configs/
- Inference utilities in src/ups/inference/
- I/O encoders in src/ups/io/

All paths are absolute and reference the actual codebase:
- `/Users/emerygunselman/Code/universal_simulator/`

