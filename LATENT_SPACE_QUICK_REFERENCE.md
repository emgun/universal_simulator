# Latent Space Architecture - Quick Reference

**Full documentation**: See `LATENT_SPACE_ARCHITECTURE.md` (918 lines, 14 sections)

## Core Concepts at a Glance

### Latent State Structure
```python
LatentState(
    z: (batch, tokens, latent_dim),      # Latent tokens
    t: scalar,                            # Current time
    cond: Dict[str, Tensor]               # Physics conditioning
)
```

### Configuration Parameters

**Token Count vs Latent Dimension Trade-off**:
- **More tokens** → Better spatial resolution (8×8, 11×11, 16×16 patches)
- **Higher dim** → Richer per-token features
- **Golden config**: Minimal (16-dim, 32 tokens) proves efficiency wins

| Config | Latent Dim | Tokens | Hidden Dim | Depths | Total Features | Batch |
|--------|-----------|--------|-----------|--------|-----------------|-------|
| 64-token | 64 | 64 | 128 | [2,2,2] | 4,096 | 8 |
| 128-token | 128 | 128 | 256 | [3,3,2] | 16,384 | 6 |
| 256-token | 192 | 256 | 384 | [4,4,4] | 49,152 | 4 |
| **Golden** | **16** | **32** | **96** | **[1,1,1]** | **512** | **12** |

### Encoder/Decoder Flow

```
Training:
  Physical Grid (128×128)
    ↓ GridEncoder
  Latent Tokens (32 tokens × 16-dim)
    ↓ LatentOperator × N steps
  Evolved Tokens
    ↓ AnyPointDecoder
  Predicted Fields (128×128)
    ↓ Loss Computation

Inference:
  Latent Tokens
    ↓ Operator × 8 candidates (parallel)
    ↓ Diffusion Sampling (optional)
    ↓ Decoder (arbitrary resolution, e.g., 512×512)
    ↓ Physics Reward Evaluation
    ↓ Beam Search Best
  Final Predictions
```

## Architecture Components

### 1. GridEncoder (Patch-based, Discretization-Agnostic)
- **Input**: Grid fields (e.g., Burgers: ρ, e on 128×128)
- **Process**: 
  1. PixelUnshuffle (4×4 patches)
  2. Per-field residual convolutions
  3. Fourier positional features
  4. Token projection
  5. Adaptive pooling to target token count
- **Output**: (batch, target_tokens, latent_dim)
- **Why PixelUnshuffle**: Invertible, preserves structure, efficient

### 2. LatentOperator (U-shaped PDE-Transformer)
- **Input**: LatentState(z, dt)
- **Process**:
  1. Time embedding (scalar Δt → continuous embedding)
  2. Optional AdaLN conditioning
  3. PDETransformerBlock (U-net):
     - Projection: latent_dim → hidden_dim
     - Down: Token pooling + transformer layers
     - Bottleneck: Intensive transformation
     - Up: Token upsampling + skip connections
     - Output projection: hidden_dim → latent_dim
  4. Residual connection: z_new = z_old + output
- **Output**: LatentState(z_new, t + dt)

### 3. AnyPointDecoder (Query-based, Perceiver-style)
- **Input**: Latent tokens + arbitrary query coordinates
- **Process**:
  1. Fourier encode query coordinates
  2. Cross-attention: queries ↔ latent tokens
  3. Per-field prediction heads
- **Output**: Predictions at any resolution
- **Key**: O(queries × tokens), not O(queries²)
- **Benefit**: Train @64×64, test @512×512, interpolate at sub-pixel

### 4. Channel-Separated Attention
- Splits features into groups: (batch, tokens, dim) → groups
- Applies multi-head attention per group
- More efficient than full self-attention
- RMSNorm preserves magnitude (better for physics)

### 5. Shifted Window Attention (Infrastructure, unused)
- Status: Implemented but NOT in active use
- Purpose: Local window-based attention (Swin Transformer pattern)
- API: partition_windows() ↔ merge_windows()
- Could optimize for grid-shaped latents in future

### 6. AdaLN Conditioning (Optional)
- Modulates latent tokens based on physics parameters
- Scale, shift, gate from per-condition MLPs
- Supports: Reynolds number, energy, boundary conditions
- Current usage: Typically disabled

## Dimension Consistency Rules

**MUST satisfy these constraints**:

1. Latent dimension equality:
   ```python
   latent.dim == operator.pdet.input_dim 
             == diffusion.latent_dim 
             == ttc.decoder.latent_dim
   ```

2. Channel separation divisibility:
   ```python
   hidden_dim % group_size == 0
   group_size % num_heads == 0
   ```

3. Example (Golden config):
   ```yaml
   latent.dim = 16
   operator.pdet.input_dim = 16         ✓
   operator.pdet.hidden_dim = 96        ✓
   operator.pdet.group_size = 12        # 96 / 12 = 8 ✓
   operator.pdet.num_heads = 6          # 12 / 6 = 2 ✓
   ```

## Token Count Effects

### 64 Tokens
- Spatial: ~8×8 patches
- Capacity: 4,096 features
- Training: Fast
- Quality: Coarse
- Use case: Quick prototyping

### 128 Tokens
- Spatial: ~11×11 patches
- Capacity: 16,384 features
- Training: Medium
- Quality: Good
- Use case: Balanced approach

### 256 Tokens
- Spatial: ~16×16 patches
- Capacity: 49,152 features
- Training: Slow
- Quality: Excellent
- Use case: Research on discretization

### Golden (32 tokens)
- Spatial: ~5×6 patches
- Capacity: 512 features (8× compression)
- Training: Very fast
- Quality: SOTA with diffusion + TTC
- Use case: **Production** (recommended)

## Memory Analysis

| Config | Parameters | Peak Memory | Training Time |
|--------|-----------|------------|-----------------|
| Golden | 200K | 50 MB | 14.5 min (A100) |
| 64-token | ? | 200 MB | Hours |
| 128-token | ? | 400 MB | Hours |
| 256-token | ? | 600 MB | 24+ hours |

## Key Architectural Principles

1. **Discretization-Agnostic**
   - Encoder: Adaptive pooling handles any grid size
   - Decoder: Fourier features + Perceiver queries work at any resolution
   - Training grid ≠ Inference grid ✓

2. **Efficiency-First**
   - Channel-separated attention vs. full attention
   - U-net pooling vs. global patterns
   - Small latent + diffusion + TTC > large latent alone

3. **Invertible Components**
   - PixelUnshuffle ↔ PixelShuffle
   - Window partition ↔ merge
   - Residual stems (identity initialization)

4. **Physics-Aware Design**
   - Fourier positional encoding (multi-scale)
   - Time embedding (continuous Δt)
   - Optional AdaLN for parameters
   - Test-time physics rewards

## File Quick Reference

| Component | File |
|-----------|------|
| LatentState | `src/ups/core/latent_state.py` |
| PDE-Transformer | `src/ups/core/blocks_pdet.py` |
| Attention | `src/ups/core/blocks_pdet.py` (RMSNorm, ChannelSeparated) |
| Window Utilities | `src/ups/core/shifted_window.py` |
| AdaLN | `src/ups/core/conditioning.py` |
| Grid Encoder | `src/ups/io/enc_grid.py` |
| Mesh Encoder | `src/ups/io/enc_mesh_particle.py` |
| Query Decoder | `src/ups/io/decoder_anypoint.py` |
| Operator | `src/ups/models/latent_operator.py` |
| Golden Config | `configs/train_burgers_golden.yaml` |
| Ablations | `configs/ablation_upt_*.yaml` |

## Common Configuration Patterns

### Starting a New Task
```yaml
# Copy golden and scale up if needed
cp configs/train_burgers_golden.yaml configs/my_new_task.yaml

# Minimal changes:
latent:
  dim: 16        # Start small
  tokens: 32

operator:
  pdet:
    input_dim: 16
    hidden_dim: 96
    depths: [1, 1, 1]
    group_size: 12
    num_heads: 6
```

### Increasing Capacity (if needed)
```yaml
# Option 1: More tokens (preserve per-token dim)
latent:
  dim: 16
  tokens: 64     # 2× tokens

# Update operator accordingly
operator.pdet.hidden_dim: 160  # Adjust for balance

# Option 2: More features per token
latent:
  dim: 32        # 2× dimension
  tokens: 32

# Update operator
operator.pdet.input_dim: 32
operator.pdet.hidden_dim: 192
```

### Reducing Memory
```yaml
# Start with golden, then reduce:
latent:
  dim: 8         # Half
  tokens: 16     # Half

operator:
  pdet:
    input_dim: 8
    hidden_dim: 48    # Match scaling
    depths: [1, 1, 1]
    group_size: 8
    num_heads: 4
```

## Validation Checklist

Before training, verify:

- [ ] `latent.dim` matches `operator.pdet.input_dim`
- [ ] `operator.pdet.hidden_dim % group_size == 0`
- [ ] `group_size % num_heads == 0`
- [ ] `diffusion.latent_dim == latent.dim`
- [ ] `ttc.decoder.latent_dim == latent.dim`
- [ ] `batch_size * accum_steps * tokens * latent_dim` fits in memory
- [ ] `training.lr` adjusted for batch size (rule: 1e-3 for batch=12)
- [ ] Random seed set for reproducibility (if needed)

Run validation:
```bash
python scripts/validate_config.py configs/my_config.yaml
```

## Common Issues & Fixes

**Issue**: `dim must be divisible by group_size`
- Adjust `hidden_dim` or `group_size`
- Example: hidden_dim=96, group_size=12 works (8 groups)

**Issue**: `group_size must be divisible by num_heads`
- Reduce `num_heads` or increase `group_size`
- Example: group_size=12, num_heads=6 works (2 per head)

**Issue**: Out of memory
- Reduce `latent.tokens` or `latent.dim`
- Reduce `batch_size`
- Reduce `operator.pdet.hidden_dim`
- Disable `training.compile`

**Issue**: Poor training convergence
- Check golden config learning rates as reference
- Verify operator final loss < 0.001 (typically ~0.0002)
- Check gradient clipping: `grad_clip: 1.0`
- Verify time embedding range (dt around 0.1 is standard)

## Recommended Reading Order

1. **This file** (quick reference)
2. **LATENT_SPACE_ARCHITECTURE.md** (sections 1-4, 6-7)
3. **LATENT_OPERATOR_ARCHITECTURE.md** (existing doc on operator)
4. Code: `src/ups/io/enc_grid.py` (understand PixelUnshuffle)
5. Code: `src/ups/io/decoder_anypoint.py` (understand Perceiver)
6. Code: `src/ups/core/blocks_pdet.py` (understand U-net)

