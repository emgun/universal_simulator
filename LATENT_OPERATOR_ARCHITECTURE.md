# Latent Operator Architecture Documentation

## Overview

The **Latent Operator** is the deterministic core of the Universal Physics Stack, responsible for evolving encoded PDE solutions forward in time within a learned latent space. It uses a **PDE-Transformer (PDE-T) backbone** with residual connections to compute single-step state transitions.

**Key Principle**: Physics evolves as `z_new = z_old + residual_step(z_old, Δt)`

---

## Architecture Hierarchy

```
LatentOperator (src/ups/models/latent_operator.py)
├── TimeEmbedding
│   └── MLPProjNet(1 → time_embed_dim → time_embed_dim)
├── time_to_latent: Linear(time_embed_dim → latent_dim)
├── PDETransformerBlock (src/ups/core/blocks_pdet.py) [CORE]
│   ├── input_proj: Linear(latent_dim → hidden_dim)
│   ├── Down-Upsample Stack
│   │   ├── down_layers[i]: TransformerLayers + Downsample
│   │   ├── bottleneck: TransformerLayers
│   │   └── up_layers[i]: Upsample + TransformerLayers + Skip
│   └── output_proj: Linear(hidden_dim → latent_dim)
├── AdaLNConditioner [OPTIONAL]
│   └── embedders: Per-condition source MLPs
└── output_norm: LayerNorm(latent_dim)
```

---

## 1. LatentOperator Implementation

**File**: `/Users/emerygunselman/Code/universal_simulator/src/ups/models/latent_operator.py`

### Class: `LatentOperator(nn.Module)`

```python
class LatentOperator(nn.Module):
    """Advance latent state by one time step using PDE-Transformer backbone."""
    
    def __init__(self, cfg: LatentOperatorConfig):
        # Initialize time embedding (scalar Δt → embedding)
        self.time_embed = TimeEmbedding(cfg.time_embed_dim)
        self.time_to_latent = nn.Linear(cfg.time_embed_dim, cfg.latent_dim)
        
        # PDE-Transformer core
        self.core = PDETransformerBlock(cfg.pdet)
        
        # Optional conditioning (Adaptive Layer Norm)
        self.conditioner = AdaLNConditioner(cfg.conditioning) if cfg.conditioning else None
        
        self.output_norm = nn.LayerNorm(cfg.latent_dim)
```

### Core Methods

#### `forward(state: LatentState, dt: torch.Tensor) -> LatentState`

Performs a **single time step** of evolution:

1. Computes residual: `residual = self.step(state, dt)`
2. Updates latent state: `z_new = z_old + residual` (residual connection)
3. Updates time: `t_new = t_old + dt`
4. Preserves conditioning: `cond_new = cond_old`

**Returns**: New `LatentState` with evolved latent tokens

#### `step(state: LatentState, dt: torch.Tensor) -> torch.Tensor`

Computes the residual term:

1. **Time embedding**: Scalar Δt → continuous embedding
   ```
   dt_embed = time_embed(dt)  # shape: (batch, time_embed_dim)
   time_feat = time_to_latent(dt_embed)[:, None, :]  # (batch, 1, latent_dim)
   ```

2. **Inject time into latent tokens**:
   ```
   z = z + time_feat  # Broadcast (batch, 1, latent_dim) across tokens
   ```

3. **Optional conditioning**:
   ```
   if conditioner is not None:
       z = conditioner.modulate(z, state.cond)  # Apply adaptive scale/shift/gate
   ```

4. **PDE-Transformer core processing**:
   ```
   residual = core(z)  # U-shaped transformer, outputs same shape as input
   ```

5. **Output normalization**:
   ```
   residual = output_norm(residual)  # LayerNorm for stability
   ```

---

## 2. Architecture: Time Evolution

### Single-Step Evolution
```
Input: LatentState
  z ∈ ℝ^(batch × tokens × latent_dim)
  dt ∈ ℝ (scalar time step)
  cond: dict of conditioning signals

Process:
  1. Encode Δt → embedding vector (continuous function)
  2. Broadcast time embedding across spatial tokens
  3. Apply adaptive conditioning (if enabled)
  4. Process through PDE-Transformer U-net
  5. Add residual: z_new = z_old + core_output

Output: LatentState
  z_new = z + residual  (updated latent tokens)
  t_new = t + dt        (updated time)
  cond_new = cond       (unchanged)
```

### Multi-Step Rollout (Autoregressive)

The operator **chains single steps** for multi-step prediction:

**File**: `/Users/emerygunselman/Code/universal_simulator/src/ups/inference/rollout_transient.py`

```python
def rollout_transient(initial_state: LatentState, operator: LatentOperator, 
                      config: RolloutConfig) -> RolloutLog:
    state = initial_state
    for step in range(config.steps):
        predicted = operator(state, dt_tensor)  # Single step
        # Optional: apply diffusion corrector for refinement
        if should_correct:
            drift = corrector(predicted, tau)
            predicted.z = predicted.z + drift
        state = predicted
    return states
```

**Key Property**: Operator can perform arbitrary number of steps by iterating single-step predictions. Each step preserves `LatentState` structure and conditioning signals.

---

## 3. Core Components: Transformer Mechanisms

### 3a. Time Embedding

**Class**: `TimeEmbedding(nn.Module)`

```python
class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim: int):
        self.proj = nn.Sequential(
            nn.Linear(1, embed_dim),      # Scalar → embedding
            nn.SiLU(),                     # Smooth activation
            nn.Linear(embed_dim, embed_dim)
        )
    
    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        # Handle batch and scalar dt
        dt = dt.view(-1, 1)  # Ensure shape (batch, 1)
        return self.proj(dt)  # (batch, embed_dim)
```

**Purpose**: Continuous embedding of timestep size, enabling learned sensitivity to different Δt values (physically important: smaller Δt may require different treatment than large steps)

---

## 4. PDE-Transformer Architecture (blocks_pdet.py)

**File**: `/Users/emerygunselman/Code/universal_simulator/src/ups/core/blocks_pdet.py`

### Design Principle

U-shaped transformer with **channel-separated self-attention** for efficiency. Inspired by Swin Transformer's local attention but operating on token sequences rather than spatial windows.

### Core Components

#### A. `RMSNorm(nn.Module)`

Root Mean Square normalization (alternative to LayerNorm):
```python
rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()
output = weight * (x / rms)
```

**Advantage**: Preserves mean signal (important for physics where magnitudes matter)

---

#### B. `ChannelSeparatedSelfAttention(nn.Module)`

Multi-head attention applied independently to **channel groups** (not all dimensions jointly):

```
Input: (batch, tokens, dim)
│
├─ Split along feature dim into groups
│  └─ Each group size: dim / num_groups
│
├─ Per-group multi-head attention
│  ├─ RMS-normalize Q, K (preserve magnitude)
│  ├─ Scaled dot-product attention
│  └─ Output projection per group
│
└─ Recombine groups
   Output: (batch, tokens, dim)
```

**Efficiency Benefit**: 
- Standard attention: O(tokens² × dim) complexity
- Channel-separated: O(tokens² × group_size) per group
- Total: O(tokens² × dim) but with smaller constant (SIMD friendly)

**Implementation Detail**:
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    B, T, C = x.shape
    # Split: (B, T, C) → (B, T, groups, group_size)
    x_groups = x.view(B, T, self.groups, self.group_size)
    # Reshape for efficient batched attention: (B*groups, T, group_size)
    x_groups = x_groups.permute(0, 2, 1, 3).contiguous().view(B*self.groups, T, self.group_size)
    
    # Attention within each group
    q = self.q_proj(self.q_norm(x_groups))
    k = self.k_proj(self.k_norm(x_groups))
    v = self.v_proj(x_groups)
    
    # Multi-head: (B*groups, T, group_size) → (B*groups, num_heads, T, head_dim)
    attn_out = F.scaled_dot_product_attention(q, k, v)
    
    # Reshape back: (B*groups, T, group_size) → (B, T, C)
    return attn_out.view(B, T, C)
```

---

#### C. `TransformerLayer(nn.Module)`

Single transformer block (attention + feedforward):

```python
class TransformerLayer(nn.Module):
    def __init__(self, dim, group_size, num_heads, mlp_ratio=2.0):
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = ChannelSeparatedSelfAttention(...)
        self.ff = FeedForward(dim, hidden_dim=int(dim*mlp_ratio))
    
    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))        # Residual attention
        x = x + self.ff(x)                          # Residual MLP
        return x
```

**Skip connections**: Both attention and MLP use residual paths (identity + processed)

---

#### D. `PDETransformerBlock(nn.Module)` - The Core

U-shaped architecture with hierarchical token processing:

```
Input: (batch, tokens, input_dim)
│
├─ input_proj: Linear(input_dim → hidden_dim)
│
├─ DOWN PHASE (reduce tokens via pooling)
│  ├─ depth_layer_0: TransformerLayers + downsample(↓2)
│  ├─ depth_layer_1: TransformerLayers + downsample(↓2)
│  └─ depth_layer_2: TransformerLayers [bottleneck, no downsample]
│
├─ BOTTLENECK (finest latent representation)
│  └─ Intensive transformation at coarsest scale
│
├─ UP PHASE (restore tokens via upsampling)
│  ├─ upsample(↑2) + skip_connection + TransformerLayers
│  ├─ upsample(↑2) + skip_connection + TransformerLayers
│  └─ [final layer]
│
├─ final_norm: LayerNorm(hidden_dim)
│
└─ output_proj: Linear(hidden_dim → input_dim)
   Output: (batch, tokens, input_dim)
```

**Key Design Choices**:

1. **Token Pooling** (downsampling):
   ```python
   def _downsample_tokens(x):
       # Average adjacent pairs: [a, b, c, d] → [(a+b)/2, (c+d)/2]
       x_even = x[:, ::2, :]
       x_odd = x[:, 1::2, :]
       return 0.5 * (x_even + x_odd)
   ```
   
   **Benefit**: Coarser representations capture long-range dependencies without attention cost

2. **Token Upsampling** (interpolation):
   ```python
   def _upsample_tokens(x, target_length):
       # Linear interpolation back to original token count
       return F.interpolate(x.transpose(1,2), size=target_length, mode='linear')
   ```

3. **Skip Connections**:
   - Saved at each downsampling layer
   - Injected (added) at corresponding upsampling layer
   - Preserves fine-grained spatial information

**Configuration Example** (from golden config):
```yaml
operator:
  pdet:
    input_dim: 16           # Must match latent.dim
    hidden_dim: 96          # 6× latent_dim for expressivity
    depths: [1, 1, 1]       # 1 layer per down/bottle/up
    group_size: 12          # Channel group size
    num_heads: 6            # Heads per group
    mlp_ratio: 2.0          # FF hidden = 2× dim
```

---

## 5. Token Handling

### Latent Token Representation

**Structure**: `z ∈ ℝ^(batch × tokens × latent_dim)`

#### Configuration Parameters:

From `/Users/emerygunselman/Code/universal_simulator/configs/train_burgers_golden.yaml`:

```yaml
latent:
  dim: 16        # Feature dimension per token
  tokens: 32     # Number of spatial tokens
```

#### How Tokens Are Created

Tokens come from **grid encoding** in the I/O layer:

**File**: `/Users/emerygunselman/Code/universal_simulator/src/ups/io/enc_grid.py`

```
Physical Grid (e.g., Burgers: 128×128 or 512×512)
│
├─ PixelUnshuffle (patch-based grouping)
│  └─ patch_size: 4 → 4×4 spatial pixels per patch
│
├─ Per-field residual convolutions
│  └─ Process patch embeddings
│
├─ Fourier positional features (optional)
│  └─ Sinusoidal encodings of patch centers
│
├─ Token projection
│  └─ All patches → Linear → latent_dim
│
├─ Adaptive token pooling (if needed)
│  └─ Reduce/expand to target token count
│
└─ Output: (batch, latent_tokens, latent_dim)
```

#### Token Dimensions in Golden Config:

- **Latent dimension**: 16 features per token
- **Token count**: 32 spatial tokens (typically: grid_height/patch_size × grid_width/patch_size)
- **Full shape**: (batch=12, tokens=32, dim=16)

**Physical Interpretation**: 
- Each token represents ~32 patches of the spatial domain
- Token count ∝ spatial resolution (coarser grids → fewer tokens)
- Latent dim ∝ information capacity per token (trade-off with computation)

---

## 6. Temporal Evolution: Δt Handling

### Time Parametrization

The operator is **conditioned on Δt** (continuous scalar):

```
forward(state: LatentState, dt: float) -> LatentState
```

**Implementation**:

1. **Continuous time embedding**:
   ```python
   dt_embed = time_embed(dt)  # Scalar → embedding vector
   ```

2. **Broadcast to tokens**:
   ```python
   time_feat = time_to_latent(dt_embed)  # → latent_dim
   z_with_time = z + time_feat[:, None, :]  # Inject across all tokens
   ```

3. **Learned sensitivity**: The embedding MLP learns how to weight different Δt values
   - Small Δt: Linear regime (small residuals)
   - Large Δt: Nonlinear/shock formation regime

### Multi-Step Rollouts

**Arbitrary sequence prediction** through iterative steps:

```python
state = initial_state
for t in range(num_steps):
    state = operator(state, dt_tensor)  # Single step forward
```

**Capabilities**:
- ✅ Variable-length trajectories
- ✅ Different Δt per step (if needed)
- ✅ Conditioning signals preserved/updated
- ✅ Long-horizon prediction (autoregressive chain)

**Limitations**:
- Error accumulation: Each step compounds prediction error
- No ground truth feedback (pure extrapolation)
- Solved via: Diffusion residual model + Test-Time Conditioning (TTC)

---

## 7. Shifted Window Mechanisms (shifted_window.py)

**File**: `/Users/emerygunselman/Code/universal_simulator/src/ups/core/shifted_window.py`

### Purpose

Provide efficient **local attention** by partitioning spatial grids into non-overlapping windows. This is the **Swin Transformer pattern** adapted for latent token sequences.

### Key Functions

#### `partition_windows(tensor, window_size, shift_size=(0,0)) -> (windows, info)`

Slices a `(B, H, W, C)` grid tensor into flattened windows:

```
Input grid: (batch, height, width, channels)
│
├─ Cyclic shift (optional, for local cross-window attention):
│  └─ roll negatively by shift_size
│
├─ Partition into non-overlapping windows:
│  └─ height/window_height × width/window_width windows
│
└─ Flatten: (batch*num_windows, window_area, channels)
```

**Example**:
- Grid: (2, 64, 64, 16) [batch=2, 64×64 spatial, 16 channels]
- Window size: (8, 8)
- Windows: (2×8×8=128 windows, 64 pixels/window, 16 channels)

#### `merge_windows(windows, info) -> tensor`

Reverses the partitioning operation:

```python
def merge_windows(windows: torch.Tensor, info: WindowPartitionInfo) -> torch.Tensor:
    # Reshape from (batch*windows, window_area, C) → (batch, H, W, C)
    # Apply inverse cyclic shift if needed
```

### Efficiency Benefit

Instead of global self-attention on all tokens:
- **Global**: O(tokens²) attention complexity (expensive)
- **Windowed**: O(window_area²) per window (cheap, parallelizable)

### Current Usage

**Status**: Defined in codebase but **not actively used in PDETransformerBlock**

The PDETransformerBlock uses:
- Token pooling (coarser representations instead of local windows)
- Channel-separated attention (efficient per-group computation)

Future enhancement could integrate windowed attention for grid-shaped inputs.

---

## Configuration Consistency Requirements

### Critical Dimension Constraints

All dimensions must **match exactly** across components:

**Config Schema**:
```yaml
latent:
  dim: 16                    # D_latent

operator:
  pdet:
    input_dim: 16            # MUST equal latent.dim
    hidden_dim: 96           # Typically 4-8× latent_dim
    depths: [1, 1, 1]        # Layers per scale
    group_size: 12           # MUST divide hidden_dim evenly
    num_heads: 6             # MUST divide group_size evenly

diffusion:
  latent_dim: 16             # MUST equal latent.dim
  hidden_dim: 96             # Typically match operator.pdet.hidden_dim

ttc:
  decoder:
    latent_dim: 16           # MUST equal latent.dim
```

### Validation

The system enforces these checks:

```python
# In LatentOperator.__init__:
if pdet_cfg.input_dim != cfg.latent_dim:
    raise ValueError("PDETransformer input_dim must match latent_dim")

# In ChannelSeparatedSelfAttention.__init__:
if dim % group_size != 0:
    raise ValueError("dim must be divisible by group_size")
if group_size % num_heads != 0:
    raise ValueError("group_size must be divisible by num_heads")
```

---

## Performance Reference (Golden Config)

**Configuration**: `/Users/emerygunselman/Code/universal_simulator/configs/train_burgers_golden.yaml`

| Metric | Value |
|--------|-------|
| Latent dimension | 16 |
| Token count | 32 |
| Hidden dimension | 96 |
| Operator final loss | ~0.00023 |
| Baseline NRMSE (no TTC) | ~0.78 |
| TTC NRMSE (8 candidates) | ~0.09 |
| Training time (A100) | ~14.5 min |
| Operator epochs | 25 |

---

## Summary: Data Flow

```
INFERENCE:
─────────
LatentState(z, t, cond)
    ↓
[TimeEmbedding] Δt → embedding
    ↓
[time_to_latent] embedding → latent_dim
    ↓
[Inject into tokens] z_cond = z + time_feature
    ↓
[AdaLNConditioner] (optional) Apply conditioning scale/shift/gate
    ↓
[PDETransformerBlock] U-shaped processing
    ├─ input_proj: latent_dim → hidden_dim
    ├─ down: [tokens→ tokens/2→ tokens/4]
    ├─ bottleneck: process coarsest scale
    ├─ up: [tokens/4→ tokens/2→ tokens]
    └─ output_proj: hidden_dim → latent_dim
    ↓
[LayerNorm] Normalize residual
    ↓
z_new = z_old + residual
    ↓
LatentState(z_new, t+dt, cond)


MULTI-STEP:
──────────
initial_state → [Operator] → state₁ → [Operator] → state₂ → ... → stateₙ
```

---

## Key Files Summary

| File | Purpose |
|------|---------|
| `src/ups/models/latent_operator.py` | Main operator class, time embedding, forward logic |
| `src/ups/core/blocks_pdet.py` | PDE-Transformer U-net backbone, attention mechanisms |
| `src/ups/core/latent_state.py` | LatentState data structure (z, t, cond) |
| `src/ups/core/conditioning.py` | AdaLNConditioner for physics-aware modulation |
| `src/ups/core/shifted_window.py` | Window partitioning utilities (currently unused) |
| `src/ups/io/enc_grid.py` | Grid encoder producing latent tokens |
| `src/ups/inference/rollout_transient.py` | Multi-step autoregressive rollout |
| `configs/train_burgers_golden.yaml` | Example configuration with all dimensions |

