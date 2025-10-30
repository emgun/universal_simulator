# Latent Space Architecture and Configuration Documentation

## Overview

The Universal Physics Stack (UPS) employs a **discretization-agnostic latent space approach** where:

1. **Encoding**: Physical fields → Low-dimensional latent tokens via grid/mesh encoders
2. **Evolution**: Latent tokens evolved via PDE-Transformer operator
3. **Decoding**: Latent tokens → Physical field values at arbitrary query points
4. **Conditioning**: Test-time physics rewards guide refinement

This document details the latent space design, encoder/decoder implementations, and how latent token configurations (64, 128, 256 tokens) affect model capacity and performance.

---

## 1. Latent Token Representation

### Core Data Structure: LatentState

**File**: `/Users/emerygunselman/Code/universal_simulator/src/ups/core/latent_state.py`

```python
@dataclass
class LatentState:
    """Bundle latent tokens with metadata."""
    z: torch.Tensor                          # Shape: (batch, tokens, dim)
    t: Optional[torch.Tensor] = None         # Current time (scalar or single-element)
    cond: Dict[str, torch.Tensor] = {}       # Conditioning metadata
```

### Tensor Dimensions

**Shape**: `(batch, tokens, latent_dim)`

Example configurations from ablation studies:

| Config | Batch | Tokens | Latent Dim | Shape | Total Elements |
|--------|-------|--------|------------|-------|-----------------|
| 64-token | 8 | 64 | 64 | (8, 64, 64) | 32,768 |
| 128-token | 6 | 128 | 128 | (6, 128, 128) | 98,304 |
| 256-token | 4 | 256 | 192 | (4, 256, 192) | 196,608 |
| Golden | 12 | 32 | 16 | (12, 32, 16) | 6,144 |

**Key Insight**: Token count and latent dimension scale differently:
- **More tokens** = finer spatial resolution (better captures local structure)
- **Higher latent_dim** = richer per-token features (better represents complex fields)

---

## 2. Grid Encoder (Discretization-Agnostic I/O)

### Purpose

Convert physical grid-based fields (Burgers, Navier-Stokes, etc.) to latent tokens with automatic discretization handling.

**File**: `/Users/emerygunselman/Code/universal_simulator/src/ups/io/enc_grid.py`

### Configuration

```python
@dataclass
class GridEncoderConfig:
    latent_len: int                          # Target token count
    latent_dim: int                          # Features per token
    field_channels: Mapping[str, int]        # Field name → channel count
    patch_size: int = 4                      # Spatial grouping
    stem_width: Optional[int] = None         # Conv width (auto if None)
    use_fourier_features: bool = True        # Positional encoding
    fourier_frequencies: Tuple[float, ...] = (1.0, 2.0)
```

### Encoding Pipeline

```
Physical Grid Fields (e.g., Burgers: 128×128, 2 fields)
│
├─ Step 1: PixelUnshuffle (Patch Grouping)
│  ├─ patch_size: 4 → groups 4×4 spatial pixels
│  ├─ Input: (B, channels, H, W)
│  └─ Output: (B, channels×16, H/4, W/4)
│
├─ Step 2: Per-Field Residual Convolutions
│  ├─ Field-specific CNN stems
│  ├─ Structure: Conv2d(3×3) → GELU → Conv2d(1×1)
│  ├─ Zero-init last layer for identity residual
│  └─ One stem per field (e.g., separate for ρ, e)
│
├─ Step 3: Fourier Positional Features (Optional)
│  ├─ Sinusoidal encoding of patch centers
│  ├─ Frequencies: (1.0, 2.0, ...) × 2π
│  └─ Adds spatial awareness without learnable parameters
│
├─ Step 4: Token Projection
│  ├─ Flatten spatial dims: (B, channels, H', W') → (B, tokens, C)
│  └─ Project features: C → latent_dim via Linear or Identity
│
├─ Step 5: Adaptive Token Pooling (if needed)
│  ├─ If num_tokens_generated ≠ latent_len
│  └─ Use F.adaptive_avg_pool1d for token count adjustment
│
└─ Output: Latent Tokens (B, latent_len, latent_dim)
```

### Implementation Details

#### PixelUnshuffle-based Encoding

```python
class GridEncoder(nn.Module):
    def __init__(self, cfg: GridEncoderConfig):
        self.pixel_unshuffle = nn.PixelUnshuffle(cfg.patch_size)
        self.per_field_stems = nn.ModuleDict()
        
        # Create a residual stem for each field
        for name, channels in cfg.field_channels.items():
            in_ch = channels * patch_size * patch_size  # e.g., 1×16=16
            stem = nn.Sequential(
                nn.Conv2d(in_ch, width, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(width, in_ch, kernel_size=1),  # Zero-init for identity
            )
            self.per_field_stems[name] = stem
        
        # Projection to latent space
        if cfg.latent_dim == self._patch_channel_total:
            self.to_latent = nn.Identity()  # Lossless when dims match
        else:
            self.to_latent = nn.LazyLinear(cfg.latent_dim)
```

**Why PixelUnshuffle?**
- Groups spatial patches into channel dimension
- Preserves spatial structure while reducing dimensionality
- Invertible: PixelShuffle reconstructs original layout
- Efficient: Single operation instead of convolutions

#### Token Generation Formula

For a grid of size `(H, W)` with patch size `P`:

```
num_tokens = (H / P) × (W / P)
```

**Examples**:
- Grid 128×128, patch_size=4 → 32×32 = 1,024 tokens
- Grid 64×64, patch_size=4 → 16×16 = 256 tokens
- Grid 32×32, patch_size=4 → 8×8 = 64 tokens

If `num_tokens ≠ target_tokens` (latent_len):
- Use `F.adaptive_avg_pool1d` to pool/upsample
- Example: 1,024 tokens → 256 tokens via adaptive pooling

#### Fourier Features (Positional Encoding)

```python
def _fourier_features(self, coords, grid_shape, batch):
    # Compute patch center coordinates
    patch_centers = coords.view(batch, Hp, patch_size, Wp, patch_size, coord_dim)
    patch_centers = patch_centers.mean(dim=(2, 4))  # Average within patches
    
    # Sinusoidal encoding at each frequency
    for freq in self.fourier_frequencies:
        angles = 2π × patch_centers × freq
        features = cat([cos(angles), sin(angles)])  # Both sine and cosine
```

**Purpose**: Encodes spatial location information without learning
- Frequency (1.0, 2.0, 4.0, 8.0) captures multi-scale positions
- Allows decoder to decode at arbitrary query points
- Invariant to grid discretization

---

## 3. Mesh/Particle Encoder

**File**: `/Users/emerygunselman/Code/universal_simulator/src/ups/io/enc_mesh_particle.py`

For unstructured mesh or particle systems:

```python
@dataclass
class MeshParticleEncoderConfig:
    latent_len: int                  # Target token count
    latent_dim: int                  # Features per token
    hidden_dim: int                  # Internal representation
    message_passing_steps: int = 3   # Graph convolution iterations
    supernodes: int = 2048           # Intermediate pooling nodes
    use_coords: bool = True          # Include particle positions
```

### Architecture

```
Node Features + Connectivity
│
├─ Flatten all fields into feature vectors
├─ Message passing (graph convolution)
│  └─ Iteratively aggregate neighbor information
├─ Supernode pooling
│  └─ Pool to 2048 intermediate nodes
├─ Latent projection
│  └─ Project to (latent_len, latent_dim)
│
└─ Output: Latent Tokens
```

**Key Difference from Grid Encoder**: Uses graph structure instead of spatial grid
- Supports irregular particle distributions
- Handles deformable meshes
- Message passing aggregates local neighbor info

---

## 4. AnyPoint Decoder (Query-based Decoding)

### Purpose

Decode latent tokens at **arbitrary spatial query coordinates** - discretization-agnostic output.

**File**: `/Users/emerygunselman/Code/universal_simulator/src/ups/io/decoder_anypoint.py`

### Configuration

```python
@dataclass
class AnyPointDecoderConfig:
    latent_dim: int                          # Input latent dimension
    query_dim: int                           # Coordinate dimension (2 for 2D)
    hidden_dim: int = 128                    # Internal projection
    num_layers: int = 2                      # Cross-attention layers
    num_heads: int = 4                       # Attention heads
    frequencies: Tuple[float, ...] = (1.0, 2.0, 4.0)
    output_channels: Mapping[str, int]       # Field name → output channels
```

### Architecture

```
Latent Tokens (from operator)
│
├─ input_proj: (latent_dim) → hidden_dim
│
Query Points (arbitrary coordinates)
│
├─ fourier_encode: coordinates → sinusoidal features
├─ query_embed: features → hidden_dim
│
├─ Cross-Attention Layers (num_layers times):
│  ├─ MultiheadAttention(queries attend to latents)
│  ├─ LayerNorm + Residual
│  └─ MLP + LayerNorm + Residual
│
├─ Output Heads (per-field)
│  ├─ Field 'rho': MLP → out_channels['rho']
│  ├─ Field 'e': MLP → out_channels['e']
│  └─ ... (one head per field)
│
└─ Predictions at Query Points
```

### Implementation

```python
class AnyPointDecoder(nn.Module):
    def forward(self, points, latent_tokens, conditioning=None):
        """
        points: (batch, num_query_points, query_dim)
        latent_tokens: (batch, num_tokens, latent_dim)
        
        Returns: Dict[field_name] → (batch, num_query_points, out_channels)
        """
        # Fourier encode query coordinates
        enriched = _fourier_encode(points, self.cfg.frequencies)
        queries = self.query_embed(enriched)
        
        # Cross-attend to latent tokens
        latents = self.latent_proj(latent_tokens)
        for attn, ln_q, ff, ln_ff in self.layers:
            attn_out, _ = attn(queries, latents, latents)
            queries = ln_q(queries + attn_out)
            ff_out = ff(queries)
            queries = ln_ff(queries + ff_out)
        
        # Decode field values
        outputs = {}
        for field_name, head in self.heads.items():
            outputs[field_name] = head(queries)
        return outputs
```

### Key Properties

1. **Discretization-Agnostic**: Query at any resolution
   - Training: Fixed grid (e.g., 64×64)
   - Inference: Arbitrary grid (32×32, 512×512, non-uniform, etc.)

2. **Continuous Decoding**: Fourier features allow smooth interpolation
   - Not restricted to grid points
   - Can decode at sub-pixel coordinates

3. **Perceiver Pattern**: Queries ↔ Latents cross-attention
   - Queries learn to extract relevant information from tokens
   - Scalable: O(queries × tokens) instead of O(queries²)

---

## 5. Latent Token Configuration Ablations

### Study: Token Count Effects (64, 128, 256)

**Purpose**: Understand how token count vs. latent dimension affects model capacity and performance.

#### Configuration 1: 64 Tokens

**File**: `/Users/emerygunselman/Code/universal_simulator/configs/ablation_upt_64tokens.yaml`

```yaml
latent:
  dim: 64
  tokens: 64

operator:
  pdet:
    input_dim: 64
    hidden_dim: 128
    depths: [2, 2, 2]
    group_size: 16          # 128 / 16 = 8 attention heads
    num_heads: 4

training:
  batch_size: 8
  accum_steps: 6
  lambda_spectral: 0.05
  
stages:
  operator:
    epochs: 25
    lr: 6.0e-4
```

**Characteristics**:
- Smallest configuration
- 64×64 = 4,096 latent features total
- Lower memory, faster training
- May lose spatial details (64 tokens ≈ 8×8 spatial resolution)

#### Configuration 2: 128 Tokens

**File**: `/Users/emerygunselman/Code/universal_simulator/configs/ablation_upt_128tokens.yaml`

```yaml
latent:
  dim: 128
  tokens: 128

operator:
  pdet:
    input_dim: 128
    hidden_dim: 256
    depths: [3, 3, 2]
    group_size: 32          # 256 / 32 = 8 attention heads
    num_heads: 8

training:
  batch_size: 6
  accum_steps: 6
  
stages:
  operator:
    epochs: 25
    lr: 5.0e-4
```

**Characteristics**:
- Mid-range configuration
- 128×128 = 16,384 latent features total
- ~2× memory vs 64-token
- Better spatial resolution (128 ≈ 11×11)
- More expressive transformer (256 hidden_dim)

#### Configuration 3: 256 Tokens (UPT-17M scale)

**File**: `/Users/emerygunselman/Code/universal_simulator/configs/ablation_upt_256tokens.yaml`

```yaml
latent:
  dim: 192
  tokens: 256

operator:
  pdet:
    input_dim: 192
    hidden_dim: 384
    depths: [4, 4, 4]
    group_size: 32          # 384 / 32 = 12 attention heads
    num_heads: 8

training:
  batch_size: 4
  accum_steps: 8
  
stages:
  operator:
    epochs: 30
    lr: 8.0e-4
```

**Characteristics**:
- Largest configuration
- 256×192 = 49,152 latent features total
- ~12× memory vs 64-token
- Finest spatial resolution (256 ≈ 16×16)
- Deepest transformer (4 layers per U-Net scale)
- Deterministic training for reproducibility

### Comparison Table

| Metric | 64-Token | 128-Token | 256-Token |
|--------|----------|-----------|-----------|
| Latent dim | 64 | 128 | 192 |
| Token count | 64 | 128 | 256 |
| Total features | 4,096 | 16,384 | 49,152 |
| Hidden dim | 128 | 256 | 384 |
| Depths | [2,2,2] | [3,3,2] | [4,4,4] |
| Batch size | 8 | 6 | 4 |
| Group size | 16 | 32 | 32 |
| Num heads | 4 | 8 | 8 |
| Approx. spatial res | 8×8 | 11×11 | 16×16 |
| Training epochs | 25 | 25 | 30 |
| Learning rate | 6.0e-4 | 5.0e-4 | 8.0e-4 |

### Token Count Effects on Architecture

**Constraint**: `group_size` must divide `hidden_dim` evenly

```
128-Token Config:
  hidden_dim = 256
  group_size = 32
  num_heads per group = 256 / 32 / 8 = 1 head per group ✓

256-Token Config:
  hidden_dim = 384
  group_size = 32
  num_heads per group = 384 / 32 / 8 = 1.5 ✗ INVALID!
  
  Fix: Reduce num_heads or increase group_size
  Valid options:
    - num_heads = 6: 384 / 32 / 6 = 2 ✓
    - group_size = 48: 384 / 48 / 8 = 1 ✓
```

**Configuration 256 uses**: `group_size=32, num_heads=8`
- Implies: `head_dim = 384 / 8 = 48` (when split into num_heads)
- Channel separation: 384 / 32 = 12 groups
- Each group: 32 features, 8 heads → head_dim = 4

---

## 6. Golden Configuration (Production)

**File**: `/Users/emerygunselman/Code/universal_simulator/configs/train_burgers_golden.yaml`

```yaml
latent:
  dim: 16
  tokens: 32

operator:
  pdet:
    input_dim: 16
    hidden_dim: 96
    depths: [1, 1, 1]
    group_size: 12          # 96 / 12 = 8 groups
    num_heads: 6            # 12 / 6 = 2 per head

diffusion:
  latent_dim: 16
  hidden_dim: 96

ttc:
  decoder:
    latent_dim: 16
    query_dim: 2
    hidden_dim: 96
    num_heads: 4
    num_layers: 3
```

**Philosophy**: Minimal viable architecture
- **16 latent features** per token
- **32 tokens** = ~5×6 spatial resolution
- **Total**: 512 latent features (vs 4,096 for 64-token)
- **Benefit**: 8× memory reduction with comparable performance

**Key Insight**: Efficiency matters more than raw capacity
- Well-designed operator can work with minimal tokens
- Diffusion + TTC compensate for limited latent capacity
- Training faster: 14.5 min vs hours for larger configs

---

## 7. Dimension Consistency Requirements

### Critical Constraints

All dimension parameters must satisfy strict divisibility:

#### Constraint 1: Latent Dimension Consistency

```python
# All of these MUST be equal:
latent.dim == operator.pdet.input_dim == diffusion.latent_dim == ttc.decoder.latent_dim

# Example:
latent.dim = 16
operator.pdet.input_dim = 16  ✓
diffusion.latent_dim = 16     ✓
ttc.decoder.latent_dim = 16   ✓
```

#### Constraint 2: Channel Separation Divisibility

```python
# In ChannelSeparatedSelfAttention:
hidden_dim % group_size == 0
group_size % num_heads == 0

# Example:
hidden_dim = 96
group_size = 12    # 96 / 12 = 8 ✓
num_heads = 6      # 12 / 6 = 2 ✓
head_dim = 12 / 6 = 2
```

#### Constraint 3: Decoder Dimension Matching

```python
# Cross-attention between queries and latents:
latent_tokens: (batch, tokens, latent_dim)
queries: (batch, num_points, hidden_dim_decoder)

# Both must project to same hidden_dim for attention:
latents = latent_proj(latent_tokens)  # → (batch, tokens, hidden_dim)
queries = query_embed(enriched_points)  # → (batch, points, hidden_dim)
```

### Validation in Code

```python
# latent_operator.py
class LatentOperator(nn.Module):
    def __init__(self, cfg: LatentOperatorConfig):
        if cfg.pdet.input_dim != cfg.latent_dim:
            raise ValueError("PDETransformer input_dim must match latent_dim")

# blocks_pdet.py
class ChannelSeparatedSelfAttention(nn.Module):
    def __init__(self, dim, group_size, num_heads):
        if dim % group_size != 0:
            raise ValueError("dim must be divisible by group_size")
        if group_size % num_heads != 0:
            raise ValueError("group_size must be divisible by num_heads")
```

---

## 8. Encoder/Decoder Data Flow

### Complete Pipeline Example (Burgers1D)

#### Forward (Training)

```
Input: Burgers fields (ρ, e) on 128×128 grid

1. ENCODING
   Grid: (batch=12, 128×128 spatial, 2 fields)
   ↓ GridEncoder
   - PixelUnshuffle (patch_size=4): 128×128 → 32×32 patches
   - Per-field stems (Conv2d residual)
   - Fourier features (frequencies: 1.0, 2.0)
   - Token projection: features → 16-dim
   - Output: (batch=12, tokens=32, dim=16)

2. EVOLUTION (Latent Operator)
   Input: LatentState(z=(12, 32, 16), t=0.0, cond={})
   ↓ LatentOperator
   - TimeEmbedding(dt=0.1): scalar → 64-dim embedding
   - time_to_latent: → 16-dim
   - PDETransformerBlock (U-net): (12, 32, 16) → (12, 32, 16)
   - Add residual: z_new = z_old + residual
   - Output: LatentState(z=(12, 32, 16), t=0.1, cond={})

3. DECODING
   Latent tokens: (12, 32, 16)
   ↓ AnyPointDecoder
   - Query points (test grid or training points)
   - Fourier encode query coords
   - Cross-attention: queries ↔ latent_tokens
   - Field heads: predict ρ, e at query points
   - Output: (batch=12, num_points, channels)

4. LOSS COMPUTATION
   Predicted fields vs. ground truth fields
   Loss backprop through entire pipeline
```

#### Inference (Test-time Conditioning)

```
Input: Initial state + physics rewards

1. Multiple Rollouts (candidates=8)
   - Parallel latent operator steps
   - Each rollout: initial → diffusion sampling → latent sequence

2. Per-Candidate Decoding
   For each candidate:
   - AnyPointDecoder(latent_tokens)
   - Evaluate on test grid (e.g., 512×512)
   - Compute conservation metrics (mass, energy)

3. Physics-based Selection
   - Score each candidate by conservation errors
   - Beam search to find best trajectory
   - Return highest-scoring predictions
```

---

## 9. Shifted Window Attention (Infrastructure)

**File**: `/Users/emerygunselman/Code/universal_simulator/src/ups/core/shifted_window.py`

### Purpose

Partition spatial grids into local windows for efficient attention (Swin Transformer pattern).

**Status**: Implemented but not actively used in PDETransformerBlock

### API

```python
def partition_windows(tensor: torch.Tensor, window_size=(8, 8), 
                     shift_size=(0, 0)) -> (windows, info):
    """
    tensor: (batch, height, width, channels)
    Returns: (windows, WindowPartitionInfo)
      windows: (batch*num_windows, window_area, channels)
      info: metadata for reconstruction
    """
    # Cyclic shift (optional)
    if shift_size != (0, 0):
        tensor = torch.roll(tensor, shifts=(-shift_size[0], -shift_size[1]))
    
    # Partition into fixed-size windows
    # Reshape and flatten: (B, H, W, C) → (B*NW, window_area, C)
```

### Log-Spaced Relative Position Bias

```python
class LogSpacedRelativePositionBias(nn.Module):
    """Per-head relative position bias with log-spaced scaling."""
    
    def __init__(self, window_size=(8, 8), num_heads=4):
        # Precompute signed log distances between all pairs
        coords_flat = coords.view(-1, 2)  # (window_area, 2)
        rel = coords_flat[:, None, :] - coords_flat[None, :, :]  # (N, N, 2)
        log_rel = torch.sign(rel) * torch.log1p(rel.abs())
        
        # Learnable head-specific weights
        self.weight = nn.Parameter(torch.zeros(num_heads, 2))
        self.bias = nn.Parameter(torch.zeros(num_heads, 1, 1))
```

**Benefits**:
- Preserves relative position information
- Log scaling compresses large distances
- Efficient: O(window²) instead of O(global²)

### Future Use Case

Could replace or augment token pooling in PDETransformerBlock:
- Windowed attention for grid-shaped latents
- Hybrid: Coarse pooling + fine window attention
- Preserve local structure while managing complexity

---

## 10. Adaptive Layer Normalization (AdaLN) Conditioning

**File**: `/Users/emerygunselman/Code/universal_simulator/src/ups/core/conditioning.py`

### Purpose

Modulate latent tokens based on physics-aware conditioning signals (optional).

### Configuration

```python
@dataclass
class ConditioningConfig:
    latent_dim: int              # Dimension of modulated features
    hidden_dim: int = 128        # Internal embedder width
    sources: Mapping[str, int]   # Condition name → input dimension
```

### Architecture

```python
class AdaLNConditioner(nn.Module):
    def __init__(self, cfg: ConditioningConfig):
        # Per-condition embedder (MLP)
        for name, in_dim in cfg.sources.items():
            embed = nn.Sequential(
                nn.Linear(in_dim, cfg.hidden_dim),
                nn.GELU(),
                nn.Linear(cfg.hidden_dim, latent_dim * 3),  # scale, shift, gate
            )
            nn.init.zeros_(embed[-1].weight)  # Start from identity
```

### Forward Pass

```python
def forward(self, normed_tokens, cond: Dict[str, torch.Tensor]):
    """
    normed_tokens: (batch, tokens, latent_dim)
    cond: {condition_name: (batch, input_dim), ...}
    
    Returns: modulated_tokens (batch, tokens, latent_dim)
    """
    # Sum embeddings from all conditions
    total = None
    for name, embed in self.embedders.items():
        if name in cond:
            contrib = embed(cond[name])  # → (batch, latent_dim * 3)
            total = contrib if total is None else total + contrib
    
    # Decompose into scale, shift, gate
    gamma, beta, gate_raw = total.chunk(3, dim=-1)
    scale = 1.0 + gamma          # Scale factor
    shift = beta                 # Shift
    gate = sigmoid(gate_raw + 2.0)  # Gating (≈0.88 initially)
    
    # Apply modulation
    return gate * (scale * normed_tokens + shift)
```

### Physics-Aware Conditioning Examples

Possible conditioning signals:
- **Reynolds number** (viscosity for NS)
- **Initial energy** (energy conservation constraint)
- **Boundary conditions** (walls, free-slip)
- **Material parameters** (density, viscosity)

**Current Usage**: Enabled in configs but typically set to `conditioning: None` (unused)

---

## 11. Memory and Computational Analysis

### Parameter Count

For golden config:
- Latent dim: 16
- Token count: 32
- Total latent features: 512

#### Operator Parameters

```
LatentOperator:
  - time_embed: 1→64→64 = 4,160
  - time_to_latent: 64→16 = 1,040
  - PDETransformerBlock:
    - input_proj: 16→96 = 1,536
    - down_layers: ~50K (depends on depths)
    - bottleneck: ~20K
    - up_layers: ~50K
    - output_proj: 96→16 = 1,536
  Total: ~130K parameters
```

#### Encoder Parameters

```
GridEncoder:
  - Per-field stems: ~200 each (3 convs)
  - to_latent: features→16 = variable
  - from_latent: 16→features = variable
  Total: ~1K-5K (lightweight)
```

#### Decoder Parameters

```
AnyPointDecoder:
  - query_embed: (~30)→96 = ~3K
  - latent_proj: 16→96 = 1,536
  - Cross-attn layers (3×): ~50K
  - Field heads (2): ~10K
  Total: ~65K parameters
```

**Total Model**: ~200K parameters (very compact!)

### Memory Usage (Training)

For batch_size=12, tokens=32, latent_dim=16:

```
Forward activations:
  - Latent tokens: 12×32×16×4 = 24 KB
  - Hidden (96-dim): 12×32×96×4 = 147 KB
  - Attention matrices: O(tokens²) = 4K elements
  Total: ~500 KB activations

Backward (gradients):
  - Same as forward: ~500 KB

Peak memory: ~50 MB (including optimizer states, batch buffer)
```

Compare to 256-token config:
- Activations: 12× larger (from 500 KB → 6 MB)
- Gradients: 12× larger
- Peak memory: ~200-300 MB

---

## 12. Summary: Configuration Impact on Model

### Effect of Token Count (64, 128, 256)

| Aspect | 64 Tokens | 128 Tokens | 256 Tokens |
|--------|-----------|------------|------------|
| **Spatial Resolution** | ~8×8 patches | ~11×11 | ~16×16 |
| **Capacity** | 4K features | 16K | 49K |
| **Training Time** | Fast | Medium | Slow |
| **Memory** | Low | Medium | High |
| **Expressivity** | Coarse | Detailed | Very detailed |
| **Physics Detail** | Averaged | Good | Excellent |
| **Generalization** | ✓ Good | ✓ Good | ? Risk of overfit |

### Effect of Latent Dimension (16, 64, 128)

| Aspect | 16-dim | 64-dim | 128-dim |
|--------|--------|--------|----------|
| **Per-Token Features** | Few | Many | Very many |
| **Compression Ratio** | Very high | Medium | Low |
| **Reconstruction Quality** | Lower | Better | Near-lossless |
| **Operator Complexity** | Simple | Complex | Very complex |
| **Diffusion Modeling** | Difficult | Easier | Easiest |

### Design Philosophy

**Golden Config** (16-dim, 32 tokens): Minimal viable
- Compress to essential physics features
- Let diffusion model handle uncertainty
- Use TTC rewards for refinement
- Result: SOTA performance with 8× speedup

**256-Token Config**: Maximum capacity
- Preserve all spatial details
- Better for complex phenomena (shock formation)
- Risk: Overfitting to training resolution
- Use case: Research on discretization independence

---

## 13. Key Files Reference

| File | Purpose |
|------|---------|
| `src/ups/core/latent_state.py` | LatentState data structure |
| `src/ups/core/blocks_pdet.py` | PDE-Transformer U-net architecture |
| `src/ups/core/conditioning.py` | AdaLN conditioning mechanism |
| `src/ups/core/shifted_window.py` | Window partitioning utilities |
| `src/ups/io/enc_grid.py` | Grid encoder (patch-based) |
| `src/ups/io/enc_mesh_particle.py` | Mesh/particle encoder (GNN-based) |
| `src/ups/io/decoder_anypoint.py` | Query-based decoder (Perceiver-style) |
| `src/ups/models/latent_operator.py` | Latent evolution operator |
| `configs/train_burgers_golden.yaml` | Production config (16-dim, 32 tokens) |
| `configs/ablation_upt_64tokens.yaml` | Ablation: 64 tokens |
| `configs/ablation_upt_128tokens.yaml` | Ablation: 128 tokens |
| `configs/ablation_upt_256tokens.yaml` | Ablation: 256 tokens (UPT-17M) |

---

## 14. Next Steps: Extending the Architecture

### Possible Enhancements

1. **Adaptive Token Count**: Learn optimal tokens per grid
2. **Hierarchical Latents**: Multi-scale token representations
3. **Physics Encoders**: Embed conservation laws in latent space
4. **Windowed Attention**: Integrate shifted_window for large grids
5. **Multi-Phase Latents**: Separate representations for different physics modes

### Known Limitations

1. **Fixed Token Count**: Cannot adapt to variable-size inputs
   - Workaround: Adaptive pooling (current)
   - Better: Dynamic token allocation

2. **Discretization Coupling**: Encoder depends on patch_size
   - Fourier features help decouple
   - Future: Fourier-only encoding (no patches)

3. **Error Accumulation**: Long-horizon predictions drift
   - Addressed by: Diffusion residual + TTC
   - Research: Predict uncertainties directly

