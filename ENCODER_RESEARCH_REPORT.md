# Encoder Architecture Research Report

**Research Date**: November 5, 2025  
**Scope**: Quick analysis of CFD-relevant encoder patterns in Universal Simulator  
**Focus Areas**: GNN capabilities, mesh handling, pooling strategies, CFD gaps

---

## Executive Summary

The Universal Simulator has two main encoders:

1. **GridEncoder** (enc_grid.py): For structured/regular grids
   - Simple, fast, pixel-unshuffle patch extraction
   - Non-learned pooling (F.adaptive_avg_pool1d)
   - Fourier coordinate features
   - CFD Limitation: Cannot handle irregular meshes

2. **MeshParticleEncoder** (enc_mesh_particle.py): For unstructured meshes/particles
   - Graph-aware with message passing (3 steps default)
   - Two-stage pooling (supernode + perceiver)
   - Coordinate embedding support
   - CFD Limitation: Pooling is non-learned, no boundary condition awareness

**Key Finding**: Current pooling strategies are purely geometric/averaged. No learned or physics-aware pooling exists.

---

## 1. ENCODER ARCHITECTURES

### A. GridEncoder (enc_grid.py, 232 lines)

**Architecture Flow**:
```
Input Fields (B, H, W, C)
         ↓
  PixelUnshuffle (patch_size=4)
         ↓
  Per-field residual stems (Conv2d)
         ↓
  Fourier feature augmentation (optional)
         ↓
  Flatten to tokens (B, H*W/patch², D)
         ↓
  Adaptive token pooling to target length
         ↓
Output latent tokens (B, latent_len, latent_dim)
```

**File Line References**:

| Operation | File:Lines | Details |
|-----------|-----------|---------|
| Config | enc_grid.py:11-19 | GridEncoderConfig dataclass |
| Init | enc_grid.py:34-76 | Module setup |
| Forward | enc_grid.py:81-107 | Main entry point |
| Patch Extraction | enc_grid.py:43-44 | PixelUnshuffle/Shuffle |
| Per-field Stems | enc_grid.py:46-59 | Residual Conv2d stems |
| Fourier Features | enc_grid.py:61-66 | Frequency buffer setup |
| Field Encoding | enc_grid.py:139-181 | _encode_fields() |
| Fourier Computation | enc_grid.py:183-214 | _fourier_features() |
| Token Pooling | enc_grid.py:216-221 | _adaptive_token_pool() |

**Pooling Details** (enc_grid.py:216-221):
```python
def _adaptive_token_pool(self, tokens: Tensor, target_len: int) -> Tensor:
    if tokens.shape[1] == target_len:
        return tokens
    tokens_t = tokens.transpose(1, 2)           # (B, D, T)
    pooled = F.adaptive_avg_pool1d(tokens_t, target_len)
    return pooled.transpose(1, 2)               # (B, target, D)
```

**Characteristics**:
- Non-learned average pooling
- No spatial awareness
- Simple, efficient, deterministic

**CFD Capabilities**:
- Works for 1D Burgers (reshaped to 2D)
- Works for 2D shallow water
- Limited to regular grids

**CFD Limitations**:
- No irregular mesh support
- No boundary condition encoding
- No physics-aware token selection

---

### B. MeshParticleEncoder (enc_mesh_particle.py, 171 lines)

**Architecture Flow**:
```
Input Fields & Mesh (B, N, C) + coords (N, D) + edges (E, 2)
         ↓
  Flatten fields + concatenate coordinates
         ↓
  Project to hidden dimension (Lazy Linear)
         ↓
  Message Passing Loop (default 3 steps):
    - Aggregate neighbor features (index_add_)
    - Degree normalize (mean aggregation)
    - Learnable linear + GELU activation
    - Residual connection
         ↓
  Stage 1: Supernode pooling (N → S, chunk-based)
         ↓
  Stage 2: Perceiver pooling (S → latent_len, adaptive avg)
         ↓
Output latent tokens (B, latent_len, latent_dim)
```

**File Line References**:

| Operation | File:Lines | Details |
|-----------|-----------|---------|
| Config | enc_mesh_particle.py:11-18 | MeshParticleEncoderConfig |
| Init | enc_mesh_particle.py:58-86 | Module setup |
| Field Flattening | enc_mesh_particle.py:21-36 | _flatten_fields() |
| Adjacency Building | enc_mesh_particle.py:39-48 | _build_adjacency() |
| Forward | enc_mesh_particle.py:87-141 | Main forward pass |
| Feature Setup | enc_mesh_particle.py:95-107 | Field + coord concatenation |
| Message Passing | enc_mesh_particle.py:114-123 | GNN convolution loop |
| Supernode Pool | enc_mesh_particle.py:155-163 | _pool_supernodes() |
| Perceiver Pool | enc_mesh_particle.py:165-170 | _perceiver_pool() |
| Reconstruction | enc_mesh_particle.py:143-153 | reconstruct() |

**Message Passing Details** (enc_mesh_particle.py:114-123):
```python
for layer in self.message_layers:
    m = torch.zeros_like(h)
    m.index_add_(1, dst_idx, h[:, src_idx, :])     # Gather neighbor features
    
    deg = torch.zeros(N, device=device)
    deg.index_add_(0, dst_idx, torch.ones_like(...))
    deg = deg.clamp_min_(1.0).view(1, N, 1)
    m = m / deg                                      # Average (not sum)
    
    m = layer(m)                                     # Learnable Linear
    h = h + F.gelu(m)                               # Residual connection
```

**Characteristics**:
- Degree-normalized aggregation (proper averaging)
- Learnable per-step transformations
- Residual connections
- Undirected edges (symmetric adjacency)

**Supernode Pooling Details** (enc_mesh_particle.py:155-163):
```python
def _pool_supernodes(self, tokens: Tensor, supernodes: int) -> Tensor:
    B, N, D = tokens.shape
    chunk = (N + supernodes - 1) // supernodes
    pad = chunk * supernodes - N
    
    if pad > 0:
        pad_tensor = tokens[:, :pad, :]
        tokens = torch.cat([tokens, pad_tensor], dim=1)
    
    tokens = tokens.view(B, supernodes, chunk, D).mean(dim=2)
    return tokens
```

**Example**: 1000 nodes → 256 supernodes
- chunk = 4, pad = 24
- Reshape (B, 1024, D) → (B, 256, 4, D)
- Mean over chunks → (B, 256, D)

**CFD Capabilities**:
- Graph-aware message passing
- Handles irregular meshes
- Coordinate embedding
- Mesh connectivity preservation (during message passing)

**CFD Limitations**:
- Non-learned pooling (geometric only)
- No boundary condition encoding
- No physics-aware importance weighting
- Supernode pooling breaks spatial coherence

---

### C. AnyPointDecoder (decoder_anypoint.py, 137 lines)

**Architecture Flow**:
```
Latent tokens (B, T, latent_dim) + query points (B, Q, query_dim)
         ↓
  Fourier encode query points
         ↓
  Project queries & latents to hidden dimension
         ↓
  Cross-Attention Layers (default 2):
    - Queries attend to latent tokens
    - Residual connections
    - Feed-forward networks
         ↓
  Per-field prediction heads
         ↓
Output field values at queries (B, Q, channels_per_field)
```

**File Line References**:

| Operation | File:Lines | Details |
|-----------|-----------|---------|
| Config | decoder_anypoint.py:35-50 | AnyPointDecoderConfig |
| Init | decoder_anypoint.py:63-87 | Module setup |
| Forward | decoder_anypoint.py:89-134 | Main forward pass |
| Fourier Encode | decoder_anypoint.py:11-32 | _fourier_encode() |

**Characteristics**:
- Perceiver-IO inspired (but simplified)
- Cross-attention based query-latent interaction
- Query Fourier embedding for positional information
- Per-field output heads

---

## 2. GNN AND GRAPH-BASED POOLING

### Currently Implemented

**Message Passing** (MeshParticleEncoder):
- Basic node feature aggregation via `index_add_`
- Degree normalization for averaging
- Learnable linear per layer
- Residual connections
- Undirected edge support

**Pooling**:
- GridEncoder: Simple adaptive average (F.adaptive_avg_pool1d)
- MeshParticleEncoder: Two-stage (chunk-based + adaptive average)
- No learnable pooling operators
- No importance weighting

### NOT Implemented (Gaps)

1. **Learned Pooling Algorithms**:
   - TopKPool (learned k)
   - DiffPool (learned soft assignment matrices)
   - SAGPool (self-attention based selection)
   - MinCutPool (graph structure preserving)

2. **Graph Attention**:
   - GAT-style multi-head attention over edges
   - Edge feature propagation

3. **Hierarchical Coarsening**:
   - Multi-level graph coarsening
   - Preserving important nodes at each level

4. **Physics-Aware Operations**:
   - Boundary-aware aggregation
   - Conservation-preserving pooling
   - Mesh quality aware operations

---

## 3. MESH/PARTICLE HANDLING FOR IRREGULAR GRIDS

### MeshParticleEncoder Approach

**Field Concatenation** (_flatten_fields, lines 21-36):
- Handles variable input dimensions (2D, 3D, 4D)
- 2D: (N, C) → add batch dim
- 3D: (B, N, C) → use as-is
- 4D: (T, B, N, C) → average over T
- Concatenates all fields along feature dimension

**Adjacency Building** (_build_adjacency, lines 39-48):
```python
def _build_adjacency(num_nodes, edges, device):
    if edges.numel() == 0:
        # No connectivity - use self-loops
        return torch.arange(num_nodes, device=device), torch.arange(num_nodes, device=device)
    
    src = edges[:, 0].to(torch.long)
    dst = edges[:, 1].to(torch.long)
    
    # Make undirected (bidirectional edges)
    undirected_src = torch.cat([src, dst])
    undirected_dst = torch.cat([dst, src])
    return undirected_src, undirected_dst
```

**Coordinate Integration** (forward, lines 99-103):
- Appends coordinates to feature vector when use_coords=True
- Allows encoder to learn spatial patterns

### Limitations

- No time-varying mesh connectivity
- No multi-resolution mesh handling
- No mesh quality metrics (aspect ratio, skewness)
- No boundary layer detection
- No edge type information (wall, inlet, outlet, periodic)
- Reconstruction only works when latent_len == node_count

---

## 4. PERCEIVER-BASED POOLING PATTERNS

### Current Implementation

Both GridEncoder and MeshParticleEncoder use identical perceiver pooling:

```python
def _perceiver_pool(self, tokens, target_len):
    tokens_t = tokens.transpose(1, 2)
    pooled = F.adaptive_avg_pool1d(tokens_t, target_len)
    return pooled.transpose(1, 2)
```

This is NOT true Perceiver-IO. It's just adaptive average pooling.

### True Perceiver-IO Features (NOT Implemented)

1. **Learned Latent Vectors**: Fixed learnable basis vectors
2. **Cross-Attention**: Queries attend to latents for selection
3. **Soft Assignment**: Probabilistic token selection vs hard bins
4. **Interpretability**: Can see which tokens are selected
5. **Learnable Pooling**: Importance weighting per dimension

### Where to Add

- File: `/Users/emerygunselman/Code/universal_simulator/src/ups/io/enc_mesh_particle.py`
- Location: After line 170 (new method) or lines 125-131 (replace supernode pooling)
- Pattern: Use learned latent vectors with cross-attention

---

## 5. CFD-SPECIFIC FEATURES AND LIMITATIONS

### Existing CFD Support

| Capability | GridEncoder | MeshParticleEncoder | Notes |
|-----------|------------|-------------------|-------|
| Structured grids | Yes | No | Regular rectangular grids |
| Unstructured grids | No | Yes | Tetrahedral, hybrid meshes |
| Particle clouds | No | Yes | Point clouds, SPH |
| Message passing | No | Yes | 3 steps default, learnable |
| Fourier features | Yes | No (embedded) | Coordinate augmentation |
| Mesh topology | No | Yes (in MP) | Edge connectivity |
| Reconstruction | Yes | Yes (limited) | exact if tokens=nodes |

### Missing CFD-Specific Patterns

#### 1. Boundary Condition Encoding
**Problem**: No distinction between BC types
**Current**: Coordinates concatenated but BC type not encoded
**Gap Location**: enc_mesh_particle.py forward() line 103
**CFD Need**: Different treatment for:
- Dirichlet (fixed value)
- Neumann (fixed gradient)
- Robin (mixed)
- Periodic
- Slip/No-slip
**Impact**: Decoder lacks BC context

#### 2. Physics-Aware Pooling
**Problem**: Chunk-based supernode pooling has no physics awareness
**Current**: Reshape and average (geometric only)
**Gap Location**: enc_mesh_particle.py lines 155-163
**CFD Need**:
- Preserve high-gradient regions
- Maintain conservation laws during pooling
- Weighted pooling by local energy/enstrophy
**Impact**: Coarse tokens may lose important flow features

#### 3. Multi-Physics Integration
**Problem**: MultiphysicsFactorGraph exists but not used in encoders
**Current**: Single physics domain
**Gap Location**: enc_mesh_particle.py + multiphysics_factor_graph.py
**CFD Need**:
- Fluid-structure coupling
- Thermal-fluid coupling
- Preserve multi-domain structure in latent space
**Impact**: Cannot encode coupled physics

#### 4. Reynolds Number Awareness
**Problem**: No Re-dependent feature normalization
**Current**: Fixed normalization regardless of flow regime
**Gap Location**: enc_mesh_particle.py preprocessing
**CFD Need**:
- Different feature scales for laminar vs turbulent
- Re-dependent compression ratios
- Adaptive message passing depth
**Impact**: Poor performance across Re regimes

#### 5. Turbulence/Subscale Modeling
**Problem**: No subscale/eddy viscosity encoding
**Current**: Direct field encoding
**Gap Location**: New preprocessing module
**CFD Need**:
- Eddy viscosity features
- Turbulent kinetic energy
- Dissipation rate
**Impact**: Cannot represent turbulence effects

#### 6. Mesh Quality and Adaptation
**Problem**: No mesh refinement awareness
**Current**: All nodes treated equally
**Gap Location**: New config parameters, forward()
**CFD Need**:
- Cell aspect ratio awareness
- Refinement level tracking
- Boundary layer detection
- Local mesh density
**Impact**: Cannot exploit mesh structure

#### 7. Spectral Information
**Problem**: No frequency domain features
**Current**: Only spatial domain
**Gap Location**: New preprocessing
**CFD Need**:
- FFT-based features for grid encoders
- Spectral energy distribution
- Wavenumber-dependent information
**Impact**: Limited to spatial patterns

---

## 6. CURRENT SUPERNODE AND TOKEN POOLING STRATEGIES

### Supernode Pooling (MeshParticleEncoder, lines 155-163)

**Algorithm**:
```
1. Input: (B, N, D) tokens from N nodes
2. Calculate chunk size: chunk = ceil(N / S)
3. Pad to multiple: N_padded = chunk * S
4. Reshape: (B, N_padded, D) → (B, S, chunk, D)
5. Average: (B, S, D) via mean(dim=2)
```

**Characteristics**:
- Non-learned hard assignment
- Sequential/chunk-based (first nodes → supernode 1, etc.)
- No spatial or importance awareness
- Circular padding for uneven division

**CFD Implications**:
- May split important flow features across supernodes
- No clustering by similarity
- Breaks spatial coherence in irregular meshes
- Chunk boundaries arbitrary

**Example**: 1000 nodes → 256 supernodes
- Chunk = 4 nodes per supernode
- First 4 nodes average → supernode 1
- Nodes 5-8 average → supernode 2
- etc.

### Perceiver Pooling (MeshParticleEncoder, lines 165-170)

**Algorithm**:
```
1. Transpose: (B, T, D) → (B, D, T)
2. Adaptive pool: (B, D, T) → (B, D, target_len)
3. Transpose back: (B, target_len, D)
```

**Characteristics**:
- Uses PyTorch's F.adaptive_avg_pool1d
- Learned implicitly via bin assignment
- No attention-based selection
- Works best if tokens ordered by importance

**CFD Implications**:
- Second-stage compression (after supernodes)
- Further loses spatial structure
- No selectivity

---

## 7. WHERE GNN-BASED POOLING COULD BE ADDED

### Option 1: Replace Supernode Pooling (RECOMMENDED)

**File**: `/Users/emerygunselman/Code/universal_simulator/src/ups/io/enc_mesh_particle.py`  
**Location**: Lines 125-131, replace `_pool_supernodes` call  
**Approach**: TopKPool or learned selection  

**Implementation Pattern**:
```python
def _learned_pool_supernodes(self, tokens, edges, supernodes):
    # Compute node importance (energy, gradient magnitude, etc.)
    importance = self._compute_importance(tokens)  # (B, N)
    
    # TopKPool: select top-k important nodes
    k = min(supernodes, tokens.shape[1])
    _, topk_idx = torch.topk(importance, k, dim=1)
    
    # Gather selected nodes
    pooled = tokens.gather(1, topk_idx.unsqueeze(-1).expand_as(tokens))
    
    return pooled  # (B, k, D)
```

**Benefits**:
- Importance-aware selection
- Preserves important features
- Maintains spatial structure

**CFD-Specific Enhancement**:
```python
def _compute_importance(self, tokens):
    # Energy norm per node
    energy = (tokens ** 2).sum(dim=-1)
    
    # Or gradient magnitude
    grad = torch.abs(tokens[:, 1:] - tokens[:, :-1]).mean(dim=-1)
    
    # Combined importance
    importance = energy + grad
    return importance
```

### Option 2: Add Learned Perceiver Pooling

**File**: Same  
**Location**: New method after line 170  
**Approach**: Cross-attention with learned latent vectors  

**Benefits**:
- Interpretable token selection
- Attention weights show importance
- Better preservation of spatial patterns

**Implementation Pattern**:
```python
class LearnedPerceivePooling(nn.Module):
    def __init__(self, tokens, target_len, hidden_dim):
        self.learned_latents = nn.Parameter(torch.randn(target_len, hidden_dim))
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
    
    def forward(self, tokens):
        # tokens: (B, N, D)
        # Cross-attention: learned_latents attend to tokens
        latent_batch = self.learned_latents.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        pooled, attn_weights = self.attention(latent_batch, tokens, tokens)
        return pooled  # (B, target_len, D)
```

### Option 3: Multi-Head Message Passing

**File**: Same  
**Location**: Lines 64-123  
**Approach**: Separate heads per field type  

**CFD-Specific**:
- Head 1: Velocity aggregation
- Head 2: Pressure aggregation
- Head 3: Vorticity aggregation
- Combine heads for final representation

### Option 4: Physics-Aware Pooling Layer

**File**: New file `src/ups/io/pool_physics.py`  
**Approach**:
```python
def physics_aware_pool(tokens, node_coords, target_len):
    # Compute conservation metrics per node
    energy = (tokens ** 2).sum(dim=-1)
    
    # Select nodes with maximum representativeness
    # (preserve global integrals)
    selected_idx = select_representative_nodes(
        energy, node_coords, target_len
    )
    return tokens[selected_idx]
```

---

## 8. KEY FILE LOCATION SUMMARY

### Core Implementation Files

| Component | File | Lines | Methods |
|-----------|------|-------|---------|
| GridEncoder | enc_grid.py | 22-107 | `__init__`, `forward`, `_encode_fields` |
| GridEncoder Pooling | enc_grid.py | 216-221 | `_adaptive_token_pool` |
| GridEncoder Fourier | enc_grid.py | 183-214 | `_fourier_features` |
| MeshParticleEncoder | enc_mesh_particle.py | 51-171 | `__init__`, `forward`, `reconstruct` |
| Message Passing | enc_mesh_particle.py | 114-123 | Inline in forward loop |
| Adjacency | enc_mesh_particle.py | 39-48 | `_build_adjacency` |
| Supernode Pool | enc_mesh_particle.py | 155-163 | `_pool_supernodes` |
| Perceiver Pool | enc_mesh_particle.py | 165-170 | `_perceiver_pool` |
| AnyPointDecoder | decoder_anypoint.py | 53-136 | `__init__`, `forward`, `decode` |
| Decoder Fourier | decoder_anypoint.py | 11-32 | `_fourier_encode` |

### Integration Points

| Usage | File | Lines | Pattern |
|-------|------|-------|---------|
| GridEncoder training | latent_pairs.py | 211-241 | `_encode_grid_sample` |
| MeshParticleEncoder training | latent_pairs.py | 478-524 | `MeshParticleLDataset` |
| Encoder export | __init__.py | 1-14 | Module exports |
| Tests | test_enc_mesh_particle.py | 1-50 | Identity and reduction tests |

---

## 9. CURRENT CFD ENCODERS IN LITERATURE

**Common Patterns Found**:

1. **Graph Neural Networks for CFD** (most similar to MeshParticleEncoder):
   - Message passing on unstructured mesh
   - Node features: velocity, pressure, coordinates
   - Edge features: mesh connectivity
   - Limitation: Standard GNN loses spatial structure in pooling

2. **Attention-Based Pooling** (not used):
   - DiffPool: Learned soft assignments
   - MinCutPool: Graph structure preservation
   - Would improve on chunk-based supernode pooling

3. **Hierarchical Encoders** (not used):
   - Multi-level coarsening
   - Preserves important nodes at each level
   - Better for large meshes (>100K nodes)

4. **Physics-Aware Encoders** (not used):
   - Conservation-preserving pooling
   - Boundary condition encoding
   - Reynolds number conditioning

---

## 10. RECOMMENDATIONS FOR CFD ENHANCEMENTS

### Priority 1: Learned Pooling (High Impact, Medium Effort)
**Goal**: Replace chunk-based supernode pooling with importance-weighted selection

**File**: `enc_mesh_particle.py`  
**Changes**:
- Add `_compute_importance()` method (lines 155-163 area)
- Implement TopKPool or learned attention-based pooling
- Preserve edges in pooled graph

**Benefit**: Better token selection, physics-aware compression

**Estimated LOC**: 40-60 lines

### Priority 2: Boundary Condition Encoding (Medium Impact, Low Effort)
**Goal**: Embed BC types in latent tokens

**File**: `enc_mesh_particle.py`  
**Changes**:
- Add `bc` parameter to forward() (line 87)
- Encode BC types as features (one-hot or embeddings)
- Concatenate with coordinates (after line 103)

**Benefit**: Decoder has BC context, better predictions at boundaries

**Estimated LOC**: 20-30 lines

### Priority 3: Multi-Physics Awareness (Medium Impact, High Effort)
**Goal**: Preserve multi-domain structure in encoder

**File**: `enc_mesh_particle.py` + new integration  
**Changes**:
- Accept domain_id or physics_type parameter
- Use MultiphysicsFactorGraph structure during encoding
- Separate latent subspaces per physics

**Benefit**: Can encode coupled phenomena

**Estimated LOC**: 100+ lines

### Priority 4: Reynolds Number Handling (Low Impact for now, Low Effort)
**Goal**: Re-dependent feature normalization

**File**: `enc_mesh_particle.py` config  
**Changes**:
- Add `reynolds_number` to config
- Scale features by Re-dependent factors
- Adaptive message passing depth

**Benefit**: Better scaling across flow regimes

**Estimated LOC**: 10-20 lines

---

## Appendix: Code Snippet Reference

### GridEncoder Basic Usage
```python
from ups.io import GridEncoder, GridEncoderConfig

cfg = GridEncoderConfig(
    latent_len=256,
    latent_dim=32,
    field_channels={'u': 1},
    patch_size=4,
    use_fourier_features=True,
)
encoder = GridEncoder(cfg)

fields = {'u': torch.randn(2, 128, 128)}  # Batch of 128×128 grids
latent = encoder(fields, coords, meta={'grid_shape': (128, 128)})
# Output: (2, 256, 32)
```

### MeshParticleEncoder Basic Usage
```python
from ups.io import MeshParticleEncoder, MeshParticleEncoderConfig

cfg = MeshParticleEncoderConfig(
    latent_len=256,
    latent_dim=32,
    hidden_dim=64,
    message_passing_steps=3,
    supernodes=2048,
)
encoder = MeshParticleEncoder(cfg)

fields = {'u': torch.randn(2, 1000, 1), 'v': torch.randn(2, 1000, 1)}
coords = torch.randn(1000, 3)  # 3D coordinates
edges = torch.randint(0, 1000, (2000, 2))  # 2000 edges

latent = encoder(fields, coords, connect=edges)
# Output: (2, 256, 32)
```

### Message Passing Visualization
```
Node 0 (features h0)
    ↓ (message)
    ↓ aggregates from neighbors 1, 2
    → Σ(h1, h2) / degree
    → Linear transform + ReLU
    → h0 + residual
```

---

## Summary Statistics

- **Total Encoder LOC**: ~550 lines (GridEncoder 232 + MeshParticleEncoder 171 + Decoder 137)
- **Message Passing LOC**: 10 lines of core logic (but effective)
- **Pooling LOC**: 20 lines total (both non-learned)
- **CFD-Specific LOC**: 0 (entirely missing)
- **GNN Features**: Basic message passing only
- **Learned Pooling**: 0 (pure averaging)
- **Physics Awareness**: 0 (geometric only)

**Gaps vs. Literature**: 
- Missing: Learned pooling (DiffPool, TopKPool)
- Missing: Physics-aware aggregation
- Missing: Boundary condition encoding
- Missing: Multi-physics coupling
- Missing: Reynolds number conditioning

