# Encoder Architecture Research Summary

## 1. CURRENT ENCODERS IN src/ups/io/

### A. GridEncoder (enc_grid.py)
**Purpose**: Encode structured grid-based fields (e.g., Burgers, shallow water on regular grids)

**Key Characteristics** (File: /Users/emerygunselman/Code/universal_simulator/src/ups/io/enc_grid.py):
- **Lines 22-107**: Main encoder class with pixel-unshuffle patch extraction
- **Lines 43-44**: Pixel unshuffle/shuffle for patch extraction
- **Lines 46-59**: Per-field residual convolutional stems
- **Lines 61-66**: Optional sinusoidal Fourier features for coordinate encoding
- **Lines 104-106**: Adaptive token pooling to target latent_len
- **Lines 216-221**: _adaptive_token_pool uses F.adaptive_avg_pool1d (SIMPLE POOLING)
- **Lines 183-214**: Fourier feature generation for coordinate augmentation

**Pooling Strategy**:
- Uses `F.adaptive_avg_pool1d` for basic averaging
- No mesh-awareness or graph-based pooling
- Simple reduction from patch tokens to target latent length

**CFD Limitations**:
- Assumes regular grids (rectangular, structured)
- Cannot handle irregular meshes or unstructured grids
- No edge information or topology awareness
- Fourier features are grid-coordinate only, not mesh-aware

---

### B. MeshParticleEncoder (enc_mesh_particle.py)
**Purpose**: Encode irregular mesh or particle-based domains (unstructured grids, clouds)

**Key Architecture** (File: /Users/emerygunselman/Code/universal_simulator/src/ups/io/enc_mesh_particle.py):

#### Message Passing (Graph Component)
- **Lines 64-66**: ModuleList of message passing layers
- **Lines 114-123**: Message passing forward pass:
  - Line 116: Aggregates features via `index_add_` to gather messages
  - Line 119-121: Degree normalization (mean aggregation)
  - Line 122: Message projection with GELU activation
  - Line 123: Residual connection (h = h + GELU(m))
- **Lines 39-48**: _build_adjacency creates undirected edge lists
- **message_passing_steps**: 3 by default (GNN-like depth)

#### Pooling Strategy (TWO LAYERS)
- **Lines 155-163**: _pool_supernodes - chunk-based pooling
  - Groups N nodes into S supernodes by reshaping/averaging
  - Hard pooling strategy (no learnable weights)
  - Lines 157-162: Chunk nodes, pad if necessary, reshape and mean
- **Lines 165-170**: _perceiver_pool - Perceiver-style adaptive pooling
  - Uses `F.adaptive_avg_pool1d` on transposed tokens
  - Reduces from supernode count to target latent_len

**CFD Capabilities**:
- Handles irregular meshes via edge connectivity
- Message passing (3 steps default) aggregates information across mesh
- Supernode pooling allows hierarchical reduction
- Perceiver-style second-stage pooling for final token compression
- Coordinate-aware (embeds x,y,z positions when use_coords=True)

**Key Config** (Lines 11-18):
```python
@dataclass
class MeshParticleEncoderConfig:
    latent_len: int           # Target number of latent tokens
    latent_dim: int           # Feature dimension per token
    hidden_dim: int           # Message passing feature dimension
    message_passing_steps: int = 3
    supernodes: int = 2048    # Intermediate pooling count
    use_coords: bool = True
```

---

### C. AnyPointDecoder (decoder_anypoint.py)
**Purpose**: Decode latent tokens to arbitrary spatial query points (Perceiver-IO style)

**Key Components** (File: /Users/emerygunselman/Code/universal_simulator/src/ups/io/decoder_anypoint.py):
- **Lines 54-87**: Cross-attention decoder architecture
- **Lines 66-67**: Query and latent projections to hidden_dim
- **Lines 69-79**: Stacked cross-attention layers with residual connections
- **Lines 81-87**: Per-field prediction heads
- **Lines 122-129**: Forward pass: Fourier encode queries → cross-attention → field heads
- **Lines 11-32**: _fourier_encode for query coordinates

---

## 2. GNN AND GRAPH-BASED POOLING CAPABILITIES

### Currently Implemented
1. **Message Passing** (MeshParticleEncoder):
   - Basic node aggregation via index_add_
   - Degree-normalized averaging
   - Learnable linear projections per layer
   - Residual connections

2. **Pooling Strategies**:
   - GridEncoder: Simple adaptive average pooling
   - MeshParticleEncoder: Two-stage (supernode + perceiver)
   - No learnable pooling operators

### NOT Implemented (Gaps)
- Graph neural networks (GCN, GAT, GraphSAGE)
- Learnable pooling (DiffPool, SAGPool, TopKPool)
- Attention-based graph pooling
- Edge feature propagation
- Multi-head message passing
- Hierarchical graph coarsening
- Flow/boundary-aware message passing

---

## 3. MESH/PARTICLE HANDLING FOR IRREGULAR GRIDS

### MeshParticleEncoder Approach
**Files and Line References**:
- **Lines 39-48**: _build_adjacency - Creates edge connectivity
  - Takes edge tensor, ensures undirected (bidirectional)
  - Handles empty edge case (fallback to self-loops)
  - Returns (src_idx, dst_idx) for aggregation

- **Lines 21-36**: _flatten_fields - Field concatenation
  - Handles variable tensor dimensions (2D, 3D, 4D)
  - Averages over time dimension if present

- **Lines 99-103**: Coordinate concatenation with features
  - Appends x,y,z to feature vector when use_coords=True

**Limitations**:
- No support for time-varying mesh connectivity
- No boundary condition encoding (edge types)
- No multi-resolution handling
- No mesh quality metrics

---

## 4. PERCEIVER-BASED POOLING PATTERNS

### Current Implementation
**GridEncoder** (Lines 216-221):
```python
def _adaptive_token_pool(self, tokens: torch.Tensor, target_len: int) -> torch.Tensor:
    if tokens.shape[1] == target_len:
        return tokens
    tokens_t = tokens.transpose(1, 2)
    pooled = F.adaptive_avg_pool1d(tokens_t, target_len)
    return pooled.transpose(1, 2)
```

**MeshParticleEncoder** (Lines 165-170):
```python
def _perceiver_pool(self, tokens: torch.Tensor, target_len: int) -> torch.Tensor:
    if tokens.shape[1] == target_len:
        return tokens
    tokens_t = tokens.transpose(1, 2)
    pooled = F.adaptive_avg_pool1d(tokens_t, target_len)
    return pooled.transpose(1, 2)
```

### NOT Perceiver-IO Features
- No learned projection basis (latents)
- No cross-attention for pooling
- Pure average pooling, not learned selection
- No token deduplication or importance weighting

---

## 5. CFD-SPECIFIC FEATURES AND LIMITATIONS

### Existing CFD Support
1. **Shallow Water Equations**: GridEncoder handles 2D structured grids
2. **Burgers Equation**: GridEncoder works on 1D chains (reshaped as 2D)
3. **Mesh Particles**: MeshParticleEncoder supports irregular graphs

### Missing CFD-Specific Patterns

#### A. Domain Decomposition
- No multi-region support
- No subdomain boundary handling
- SingleMeshParticleEncoder processes entire domain

#### B. Boundary Conditions
- No BC encoding in latent space
- No distinction between:
  - Dirichlet (fixed value)
  - Neumann (fixed gradient)
  - Robin (mixed)
  - Periodic boundaries
- **Where to add**: MeshParticleEncoder.forward() accept bc parameter

#### C. Physics-Aware Pooling
- No conservation-aware merging
- No mass/energy preservation during pooling
- Chunk-based supernode pooling is agnostic to physics
- **Where to add**: New _physics_aware_pool method

#### D. Mesh Quality and Adaptation
- No cell aspect ratio awareness
- No refinement level tracking
- No boundary layer clustering
- GridEncoder doesn't distinguish between different grid types

#### E. Turbulence/Eddy Viscosity
- No subscale modeling in encoder
- No feature scaling for different Re regimes
- **Potential enhancement**: Re-dependent normalization

#### F. Multi-Physics Coupling
- MultiphysicsFactorGraph exists (lines 25-47 in multiphysics_factor_graph.py)
  - But NOT integrated into encoders
  - Only at inference level
  - **Gap**: Encoder doesn't preserve multi-physics structure

---

## 6. CURRENT SUPERNODE AND TOKEN POOLING STRATEGIES

### Supernode Pooling (MeshParticleEncoder.Lines 155-163)
**Algorithm**:
1. Calculate chunk size: `chunk = ceil(N / S)`
2. Pad to align: `N_padded = chunk * S`
3. Reshape to (B, S, chunk, D)
4. Mean across chunk dimension

**Characteristics**:
- Sequential/spatial grouping (not learned)
- Hard assignment (each node → exactly one supernode)
- Fixed output size (S supernodes)
- O(N) complexity

**CFD Implications**:
- No geometric awareness (doesn't consider node positions)
- Chunk boundaries may split important features
- Works for uniform particle clouds, poor for refined regions

### Perceiver Pooling (MeshParticleEncoder.Lines 165-170)
**Algorithm**:
1. Adaptive average pooling via PyTorch's pool1d
2. Reduces from current token count to target_len

**Characteristics**:
- Learned via implicit bin assignment
- No attention-based selection
- Works best when tokens are ordered by importance
- O(1) relative to target_len

---

## 7. WHERE GNN-BASED POOLING COULD BE ADDED

### Option 1: Replace Supernode Pooling
**File**: `/Users/emerygunselman/Code/universal_simulator/src/ups/io/enc_mesh_particle.py`
**Location**: After line 123 (after message passing)
**Implementation Pattern**:
```python
def _learned_pool_supernodes(self, tokens: torch.Tensor, edges: torch.Tensor, supernodes: int) -> torch.Tensor:
    """Replace chunk-based pooling with learned GNN-based pooling."""
    # Learns which nodes to keep/merge based on graph structure
    # Uses: TopKPool, DiffPool, or SAGPool
```

### Option 2: Add Graph Attention Pooling
**File**: Same as above
**Location**: New method after _perceiver_pool
**Approach**:
- Nodes vote on which tokens are most important
- Attention-weighted selection preserves high-impact nodes
- Maintains graph connectivity in pooled representation

### Option 3: Multi-Head Message Passing
**File**: Same, lines 64-123
**Approach**:
- Split hidden_dim into multiple heads
- Each head aggregates different aspects (velocity, pressure, vorticity, etc.)
- CFD-specific: separate heads per field

### Option 4: Physics-Aware Pooling Layer
**New file**: `src/ups/io/pool_physics.py`
**Idea**:
- Compute local energy, mass, enstrophy per node
- Pool nodes with lowest variance in conservation metrics
- Preserves global integrals during pooling

---

## 8. KEY FILE LOCATION SUMMARY

| Component | File | Lines | Key Methods |
|-----------|------|-------|------------|
| GridEncoder | enc_grid.py | 22-107 | forward, _adaptive_token_pool |
| MeshParticleEncoder | enc_mesh_particle.py | 51-171 | forward, _pool_supernodes, _perceiver_pool |
| Message Passing | enc_mesh_particle.py | 114-123 | Forward loop over message_layers |
| Adjacency Building | enc_mesh_particle.py | 39-48 | _build_adjacency |
| AnyPointDecoder | decoder_anypoint.py | 53-136 | forward, _fourier_encode |
| Fourier Features (Grid) | enc_grid.py | 183-214 | _fourier_features |
| Fourier Features (Query) | decoder_anypoint.py | 11-32 | _fourier_encode |

---

## RECOMMENDATIONS FOR CFD ENHANCEMENTS

### Priority 1: Boundary Condition Awareness
- Add bc parameter to MeshParticleEncoder.forward()
- Embed boundary types (Dirichlet/Neumann/periodic) in token features
- File: enc_mesh_particle.py, add after line 103

### Priority 2: Learned Pooling
- Replace supernode pooling with TopKPool or learned selection
- Weights nodes by local importance (energy, vorticity, pressure gradients)
- File: new method in enc_mesh_particle.py

### Priority 3: Multi-Physics Awareness
- Connect to MultiphysicsFactorGraph during encoding
- Preserve domain structure in latent tokens
- File: enc_mesh_particle.py integration with multiphysics_factor_graph.py

### Priority 4: Reynolds Number Handling
- Add Re normalization to feature scaling
- Field-specific preprocessing for different regimes
- File: enc_mesh_particle.py configuration extension
