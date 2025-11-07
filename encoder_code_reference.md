# Encoder Code Reference: Detailed Implementation Walkthrough

## File Locations and Architecture Overview

All encoder implementations are in `/Users/emerygunselman/Code/universal_simulator/src/ups/io/`:
- `enc_grid.py` - GridEncoder (structured grids)
- `enc_mesh_particle.py` - MeshParticleEncoder (unstructured/particles)
- `decoder_anypoint.py` - AnyPointDecoder (query-based decoding)
- `__init__.py` - Module exports

---

## GridEncoder Deep Dive

### File: enc_grid.py

#### Core Structure (Lines 22-107)
```
GridEncoder.__init__():
  - self.pixel_unshuffle (line 43) - Patch extraction via pixel reorganization
  - self.pixel_shuffle (line 44) - Inverse for reconstruction
  - self.per_field_stems (lines 46-59) - ModuleDict of Conv2d residual stems
  - self.to_latent, self.from_latent (lines 71-75) - Projection layers
  - self.fourier_frequencies (lines 61-66) - Buffer for Fourier features

GridEncoder.forward():
  - Line 91: Infer grid shape from metadata
  - Lines 96: Call _encode_fields() for feature extraction
  - Lines 103: Project to latent dimension
  - Lines 104-106: Adaptive pooling if tokens != target length
```

#### Key Implementation Details

**Pixel Unshuffle Mechanism** (Lines 167-169):
- Input: (B, H, W) with C channels
- PixelUnshuffle(patch_size=4): (B, C, H, W) → (B, C*16, H/4, W/4)
- This groups 4x4 patches into 16 feature channels

**Per-Field Stems** (Lines 48-59):
```python
for name, channels in field_channels.items():
    in_ch = channels * patch * patch  # e.g., 1*16 = 16 for scalar field
    width = stem_width or max(in_ch, 32)
    stem = Sequential(
        Conv2d(in_ch, width, 3, padding=1),  # Expand
        GELU(),
        Conv2d(width, in_ch, 1),            # Project back (residual)
    )
    nn.init.zeros_(stem[-1].weight)  # Zero-init for near-identity residual
```

**Fourier Features** (Lines 183-214):
- Line 200: Reshape coords to patch grid (B, Hp, Wp, coord_dim)
- Line 202: Compute patch centers via mean over patch spatial dims
- Lines 211-212: Generate sin/cos features for each frequency
- Allows encoder to learn coordinate-aware patches

#### Pooling Strategy (Lines 216-221)
```python
def _adaptive_token_pool(self, tokens, target_len):
    tokens_t = tokens.transpose(1, 2)  # (B, D, T)
    pooled = F.adaptive_avg_pool1d(tokens_t, target_len)  # PyTorch's built-in
    return pooled.transpose(1, 2)  # Back to (B, T, D)
```
- Uses PyTorch's built-in adaptive pooling (averages bins)
- No spatial/geometric awareness

#### CFD Usage Example
```yaml
# For Burgers equation (1D, reshaped to 2D)
field_channels: {u: 1}
patch_size: 4
latent_len: 256        # Fewer tokens than input points
latent_dim: 32         # 32-dimensional embedding
use_fourier_features: true
fourier_frequencies: [1.0, 2.0]
```

---

## MeshParticleEncoder Deep Dive

### File: enc_mesh_particle.py

#### Core Data Structures

**Config** (Lines 11-18):
```python
@dataclass
class MeshParticleEncoderConfig:
    latent_len: int           # Target output tokens (e.g., 256)
    latent_dim: int           # Feature dim (e.g., 32)
    hidden_dim: int           # GNN feature dim (e.g., 64)
    message_passing_steps: int = 3  # Graph convolution depth
    supernodes: int = 2048    # First pooling stage count
    use_coords: bool = True   # Embed x,y,z coordinates
```

#### Adjacency Building (Lines 39-48)
```python
def _build_adjacency(num_nodes, edges, device):
    if edges.numel() == 0:
        # No connectivity info - use self-loops only
        rows = torch.arange(num_nodes, device=device)
        return rows, rows  # (self_idx, self_idx)
    
    src = edges[:, 0].to(torch.long)      # Source nodes
    dst = edges[:, 1].to(torch.long)      # Destination nodes
    
    # Make undirected
    undirected_src = torch.cat([src, dst])
    undirected_dst = torch.cat([dst, src])
    return undirected_src, undirected_dst
```

**CFD Implication**: This allows message passing in both directions on the mesh.

#### Forward Pass Architecture (Lines 87-141)

**Step 1: Feature Setup** (Lines 95-107):
```python
feat = _flatten_fields(fields).to(device)      # Concatenate all fields
if cfg.use_coords:
    coord_feat = coords.to(device)             # Add x,y,z
    feat = torch.cat([feat, coord_feat], dim=-1)
```

**Step 2: Projection to Hidden Dimension** (Lines 105-107):
```python
self._ensure_projections(feat.size(-1))  # Lazy Linear creation
h = self.node_proj(feat)                 # Project features
```

**Step 3: Message Passing Loop** (Lines 114-123):
```python
for layer in self.message_layers:
    # Aggregate incoming messages to each node
    m = torch.zeros_like(h)
    m.index_add_(1, dst_idx, h[:, src_idx, :])
    
    # Normalize by degree (average rather than sum)
    deg = torch.zeros(N, device=device)
    deg.index_add_(0, dst_idx, torch.ones_like(dst_idx, dtype=deg.dtype))
    deg = deg.clamp_min_(1.0).view(1, N, 1)
    m = m / deg
    
    # Apply learnable transformation
    m = layer(m)  # Linear transformation
    h = h + F.gelu(m)  # Residual connection
```

**Step 4: Two-Stage Pooling** (Lines 125-131):
```python
tokens = h                           # N nodes → N tokens
S = min(cfg.supernodes, N)
if S < N:
    tokens = self._pool_supernodes(tokens, S)  # N → S
pooled_len = tokens.shape[1]
if pooled_len != cfg.latent_len:
    tokens = self._perceiver_pool(tokens, cfg.latent_len)  # S → latent_len
```

#### Supernode Pooling (Lines 155-163)
```python
def _pool_supernodes(self, tokens, supernodes):
    B, N, D = tokens.shape
    chunk = (N + supernodes - 1) // supernodes  # ceil division
    pad = chunk * supernodes - N
    
    if pad > 0:
        # Pad with first few nodes (circular padding)
        pad_tensor = tokens[:, :pad, :]
        tokens = torch.cat([tokens, pad_tensor], dim=1)
    
    # Reshape to (B, S, chunk, D) and mean over chunk
    tokens = tokens.view(B, supernodes, chunk, D).mean(dim=2)
    return tokens
```

**Example**:
- 1000 nodes, 256 supernodes
- chunk = ceil(1000/256) = 4
- pad = 4*256 - 1000 = 24
- Reshape to (B, 256, 4, D) and average → (B, 256, D)

#### Perceiver Pooling (Lines 165-170)
```python
def _perceiver_pool(self, tokens, target_len):
    if tokens.shape[1] == target_len:
        return tokens
    tokens_t = tokens.transpose(1, 2)      # (B, D, current_len)
    pooled = F.adaptive_avg_pool1d(tokens_t, target_len)  # → (B, D, target)
    return pooled.transpose(1, 2)          # → (B, target, D)
```

#### Reconstruction (Lines 143-153)
```python
def reconstruct(self, latent):
    cache = self._cache
    tokens = self.latent_to_hidden(latent)  # latent_dim → hidden_dim
    
    # Can only reconstruct if tokens match original node count
    if tokens.shape[1] != cache["node_count"]:
        raise ValueError("Need exact token count for reconstruction")
    
    node_feat = self.output_proj(tokens)
    return node_feat  # (B, N, feat_dim)
```

**Limitation**: Only works when latent_len == node_count (no interpolation needed).

---

## AnyPointDecoder Implementation

### File: decoder_anypoint.py

#### Architecture (Lines 63-87)
```python
class AnyPointDecoder(nn.Module):
    def __init__(self, cfg):
        # Fourier embedding of query coordinates
        query_feat_dim = cfg.query_dim + 2 * len(cfg.frequencies) * cfg.query_dim
        self.query_embed = nn.Linear(query_feat_dim, cfg.hidden_dim)
        self.latent_proj = nn.Linear(cfg.latent_dim, cfg.hidden_dim)
        
        # Stacked cross-attention layers
        self.layers = nn.ModuleList()
        for _ in range(cfg.num_layers):
            attn = nn.MultiheadAttention(cfg.hidden_dim, cfg.num_heads, batch_first=True)
            ln_query = nn.LayerNorm(cfg.hidden_dim)
            ff = Sequential(...)
            ln_ff = nn.LayerNorm(cfg.hidden_dim)
            self.layers.append(ModuleList([attn, ln_query, ff, ln_ff]))
        
        # Per-field prediction heads
        self.heads = nn.ModuleDict()
        for name, out_ch in cfg.output_channels.items():
            self.heads[name] = Sequential(...)
```

#### Forward Pass (Lines 89-134)
```python
def forward(self, points, latent_tokens, conditioning=None):
    # points: (B, Q, query_dim) - arbitrary query coordinates
    # latent_tokens: (B, T, latent_dim) - encoder output
    
    # Optionally append conditioning signals
    latents = latent_tokens
    if conditioning:
        cond_feats = [t.expand_as(latent_tokens) for t in conditioning.values()]
        latents = torch.cat([latent_tokens, *cond_feats], dim=-1)
    
    # Project latents and queries to hidden space
    latents = self.latent_proj(latents)      # (B, T, hidden)
    enriched_points = _fourier_encode(points, frequencies)  # Fourier features
    queries = self.query_embed(enriched_points)  # (B, Q, hidden)
    
    # Apply cross-attention layers
    for attn, ln_q, ff, ln_ff in self.layers:
        attn_out, _ = attn(queries, latents, latents)  # Cross-attention
        queries = ln_q(queries + attn_out)             # Residual + LayerNorm
        ff_out = ff(queries)                           # Feed-forward
        queries = ln_ff(queries + ff_out)              # Residual + LayerNorm
    
    # Predict fields at query points
    outputs = {}
    for name, head in self.heads.items():
        outputs[name] = head(queries)
    return outputs
```

#### Fourier Encoding (Lines 11-32)
```python
def _fourier_encode(coords, frequencies):
    # coords: (B, Q, coord_dim)
    freqs = torch.tensor(frequencies, dtype=coords.dtype, device=coords.device)
    freqs = freqs.view(1, 1, -1, 1)  # (1, 1, F, 1)
    
    coords_expanded = coords.unsqueeze(-2)  # (B, Q, 1, coord_dim)
    scaled = 2*pi * coords_expanded * freqs  # (B, Q, F, coord_dim)
    
    sin_feat = torch.sin(scaled)
    cos_feat = torch.cos(scaled)
    
    # Concatenate: [raw, sin_1, cos_1, sin_2, cos_2, ...]
    encoded = torch.cat([coords, 
                        sin_feat.flatten(-2), 
                        cos_feat.flatten(-2)], dim=-1)
    return encoded
```

**Example Output Feature Count**:
- Input: 3D coordinates (x, y, z)
- 3 frequencies
- Output: 3 + 2*(3*3) = 3 + 18 = 21-dimensional feature vector

---

## Integration Flow: Training Data Path

### File: src/ups/data/latent_pairs.py (Key Usages)

**GridEncoder Usage** (Lines 211-241):
```python
def _encode_grid_sample(encoder: GridEncoder, sample, device=None):
    # Single sample encoding
    fields = sample['fields']  # Dict of field tensors
    meta = {'grid_shape': (H, W)}
    latent = encoder(fields, coords_batch, meta=meta)
    return latent
```

**MeshParticleEncoder Usage** (Lines 478-524):
```python
class MeshParticleLDataset:
    def __init__(self, encoder: MeshParticleEncoder):
        self.encoder = encoder
        
    def __getitem__(self, idx):
        sample = self.base[idx]
        field_batch = self._extract_fields(sample)
        coords = sample['coords']
        connect = sample.get('edges', None)  # Mesh connectivity
        
        latent_step = self.encoder(field_batch, coords, connect=connect)
        return latent_step
```

---

## CFD-Specific Encoding Patterns

### Example 1: 2D Burgers Equation
```python
# Grid-based (regular mesh)
GridEncoder(
    latent_len=256,
    latent_dim=32,
    field_channels={'u': 1, 'v': 1},
    patch_size=4,
    use_fourier_features=True,
    fourier_frequencies=(1.0, 2.0, 4.0),
)
# Input: (B, H, W) = (2, 128, 128)
# Output: (B, 256, 32)
```

### Example 2: Navier-Stokes on Unstructured Mesh
```python
# Mesh-based (irregular grid)
MeshParticleEncoder(
    latent_len=256,
    latent_dim=32,
    hidden_dim=64,
    message_passing_steps=5,  # Deeper for complex domains
    supernodes=2048,
    use_coords=True,
)
# Input:
#   fields={'u': (B, N, 1), 'v': (B, N, 1), 'p': (B, N, 1)}
#   coords: (N, 3)
#   edges: (E, 2)
# Output: (B, 256, 32)
```

### Example 3: Multi-Physics (Thermal CFD)
```python
# Fluid + Heat coupled
MeshParticleEncoder(
    latent_len=512,  # More tokens for complex physics
    latent_dim=64,
    hidden_dim=128,
    message_passing_steps=6,  # Longer range for thermal coupling
    supernodes=4096,
    use_coords=True,
)
# Input:
#   fields={
#       'u': (B, N, 1), 'v': (B, N, 1), 'p': (B, N, 1),  # Fluid
#       'T': (B, N, 1), 'k': (B, N, 1)                   # Thermal
#   }
# Output: (B, 512, 64)
```

---

## Performance Characteristics

### Memory Usage
- GridEncoder: ~Linear in grid resolution (H*W)
- MeshParticleEncoder: ~Linear in node count (N) + edges (E)

### Computation
- GridEncoder: Conv2d patch extraction (O(HW)) + pooling (O(T))
- MeshParticleEncoder: Message passing (O(steps * (N + E)) + pooling (O(N log S))

### Typical Latent Compression Ratios
- GridEncoder: 128×128 grid → 256 tokens (64× compression)
- MeshParticleEncoder: 10,000 nodes → 256 tokens (39× compression)

---

## Testing Reference

Test file: `tests/unit/test_enc_mesh_particle.py`

**Test 1: Identity Path** (Lines 15-32)
- Tests that encoder→decoder gives exact reconstruction
- Requires: latent_len = node_count, no message passing, identity projections

**Test 2: Token Reduction** (Lines 35-49)
- Tests that encoder can reduce large mesh to small token count
- Verifies output shape matches config

---

## Key Takeaways

1. **GridEncoder**: Simple, fast, for structured/regular grids
2. **MeshParticleEncoder**: Complex, graph-aware, for unstructured/particles
3. **AnyPointDecoder**: Perceiver-style query-based decoding
4. **Pooling Bottleneck**: Current pooling is non-learned, purely geometric
5. **CFD Gap**: No boundary condition encoding, no physics-aware pooling
