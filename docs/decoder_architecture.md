# AnyPointDecoder Architecture Documentation

## Overview

The `AnyPointDecoder` is a **Perceiver-style cross-attention decoder** that decodes latent tokens to arbitrary spatial query points. It implements discretization-agnostic decoding, allowing inference at any resolution without retraining.

**Location**: `/Users/emerygunselman/Code/universal_simulator/src/ups/io/decoder_anypoint.py`

**Key Characteristics**:
- Perceiver-IO architecture with cross-attention mechanism
- Queries (arbitrary spatial points) attend to latent tokens (from encoder)
- Supports multi-field output prediction (e.g., velocity, pressure, density)
- Optional conditioning signals for physics constraints
- Fourier positional encoding for query coordinates

---

## Architecture Components

### 1. Configuration: `AnyPointDecoderConfig`

**Location**: Lines 35-50 of `decoder_anypoint.py`

```python
@dataclass
class AnyPointDecoderConfig:
    latent_dim: int                                    # Latent token feature dimension
    query_dim: int                                     # Query coordinate dimension (e.g., 2 for 2D)
    hidden_dim: int = 128                              # Cross-attention hidden dimension
    num_layers: int = 2                                # Number of cross-attention blocks
    num_heads: int = 4                                 # Attention heads
    frequencies: Tuple[float, ...] = (1.0, 2.0, 4.0)  # Fourier frequencies for positional encoding
    mlp_hidden_dim: int = 128                          # MLP width in prediction heads
    output_channels: Mapping[str, int] = None          # Dict: {field_name -> num_channels}
```

**Validation**:
- `output_channels` must be non-empty (at least one output field)
- `hidden_dim` must be divisible by `num_heads` (for multihead attention)

**Usage Example** (from `rollout_ttc.py`):
```python
decoder_config = AnyPointDecoderConfig(
    latent_dim=64,                                    # Matches operator latent_dim
    query_dim=2,                                      # 2D spatial coordinates
    hidden_dim=128,
    num_layers=2,
    num_heads=4,
    mlp_hidden_dim=128,
    frequencies=(1.0, 2.0, 4.0),
    output_channels={"u": 2, "p": 1},  # Velocity (2-channel), pressure (1-channel)
)
decoder = AnyPointDecoder(decoder_config).to(device)
```

---

### 2. Fourier Positional Encoding

**Function**: `_fourier_encode()` (Lines 11-32)

Transforms raw query coordinates into sinusoidal positional features:

```python
def _fourier_encode(coords: torch.Tensor, frequencies: Sequence[float]) -> torch.Tensor:
    """
    coords: (batch, num_points, coord_dim)
    frequencies: [1.0, 2.0, 4.0] (example)
    Returns: (batch, num_points, enriched_dim)
    """
```

**Encoding Process**:
1. For each frequency `f` and coordinate dimension:
   - Compute: `sin(2π × f × coord)` and `cos(2π × f × coord)`
2. Concatenate all sin/cos features with original coordinates
3. **Output dimension**: `query_dim + 2 × query_dim × len(frequencies)`

**Example** (2D coordinates with 3 frequencies):
- Input: `(batch, num_points, 2)` → `(batch, num_points, 2 + 2×2×3)` = `(batch, num_points, 14)`
- Raw coords: `[x, y]`
- After encoding: `[x, y, sin(2πx), sin(4πx), sin(8πx), cos(2πx), cos(4πx), cos(8πx), sin(2πy), sin(4πy), sin(8πy), cos(2πy), cos(4πy), cos(8πy)]`

This enables the decoder to handle arbitrary resolutions through rich frequency-domain features.

---

### 3. Query Embedding

**Initialization** (Lines 66, 122-123):
```python
self.query_embed = nn.Linear(
    cfg.query_dim + 2 * len(cfg.frequencies) * cfg.query_dim,  # Enriched dimension
    cfg.hidden_dim
)
```

**Usage** (Forward):
```python
enriched_points = _fourier_encode(points, self.cfg.frequencies)  # (B, Q, enriched_dim)
queries = self.query_embed(enriched_points)                       # (B, Q, hidden_dim)
```

---

### 4. Latent Projection

**Initialization** (Line 67):
```python
self.latent_proj = nn.Linear(cfg.latent_dim, cfg.hidden_dim)
```

**Purpose**: Projects latent tokens from encoder space to the decoder's hidden space.

**With Conditioning** (Lines 116-120):
If conditioning signals are provided, they are concatenated onto latent tokens before projection:
```python
if conditioning:
    cond_feats = [tensor.expand_as(latent_tokens) for tensor in conditioning.values()]
    latents = torch.cat([latent_tokens, *cond_feats], dim=-1)
latents = self.latent_proj(latents)  # (B, num_tokens, hidden_dim)
```

---

### 5. Cross-Attention Blocks

**Architecture** (Lines 69-79):
```python
for _ in range(cfg.num_layers):
    attn = nn.MultiheadAttention(cfg.hidden_dim, cfg.num_heads, batch_first=True)
    ln_query = nn.LayerNorm(cfg.hidden_dim)
    ln_ff = nn.LayerNorm(cfg.hidden_dim)
    ff = nn.Sequential(
        nn.Linear(cfg.hidden_dim, cfg.mlp_hidden_dim),
        nn.GELU(),
        nn.Linear(cfg.mlp_hidden_dim, cfg.hidden_dim),
    )
    self.layers.append(nn.ModuleList([attn, ln_query, ff, ln_ff]))
```

**Attention Mechanism** (Standard Transformer block):
```
1. Cross-Attention: queries (points) attend to latents (tokens)
   attn_out, _ = attn(queries, latents, latents)
2. Residual + LayerNorm: queries = LayerNorm(queries + attn_out)
3. Feed-Forward Network: ff_out = FF(queries)
4. Residual + LayerNorm: queries = LayerNorm(queries + ff_out)
```

**Key Property**: 
- **Queries** have shape `(B, Q, hidden_dim)` – one query per point
- **Latents** have shape `(B, num_tokens, hidden_dim)` – from encoder
- Each query learns a weighted combination of latent tokens

---

### 6. Prediction Heads

**Initialization** (Lines 81-87):
```python
self.heads = nn.ModuleDict()
for name, out_ch in cfg.output_channels.items():
    self.heads[name] = nn.Sequential(
        nn.Linear(cfg.hidden_dim, cfg.mlp_hidden_dim),
        nn.GELU(),
        nn.Linear(cfg.mlp_hidden_dim, out_ch),
    )
```

**Design**:
- One 2-layer MLP head per output field
- Each head maps `hidden_dim` → `output_channels[field_name]`
- No activation on final layer (allows unbounded output values)

**Usage** (Forward):
```python
outputs = {}
for name, head in self.heads.items():
    outputs[name] = head(queries)  # (B, Q, out_ch)
return outputs  # Dict: {field_name -> (B, Q, out_ch)}
```

---

## Forward Pass: Complete Data Flow

**Function**: `forward()` / `decode()` (Lines 89-136)

```python
def forward(
    self,
    points: torch.Tensor,                              # (B, Q, query_dim)
    latent_tokens: torch.Tensor,                       # (B, num_tokens, latent_dim)
    conditioning: Optional[Mapping[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
```

**Input Validation** (Line 111-112):
- `latent_tokens.dim() == 3` and `points.dim() == 3`
- Shapes: `(batch, seq_len, feature_dim)`

**Step-by-Step Data Flow**:

```
1. INPUT
   points: (B, Q, query_dim)
   latent_tokens: (B, num_tokens, latent_dim)
   
2. CONDITIONING (Optional)
   if conditioning:
       latents = cat(latent_tokens, [cond1, cond2, ...])
   else:
       latents = latent_tokens
   Shape after concat: (B, num_tokens, latent_dim + sum(cond_dims))

3. LATENT PROJECTION
   latents = self.latent_proj(latents)  # (B, num_tokens, hidden_dim)

4. FOURIER ENCODING (Query Points)
   enriched_points = _fourier_encode(points, frequencies)
   Shape: (B, Q, query_dim + 2×query_dim×len(frequencies))

5. QUERY EMBEDDING
   queries = self.query_embed(enriched_points)  # (B, Q, hidden_dim)

6. CROSS-ATTENTION LOOP (num_layers times)
   for attn, ln_q, ff, ln_ff in self.layers:
       attn_out, _ = attn(queries, latents, latents)
       queries = ln_q(queries + attn_out)      # Residual + LayerNorm
       ff_out = ff(queries)
       queries = ln_ff(queries + ff_out)       # Residual + LayerNorm

7. PREDICTION HEADS
   for each field_name in output_channels:
       outputs[field_name] = self.heads[field_name](queries)
       Shape: (B, Q, output_channels[field_name])

8. OUTPUT
   Dict: {field_name -> (B, Q, out_ch)}
   Example: {"u": (B, Q, 2), "p": (B, Q, 1)}
```

---

## Integration with UPS Pipeline

### 1. Encoding Stage (GridEncoder)

**File**: `/Users/emerygunselman/Code/universal_simulator/src/ups/io/enc_grid.py`

Grid-based PDEs are encoded:
```
Physical fields (B, H, W, C) 
→ GridEncoder 
→ Latent tokens (B, num_tokens, latent_dim)
```

### 2. Evolution Stage (LatentOperator)

**File**: `/Users/emerygunselman/Code/universal_simulator/src/ups/models/latent_operator.py`

Latent tokens evolve in time:
```
LatentState(z=(B, num_tokens, latent_dim)) 
→ LatentOperator(dt) 
→ LatentState(z=(B, num_tokens, latent_dim))
```

### 3. Decoding Stage (AnyPointDecoder)

**Reward Model Usage** (`reward_models.py`, lines 47-104):
```python
def _decode_fields(
    decoder: AnyPointDecoder,
    state: LatentState,
    query_points: torch.Tensor,  # Grid coordinates (1, H×W, 2)
    device: torch.device,
    height: int,
    width: int,
) -> Dict[str, torch.Tensor]:
    tokens = state.z.to(device)                       # (B, num_tokens, latent_dim)
    cond = {k: v.to(device) for k, v in state.cond.items()}
    query = query_points.expand(tokens.size(0), -1, -1)  # (B, H×W, 2)
    outputs = decoder(query, tokens, conditioning=cond)   # Dict of decoded fields
    for name, values in outputs.items():
        decoded[name] = values.view(tokens.size(0), height, width, -1)  # (B, H, W, C)
    return decoded
```

**Arbitrary Resolution Decoding**:
```python
# Decoder supports ANY query resolution, not just training grid
query_points = sample_random_points(resolution=512)  # Different from training (64×64)
outputs = decoder(query_points, latent_tokens)
```

---

## Test-Time Conditioning (TTC)

**File**: `/Users/emerygunselman/Code/universal_simulator/src/ups/inference/rollout_ttc.py`

### Decoder in TTC Workflow

1. **Initialization** (lines 215-225):
   ```python
   decoder_config = AnyPointDecoderConfig(
       latent_dim=decoder_cfg.get("latent_dim", latent_dim),
       query_dim=decoder_cfg.get("query_dim", 2),
       hidden_dim=decoder_cfg.get("hidden_dim", 128),
       num_layers=decoder_cfg.get("num_layers", 2),
       num_heads=decoder_cfg.get("num_heads", 4),
       mlp_hidden_dim=decoder_cfg.get("mlp_hidden_dim", 128),
       frequencies=tuple(decoder_cfg.get("frequencies", (1.0, 2.0, 4.0))),
       output_channels=decoder_cfg["output_channels"],
   )
   decoder = AnyPointDecoder(decoder_config).to(device)
   ```

2. **Used in Reward Models** (AnalyticalRewardModel, FeatureCriticRewardModel):
   - Decodes latent trajectories to physical space
   - Evaluates conservation laws (mass, momentum, energy)
   - Scores candidate rollouts for beam search

---

## Latent State Structure

**File**: `/Users/emerygunselman/Code/universal_simulator/src/ups/core/latent_state.py`

```python
@dataclass
class LatentState:
    z: torch.Tensor           # Latent tokens (batch, num_tokens, latent_dim)
    t: Optional[torch.Tensor] # Time value (scalar)
    cond: Dict[str, torch.Tensor]  # Conditioning (e.g., boundary conditions)
```

**Decoder accepts LatentState via `_decode_fields()`**:
- Extracts `state.z` → latent tokens
- Passes `state.cond` → conditioning parameters
- Returns decoded physical fields

---

## Output Fields

### No Built-in Constraints

The decoder **does not clamp or constrain outputs**. It predicts raw field values:

```python
# Example: velocity field (2 channels)
outputs["u"] = (B, Q, 2)  # No clamping, can be any float values

# Example: pressure field (1 channel)
outputs["p"] = (B, Q, 1)  # No clamping, can be any float values
```

**Physics Constraints** are enforced elsewhere:
- **Physics Guards** (`physics_guards.py`): Post-processing constraints
- **Test-Time Conditioning**: Reward models penalize violations (e.g., negative mass)
- **Diffusion Residual**: Refines predictions with learned uncertainty

### Configurable Multi-Field Output

```python
output_channels = {
    "u": 2,      # 2D velocity
    "p": 1,      # Pressure
    "rho": 1,    # Density
    "c": 1,      # Concentration
}
outputs = decoder(query_points, latent_tokens)
# Returns: {"u": (B, Q, 2), "p": (B, Q, 1), "rho": (B, Q, 1), "c": (B, Q, 1)}
```

---

## Key Properties

| Property | Value |
|----------|-------|
| **Attention Type** | Cross-attention (queries → latents) |
| **Positional Encoding** | Fourier basis (sinusoidal) |
| **Query Handling** | Arbitrary spatial points via MLPs |
| **Conditioning** | Via concatenation + adaptive layer norm (elsewhere) |
| **Output** | Multiple physics fields (velocity, pressure, density, etc.) |
| **Constraints** | None (enforced post-hoc by physics guards/TTC) |
| **Architecture** | Perceiver-IO block repeated num_layers times |
| **Gradient Flow** | Full (all components trainable) |

---

## Unit Tests

**File**: `/Users/emerygunselman/Code/universal_simulator/tests/unit/test_decoder_anypoint.py`

### Test 1: Shape and Gradient Flow
```python
def test_decoder_shapes_and_grads():
    cfg = AnyPointDecoderConfig(
        latent_dim=64,
        query_dim=3,
        hidden_dim=64,
        num_layers=2,
        num_heads=8,
        output_channels={"u": 2, "p": 1},
    )
    decoder = AnyPointDecoder(cfg)
    latent = torch.randn(4, 32, cfg.latent_dim, requires_grad=True)
    points = torch.rand(4, 17, cfg.query_dim)
    
    outputs = decoder(points, latent)
    assert outputs["u"].shape == (4, 17, 2)  # ✓
    assert outputs["p"].shape == (4, 17, 1)  # ✓
    
    loss = outputs["u"].sum() + outputs["p"].sum()
    loss.backward()
    assert latent.grad is not None  # ✓ Gradients flow
```

### Test 2: Constant Output via Bias
```python
def test_decoder_constant_output_via_bias():
    # With all weights zeroed and final bias=2.5
    # outputs should be constant (2.5)
    # Tests that MLP structure works correctly
```

---

## Configuration Examples

### Production Burgers Equation (from fast_to_sota.py)

```yaml
decoder:
  latent_dim: 32           # Matches operator.pdet.input_dim
  query_dim: 2             # 2D spatial coordinates
  hidden_dim: 128
  num_layers: 2
  num_heads: 4
  mlp_hidden_dim: 128
  frequencies: [1.0, 2.0, 4.0]
  output_channels:
    u: 1                   # Velocity (1D Burgers)
```

### 2D Navier-Stokes

```yaml
decoder:
  latent_dim: 64
  query_dim: 2             # (x, y)
  hidden_dim: 256
  num_layers: 3
  num_heads: 8
  mlp_hidden_dim: 256
  frequencies: [1.0, 2.0, 4.0, 8.0]
  output_channels:
    u: 2                   # Velocity (2D)
    p: 1                   # Pressure
```

---

## Summary

The **AnyPointDecoder** is a lightweight, flexible neural decoder that:

1. **Accepts arbitrary query points** (via Fourier encoding)
2. **Uses cross-attention** to extract relevant information from latent tokens
3. **Predicts multiple physics fields** simultaneously
4. **Supports conditioning** for physics constraints
5. **Enables zero-shot super-resolution** (e.g., eval at 512×512 after training on 64×64)
6. **Integrates cleanly** into the UPS pipeline (encode → evolve → decode)

It achieves **discretization-agnostic I/O** by learning in latent space rather than enforcing rigid grid structure.

