# UPT Inverse Loss Implementation and Memory Requirements Analysis

## Executive Summary

The UPT (Universal Physics Transformer) implementation includes two inverse loss functions (`lambda_inv_enc` and `lambda_inv_dec`) that enable training with encoder-decoder invertibility constraints. These losses are critical for ensuring the latent space properly reconstructs physical fields and maintains information fidelity. Query sampling with `num_queries: 2048` reduces memory overhead during training.

---

## 1. Lambda Parameters Usage

### 1.1 `lambda_inv_enc` - Inverse Encoding Loss Weight

**Purpose**: Ensures encoded latent can be decoded back to original physical fields.

**Configuration Location**:
- `/Users/emerygunselman/Code/universal_simulator/scripts/train.py:658`
  ```python
  "lambda_inv_enc": float(train_cfg.get("lambda_inv_enc", 0.0)),
  ```

**Usage Pattern**:
- Loaded from config under `training.lambda_inv_enc`
- Passed to loss computation pipeline
- Applied with curriculum learning weight schedule

**Detection/Enabled**:
- `/Users/emerygunselman/Code/universal_simulator/scripts/train.py:498-501`
  ```python
  use_inverse_losses = (
      bool(cfg.get("training", {}).get("use_inverse_losses", False)) or
      cfg.get("training", {}).get("lambda_inv_enc", 0.0) > 0 or
      cfg.get("training", {}).get("lambda_inv_dec", 0.0) > 0
  )
  ```

### 1.2 `lambda_inv_dec` - Inverse Decoding Loss Weight

**Purpose**: Ensures decoded fields can be re-encoded back to original latent representation.

**Configuration Location**:
- `/Users/emerygunselman/Code/universal_simulator/scripts/train.py:659`
  ```python
  "lambda_inv_dec": float(train_cfg.get("lambda_inv_dec", 0.0)),
  ```

**Usage Pattern**:
- Loaded from config under `training.lambda_inv_dec`
- Passed to loss computation pipeline
- Applied with curriculum learning weight schedule

**Typical Configuration Values** (from `/Users/emerygunselman/Code/universal_simulator/configs/train_burgers_upt_full.yaml`):
- `lambda_inv_enc: 0.01`
- `lambda_inv_dec: 0.01`
- `use_inverse_losses: true`
- `inverse_loss_frequency: 1` (apply every batch)
- `inverse_loss_warmup_epochs: 5`
- `inverse_loss_max_weight: 0.05`

---

## 2. Inverse Loss Function Implementations

### 2.1 `inverse_encoding_loss()` Function

**File**: `/Users/emerygunselman/Code/universal_simulator/src/ups/training/losses.py:25-82`

**Function Signature**:
```python
def inverse_encoding_loss(
    input_fields: Mapping[str, torch.Tensor],        # Original physical fields (B, N, C)
    latent: torch.Tensor,                             # Encoded latent (B, tokens, latent_dim)
    decoder: nn.Module,                               # AnyPointDecoder
    input_positions: torch.Tensor,                    # Spatial coordinates (B, N, coord_dim)
    weight: float = 1.0,                              # Loss weight multiplier
    num_queries: Optional[int] = None,                # Number of query points to sample
    query_strategy: str = "uniform",                  # "uniform" or "stratified"
    grid_shape: Optional[Tuple[int, int]] = None,    # (H, W) for stratified sampling
) -> torch.Tensor:
```

**Flow**:
1. Input: Original physical fields (encoded to latent) → latent representation
2. Decode: latent → decoder → reconstructed_fields (in physical space)
3. Loss: MSE(reconstructed_fields, input_fields) at sampled query points

**Query Sampling** (lines 57-70):
```python
if num_queries is not None and num_queries < input_positions.shape[1]:
    from ups.training.query_sampling import apply_query_sampling
    input_fields_sampled, input_positions_sampled = apply_query_sampling(
        input_fields,
        input_positions,
        num_queries=num_queries,
        strategy=query_strategy,
        grid_shape=grid_shape,
    )
else:
    input_fields_sampled = input_fields
    input_positions_sampled = input_positions
```

**Decoder Call** (line 73):
```python
reconstructed = decoder(input_positions_sampled, latent)
```

**Memory Cost**:
- Decoder forward: O(B × num_queries × hidden_dim)
- Field tensors: O(B × num_queries × field_channels)
- With num_queries=2048, 64×64 grid: Saves ~50% decoder memory

### 2.2 `inverse_decoding_loss()` Function

**File**: `/Users/emerygunselman/Code/universal_simulator/src/ups/training/losses.py:85-135`

**Function Signature**:
```python
def inverse_decoding_loss(
    latent: torch.Tensor,                             # Latent (B, tokens, latent_dim)
    decoder: nn.Module,                               # AnyPointDecoder
    encoder: nn.Module,                               # GridEncoder or MeshParticleEncoder
    query_positions: torch.Tensor,                    # (B, N, coord_dim) for decoding
    coords: torch.Tensor,                             # Full grid (B, H*W, coord_dim) for re-encoding
    meta: dict,                                       # Metadata with 'grid_shape'
    weight: float = 1.0,                              # Loss weight multiplier
    num_queries: Optional[int] = None,                # IGNORED (kept for API compatibility)
    query_strategy: str = "uniform",                  # IGNORED
    grid_shape: Optional[Tuple[int, int]] = None,    # IGNORED
) -> torch.Tensor:
```

**Flow**:
1. Decode: latent → decoder → decoded_fields (in physical space)
2. Re-encode: decoded_fields → encoder → reconstructed_latent
3. Loss: MSE(reconstructed_latent, latent.detach()) in latent space

**Key Design Decision** (lines 124-126):
```python
# NOTE: We intentionally ignore num_queries for inverse decoding loss
# because the encoder requires a full grid. Query sampling only makes
# sense for inverse encoding where we compare in physical space.
```

**Decoder Call** (line 129):
```python
decoded_fields = decoder(query_positions, latent)
```

**Encoder Call** (line 132):
```python
latent_reconstructed = encoder(decoded_fields, coords, meta=meta)
```

**Detach Call** (line 135):
```python
return weight * mse(latent_reconstructed, latent.detach())
```
- Prevents double backprop through encoder
- Saves memory: O(B × tokens × latent_dim) gradient accumulation avoided

---

## 3. Query Sampling Implementation

### 3.1 Query Sampling Module

**File**: `/Users/emerygunselman/Code/universal_simulator/src/ups/training/query_sampling.py`

**Core Function**: `apply_query_sampling()` (lines 106-153)

```python
def apply_query_sampling(
    fields: Dict[str, torch.Tensor],                  # {name: (B, N, C)}
    coords: torch.Tensor,                             # (B, N, coord_dim)
    num_queries: int,                                 # Target number of samples
    strategy: str = "uniform",                        # "uniform" or "stratified"
    grid_shape: Optional[Tuple[int, int]] = None,    # (H, W) for stratified
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
```

**Strategies**:

#### Uniform Sampling (lines 15-36):
```python
def sample_uniform_queries(
    total_points: int,
    num_queries: int,
    device: torch.device = None,
) -> torch.Tensor:
    if num_queries >= total_points:
        return torch.arange(total_points, device=device)
    indices = torch.randperm(total_points, device=device)[:num_queries]
    return indices
```

#### Stratified Sampling (lines 39-103):
```python
def sample_stratified_queries(
    grid_shape: Tuple[int, int],
    num_queries: int,
    device: torch.device = None,
) -> torch.Tensor:
    H, W = grid_shape
    total_points = H * W
    
    if num_queries >= total_points:
        return torch.arange(total_points, device=device)
    
    blocks_per_dim = max(1, int(num_queries ** 0.5))
    block_h = H // blocks_per_dim
    block_w = W // blocks_per_dim
    queries_per_block = max(1, num_queries // (blocks_per_dim ** 2))
    
    # Sample proportionally from each block...
```

### 3.2 Usage in Training

**File**: `/Users/emerygunselman/Code/universal_simulator/scripts/train.py:574-577`

```python
query_sample_cfg = train_cfg.get("query_sampling", {})
use_query_sampling = query_sample_cfg.get("enabled", False)
num_queries = query_sample_cfg.get("num_queries", None) if use_query_sampling else None
query_strategy = query_sample_cfg.get("strategy", "uniform")
```

**Passed to Loss Computation** (lines 734-735):
```python
num_queries=num_queries,
query_strategy=query_strategy,
```

### 3.3 Configuration: `num_queries: 2048`

**From config** `/Users/emerygunselman/Code/universal_simulator/configs/train_burgers_upt_full.yaml:95-102`:

```yaml
query_sampling:
  enabled: true
  num_queries: 2048              # Sample 2k points per batch (vs. 4096 dense for 64×64 grid)
  strategy: uniform              # Start with uniform, can try "stratified"
  
  # Query sampling currently applies to inverse_encoding_loss only.
  # inverse_decoding_loss always uses full grid (GridEncoder requirement).
  # This still provides 20-30% speedup since inverse_encoding dominates cost.
```

**Memory Impact**:
- Full 64×64 grid: 4096 points per batch
- Sampled: 2048 points (50% reduction)
- Decoder memory: O(B × 2048 × hidden_dim) vs O(B × 4096 × hidden_dim)
- Speedup: 20-30% (inverse_encoding dominates inverse_decoding cost)

---

## 4. Loss Bundle Integration

### 4.1 Loss Bundle Computation

**File**: `/Users/emerygunselman/Code/universal_simulator/src/ups/training/losses.py:267-369`

**Function**: `compute_operator_loss_bundle()`

**Inverse Loss Integration** (lines 331-349):

```python
# Apply curriculum learning to inverse loss weights if epoch is provided
lambda_inv_enc = weights.get("lambda_inv_enc", 0.0)
lambda_inv_dec = weights.get("lambda_inv_dec", 0.0)

if current_epoch is not None:
    warmup_epochs = weights.get("inverse_loss_warmup_epochs", 15)
    max_weight = weights.get("inverse_loss_max_weight", 0.05)
    
    lambda_inv_enc = compute_inverse_loss_curriculum_weight(
        current_epoch, lambda_inv_enc, warmup_epochs, max_weight
    )
    lambda_inv_dec = compute_inverse_loss_curriculum_weight(
        current_epoch, lambda_inv_dec, warmup_epochs, max_weight
    )

# UPT Inverse Encoding Loss (with optional query sampling)
if all(x is not None for x in [input_fields, encoded_latent, decoder, input_positions]):
    comp["L_inv_enc"] = inverse_encoding_loss(
        input_fields, encoded_latent, decoder, input_positions,
        weight=lambda_inv_enc,
        num_queries=num_queries,
        query_strategy=query_strategy,
        grid_shape=grid_shape,
    )

# UPT Inverse Decoding Loss (with optional query sampling)
if all(x is not None for x in [encoded_latent, decoder, encoder, query_positions, coords, meta]):
    comp["L_inv_dec"] = inverse_decoding_loss(
        encoded_latent, decoder, encoder, query_positions, coords, meta,
        weight=lambda_inv_dec,
        num_queries=num_queries,
        query_strategy=query_strategy,
        grid_shape=grid_shape,
    )
```

### 4.2 Curriculum Learning Schedule

**File**: `/Users/emerygunselman/Code/universal_simulator/src/ups/training/losses.py:233-264`

```python
def compute_inverse_loss_curriculum_weight(
    epoch: int,
    base_weight: float,
    warmup_epochs: int = 15,
    max_weight: float = 0.05,
) -> float:
    """Compute curriculum-scheduled weight for inverse losses.
    
    Epochs 0-warmup_epochs: weight = 0 (pure forward training)
    Epochs warmup_epochs to warmup_epochs*2: linear ramp from 0 to base_weight
    Epochs > warmup_epochs*2: weight = min(base_weight, max_weight)
    """
    if epoch < warmup_epochs:
        return 0.0
    elif epoch < warmup_epochs * 2:
        progress = (epoch - warmup_epochs) / warmup_epochs
        return min(base_weight * progress, max_weight)
    else:
        return min(base_weight, max_weight)
```

**Config Parameters**:
```yaml
inverse_loss_warmup_epochs: 5        # Warmup period
inverse_loss_max_weight: 0.05        # Cap on maximum weight
```

---

## 5. Training Loop Integration

### 5.1 Encoder/Decoder Instantiation

**File**: `/Users/emerygunselman/Code/universal_simulator/scripts/train.py:496-553`

**Encoder Creation**:
```python
encoder_cfg = GridEncoderConfig(
    latent_len=latent_cfg.get("tokens", 32),
    latent_dim=latent_cfg.get("dim", 16),
    field_channels=field_channels,
    patch_size=data_cfg.get("patch_size", 4),
)
encoder = (shared_encoder or GridEncoder(encoder_cfg)).to(device)
encoder.eval()  # Encoder is frozen during operator stage
```

**Decoder Creation**:
```python
decoder_cfg = AnyPointDecoderConfig(
    latent_dim=latent_cfg.get("dim", 16),
    query_dim=2,
    hidden_dim=ttc_decoder_cfg.get("hidden_dim", latent_cfg.get("dim", 16) * 4),
    num_layers=ttc_decoder_cfg.get("num_layers", 2),
    num_heads=ttc_decoder_cfg.get("num_heads", 4),
    frequencies=tuple(ttc_decoder_cfg.get("frequencies", [1.0, 2.0, 4.0])),
    mlp_hidden_dim=ttc_decoder_cfg.get("mlp_hidden_dim", 128),
    output_channels=field_channels,
)
decoder = AnyPointDecoder(decoder_cfg).to(device)
decoder.eval()  # Decoder not trained during operator stage
```

### 5.2 Per-Batch Loss Computation

**File**: `/Users/emerygunselman/Code/universal_simulator/scripts/train.py:650-774`

**Loss Weights Setup** (lines 656-665):
```python
loss_weights = {
    "lambda_forward": 1.0,
    "lambda_inv_enc": float(train_cfg.get("lambda_inv_enc", 0.0)),
    "lambda_inv_dec": float(train_cfg.get("lambda_inv_dec", 0.0)),
    "lambda_spectral": lam_spec,
    "lambda_rollout": lam_rollout,
    "inverse_loss_warmup_epochs": int(train_cfg.get("inverse_loss_warmup_epochs", 15)),
    "inverse_loss_max_weight": float(train_cfg.get("inverse_loss_max_weight", 0.05)),
}
```

**Conditional Inverse Loss Computation** (lines 683-684):
```python
inv_freq = int(train_cfg.get("inverse_loss_frequency", 1) or 1)
use_inv_now = use_inverse_losses and (inv_freq > 0) and (i % inv_freq == 0)
```

**Loss Bundle Call** (lines 746-774):
```python
loss_bundle = compute_operator_loss_bundle(
    input_fields=input_fields_physical if use_inv_now else None,
    encoded_latent=state.z if use_inv_now else None,
    decoder=decoder if use_inv_now else None,
    input_positions=coords if use_inv_now else None,
    encoder=encoder if use_inv_now else None,
    query_positions=coords if use_inv_now else None,
    coords=coords if use_inv_now else None,
    meta=meta if use_inv_now else None,
    pred_next=next_state.z,
    target_next=target,
    pred_rollout=rollout_pred,
    target_rollout=rollout_tgt,
    spectral_pred=next_state.z if lam_spec > 0 else None,
    spectral_target=target if lam_spec > 0 else None,
    weights=loss_weights,
    current_epoch=epoch,
    num_queries=num_queries,
    query_strategy=query_strategy,
    grid_shape=grid_shape,
)
```

---

## 6. Memory Overhead Analysis

### 6.1 Decoder Forward Pass Memory

**File**: `/Users/emerygunselman/Code/universal_simulator/src/ups/io/decoder_anypoint.py:94-134`

**Decoder Architecture**:
```python
def forward(
    self,
    points: torch.Tensor,              # (B, Q, query_dim) where Q = num_queries
    latent_tokens: torch.Tensor,       # (B, tokens, latent_dim)
    conditioning: Optional[Mapping] = None,
) -> Dict[str, torch.Tensor]:
    B, Q, _ = points.shape
    latents = latent_tokens
    
    # Projection
    latents = self.latent_proj(latents)                    # (B, tokens, hidden_dim)
    enriched_points = _fourier_encode(points, ...)         # (B, Q, feat_dim)
    queries = self.query_embed(enriched_points)            # (B, Q, hidden_dim)
    
    # Cross-attention layers
    for attn, ln_q, ff, ln_ff in self.layers:
        attn_out, _ = attn(queries, latents, latents)      # (B, Q, hidden_dim)
        queries = ln_q(queries + attn_out)
        ff_out = ff(queries)
        queries = ln_ff(queries + ff_out)
    
    # Output heads
    outputs: Dict[str, torch.Tensor] = {}
    for name, head in self.heads.items():
        x = head(queries)                                   # (B, Q, out_channels)
        outputs[name] = x
    
    return outputs
```

**Memory Components**:
1. **Fourier Encoding** (line 127):
   - Input: (B, Q, 2) coordinates
   - Output: (B, Q, 2 + 2*len(frequencies)*2)
   - With frequencies=(1.0, 2.0, 4.0): (B, Q, 14)
   - Memory: O(B × Q × 14) floats

2. **Query Embedding** (line 128):
   - Input: (B, Q, 14)
   - Output: (B, Q, hidden_dim)
   - Memory: O(B × Q × hidden_dim)

3. **Cross-Attention** (line 131):
   - Query: (B, Q, hidden_dim)
   - Key/Value: (B, tokens, hidden_dim)
   - Attention scores: (B, num_heads, Q, tokens)
   - Memory: O(B × num_heads × Q × tokens) + intermediate outputs
   - **Key: linear in Q (num_queries)**

4. **Per-Layer Outputs**:
   - Attention output: (B, Q, hidden_dim)
   - MLP output: (B, Q, mlp_hidden_dim)
   - Memory: O(B × Q × max(hidden_dim, mlp_hidden_dim))

**Memory Savings with Query Sampling**:
- Without sampling (Q=4096): O(B × 4096 × hidden_dim) = O(B × 4096 × 256) ≈ 1M floats/batch
- With sampling (Q=2048): O(B × 2048 × 256) ≈ 512K floats/batch
- **Reduction: 50%**

### 6.2 Encoder Forward Pass Memory

**File**: `/Users/emerygunselman/Code/universal_simulator/src/ups/io/enc_grid.py:81-107`

```python
def forward(
    self,
    fields: Dict[str, torch.Tensor],   # {name: (B, H*W, C)}
    coords: torch.Tensor,               # (B, H*W, 2)
    meta: Optional[Mapping] = None,
) -> torch.Tensor:
    grid_shape = self._infer_grid_shape(meta)
    H, W = grid_shape
    
    tokens, features = self._encode_fields(fields, coords, grid_shape)  # (B, tokens, patch_channels)
    features = features.to(target_device)
    latent = self.to_latent(features)   # (B, tokens, latent_dim)
    
    if tokens != self.cfg.latent_len:
        latent = self._adaptive_token_pool(latent, self.cfg.latent_len)
    return latent
```

**Note**: Inverse decoding loss ALWAYS uses full grid (no query sampling):
- Encoder requires full H×W grid
- Cannot subsample and re-encode
- This is why `num_queries` parameter is ignored in `inverse_decoding_loss()`

**Memory**: O(B × H × W × hidden_channels) for intermediate features

### 6.3 Backward Pass Memory

**File**: `/Users/emerygunselman/Code/universal_simulator/scripts/train.py:806-819`

```python
# Backward with gradient accumulation
if use_amp:
    scaler.scale(loss / accum_steps).backward()
else:
    (loss / accum_steps).backward()

do_step = ((i + 1) % accum_steps == 0) or ((i + 1) == num_batches)
if do_step:
    if use_amp:
        if clip_val is not None:
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(operator.parameters(), ...)
        scaler.step(optimizer)
        scaler.update()
```

**Memory Impact**:
- Activation recomputation: O(B × tokens × latent_dim) for each layer
- Gradient storage: O(parameters_count)
- With `detach()` on encoder latent (line 135 in losses.py): Saves O(B × tokens × latent_dim) gradient

### 6.4 Reference Fields Storage

**File**: `/Users/emerygunselman/Code/universal_simulator/scripts/train.py:696-703`

```python
# Store reference fields at first batch (for conservation checks)
if not reference_fields_storage and input_fields_physical is not None:
    reference_fields_storage.update({
        k: v.detach().clone() for k, v in input_fields_physical.items()
    })
reference_fields = reference_fields_storage
```

**Memory Cost**:
- Clone: O(B × H × W × field_channels) for each field
- Persistent storage for entire training epoch
- With detach: Prevents gradient accumulation

---

## 7. Summary Table: Memory-Intensive Operations

| Component | Location | Memory Cost | Query Sampling Impact | Notes |
|-----------|----------|-------------|----------------------|-------|
| Inverse Encoding Loss | losses.py:25-82 | O(B × Q × hidden_dim) | -50% with Q=2048 | Only applied if enabled |
| Inverse Decoding Loss | losses.py:85-135 | O(B × H×W × hidden_dim) | No sampling | Requires full grid |
| Query Sampling | query_sampling.py:106-153 | O(Q) indices | Reduces downstream memory | Uniform or stratified |
| Decoder Forward | decoder_anypoint.py:94-134 | O(B × Q × hidden_dim) | -50% with Q=2048 | Cross-attention dominates |
| Encoder Forward | enc_grid.py:81-107 | O(B × H×W × hidden_channels) | No sampling | Always full grid |
| Encoder Backward | train.py:806 | O(B × tokens × latent_dim) | None | Detached to avoid double backprop |
| Reference Fields | train.py:700-701 | O(B × H×W × field_channels) | None | Single clone per epoch |

---

## 8. Configuration Examples

### 8.1 Recommended Configuration

**From** `/Users/emerygunselman/Code/universal_simulator/configs/train_burgers_upt_full.yaml`:

```yaml
training:
  # Inverse losses
  lambda_inv_enc: 0.01              # Encoding reconstruction weight
  lambda_inv_dec: 0.01              # Decoding reconstruction weight
  use_inverse_losses: true
  inverse_loss_frequency: 1         # Apply every batch
  inverse_loss_warmup_epochs: 5     # Warmup before full strength
  inverse_loss_max_weight: 0.05     # Cap maximum weight

  # Query sampling (Phase 4.1)
  query_sampling:
    enabled: true
    num_queries: 2048               # 50% of 64×64 grid
    strategy: uniform               # uniform or stratified
```

### 8.2 Multi-Task Configuration

**From** `/Users/emerygunselman/Code/universal_simulator/configs/train_pdebench_2task_192d.yaml`:

```yaml
training:
  query_sampling:
    num_queries: 1024               # OPTIMIZED: Reduced from 2048 for 2x speedup
```

---

## 9. References

### Key Files with Line Numbers

**Loss Functions**:
- `inverse_encoding_loss()`: `/Users/emerygunselman/Code/universal_simulator/src/ups/training/losses.py:25-82`
- `inverse_decoding_loss()`: `/Users/emerygunselman/Code/universal_simulator/src/ups/training/losses.py:85-135`
- `compute_operator_loss_bundle()`: `/Users/emerygunselman/Code/universal_simulator/src/ups/training/losses.py:267-369`
- Curriculum schedule: `/Users/emerygunselman/Code/universal_simulator/src/ups/training/losses.py:233-264`

**Query Sampling**:
- `apply_query_sampling()`: `/Users/emerygunselman/Code/universal_simulator/src/ups/training/query_sampling.py:106-153`
- `sample_uniform_queries()`: `/Users/emerygunselman/Code/universal_simulator/src/ups/training/query_sampling.py:15-36`
- `sample_stratified_queries()`: `/Users/emerygunselman/Code/universal_simulator/src/ups/training/query_sampling.py:39-103`

**Training Loop**:
- Parameter extraction: `/Users/emerygunselman/Code/universal_simulator/scripts/train.py:496-577`
- Loss computation: `/Users/emerygunselman/Code/universal_simulator/scripts/train.py:650-774`
- Backward pass: `/Users/emerygunselman/Code/universal_simulator/scripts/train.py:806-819`

**Models**:
- `AnyPointDecoder`: `/Users/emerygunselman/Code/universal_simulator/src/ups/io/decoder_anypoint.py:58-160`
- `GridEncoder`: `/Users/emerygunselman/Code/universal_simulator/src/ups/io/enc_grid.py:22-137`

**Configuration**:
- Example with all features: `/Users/emerygunselman/Code/universal_simulator/configs/train_burgers_upt_full.yaml:86-126`
- Multi-task optimized: `/Users/emerygunselman/Code/universal_simulator/configs/train_pdebench_2task_192d.yaml`

