# UPT Integration Analysis: Gaps and Improvement Recommendations

**Date**: 2025-01-23
**Purpose**: Systematic analysis comparing Universal Physics Stack (UPS) with Universal Physics Transformers (UPT) architecture to identify optimal improvements

**Documents Referenced**:
- `UPT_docs/UPT_Implementation_Plan.md` - UPT implementation guide
- `UPT_docs/UPT_Arch_Train_Scaling.md` - UPT architecture and scaling playbook
- Current UPS codebase in `src/ups/`

---

## Executive Summary

The Universal Physics Stack (UPS) already implements many UPT architectural principles but has **critical gaps** in training objectives that limit its ability to perform efficient latent-space-only rollouts. The most impactful improvements would be:

1. **[CRITICAL]** Implement inverse encoding and inverse decoding losses
2. **[HIGH]** Increase latent token count from 16-32 to 256-512
3. **[MEDIUM]** Simplify approximator architecture to pure transformer
4. **[LOW]** Adopt UPT encoder architecture for CFD/mesh domains

---

## 1. Architecture Comparison

### 1.1 Encoder Architecture

| Aspect | UPT (CFD Reference) | UPS GridEncoder | UPS MeshParticleEncoder | Gap Level |
|--------|---------------------|-----------------|-------------------------|-----------|
| **Input Processing** | Graph-based | Grid patches (pixel unshuffle) | Message passing on graph | ✓ Different domains |
| **Pooling Stages** | GNN → Transformer → Perceiver | Patch → Residual stems → Adaptive pool | Message passing → Supernode → Perceiver | ✓ Similar |
| **Supernodes** | 512-2048 | N/A (patches) | 2048 (configurable) | ✓ Match |
| **Latent Tokens** | 256-512 | 16-32 | 8-32 | ❌ **4-16x smaller!** |
| **Latent Dim** | 192-384 | 32 | 16-32 | ❌ **6-12x smaller!** |
| **Fourier Features** | Implicit in perceiver | Explicit (sin/cos) | Not used | ✓ Different approaches |
| **Conditioning** | Dit* blocks | Not implemented | Not implemented | ⚠️ Missing |

**Key Files**:
- UPS: `src/ups/io/enc_grid.py`, `src/ups/io/enc_mesh_particle.py`
- UPT Reference: `encoders/cfd_gnn_pool_transformer_perceiver.py`

**Analysis**:
- **GridEncoder** is optimized for structured grids (Burgers, NS) using patch-based compression
- **MeshParticleEncoder** is closer to UPT's CFD encoder but uses simpler message passing
- **Critical Gap**: UPS uses 16-32 latent tokens vs UPT's 256-512 → may limit capacity for complex PDEs

### 1.2 Approximator/Operator Architecture

| Aspect | UPT | UPS LatentOperator | Gap Level |
|--------|-----|-------------------|-----------|
| **Core Design** | Pure transformer blocks | PDE-Transformer (U-shaped) | ⚠️ Different |
| **Architecture** | Stack of transformer layers | Down → Bottleneck → Up with skip connections | ⚠️ More complex |
| **Token Count** | Fixed (256-512) | Hierarchical (pooling/upsampling) | ⚠️ Variable |
| **Depth** | 4-12 layers | 3-9 layers (per phase) | ✓ Similar |
| **Latent Dim** | 192-384 | 32 | ❌ **6-12x smaller!** |
| **Time Embedding** | Learned embedding | TimeEmbedding (sin + MLP) | ✓ Similar |
| **Conditioning** | AdaLN | AdaLN (AdaLNConditioner) | ✓ Match |
| **Efficiency** | Standard attention | Shifted window (available but unused) | ✓ Similar |
| **Drop Path** | 0.0-0.2 | Not implemented | ⚠️ Missing |

**Key Files**:
- UPS: `src/ups/models/latent_operator.py`, `src/ups/core/blocks_pdet.py`
- UPT Reference: Appendix pseudo-code (approximator transformer)

**Analysis**:
- **UPS uses U-shaped architecture** (similar to U-Net) which may provide better multi-scale processing
- **UPT uses simple stacked transformer** which is easier to scale and has clearer latent space semantics
- **Critical Gap**: UPS latent dim is 6-12x smaller → may explain why UPS needs more complex architecture
- **Drop-path regularization** is mentioned in UPT but not implemented in UPS

### 1.3 Decoder Architecture

| Aspect | UPT | UPS AnyPointDecoder | Gap Level |
|--------|-----|-------------------|-----------|
| **Core Design** | Perceiver cross-attention | Perceiver cross-attention | ✓ **Match!** |
| **Query Encoding** | Position MLP + pos-enc | Fourier encoding + MLP | ✓ Similar |
| **Cross-Attention** | Queries attend to latent | Queries attend to latent | ✓ Match |
| **Layers** | 2-4 | 2 (configurable) | ✓ Similar |
| **Heads** | 4-8 | 4 (configurable) | ✓ Match |
| **Output Heads** | Per-field MLPs | Per-field linear heads | ✓ Match |
| **Clamping** | Optional log-clamp | Not implemented | ⚠️ Minor gap |

**Key Files**:
- UPS: `src/ups/io/decoder_anypoint.py`
- UPT Reference: `decoders/cfd_transformer_perceiver.py`

**Analysis**:
- **Decoder architectures are nearly identical!** Both use perceiver-IO pattern
- UPS implementation is clean and follows UPT design principles
- Minor gap: UPT mentions optional log-clamping for stability (not implemented in UPS)

---

## 2. Training Procedure Comparison

### 2.1 Loss Functions

| Loss Component | UPT | UPS | Gap Level |
|----------------|-----|-----|-----------|
| **Forward Prediction** | MSE/MAE at query points | One-step MSE (`L_one_step`) | ✓ Match |
| **Inverse Encoding** | Reconstruct input from latent | **NOT IMPLEMENTED** | ❌ **CRITICAL!** |
| **Inverse Decoding** | Reconstruct latent from decoder output | **NOT IMPLEMENTED** | ❌ **CRITICAL!** |
| **Rollout Loss** | Multi-step latent rollout | Multi-step MSE (`L_rollout`) | ✓ Match |
| **Spectral Loss** | Optional | Spectral energy (`L_spec`) | ✓ Match |
| **Consistency Loss** | Not mentioned | Mean preservation (`L_cons`) | ✓ Extra |
| **Physics Priors** | Divergence/continuity penalties | Not implemented | ⚠️ Missing |
| **Latent Norm Penalty** | Mentioned | Not implemented | ⚠️ Missing |

**Key Files**:
- UPS: `src/ups/training/losses.py`
- UPT Reference: Section 5.2 (Inverse reconstruction losses)

**Analysis**:
- **CRITICAL GAP: Missing inverse encoding and inverse decoding losses**
  - UPT explicitly states these are "critical for latent rollout"
  - These losses enable the E/A/D partition to be disentangled
  - Without them, encoder/decoder may not be invertible → latent rollout less stable

**UPT Inverse Loss Definitions** (from Section 5.2):
```
Inverse Encoding Loss:
  1. Encode input fields to latent: z = E(u)
  2. Decode latent back to input positions: u_recon = D(z, input_positions)
  3. Loss: MSE(u_recon, u)
  Purpose: Ensures encoder output is reconstructible via decoder

Inverse Decoding Loss:
  1. Decode latent to query positions: u_pred = D(z, query_positions)
  2. Re-encode predicted fields to latent: z_recon = E(u_pred)
  3. Loss: MSE(z_recon, z)
  Purpose: Ensures decoder outputs are encodable back to latent
```

**Current UPS Losses** (from `losses.py`):
- `inverse_encoding_loss`: EXISTS but different semantics (reconstructs latent from latent)
- `inverse_decoding_loss`: EXISTS but different semantics (MSE between physical fields)
- Neither implements the true E/A/D disentanglement described in UPT

### 2.2 Training Pipeline

| Aspect | UPT | UPS | Gap Level |
|--------|-----|-----|-----------|
| **Training Stages** | Single-stage with joint losses | Multi-stage (operator → diffusion → distill) | ⚠️ Different philosophy |
| **Latent Rollout Training** | Explicit inverse losses | Rollout loss but no inverse | ❌ Gap |
| **Diffusion Model** | Optional residual correction | Core component | ✓ UPS more advanced |
| **Consistency Distillation** | Not mentioned | Implemented | ✓ UPS more advanced |
| **Optimizer** | AdamW + cosine | AdamW + cosine | ✓ Match |
| **Learning Rate** | 1e-3 (typical) | 1e-3 (typical) | ✓ Match |
| **AMP** | Mentioned | Implemented (bf16) | ✓ Match |
| **EMA** | 0.999 | 0.999 (optional) | ✓ Match |
| **Gradient Clipping** | 1.0 | 1.0 | ✓ Match |

**Key Files**:
- UPS: `src/ups/training/loop_train.py`, `scripts/train.py`
- UPT Reference: Section 10 (Training loop)

**Analysis**:
- **UPS has MORE advanced training pipeline** with diffusion + distillation
- **UPT focuses on core operator training** with proper inverse losses
- **Key Insight**: UPS could benefit from adding UPT's inverse losses to operator stage

### 2.3 Data Handling

| Aspect | UPT | UPS | Gap Level |
|--------|-----|-----|-----------|
| **Latent Caching** | Not explicitly mentioned | Implemented (`latent_pairs.py`) | ✓ UPS advantage |
| **Query Sampling** | 2-8k query points per batch | Not explicitly used | ⚠️ Different approach |
| **Time Stride** | Not mentioned | Implemented | ✓ UPS feature |
| **Rollout Horizon** | Curriculum (increase over training) | Configurable | ✓ Similar |
| **Multi-Task Training** | Not mentioned | Implemented (ConcatDataset) | ✓ UPS feature |
| **Normalization Stats** | Computed and cached | Computed and cached | ✓ Match |

**Key Files**:
- UPS: `src/ups/data/latent_pairs.py`, `src/ups/data/parallel_cache.py`

**Analysis**:
- **UPS has superior data infrastructure** with caching, multi-task support
- **UPT uses query-based sampling** which UPS doesn't explicitly implement (decoder is query-based but training isn't)

---

## 3. Scale Comparison

### 3.1 Model Capacity

| Configuration | UPT-8M | UPT-17M | UPT-68M | UPS (Golden) | Gap |
|---------------|--------|---------|---------|--------------|-----|
| **Latent Tokens** | 256 | 256-512 | 512-768 | 16 | ❌ **16-48x smaller** |
| **Latent Dim** | 128 | 192 | 384 | 32 | ❌ **4-12x smaller** |
| **Approx Hidden** | 128 | 192 | 384 | 64 | ❌ **2-6x smaller** |
| **Approx Depth** | 4 | 4 | 8-12 | ~6-9 (U-net) | ✓ Comparable |
| **Num Heads** | 4 | 4-6 | 6-8 | 4 | ✓ Match small |
| **Supernodes** | 512-1024 | 1024 | 2048 | N/A (grid) / 2048 (mesh) | ✓ Match mesh |

**Analysis**:
- **UPS operates at MUCH smaller capacity** than even UPT-8M
- This may explain:
  - Why UPS needs diffusion residual (not enough deterministic capacity)
  - Why UPS uses U-shaped architecture (compensates for fewer tokens)
  - Why UPS struggles with complex PDEs

**Recommendation**: Scale up to UPT-17M config as baseline:
- Increase `latent.tokens` from 16 → 256
- Increase `latent.dim` from 32 → 192
- Simplify architecture to pure transformer (if tokens increased)

---

## 4. Detailed Gap Analysis

### 4.1 CRITICAL Gaps (Immediate Impact)

#### Gap 1: Missing Inverse Encoding Loss ❌
**Description**: UPT requires inverse encoding loss to ensure latent representations are decodable

**Current State**:
- `losses.py` has `inverse_encoding_loss()` function but it reconstructs latent from latent (not physical from latent)
- Training pipeline doesn't use true inverse encoding

**UPT Requirement** (Section 5.2):
```
Inverse Encoding Loss:
  z = Encoder(u_input)
  u_reconstructed = Decoder(z, input_positions)
  L_inv_enc = MSE(u_reconstructed, u_input)
```

**Implementation Path**:
1. Modify `inverse_encoding_loss()` in `losses.py`:
   ```python
   def inverse_encoding_loss(
       input_fields: Mapping[str, Tensor],
       latent: Tensor,
       decoder: AnyPointDecoder,
       input_positions: Tensor,
       weight: float = 1.0
   ) -> Tensor:
       # Decode latent back to input positions
       reconstructed = decoder(input_positions, latent)

       # MSE between reconstructed and original fields
       losses = []
       for name in input_fields:
           losses.append(mse(reconstructed[name], input_fields[name]))
       return weight * torch.stack(losses).mean()
   ```

2. Update `train_operator()` in `scripts/train.py` to use this loss
3. Add `lambda_inv_enc` weight to config (typical: 0.1-1.0)

**Expected Impact**:
- Improved latent rollout stability (correlation time)
- Better zero-shot super-resolution
- Clearer encoder/decoder separation of concerns

---

#### Gap 2: Missing Inverse Decoding Loss ❌
**Description**: UPT requires inverse decoding loss to ensure decoder outputs are re-encodable

**Current State**:
- `losses.py` has `inverse_decoding_loss()` but it only does MSE between physical fields (not latent reconstruction)

**UPT Requirement** (Section 5.2):
```
Inverse Decoding Loss:
  u_decoded = Decoder(z, query_positions)
  z_reconstructed = Encoder(u_decoded)
  L_inv_dec = MSE(z_reconstructed, z)
```

**Implementation Path**:
1. Modify `inverse_decoding_loss()` in `losses.py`:
   ```python
   def inverse_decoding_loss(
       latent: Tensor,
       decoder: AnyPointDecoder,
       encoder: GridEncoder,
       query_positions: Tensor,
       coords: Tensor,
       meta: dict,
       weight: float = 1.0
   ) -> Tensor:
       # Decode to physical space
       decoded_fields = decoder(query_positions, latent)

       # Re-encode to latent
       latent_reconstructed = encoder(decoded_fields, coords, meta=meta)

       # MSE in latent space
       return weight * mse(latent_reconstructed, latent.detach())
   ```

2. Update operator training loop to include this loss
3. Add `lambda_inv_dec` weight to config (typical: 0.1-1.0)

**Expected Impact**:
- Ensures E/A/D partition is truly disentangled
- Enables pure latent rollout at inference (encode once, step in latent, decode at end)
- Reduces encoder/decoder coupling

---

#### Gap 3: Latent Token Count Too Small ❌
**Description**: UPS uses 16-32 latent tokens vs UPT's 256-512

**Current State**:
- Golden config: `latent.tokens = 16`
- This is **16-32x smaller** than UPT recommendations

**Impact**:
- Limited model capacity for complex PDEs
- Information bottleneck between encoder/decoder
- May explain need for diffusion residual model

**Implementation Path**:
1. **Staged Scale-Up** (to avoid breaking existing checkpoints):

   **Stage 1: Medium (UPT-17M equivalent)**
   ```yaml
   latent:
     tokens: 256  # Up from 16
     dim: 192     # Up from 32

   operator:
     pdet:
       input_dim: 192      # Match latent.dim
       hidden_dim: 384     # 2x latent.dim
       depths: [4, 4, 4]   # Simplify from U-net
   ```

   **Stage 2: Large (UPT-68M equivalent)**
   ```yaml
   latent:
     tokens: 512
     dim: 384

   operator:
     pdet:
       input_dim: 384
       hidden_dim: 768
       depths: [8, 8, 8]
   ```

2. **Update Encoder Configs**:
   ```python
   # In configs/train_burgers_upt17m.yaml
   encoder:
     type: grid
     latent_len: 256    # Up from 16
     latent_dim: 192    # Up from 32
     patch_size: 4
   ```

3. **Benchmark**: Run ablation study comparing 16/32/64/128/256 tokens on Burgers

**Expected Impact**:
- Higher capacity → better performance on complex PDEs
- May reduce need for diffusion residual
- Longer training time (more parameters)

**Estimate**: UPT-17M config (~30M params vs current ~2M) → 15x more parameters, likely 3-5x longer training

---

### 4.2 HIGH Priority Gaps (Significant Impact)

#### Gap 4: Latent Dimension Too Small ⚠️
**Description**: UPS uses latent_dim=32 vs UPT's 192-384

**Recommendation**: Increase to 192 (UPT-17M) or 384 (UPT-68M)

**Implementation**: Update all dimension configs to match:
```yaml
latent:
  dim: 192

operator:
  pdet:
    input_dim: 192
    hidden_dim: 384  # 2x

diffusion:
  latent_dim: 192

decoder:
  latent_dim: 192
```

**Trade-off**: 6x more parameters → longer training, higher memory

---

#### Gap 5: Approximator Architecture Complexity ⚠️
**Description**: UPS uses U-shaped PDE-Transformer vs UPT's pure transformer stack

**Current State**:
- `blocks_pdet.py`: Down phase (pooling) → Bottleneck → Up phase (unpooling)
- Complex skip connections and token count changes

**UPT Approach**:
- Fixed token count throughout
- Simple stacked transformer blocks
- Easier to scale and reason about

**Recommendation**:
- **If tokens remain 16-32**: Keep U-shaped architecture (needed for efficiency)
- **If tokens scale to 256-512**: Simplify to pure transformer stack

**Implementation** (if scaling tokens):
```python
class SimpleLatentTransformer(nn.Module):
    def __init__(self, latent_dim, depth, num_heads, drop_path=0.0):
        self.layers = nn.ModuleList([
            TransformerLayer(
                latent_dim,
                num_heads,
                mlp_ratio=4.0,
                drop_path=drop_path * i / depth  # Stochastic depth
            )
            for i in range(depth)
        ])

    def forward(self, z, cond=None):
        for layer in self.layers:
            z = layer(z, cond)
        return z
```

---

#### Gap 6: Missing Drop-Path Regularization ⚠️
**Description**: UPT recommends drop-path 0.0-0.2 for deep networks

**Implementation**:
1. Add `DropPath` to transformer layers (already in PyTorch/timm)
2. Add config parameter:
   ```yaml
   operator:
     pdet:
       drop_path: 0.1  # Linear schedule from 0 to 0.1
   ```

**Expected Impact**: Better generalization for deep (8-12 layer) models

---

### 4.3 MEDIUM Priority Gaps (Incremental Improvements)

#### Gap 7: Missing Physics Priors ⚠️
**Description**: UPT mentions divergence/continuity penalties for fluid flows

**Recommendation**: Add physics-informed losses for CFD domains
```python
def divergence_penalty(velocity_field: Tensor, grid_spacing: float) -> Tensor:
    # Compute ∇·u via finite differences
    du_dx = torch.gradient(velocity_field[..., 0], spacing=grid_spacing, dim=-2)
    dv_dy = torch.gradient(velocity_field[..., 1], spacing=grid_spacing, dim=-1)
    divergence = du_dx + dv_dy
    return divergence.abs().mean()
```

---

#### Gap 8: Missing Latent Norm Penalty ⚠️
**Description**: UPT uses latent norm regularization

**Implementation**:
```python
def latent_norm_penalty(latent: Tensor, weight: float = 1e-4) -> Tensor:
    norm = latent.norm(p=2, dim=-1).mean()
    return weight * norm
```

Add to operator training loss bundle.

---

#### Gap 9: Decoder Clamping Not Implemented ⚠️
**Description**: UPT mentions optional log-clamping for stability

**Implementation** (in `decoder_anypoint.py`):
```python
# In output heads
if self.cfg.use_log_clamp:
    outputs[name] = torch.sign(x) * torch.log1p(x.abs() / self.cfg.clamp_threshold)
```

---

#### Gap 10: Query-Based Training Not Implemented ⚠️
**Description**: UPT trains with 2-8k random query points per batch

**Current UPS**: Trains on full grid latent pairs

**UPT Approach**:
- Sample query points randomly per batch
- Enables training on arbitrary discretizations
- Better generalization to unseen resolutions

**Implementation**:
```python
# In training loop
def sample_query_points(grid_shape, num_queries=4096):
    B, H, W = grid_shape
    indices = torch.randint(0, H*W, (num_queries,))
    return indices

# In loss computation
query_indices = sample_query_points(batch_shape, num_queries=4096)
query_coords = coords[:, query_indices, :]
target_values = fields[:, query_indices, :]
```

---

### 4.4 LOW Priority Gaps (Domain-Specific)

#### Gap 11: CFD-Specific Encoder Architecture
**Description**: UPT uses GNN pooling for CFD meshes

**Current UPS**: `MeshParticleEncoder` uses simple message passing

**Recommendation**: Only implement if targeting CFD benchmarks (ShapeNet-Car, OpenFOAM)

---

## 5. Prioritized Improvement Roadmap

### Phase 1: Critical Fixes (Immediate - 1-2 weeks)

**Goal**: Fix training losses to enable true latent rollout

1. **Implement True Inverse Encoding Loss**
   - File: `src/ups/training/losses.py`
   - Requires: Decoder access in training loop
   - Config: Add `lambda_inv_enc = 0.5`
   - Validation: Check reconstruction MSE on validation set

2. **Implement True Inverse Decoding Loss**
   - File: `src/ups/training/losses.py`
   - Requires: Encoder access in training loop
   - Config: Add `lambda_inv_dec = 0.5`
   - Validation: Check latent reconstruction MSE

3. **Update Operator Training Loop**
   - File: `scripts/train.py` (`train_operator()` function)
   - Changes: Pass encoder+decoder to loss computation
   - Validation: Ensure losses decrease during training

4. **Create Ablation Config**
   - File: `configs/train_burgers_upt_losses.yaml`
   - Based on: `train_burgers_golden.yaml`
   - Changes: Add inverse loss weights
   - Benchmark: Compare NRMSE with/without inverse losses

**Expected Outcome**: Improved latent rollout stability, validation NRMSE improvement of 10-20%

---

### Phase 2: Scale-Up Experiment (2-4 weeks)

**Goal**: Test UPT-17M configuration

1. **Create UPT-17M Config**
   - File: `configs/train_burgers_upt17m.yaml`
   - Changes:
     ```yaml
     latent:
       tokens: 256  # Up from 16
       dim: 192     # Up from 32

     operator:
       pdet:
         input_dim: 192
         hidden_dim: 384
         depths: [4, 4, 4]  # Simplified from U-net
         num_heads: 6
         drop_path: 0.1
     ```

2. **Update Encoder/Decoder**
   - Encoder: `latent_len=256, latent_dim=192`
   - Decoder: `latent_dim=192, hidden_dim=256, num_heads=6`

3. **Run Ablation Study**
   - Test: 16, 32, 64, 128, 256 tokens
   - Metrics: NRMSE, training time, memory usage
   - Find sweet spot for Burgers equation

4. **Benchmark on PDEBench**
   - Compare against baseline (16 tokens)
   - Track: NRMSE, correlation time, rollout stability

**Expected Outcome**: 20-40% NRMSE improvement, identify optimal token count

---

### Phase 3: Architecture Simplification (4-6 weeks)

**Goal**: Simplify approximator to pure transformer (if Phase 2 successful)

1. **Implement SimpleLatentTransformer**
   - File: `src/ups/models/simple_operator.py`
   - Architecture: Stacked transformer blocks with drop-path
   - Remove: Down/up phases, skip connections

2. **Compare Architectures**
   - Baseline: U-shaped PDE-Transformer (16 tokens)
   - Simple: Pure transformer (256 tokens)
   - Hybrid: Pure transformer (64 tokens)

3. **Validate on Multiple PDEs**
   - Burgers, Navier-Stokes, Diffusion-Reaction
   - Track: Performance, training time, scalability

**Expected Outcome**: Cleaner architecture with comparable/better performance

---

### Phase 4: Advanced Features (6-8 weeks)

**Goal**: Add UPT's advanced training features

1. **Query-Based Training**
   - Sample random query points per batch
   - Enable arbitrary discretization generalization

2. **Physics Priors**
   - Divergence penalty for incompressible flows
   - Conservation law penalties

3. **Latent Regularization**
   - Latent norm penalty
   - Optional decoder clamping

4. **CFD Encoder** (if targeting fluid benchmarks)
   - GNN pooling module
   - Transformer on supernodes
   - Perceiver pooling to latents

**Expected Outcome**: Match or exceed UPT benchmark performance

---

## 6. Risk Assessment

### High-Risk Changes

1. **Scaling to 256+ tokens**:
   - Risk: OOM errors, slow training
   - Mitigation: Gradient accumulation, bf16 precision, smaller batch size
   - Fallback: 64 or 128 tokens as compromise

2. **Inverse losses breaking convergence**:
   - Risk: Conflicting gradients between forward/inverse objectives
   - Mitigation: Careful loss weight tuning (start with 0.1, increase gradually)
   - Fallback: Disable if operator loss doesn't converge

### Medium-Risk Changes

1. **Architecture simplification**:
   - Risk: Pure transformer worse than U-net for small token counts
   - Mitigation: Only apply if scaling to 256+ tokens
   - Fallback: Keep U-net architecture

2. **Query-based training**:
   - Risk: Slower convergence with sparse supervision
   - Mitigation: Start with dense queries (8k), reduce gradually
   - Fallback: Keep current full-grid training

---

## 7. Success Metrics

### Phase 1 Success Criteria
- [ ] Inverse encoding loss implemented and decreasing
- [ ] Inverse decoding loss implemented and decreasing
- [ ] Validation NRMSE improves by ≥10% over baseline
- [ ] Latent rollout stability (correlation time) improves

### Phase 2 Success Criteria
- [ ] Training completes with 256 tokens (no OOM)
- [ ] Validation NRMSE improves by ≥20% over 16-token baseline
- [ ] Training time increases by ≤5x
- [ ] Model fits on single A100 GPU (40GB)

### Phase 3 Success Criteria
- [ ] Pure transformer matches U-net performance at 256 tokens
- [ ] Training time comparable or faster
- [ ] Architecture is simpler (fewer lines of code)

### Phase 4 Success Criteria
- [ ] Query-based training enables zero-shot super-resolution
- [ ] Physics priors improve conservation metrics
- [ ] Performance matches or exceeds UPT paper benchmarks

---

## 8. Recommended First Steps (This Week)

### Step 1: Implement Inverse Losses (Day 1-2)

```bash
# 1. Create feature branch
git checkout -b feature/upt-inverse-losses

# 2. Update losses.py
# Implement true inverse_encoding_loss and inverse_decoding_loss

# 3. Add unit tests
pytest tests/unit/test_losses.py -v -k inverse
```

### Step 2: Update Training Loop (Day 3-4)

```bash
# 4. Modify train_operator() in scripts/train.py
# Pass encoder+decoder to loss computation

# 5. Create ablation config
cp configs/train_burgers_golden.yaml configs/train_burgers_upt_losses.yaml
# Edit: Add lambda_inv_enc=0.5, lambda_inv_dec=0.5

# 6. Validate locally
python scripts/train.py --config configs/train_burgers_upt_losses.yaml --stage operator --epochs 1
```

### Step 3: Run Ablation Experiment (Day 5-7)

```bash
# 7. Launch full training run
python scripts/vast_launch.py launch \
  --config configs/train_burgers_upt_losses.yaml \
  --auto-shutdown

# 8. Compare results
python scripts/compare_runs.py <baseline_run_id> <upt_losses_run_id>

# 9. Document findings
# Update experiments/YYYY-MM-DD-upt-inverse-losses/notes.md
```

---

## 9. Configuration Templates

### Minimal UPT-Inspired Config (16 tokens + inverse losses)
```yaml
# configs/train_burgers_upt_minimal.yaml
latent:
  tokens: 16
  dim: 32

training:
  lambda_inv_enc: 0.5      # NEW
  lambda_inv_dec: 0.5      # NEW
  lambda_spectral: 0.01
  lambda_relative: 0.01
  lambda_rollout: 0.1
```

### UPT-17M Config (256 tokens)
```yaml
# configs/train_burgers_upt17m.yaml
latent:
  tokens: 256
  dim: 192

operator:
  pdet:
    input_dim: 192
    hidden_dim: 384
    depths: [4, 4, 4]
    num_heads: 6
    drop_path: 0.1

encoder:
  type: grid
  latent_len: 256
  latent_dim: 192
  patch_size: 4

decoder:
  latent_dim: 192
  hidden_dim: 256
  num_heads: 6

training:
  batch_size: 8            # Reduced from 12 (more parameters)
  lambda_inv_enc: 0.5
  lambda_inv_dec: 0.5
  lambda_spectral: 0.01
```

---

## 10. Summary Table: All Gaps

| Gap # | Description | Priority | Complexity | Expected Impact |
|-------|-------------|----------|-----------|-----------------|
| 1 | Missing inverse encoding loss | CRITICAL | Medium | +15-25% NRMSE |
| 2 | Missing inverse decoding loss | CRITICAL | Medium | +10-20% NRMSE |
| 3 | Latent tokens too small (16 vs 256) | CRITICAL | High | +20-40% NRMSE |
| 4 | Latent dim too small (32 vs 192) | HIGH | High | +15-30% NRMSE |
| 5 | Complex U-shaped vs pure transformer | HIGH | High | Architecture clarity |
| 6 | Missing drop-path regularization | HIGH | Low | Better generalization |
| 7 | Missing physics priors | MEDIUM | Medium | Conservation metrics |
| 8 | Missing latent norm penalty | MEDIUM | Low | Regularization |
| 9 | Decoder clamping not implemented | MEDIUM | Low | Stability |
| 10 | Query-based training not used | MEDIUM | High | Zero-shot super-res |
| 11 | CFD encoder architecture gap | LOW | High | CFD benchmarks only |

---

## 11. References

**UPT Implementation Docs**:
- `UPT_docs/UPT_Implementation_Plan.md` - Full implementation guide
- `UPT_docs/UPT_Arch_Train_Scaling.md` - Architecture and scaling playbook

**UPS Current Implementation**:
- `src/ups/io/enc_grid.py` - Grid encoder (lines 1-221)
- `src/ups/io/enc_mesh_particle.py` - Mesh/particle encoder (lines 1-170)
- `src/ups/io/decoder_anypoint.py` - Query decoder (lines 1-136)
- `src/ups/models/latent_operator.py` - Latent operator (lines 1-156)
- `src/ups/core/blocks_pdet.py` - PDE-Transformer blocks (lines 1-450)
- `src/ups/training/losses.py` - Loss functions (lines 1-180)
- `scripts/train.py` - Training orchestrator (lines 1-500)

**Configuration**:
- `configs/train_burgers_golden.yaml` - Current production config
- `configs/README.md` - Config management guide

---

**Next Action**: Implement Phase 1 (inverse losses) and validate on Burgers equation before proceeding to scale-up experiments.
