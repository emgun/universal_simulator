# Decoder & Regularization Research - Index

This directory contains comprehensive research and analysis of the decoder implementation and regularization mechanisms in the UPS training pipeline.

## Documents in This Research

### 1. [decoder_regularization_research.md](decoder_regularization_research.md) - Full Research Report
**Length**: 457 lines | **Size**: 16KB | **Audience**: Detailed analysis seekers

Complete analysis of decoder architecture, loss functions, and regularization strategies.

**Sections**:
- Decoder architecture deep dive (output heads, activation functions)
- Output constraints examination (current gaps)
- 8 loss function catalog with formulas
- 4 types of regularization (weight decay, drop-path, gradient clipping, EMA)
- Transformer block regularization comparison
- 9 key insights and recommendations

**Best for**: Understanding the full picture, finding patterns to copy, production design decisions

---

### 2. [decoder_regularization_quick_ref.md](decoder_regularization_quick_ref.md) - Quick Reference
**Length**: 82 lines | **Size**: 3.1KB | **Audience**: Quick lookup

One-page summary with critical information at a glance.

**Sections**:
- Decoder status (unconstrained, GELU, no output activation)
- 3 critical gaps (clamping, latent norm, drop-path)
- 6 active regularization methods with values
- 2 code patterns (log-clamping, drop-path)
- 5 loss function types
- Where to add regularization (4 locations)
- Quick facts (2 architectures, 2 optimizers, 3 stages)

**Best for**: Quick reference during implementation, status check, decision making

---

### 3. [decoder_regularization_locations.md](decoder_regularization_locations.md) - Code Location Index
**Length**: 282 lines | **Size**: 10KB | **Audience**: Implementation-focused

Exact file:line references for all components in the research.

**Sections**:
- Decoder heads: exact file:line locations and components
- 8 loss functions with implementation details
- Loss bundle computation flow
- 5 active regularization methods with line references
- PDETransformerBlock (U-shaped, missing drop-path)
- PureTransformer (stacked, with drop-path)
- DropPath module reference
- Training loop integration (operator training step-by-step)
- Configuration files with regularization parameters
- Summary table: implemented vs. missing features
- Comprehensive code location index

**Best for**: Implementation, debugging, finding where to add features

---

## Quick Navigation

### I want to understand...

**How the decoder works**
→ `decoder_regularization_research.md` Section 1

**What regularization is missing**
→ `decoder_regularization_quick_ref.md` "Critical Gaps"

**Exact locations to modify code**
→ `decoder_regularization_locations.md` "Summary Table"

**Where to add output clamping**
→ Both quick ref and locations documents point to `decoder_anypoint.py:131-134`

**Where to add latent norm penalty**
→ Both documents point to `scripts/train.py:650-680`

**Why drop-path is missing in production**
→ `decoder_regularization_research.md` Section 5.1

---

## Key Findings at a Glance

### Decoder Status
- Architecture: Perceiver-style cross-attention decoder
- Output heads: 2-layer MLPs with GELU (lines 81-87)
- **Constraints**: NONE (can output negative values for density/energy)

### Regularization Summary
| Method | Status | Value | Location |
|--------|--------|-------|----------|
| Weight Decay | Active | 0.03 (op), 0.015 (diff) | Config |
| Gradient Clip | Active | 1.0 | `train.py:699-705` |
| EMA | Active | 0.999 decay | `loop_train.py:49-54` |
| Spectral Loss | Active | 0.05 weight | `losses.py:112-116` |
| Drop-Path | Missing (U-Net) | 0.1 (UPT only) | `pure_transformer.py` |
| Latent Norm | Missing | Not implemented | N/A |
| Output Clamping | Missing | Not implemented | N/A |

### Architectures
- **PDETransformerBlock** (U-shaped, default): 5 layers, NO drop-path
- **PureTransformer** (stacked, UPT): 4-12 layers, WITH drop-path

---

## Implementation Quick Start

### To Add Output Clamping
1. Open: `src/ups/io/decoder_anypoint.py`
2. Go to: Lines 131-134
3. Copy pattern from: `src/ups/models/physics_guards.py:31-32`
4. Details in: `decoder_regularization_locations.md` "Where to Add Log-Clamping"

### To Add Latent Norm Penalty
1. Open: `scripts/train.py`
2. Go to: Lines 620-629 (weight construction) and after 676 (loss computation)
3. Pattern in: `decoder_regularization_quick_ref.md`
4. Details in: `decoder_regularization_locations.md` "Where to Add Latent Norm Penalty"

### To Add Drop-Path to U-Net
1. Open: `src/ups/core/blocks_pdet.py`
2. Go to: Lines 133-134 (TransformerLayer)
3. Copy pattern from: `src/ups/models/pure_transformer.py:82-142`
4. Details in: `decoder_regularization_research.md` Section 5

---

## Loss Function Catalog

8 loss types are defined in `src/ups/training/losses.py`:

1. **MSE** (line 21-22): Base MSE loss utility
2. **Inverse Encoding** (line 25-60): Decoder reconstruction MSE
3. **Inverse Decoding** (line 63-99): Encoder invertibility MSE
4. **One-Step** (line 102-103): Single prediction loss
5. **Rollout** (line 106-109): Multi-step trajectory loss
6. **Spectral** (line 112-116): FFT magnitude matching (active, weight=0.05)
7. **Consistency** (line 119-122): Spatial mean constraint (defined, unused)
8. **Edge Total Variation** (line 125-131): Graph smoothness (defined, unused)

**Main aggregator**: `compute_operator_loss_bundle()` (line 168-251)

**Curriculum learning**: Inverse losses warm up over 15 epochs, then ramp linearly to max weight of 0.05

---

## Files Mentioned in This Research

### Core Files
- `src/ups/io/decoder_anypoint.py` - Decoder implementation
- `src/ups/training/losses.py` - Loss functions
- `scripts/train.py` - Training loop
- `src/ups/core/blocks_pdet.py` - U-shaped transformer
- `src/ups/models/pure_transformer.py` - Stacked transformer
- `src/ups/core/drop_path.py` - Drop-path module
- `src/ups/models/physics_guards.py` - Physics constraints (pattern source)

### Configuration Files
- `configs/train_burgers_golden.yaml` - Production config
- `configs/train_burgers_upt_256tokens_pure.yaml` - UPT config

### Other
- `src/ups/training/loop_train.py` - Training loop utilities
- `src/ups/models/latent_operator.py` - Operator architecture selector

---

## Document Metadata

**Created**: November 5, 2025
**Codebase**: universal_simulator (Feature branch: feature--UPT)
**Platform**: macOS / Darwin
**Thoroughness**: Medium

**Related research**:
- UPT (Universal Physics Transformer) implementation status
- Inverse loss curriculum learning mechanisms
- Architecture comparison (U-shaped vs. stacked)

---

## How to Use These Documents

1. **Start here** if new to the codebase: `decoder_regularization_quick_ref.md`
2. **For implementation**: `decoder_regularization_locations.md`
3. **For understanding**: `decoder_regularization_research.md`
4. **For decisions**: `decoder_regularization_quick_ref.md` then detailed reports
5. **For debugging**: `decoder_regularization_locations.md` then relevant sections

---

## Questions These Documents Answer

- What does the decoder do? (Research report Section 1)
- How are outputs constrained? (Quick ref, Research Section 2)
- What regularization exists? (Quick ref, Research Section 4)
- What's missing? (Quick ref "Critical Gaps", Research Section 8)
- Where do I add new regularization? (All documents have "Where to Add" sections)
- Why are there two architectures? (Research Section 5)
- How does curriculum learning work? (Research Section 3.2 and Locations Section)
- What are all the loss functions? (Research Section 3, Locations detail)

---

## Next Steps

Based on this research, potential improvements are:

**Priority 1 - Quick Wins**:
1. Add log-clamping to decoder outputs for positive fields (density, energy)
2. Add drop-path to PDETransformerBlock (copy from PureTransformer)

**Priority 2 - Medium Effort**:
3. Add latent norm penalty with configurable weight
4. Add per-layer weight decay scaling

**Priority 3 - Advanced**:
5. Field-specific output constraints (density bounds, energy bounds)
6. Adaptive regularization weight scheduling
7. Spectral regularization for high-frequency stability

See `decoder_regularization_research.md` Section 9 for detailed recommendations.
