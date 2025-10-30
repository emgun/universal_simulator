# Latent Space Architecture Documentation Index

## Overview

This documentation set comprehensively covers the latent space architecture of the Universal Physics Stack (UPS), including:
- Latent token representations and configurations (64, 128, 256, golden)
- Discretization-agnostic encoder/decoder design
- PDE-Transformer core architecture
- Shifted window attention mechanisms
- Configuration consistency rules and validation

## Documentation Files

### 1. LATENT_SPACE_QUICK_REFERENCE.md (317 lines, 9.1 KB)
**Best for**: Quick lookup, configuration templates, validation checklist

**Contains**:
- Core concepts at a glance
- Configuration parameter tables
- Encoder/decoder flow diagrams
- Architecture component overview
- Dimension consistency rules with examples
- Token count effects comparison (64/128/256/golden)
- Memory analysis
- Key architectural principles
- File quick reference table
- Common configuration patterns
- Validation checklist
- Common issues and fixes
- Recommended reading order

**When to use**: 
- First-time orientation (10 minutes)
- Configuration debugging
- Before training checklist
- Quick parameter lookup

---

### 2. LATENT_SPACE_ARCHITECTURE.md (918 lines, 27 KB)
**Best for**: Deep understanding, implementation details, design decisions

**Contains**:

#### Section 1: Latent Token Representation
- LatentState data structure (src/ups/core/latent_state.py)
- Tensor shape and dimensions
- Configuration examples (64-token, 128-token, 256-token, golden)
- Trade-offs: tokens vs. latent dimension

#### Section 2: Grid Encoder
- Purpose and discretization-agnostic design
- GridEncoderConfig parameters
- Complete encoding pipeline:
  - PixelUnshuffle (patch grouping, invertible)
  - Per-field residual stems
  - Fourier positional features
  - Token projection
  - Adaptive pooling
- Implementation code (src/ups/io/enc_grid.py)
- Token generation formula
- Fourier feature design

#### Section 3: Mesh/Particle Encoder
- MeshParticleEncoderConfig
- Graph convolution pipeline
- Message passing aggregation
- Supernode pooling
- Unstructured data support

#### Section 4: AnyPoint Decoder
- Perceiver-style query-based design
- AnyPointDecoderConfig parameters
- Complete architecture:
  - Fourier encoding of query coordinates
  - Query embedding
  - Cross-attention layers
  - Per-field output heads
- Discretization-agnostic properties
- Continuous decoding capability
- Complexity analysis: O(queries × tokens)

#### Section 5: Token Configuration Ablations
- 64-Token configuration analysis
- 128-Token configuration analysis
- 256-Token configuration analysis
- Comprehensive comparison table
- Spatial resolution estimates
- Token count effect on PDET architecture
- Divisibility constraints explanation

#### Section 6: Golden Configuration (Production)
- Minimal viable architecture philosophy
- Configuration values and rationale
- 8× memory reduction achievement
- Diffusion and TTC compensation
- Training efficiency benefits

#### Section 7: Dimension Consistency Requirements
- Critical constraints and validation
- Three main rules:
  - Latent dimension equality
  - Channel separation divisibility
  - Decoder dimension matching
- Validation code examples
- Error handling

#### Section 8: Complete Encoder/Decoder Data Flow
- Training pipeline (physical grid → latent → evolution → decoder → loss)
- Inference pipeline (initial state → multiple rollouts → physics selection)
- Multi-step evolution process
- Latent caching implications

#### Section 9: Shifted Window Attention (Infrastructure)
- Purpose and design motivation
- Swin Transformer pattern adaptation
- API: partition_windows() and merge_windows()
- LogSpacedRelativePositionBias
- Current usage status (implemented but inactive)
- Future integration possibilities

#### Section 10: Adaptive Layer Norm Conditioning
- Optional physics-aware modulation
- AdaLNConditioner design
- Scale/shift/gate generation
- Example use cases (Reynolds number, energy, boundary conditions)
- Current usage (typically disabled)

#### Section 11: Memory and Computational Analysis
- Parameter count breakdown:
  - Operator: ~130K
  - Encoder: ~1-5K
  - Decoder: ~65K
  - Total: ~200K for golden config
- Memory usage estimation
- Forward activation costs
- Backward gradient costs
- Peak memory comparison across configs

#### Section 12: Configuration Impact Summary
- Token count effects table
- Latent dimension effects table
- Design philosophy comparison
- Quality vs. efficiency trade-offs
- Generalization considerations

#### Section 13: Key Files Reference
- Complete file listing with purpose
- Cross-references to implementation
- Configuration file locations
- Ablation study references

#### Section 14: Future Extensions
- Possible enhancements
- Known limitations and workarounds
- Research directions

**When to use**:
- Understanding architecture details
- Implementing new components
- Debugging complex issues
- Research and publication
- Code review preparation

---

## Related Documentation

### Existing Documentation
- `LATENT_OPERATOR_ARCHITECTURE.md` (918 lines) - Focus on operator evolution dynamics
- `LATENT_OPERATOR_QUICK_REFERENCE.md` - Operator-specific quick reference

### Configuration Examples
- `configs/train_burgers_golden.yaml` - Production configuration (16-dim, 32 tokens)
- `configs/ablation_upt_64tokens.yaml` - Token ablation (64-token, 64-dim)
- `configs/ablation_upt_128tokens.yaml` - Token ablation (128-token, 128-dim)
- `configs/ablation_upt_256tokens.yaml` - Token ablation (256-token, 192-dim, UPT-17M)

### Implementation Files
```
src/ups/core/
├── latent_state.py         # LatentState dataclass
├── blocks_pdet.py          # PDE-Transformer, attention, normalization
├── conditioning.py         # AdaLN conditioning
└── shifted_window.py       # Window partitioning utilities

src/ups/io/
├── enc_grid.py             # Grid encoder
├── enc_mesh_particle.py    # Mesh/particle encoder
└── decoder_anypoint.py     # Query-based decoder

src/ups/models/
└── latent_operator.py      # Latent evolution operator
```

---

## Quick Start Guide

### For Understanding the Architecture (1-2 hours)
1. Read: **LATENT_SPACE_QUICK_REFERENCE.md** (10 min)
2. Read: **LATENT_SPACE_ARCHITECTURE.md** sections 1-4 (20 min)
3. Skim: **LATENT_SPACE_ARCHITECTURE.md** sections 5-6 (10 min)
4. Review: **LATENT_SPACE_ARCHITECTURE.md** sections 7-8 (20 min)
5. Code review:
   - `src/ups/io/enc_grid.py` (20 min)
   - `src/ups/core/blocks_pdet.py` (20 min)
   - `src/ups/io/decoder_anypoint.py` (15 min)

### For Creating a New Configuration
1. Copy golden config: `configs/train_burgers_golden.yaml`
2. Verify with checklist from **LATENT_SPACE_QUICK_REFERENCE.md**
3. Validate: `python scripts/validate_config.py configs/my_config.yaml`
4. Reference: **LATENT_SPACE_ARCHITECTURE.md** section 7 if errors

### For Debugging Configuration Issues
1. Check: **LATENT_SPACE_QUICK_REFERENCE.md** validation checklist
2. Verify: Dimension consistency rules (section 7)
3. Look up: Common issues and fixes in quick reference
4. Reference: Full architecture doc for deeper understanding

### For Extending the Architecture
1. Study: **LATENT_SPACE_ARCHITECTURE.md** sections 1-4 (components)
2. Study: **LATENT_SPACE_ARCHITECTURE.md** section 14 (future work)
3. Review: Implementation files for each component
4. Validate: All dimension constraints apply
5. Test: Run validate_config.py with new parameters

---

## Key Takeaways

### Golden Configuration (Recommended for Production)
- **Latent dim**: 16 (ultra-compact)
- **Token count**: 32 (coarse spatial)
- **Total features**: 512 (8× compression vs. 64-token)
- **Training time**: 14.5 min on A100
- **Memory**: ~50 MB peak
- **Philosophy**: Minimal viable + diffusion + TTC = SOTA performance

### Ablation Configurations (Research)
- **64-token**: Fast prototyping, coarse results
- **128-token**: Balanced quality and speed
- **256-token**: Maximum quality, highest cost

### Architectural Principles
1. **Discretization-agnostic**: Same architecture for different grids/PDEs
2. **Efficiency-first**: Small latents with compensation mechanisms
3. **Invertible components**: Preserve reconstruction ability
4. **Physics-aware**: Fourier features, time embedding, optional conditioning

### Critical Constraints
- All latent dimensions must match across components
- `hidden_dim` must be divisible by `group_size`
- `group_size` must be divisible by `num_heads`
- These are validated at runtime with clear error messages

---

## Table of Contents (By Topic)

### Data Structures
- LatentState: ARCHITECTURE.md Section 1
- GridEncoderConfig: ARCHITECTURE.md Section 2
- AnyPointDecoderConfig: ARCHITECTURE.md Section 4
- PDETransformerConfig: OPERATOR_ARCHITECTURE.md

### Encoders
- GridEncoder: ARCHITECTURE.md Section 2
- MeshParticleEncoder: ARCHITECTURE.md Section 3
- Fourier features: ARCHITECTURE.md Section 2

### Decoders
- AnyPointDecoder: ARCHITECTURE.md Section 4
- Perceiver pattern: ARCHITECTURE.md Section 4

### Core Transformer
- PDETransformerBlock: OPERATOR_ARCHITECTURE.md, Section 4
- Channel-separated attention: OPERATOR_ARCHITECTURE.md, Section 4B
- RMSNorm: OPERATOR_ARCHITECTURE.md, Section 4A
- Token pooling/upsampling: OPERATOR_ARCHITECTURE.md, Section 4

### Conditioning & Control
- AdaLN conditioning: ARCHITECTURE.md Section 10
- Time embedding: OPERATOR_ARCHITECTURE.md, Section 3A
- Shifted windows: ARCHITECTURE.md Section 9

### Configurations
- Golden (production): ARCHITECTURE.md Section 6, QUICK_REFERENCE.md
- Token ablations: ARCHITECTURE.md Section 5, QUICK_REFERENCE.md
- Configuration patterns: QUICK_REFERENCE.md

### Validation & Debugging
- Dimension constraints: ARCHITECTURE.md Section 7, QUICK_REFERENCE.md
- Validation checklist: QUICK_REFERENCE.md
- Common issues: QUICK_REFERENCE.md

---

## Document Statistics

| File | Lines | Size | Sections | Topics |
|------|-------|------|----------|--------|
| LATENT_SPACE_ARCHITECTURE.md | 918 | 27 KB | 14 | Core architecture |
| LATENT_SPACE_QUICK_REFERENCE.md | 317 | 9.1 KB | 11 | Configuration & lookup |
| LATENT_OPERATOR_ARCHITECTURE.md | 607 | 22 KB | 7 | Evolution operator |
| **Total** | **1,842** | **58 KB** | **25+** | Complete system |

---

## Contributing to Documentation

When updating these docs:
1. Verify all file paths are absolute and correct
2. Validate all code examples compile and run
3. Check all dimension calculations
4. Cross-reference with implementation
5. Update related docs (main docs may reference these)
6. Run validation script to ensure config examples work

---

## Support & Questions

### Finding Information
1. **Configuration questions** → QUICK_REFERENCE.md
2. **Architecture understanding** → ARCHITECTURE.md
3. **Operator details** → LATENT_OPERATOR_ARCHITECTURE.md
4. **Validation errors** → QUICK_REFERENCE.md "Common Issues"
5. **Implementation** → Actual source code + ARCHITECTURE.md

### Validation
```bash
python scripts/validate_config.py configs/my_config.yaml
```

### Common Patterns
See QUICK_REFERENCE.md sections:
- "Common Configuration Patterns"
- "Validation Checklist"
- "Common Issues & Fixes"

---

**Last Updated**: October 28, 2025
**Status**: Complete and production-ready
**Version**: 1.0
