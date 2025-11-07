# Encoder Architecture Research - Documentation Index

**Date**: November 5, 2025  
**Scope**: Quick research on CFD-relevant encoder patterns  
**Status**: Complete

---

## Research Documents

This research has produced 4 comprehensive documents analyzing the encoder implementations in the Universal Simulator:

### 1. **ENCODER_RESEARCH_REPORT.md** (23 KB, Primary Reference)
   - **Purpose**: Comprehensive research report with detailed analysis
   - **Sections**: 10 major sections covering architecture, capabilities, gaps, recommendations
   - **Audience**: Technical deep-dive
   - **Key Content**:
     - Executive summary
     - Architecture flows with diagrams
     - File:line references
     - CFD capabilities vs gaps analysis
     - Pooling strategy comparison
     - 4 options for GNN-based pooling enhancement
     - 4 priority recommendations with effort estimates
   - **Read This For**: Complete understanding of encoders and recommendations

### 2. **encoder_research_summary.md** (11 KB, Structured Overview)
   - **Purpose**: Structured analysis of encoder components
   - **Format**: Numbered sections with code snippets
   - **Key Content**:
     - Current encoders breakdown (GridEncoder, MeshParticleEncoder, AnyPointDecoder)
     - GNN capabilities and gaps
     - Mesh/particle handling details
     - Perceiver pooling patterns
     - CFD-specific features and limitations
     - Supernode pooling algorithm explanation
     - Where to add GNN-based pooling (4 options)
     - File location summary table
   - **Read This For**: Mid-level technical overview with code references

### 3. **encoder_code_reference.md** (13 KB, Implementation Guide)
   - **Purpose**: Line-by-line code walkthrough
   - **Format**: Detailed implementation details for each encoder
   - **Key Content**:
     - GridEncoder: Pixel-unshuffle, stems, Fourier features, pooling
     - MeshParticleEncoder: Field concatenation, adjacency building, message passing, two-stage pooling
     - AnyPointDecoder: Cross-attention decoder architecture
     - Message passing code analysis
     - Supernode/perceiver pooling algorithms
     - Integration flow in training pipeline
     - CFD-specific encoding patterns with examples
     - Performance characteristics
     - Testing reference
   - **Read This For**: Understanding exact implementations and integration points

### 4. **encoder_quick_reference.txt** (10 KB, Quick Lookup)
   - **Purpose**: Quick reference card for fast lookups
   - **Format**: Structured text with key info, line numbers, code snippets
   - **Key Content**:
     - File locations and sizes
     - Key methods with line ranges
     - Pooling strategies comparison table
     - GNN capabilities checklist
     - Mesh/particle handling steps
     - CFD gaps organized by feature area
     - Where to add pooling improvements
     - Testing reference
     - Performance notes
     - Prioritized recommendations
   - **Read This For**: Quick lookup, reference card during development

---

## Key Findings Summary

### Encoders in Universal Simulator

**1. GridEncoder** (`src/ups/io/enc_grid.py`, 232 lines)
- For: Structured/regular grids
- Method: Pixel-unshuffle patches → residual stems → Fourier features → adaptive pooling
- Pooling: Non-learned F.adaptive_avg_pool1d
- CFD Use: Burgers equation, shallow water (regular meshes only)
- Gap: No irregular mesh support

**2. MeshParticleEncoder** (`src/ups/io/enc_mesh_particle.py`, 171 lines)
- For: Unstructured meshes, particles, point clouds
- Method: Message passing (3 steps) → two-stage pooling (supernode + perceiver)
- Pooling: Chunk-based supernode (non-learned) + adaptive average
- CFD Use: Irregular meshes, particle systems
- Gap: No learned pooling, no BC encoding, no physics awareness

**3. AnyPointDecoder** (`src/ups/io/decoder_anypoint.py`, 137 lines)
- For: Query-based decoding at arbitrary points
- Method: Perceiver-IO inspired with cross-attention
- CFD Use: Decode latent tokens to physical space

### Current Capabilities vs Gaps

**Implemented (Existing)**:
- Basic message passing on graphs (degree-normalized aggregation)
- Two-stage pooling (geometric only)
- Fourier coordinate features
- Field concatenation
- Cross-attention decoding

**NOT Implemented (Gaps)**:
- Learned pooling algorithms (TopKPool, DiffPool, SAGPool)
- Physics-aware pooling (conservation preservation)
- Boundary condition encoding
- Multi-physics integration
- Reynolds number conditioning
- Mesh quality awareness
- Graph attention (GAT-style)
- Multi-head message passing

### Where GNN-Based Pooling Could Be Added

Four options with increasing complexity:

**Option 1: Replace Supernode Pooling** (RECOMMENDED)
- Location: enc_mesh_particle.py lines 125-131
- Approach: TopKPool with importance weighting
- Effort: 40-60 LOC
- Impact: Better token selection, physics-aware

**Option 2: Learned Perceiver Pooling**
- Location: enc_mesh_particle.py after line 170
- Approach: Cross-attention with learned latent vectors
- Effort: 50-70 LOC
- Impact: Interpretable selection

**Option 3: Multi-Head Message Passing**
- Location: enc_mesh_particle.py lines 64-123
- Approach: Separate heads per field type
- Effort: 30-40 LOC
- Impact: Field-aware aggregation

**Option 4: Physics-Aware Pooling Layer**
- Location: New file pool_physics.py
- Approach: Conservation-preserving selection
- Effort: 60-80 LOC
- Impact: Preserves physics integrals

---

## File Organization

```
Project Root: /Users/emerygunselman/Code/universal_simulator/

Encoder Implementation:
  src/ups/io/
    - enc_grid.py (232 lines) - GridEncoder
    - enc_mesh_particle.py (171 lines) - MeshParticleEncoder
    - decoder_anypoint.py (137 lines) - AnyPointDecoder
    - __init__.py (14 lines) - Exports

Related Files:
  src/ups/data/latent_pairs.py - Encoder usage in training
  src/ups/models/multiphysics_factor_graph.py - Multi-physics (not integrated)
  tests/unit/test_enc_mesh_particle.py - Tests

Research Documents (NEW):
  ENCODER_RESEARCH_REPORT.md - Main comprehensive report
  encoder_research_summary.md - Structured analysis
  encoder_code_reference.md - Implementation guide
  encoder_quick_reference.txt - Quick reference card
```

---

## Quick Navigation

### If you want to...

**Understand the overall architecture**
→ Read: ENCODER_RESEARCH_REPORT.md sections 1-2

**See code details and line numbers**
→ Read: encoder_code_reference.md or encoder_quick_reference.txt

**Find where to add a feature**
→ Read: encoder_quick_reference.txt section "WHERE TO ADD GNN-BASED POOLING"

**Understand message passing**
→ Read: encoder_code_reference.md "Message Passing Deep Dive"

**Understand pooling strategies**
→ Read: encoder_quick_reference.txt "POOLING STRATEGIES"

**See CFD gaps**
→ Read: ENCODER_RESEARCH_REPORT.md section 5 OR encoder_quick_reference.txt "CFD-SPECIFIC GAPS ANALYSIS"

**Get implementation recommendations**
→ Read: ENCODER_RESEARCH_REPORT.md section 10 AND encoder_quick_reference.txt "RECOMMENDATIONS"

**See specific code examples**
→ Read: encoder_code_reference.md "CFD-Specific Encoding Patterns"

---

## Key Statistics

**Codebase Size**:
- Total encoder LOC: ~550 lines
- Message passing: 10 lines core logic
- Pooling: 20 lines total (both non-learned)
- CFD-specific: 0 lines (entirely missing)

**Current Implementation**:
- GNN features: Basic message passing only
- Learned pooling: None (pure averaging)
- Physics awareness: None (geometric only)
- BC encoding: None
- Multi-physics: None

**Gaps vs Literature**:
- Missing: 5+ learned pooling algorithms
- Missing: Physics-aware aggregation
- Missing: Boundary condition encoding
- Missing: Multi-physics coupling
- Missing: Reynolds number conditioning

---

## Research Methodology

This research was conducted using:
1. Glob pattern matching for file discovery
2. Grep regex searching for related components
3. Line-by-line code reading and analysis
4. Data flow tracing through encoder pipelines
5. Comparison with CFD literature patterns
6. Integration point identification via latent_pairs.py

**Scope**: Quick analysis (1 hour)
**Completeness**: All encoder implementations documented
**Accuracy**: Line-by-line verified with actual code

---

## Next Steps for Development

### Priority 1: Learned Pooling
- Highest impact for CFD accuracy
- Replace chunk-based supernode pooling
- Implement TopKPool or SAGPool
- Preserve spatial coherence

### Priority 2: Boundary Condition Encoding  
- Low effort, good impact
- Add BC parameter to forward()
- Distinguish Dirichlet/Neumann/periodic
- Help decoder with boundary predictions

### Priority 3: Physics-Aware Features
- Create pool_physics.py module
- Preserve conservation laws during pooling
- Improve long-horizon predictions

### Priority 4: Multi-Physics Integration
- Connect encoders to MultiphysicsFactorGraph
- Support fluid-structure, thermal-fluid coupling

---

## Document Maintenance

**Last Updated**: November 5, 2025
**Research Status**: Complete
**Verification**: All line numbers verified against actual code

When updating encoders, please:
1. Update the corresponding document section
2. Verify line numbers are still accurate
3. Update CFD capabilities table if needed
4. Keep "File Organization" section current

---

## Contact & Questions

For questions about this research:
- Check the specific document sections listed above
- Line numbers are verified and accurate as of Nov 5, 2025
- All code snippets are from actual implementation
- Examples use actual encoder configuration patterns
