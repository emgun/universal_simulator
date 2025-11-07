# Decoder & Regularization - Quick Reference

## One-Page Summary

### DECODER OUTPUT ARCHITECTURE
**File**: `src/ups/io/decoder_anypoint.py:81-87`
- Output heads: 2-layer MLPs with GELU (no output activation/constraints)
- **Status**: Completely unconstrained (can output negative values for density/energy)

### CRITICAL GAPS

| Gap | Location | Impact |
|-----|----------|--------|
| No output clamping | decoder_anypoint.py:131-134 | Density/energy can be negative |
| No latent norm penalty | scripts/train.py:650-680 | Missing regularization on latent amplitude |
| No drop-path in U-Net | blocks_pdet.py:124-135 | Missing stochastic depth in default architecture |

### EXISTING REGULARIZATION ACTIVE

| Method | Config Value | Location |
|--------|--------------|----------|
| Weight Decay | 0.03 (operator), 0.015 (diffusion) | configs/train_burgers_golden.yaml:111,122,139 |
| Gradient Clipping | 1.0 | scripts/train.py:699-705 |
| Spectral Loss | 0.05 | configs/train_burgers_golden.yaml:92 |
| Inverse Loss Curriculum | Warmup 15 epochs | src/ups/training/losses.py:134-165 |
| EMA | 0.999 decay | configs/train_burgers_golden.yaml:86 |
| Drop-Path | 0.1 (8-layer only) | configs/train_burgers_upt_256tokens_pure.yaml:53 |

### IMPLEMENTATION REFERENCE PATTERNS

**Log-Clamping Pattern** (exists but unused):
```python
# From physics_guards.py:31-32
def positify(values: torch.Tensor, min_value: float = 1e-6) -> torch.Tensor:
    return torch.exp(torch.clamp(torch.log(values.clamp_min(min_value)), min=-20.0))
```

**Drop-Path Pattern** (used in PureTransformer, missing in PDETBlock):
```python
# From pure_transformer.py - working reference
self.drop_path_attn = DropPath(drop_path)
x = x + self.drop_path_attn(self.attn(self.norm1(x)))
```

### LOSS FUNCTION OVERVIEW
**File**: `src/ups/training/losses.py`

Core losses:
- `L_forward`: One-step prediction (always)
- `L_inv_enc`: Decoder reconstruction (optional, curriculum)
- `L_inv_dec`: Encoder invertibility (optional, curriculum)
- `L_spec`: Spectral energy matching (optional, weighted 0.05)
- `L_rollout`: Multi-step prediction (optional)

Curriculum for inverse losses:
- Epochs 0-15: weight = 0
- Epochs 15-30: linear ramp
- Epochs 30+: weight = min(base, 0.05)

### WHERE TO ADD NEW REGULARIZATION

| Regularization | File | Lines | Type |
|---------------|------|-------|------|
| Output clamping | decoder_anypoint.py | 131-134 | Model change |
| Latent norm penalty | scripts/train.py | 650-680 | Loss change |
| Drop-path in U-Net | blocks_pdet.py | 124-135 | Model change |
| Regularization weight | configs/train_burgers_golden.yaml | new | Config change |

---

## Quick Facts

- **2 architectures**: PDETransformerBlock (U-Net, production) vs PureTransformer (stack, UPT)
- **2 optimizers**: AdamW (standard) and Muon (optional, experimental)
- **3 training stages**: Operator → Diffusion Residual → Consistency Distillation
- **4 latent loss types**: Forward, Inverse-Encoding, Inverse-Decoding, Spectral
- **Weight decay values**: 0.03 (strongest) for operator, 0.015 for refinement stages

---

## Full Details
See: `docs/decoder_regularization_research.md`
