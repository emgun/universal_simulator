---
date: 2025-11-03T00:00:00Z
researcher: emgun
git_commit: d3622581ecd4f1fa9cced86c232ed59e35b76e71
branch: feature--UPT
repository: universal_simulator
topic: "UPT Phase 3: Architecture Simplification and Configurability"
tags: [implementation-plan, upt, phase3, architecture, transformers]
status: draft
last_updated: 2025-11-03
last_updated_by: emgun
---

# UPT Phase 3: Architecture Simplification and Configurability - Implementation Plan

**Date**: 2025-11-03
**Researcher**: emgun
**Git Commit**: d3622581ecd4f1fa9cced86c232ed59e35b76e71
**Branch**: feature--UPT
**Repository**: universal_simulator

---

## Overview

This plan implements **Phase 3: Architecture Simplification** from the UPT integration roadmap. Based on successful Phase 2 results (128 tokens → 20% NRMSE improvement), we now add architectural flexibility to support both U-shaped and pure transformer approximators, with configurable attention mechanisms and stochastic depth regularization.

**Core Goals**:
1. Implement pure stacked transformer approximator (UPT-style)
2. Add standard multi-head self-attention option (alongside channel-separated)
3. Implement drop-path/stochastic depth regularization
4. Make architecture type configurable (U-shaped vs Pure)
5. Validate on both 128-token (Phase 2 winner) and 256-512 token configurations

**Key Design Principle**: **Maintain backward compatibility** - All existing configs should continue to work unchanged.

---

## Current State Analysis

### Existing Architecture (src/ups/models/latent_operator.py, src/ups/core/blocks_pdet.py)

**PDETransformerBlock** (U-Shaped):
- Architecture: Down (32→16→8 tokens) → Bottleneck (8 tokens) → Up (8→16→32 tokens)
- Skip connections between corresponding down/up stages
- 5 transformer layers (golden config: `depths=[1,1,1]`)
- Channel-separated self-attention with RMS-normalized Q/K
- Dimensions: latent_dim=16, hidden_dim=96 (6x multiplier)
- **No drop-path regularization**

**Strengths**:
- ✅ Production-tested (25× NRMSE improvement on Burgers 16-dim)
- ✅ Phase 2 validated (20% improvement at 128 tokens)
- ✅ Efficient with small token counts (16-128)
- ✅ Multi-scale processing via hierarchical structure

**Gaps Identified**:
- ❌ No pure stacked transformer option (UPT recommendation for 256-512 tokens)
- ❌ No drop-path/stochastic depth (UPT recommends 0.1-0.2 for deep networks)
- ❌ No standard multi-head attention option (only channel-separated)
- ❌ Architecture not configurable (U-shaped is hard-coded)

### Phase 2 Results Context

From ablation study (Oct 28-30, 2025):
- **128 tokens**: NRMSE 0.0577 (20% improvement) ⭐ **Current winner**
- **256 tokens**: NRMSE 0.0596 (17% improvement)
- **64 tokens**: NRMSE 0.0732 (comparable to baseline 0.072)

**Observation**: 128 tokens is in "transition zone" - better than baseline but below UPT's 256-512 threshold for pure transformer recommendation.

---

## Desired End State

After Phase 3 completion, the system will support:

### 1. Configurable Architecture Types

```yaml
operator:
  type: pdet_unet      # U-shaped (current default)
  # OR
  type: pdet_stack     # Pure stacked transformer (new)
```

### 2. Configurable Attention Mechanisms

```yaml
operator:
  pdet:
    attention_type: channel_separated  # Current (group-based with RMSNorm Q/K)
    # OR
    attention_type: standard           # Standard multi-head attention (new)
```

### 3. Drop-Path Regularization

```yaml
operator:
  pdet:
    drop_path: 0.1     # Stochastic depth rate (new, default: 0.0)
```

### 4. Token Count Recommendations

Configuration templates for:
- **16-32 tokens**: U-shaped architecture (current production)
- **128 tokens**: Test both U-shaped and pure (Phase 2 winner)
- **256-512 tokens**: Pure transformer (UPT recommendation)

### 5. Verification Criteria

**Automated Tests**:
- [ ] All existing configs pass without changes (backward compatibility)
- [ ] Unit tests pass: `pytest tests/unit/test_operator.py -v`
- [ ] Integration tests pass: `pytest tests/integration/test_training_pipeline.py -v`
- [ ] Config validation: `python scripts/validate_config.py <new_configs>`
- [ ] Dry-run estimates complete: `python scripts/dry_run.py --estimate-only`

**Manual Verification**:
- [ ] 128-token pure transformer matches or beats U-shaped (NRMSE comparison)
- [ ] 256-token pure transformer validates UPT hypothesis (≥20% improvement)
- [ ] Drop-path improves generalization on deep networks (test-val gap narrows)
- [ ] Training time comparable or improved for pure transformer

---

## What We're NOT Doing

To prevent scope creep, explicitly out-of-scope:

1. **Not replacing existing architecture**: U-shaped PDE-Transformer remains default and production
2. **Not removing channel-separated attention**: Existing attention mechanism stays
3. **Not changing encoder/decoder**: Only approximator changes
4. **Not modifying training loops**: Works with existing `train_operator()` logic
5. **Not requiring config migration**: All existing configs work unchanged
6. **Not implementing new loss functions**: Phase 1 inverse losses already done
7. **Not adding query-based training**: Phase 4 feature (deferred)
8. **Not implementing CFD-specific encoders**: Phase 4 feature (deferred)

---

## Implementation Approach

### High-Level Strategy

**Additive, not replacement**: Implement new components alongside existing architecture, controlled by config flags.

**Three-layer implementation**:
1. **Bottom layer**: New reusable components (DropPath, StandardAttention)
2. **Middle layer**: New approximator class (PureTransformer)
3. **Top layer**: Factory pattern for architecture selection

**Testing strategy**: Incremental validation at each layer before moving up.

---

## Phase 3.1: Core Components (Week 1-2)

**Goal**: Implement reusable building blocks for pure transformer architecture.

### Changes Required

#### 1. Implement DropPath (Stochastic Depth)

**File**: `src/ups/core/drop_path.py` (NEW)

**Implementation**:

```python
"""Drop-path (stochastic depth) regularization.

Reference: Deep Networks with Stochastic Depth (arXiv:1603.09382)
UPT recommendation: 0.1-0.2 for 8-12 layer networks
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DropPath(nn.Module):
    """Drop paths (stochastic depth) per sample.

    When drop_prob > 0, randomly drops entire paths (residual branches)
    during training. At test time, scales by (1 - drop_prob) for expected value.

    Args:
        drop_prob: Probability of dropping a path. Default: 0.0 (disabled).
        scale_by_keep: If True, scale by 1/(1-drop_prob) during training.
                       Default: True (standard practice).

    Example:
        >>> drop_path = DropPath(drop_prob=0.1)
        >>> residual = some_layer(x)
        >>> x = x + drop_path(residual)  # Randomly drop residual 10% of time
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply drop-path to input tensor.

        Args:
            x: Input tensor of shape (B, *, D) where * can be any dims.

        Returns:
            Tensor with same shape, possibly dropped or scaled.
        """
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob

        # Create random tensor with shape (B, 1, ..., 1) to broadcast
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(
            shape, dtype=x.dtype, device=x.device
        )
        random_tensor.floor_()  # Binarize: 0 or 1

        if self.scale_by_keep:
            # Scale by 1/keep_prob to maintain expected value
            output = x.div(keep_prob) * random_tensor
        else:
            output = x * random_tensor

        return output

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob}"
```

**Tests**: `tests/unit/test_drop_path.py` (NEW)

```python
import pytest
import torch
from src.ups.core.drop_path import DropPath


def test_drop_path_disabled():
    """Drop-path with prob=0.0 should return input unchanged."""
    drop = DropPath(drop_prob=0.0)
    x = torch.randn(4, 16, 32)
    out = drop(x)
    assert torch.allclose(out, x)


def test_drop_path_inference():
    """Drop-path in eval mode should return input unchanged."""
    drop = DropPath(drop_prob=0.5)
    drop.eval()
    x = torch.randn(4, 16, 32)
    out = drop(x)
    assert torch.allclose(out, x)


def test_drop_path_training():
    """Drop-path in training should randomly zero some samples."""
    drop = DropPath(drop_prob=0.5)
    drop.train()
    x = torch.ones(100, 16, 32)  # All ones

    # Run multiple times, check that some samples are zeroed
    n_dropped = 0
    for _ in range(10):
        out = drop(x)
        # Check if any batch elements are zeroed (all zeros in that sample)
        batch_norms = out.view(100, -1).norm(dim=1)
        n_dropped += (batch_norms == 0).sum().item()

    # With prob=0.5, expect ~50% dropped over 1000 samples
    drop_rate = n_dropped / 1000
    assert 0.3 < drop_rate < 0.7, f"Drop rate {drop_rate} not near 0.5"


def test_drop_path_scaling():
    """Drop-path should scale by 1/keep_prob to maintain expected value."""
    drop = DropPath(drop_prob=0.5, scale_by_keep=True)
    drop.train()
    x = torch.ones(1000, 16, 32)

    # Average over many runs should approximate input mean
    outputs = []
    for _ in range(100):
        outputs.append(drop(x))

    avg_output = torch.stack(outputs).mean(dim=0)
    expected = x  # Should average back to input

    # Check mean is close (within 10% due to randomness)
    assert torch.allclose(avg_output.mean(), expected.mean(), rtol=0.1)
```

---

#### 2. Implement Standard Multi-Head Self-Attention

**File**: `src/ups/core/attention.py` (NEW)

**Implementation**:

```python
"""Standard and custom attention mechanisms for UPS.

This module provides multiple attention implementations:
- StandardSelfAttention: Standard multi-head self-attention
- ChannelSeparatedSelfAttention: Existing implementation (re-exported)
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardSelfAttention(nn.Module):
    """Standard multi-head self-attention with optional RMSNorm.

    Follows the standard transformer attention mechanism:
    - Linear projections for Q, K, V
    - Scaled dot-product attention with multiple heads
    - Output projection

    Optionally applies RMSNorm to queries and keys for stability
    (recommended by UPT for 8+ layer networks).

    Args:
        dim: Input/output dimension.
        num_heads: Number of attention heads.
        qkv_bias: Add bias to Q, K, V projections. Default: True.
        qk_norm: Apply RMSNorm to Q, K before attention. Default: False.
        attn_drop: Dropout rate for attention weights. Default: 0.0.
        proj_drop: Dropout rate for output projection. Default: 0.0.

    Example:
        >>> attn = StandardSelfAttention(dim=192, num_heads=6)
        >>> x = torch.randn(4, 256, 192)  # (B, T, D)
        >>> out = attn(x)  # (4, 256, 192)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Q, K, V projections (combined for efficiency)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        # Optional Q/K normalization (UPT recommendation for stability)
        self.qk_norm = qk_norm
        if qk_norm:
            from src.ups.core.blocks_pdet import RMSNorm
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, D).

        Returns:
            Output tensor of shape (B, T, D).
        """
        B, T, D = x.shape

        # Project to Q, K, V
        qkv = self.qkv(x)  # (B, T, 3*D)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, head_dim)
        q, k, v = qkv.unbind(0)  # Each: (B, H, T, head_dim)

        # Optional Q/K normalization
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Scaled dot-product attention (Flash Attention compatible)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=False,
        )  # (B, H, T, head_dim)

        # Reshape and project output
        attn_out = attn_out.transpose(1, 2)  # (B, T, H, head_dim)
        attn_out = attn_out.reshape(B, T, D)  # (B, T, D)

        out = self.proj(attn_out)
        out = self.proj_drop(out)

        return out


# Re-export ChannelSeparatedSelfAttention for consistency
from src.ups.core.blocks_pdet import ChannelSeparatedSelfAttention

__all__ = ["StandardSelfAttention", "ChannelSeparatedSelfAttention"]
```

**Tests**: `tests/unit/test_attention.py` (NEW)

```python
import pytest
import torch
from src.ups.core.attention import StandardSelfAttention


def test_standard_attention_shape():
    """Standard attention should preserve input shape."""
    attn = StandardSelfAttention(dim=192, num_heads=6)
    x = torch.randn(4, 256, 192)
    out = attn(x)
    assert out.shape == x.shape


def test_standard_attention_heads_divisibility():
    """Should raise error if dim not divisible by num_heads."""
    with pytest.raises(AssertionError):
        StandardSelfAttention(dim=192, num_heads=7)  # 192 not divisible by 7


def test_standard_attention_with_qk_norm():
    """QK normalization should stabilize attention."""
    attn = StandardSelfAttention(dim=192, num_heads=6, qk_norm=True)
    x = torch.randn(4, 256, 192)
    out = attn(x)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_standard_attention_dropout():
    """Dropout should be applied in training mode."""
    attn = StandardSelfAttention(dim=192, num_heads=6, attn_drop=0.5, proj_drop=0.5)
    attn.train()

    x = torch.randn(4, 256, 192)
    out1 = attn(x)
    out2 = attn(x)

    # With dropout, outputs should differ
    assert not torch.allclose(out1, out2)


def test_standard_attention_inference():
    """Dropout should be disabled in eval mode."""
    attn = StandardSelfAttention(dim=192, num_heads=6, attn_drop=0.5, proj_drop=0.5)
    attn.eval()

    x = torch.randn(4, 256, 192)
    out1 = attn(x)
    out2 = attn(x)

    # Without dropout, outputs should be identical
    assert torch.allclose(out1, out2)


def test_standard_attention_vs_channel_separated():
    """Compare output statistics with channel-separated attention."""
    from src.ups.core.blocks_pdet import ChannelSeparatedSelfAttention

    dim = 192
    num_heads = 6
    x = torch.randn(4, 256, dim)

    # Standard attention
    attn_std = StandardSelfAttention(dim=dim, num_heads=num_heads)
    out_std = attn_std(x)

    # Channel-separated attention
    attn_chan = ChannelSeparatedSelfAttention(dim=dim, group_size=32, num_heads=num_heads)
    out_chan = attn_chan(x)

    # Both should have similar statistical properties
    assert out_std.shape == out_chan.shape
    assert torch.allclose(out_std.mean(), out_chan.mean(), atol=0.5)
    assert torch.allclose(out_std.std(), out_chan.std(), rtol=0.3)
```

---

### Success Criteria

#### Automated Verification:
- [x] Drop-path unit tests pass: `pytest tests/unit/test_drop_path.py -v` (6/6 passed)
- [x] Standard attention unit tests pass: `pytest tests/unit/test_attention.py -v` (9/9 passed)
- [x] No regressions in existing tests: `pytest tests/unit/ -v` (132/133 passed, 1 pre-existing zarr failure)
- [x] Type checking passes: `mypy src/ups/core/drop_path.py src/ups/core/attention.py` (critical checks pass)
- [x] Linting passes: `ruff check src/ups/core/` (E/F checks pass, minor N806/N812 style warnings consistent with codebase)
- [x] Documentation builds: Check docstrings render correctly (docstrings present and formatted)

#### Manual Verification:
- [ ] Drop-path reduces overfitting in toy experiment (train-val gap narrows)
- [ ] Standard attention produces reasonable attention patterns (visualize with small test)
- [ ] Components integrate cleanly with existing TransformerLayer
- [ ] Memory usage comparable to existing attention

**Implementation Note**: After completing this phase, mark as complete and pause for manual review before proceeding to Phase 3.2.

---

## Phase 3.2: Pure Transformer Architecture (Week 3-4)

**Goal**: Implement pure stacked transformer approximator as alternative to U-shaped.

### Changes Required

#### 1. Implement Pure Transformer Approximator

**File**: `src/ups/models/pure_transformer.py` (NEW)

**Implementation**:

```python
"""Pure stacked transformer approximator for UPT.

This module implements a simplified alternative to the U-shaped PDETransformer.
Instead of hierarchical downsampling/upsampling with skip connections, this uses
a simple stack of transformer layers operating on a fixed number of latent tokens.

Recommended for 256-512 latent tokens (UPT guideline).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn

from src.ups.core.attention import StandardSelfAttention, ChannelSeparatedSelfAttention
from src.ups.core.drop_path import DropPath


@dataclass
class PureTransformerConfig:
    """Configuration for pure stacked transformer.

    Args:
        input_dim: Latent dimension (must match latent.dim from config).
        hidden_dim: Hidden dimension for transformer layers.
        depth: Number of transformer layers (e.g., 4, 8, 12).
        num_heads: Number of attention heads per layer.
        attention_type: Type of attention mechanism.
            - "standard": Standard multi-head self-attention
            - "channel_separated": Channel-separated attention (existing)
        group_size: For channel_separated attention, size of channel groups.
                    Ignored for standard attention.
        mlp_ratio: FFN hidden dimension ratio (hidden = dim * mlp_ratio).
        qk_norm: Apply RMSNorm to Q, K in standard attention.
        drop_path: Stochastic depth rate (linearly scaled across depth).
                   0.0 = disabled, 0.1-0.2 = recommended for deep networks.
        dropout: Dropout rate for attention and FFN. Default: 0.0.

    Example configs:
        Small (4 layers, 128 tokens):
            input_dim=192, hidden_dim=192, depth=4, num_heads=4, drop_path=0.0

        Medium (8 layers, 256 tokens):
            input_dim=256, hidden_dim=256, depth=8, num_heads=6, drop_path=0.1

        Large (12 layers, 512 tokens):
            input_dim=384, hidden_dim=384, depth=12, num_heads=8, drop_path=0.15
    """
    input_dim: int
    hidden_dim: int
    depth: int
    num_heads: int
    attention_type: Literal["standard", "channel_separated"] = "standard"
    group_size: int = 32
    mlp_ratio: float = 4.0
    qk_norm: bool = False
    drop_path: float = 0.0
    dropout: float = 0.0


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm, attention, FFN, and drop-path.

    Architecture:
        x = x + DropPath(Attention(LayerNorm(x)))
        x = x + DropPath(FFN(LayerNorm(x)))

    Args:
        dim: Input/output dimension.
        num_heads: Number of attention heads.
        attention_type: "standard" or "channel_separated".
        group_size: For channel_separated attention.
        mlp_ratio: FFN expansion ratio.
        qk_norm: Apply RMSNorm to Q, K (standard attention only).
        drop_path: Stochastic depth rate.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        attention_type: str,
        group_size: int,
        mlp_ratio: float,
        qk_norm: bool,
        drop_path: float,
        dropout: float,
    ):
        super().__init__()

        # Pre-norm for attention
        self.norm1 = nn.LayerNorm(dim)

        # Attention mechanism (configurable)
        if attention_type == "standard":
            self.attn = StandardSelfAttention(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=True,
                qk_norm=qk_norm,
                attn_drop=dropout,
                proj_drop=dropout,
            )
        elif attention_type == "channel_separated":
            self.attn = ChannelSeparatedSelfAttention(
                dim=dim,
                group_size=group_size,
                num_heads=num_heads,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown attention_type: {attention_type}")

        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Pre-norm for FFN
        self.norm2 = nn.LayerNorm(dim)

        # FFN (MLP)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout),
        )

        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, T, D).

        Returns:
            Output tensor (B, T, D).
        """
        # Attention block with residual and drop-path
        x = x + self.drop_path1(self.attn(self.norm1(x)))

        # FFN block with residual and drop-path
        x = x + self.drop_path2(self.mlp(self.norm2(x)))

        return x


class PureTransformer(nn.Module):
    """Pure stacked transformer for latent space evolution.

    This is a simplified alternative to U-shaped PDETransformer.
    Recommended for 256-512 latent tokens (UPT guideline).

    Architecture:
        Input (B, T, input_dim)
            ↓ input_proj
        (B, T, hidden_dim)
            ↓ depth × TransformerBlock (fixed token count T)
        (B, T, hidden_dim)
            ↓ output_norm + output_proj
        Output (B, T, input_dim)

    Key differences from U-shaped:
    - Fixed token count throughout (no pooling/unpooling)
    - No skip connections (purely sequential processing)
    - Simpler architecture (easier to reason about and scale)
    - Linear drop-path schedule (0 at layer 0, max at layer depth-1)

    Args:
        cfg: Configuration dataclass.

    Example:
        >>> cfg = PureTransformerConfig(
        ...     input_dim=192, hidden_dim=192, depth=8, num_heads=6, drop_path=0.1
        ... )
        >>> model = PureTransformer(cfg)
        >>> x = torch.randn(4, 256, 192)
        >>> out = model(x)  # (4, 256, 192)
    """

    def __init__(self, cfg: PureTransformerConfig):
        super().__init__()
        self.cfg = cfg

        # Dimension validation
        if cfg.input_dim != cfg.hidden_dim:
            # Allow different dims, use projection
            self.input_proj = nn.Linear(cfg.input_dim, cfg.hidden_dim)
            self.output_proj = nn.Linear(cfg.hidden_dim, cfg.input_dim)
        else:
            self.input_proj = nn.Identity()
            self.output_proj = nn.Identity()

        # Stacked transformer blocks with linear drop-path schedule
        drop_path_rates = [
            cfg.drop_path * i / max(cfg.depth - 1, 1) for i in range(cfg.depth)
        ]

        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=cfg.hidden_dim,
                num_heads=cfg.num_heads,
                attention_type=cfg.attention_type,
                group_size=cfg.group_size,
                mlp_ratio=cfg.mlp_ratio,
                qk_norm=cfg.qk_norm,
                drop_path=drop_path_rates[i],
                dropout=cfg.dropout,
            )
            for i in range(cfg.depth)
        ])

        # Output normalization
        self.output_norm = nn.LayerNorm(cfg.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Latent tokens (B, T, input_dim).

        Returns:
            Residual latent tokens (B, T, input_dim).
        """
        # Project to hidden dimension
        x = self.input_proj(x)

        # Apply transformer layers sequentially
        for layer in self.layers:
            x = layer(x)

        # Output normalization and projection
        x = self.output_norm(x)
        x = self.output_proj(x)

        return x
```

**Tests**: `tests/unit/test_pure_transformer.py` (NEW)

```python
import pytest
import torch
from src.ups.models.pure_transformer import (
    PureTransformer,
    PureTransformerConfig,
    TransformerBlock,
)


def test_pure_transformer_shape():
    """Pure transformer should preserve (B, T, D) shape."""
    cfg = PureTransformerConfig(
        input_dim=64, hidden_dim=128, depth=4, num_heads=4, drop_path=0.0
    )
    model = PureTransformer(cfg)

    x = torch.randn(4, 128, 64)  # (B=4, T=128, D=64)
    out = model(x)

    assert out.shape == x.shape


def test_pure_transformer_fixed_tokens():
    """Token count should remain fixed (no pooling/unpooling)."""
    cfg = PureTransformerConfig(
        input_dim=64, hidden_dim=64, depth=8, num_heads=4, drop_path=0.1
    )
    model = PureTransformer(cfg)

    # Test various token counts
    for tokens in [32, 64, 128, 256, 512]:
        x = torch.randn(2, tokens, 64)
        out = model(x)
        assert out.shape[1] == tokens


def test_pure_transformer_standard_attention():
    """Test with standard multi-head attention."""
    cfg = PureTransformerConfig(
        input_dim=192,
        hidden_dim=192,
        depth=4,
        num_heads=6,
        attention_type="standard",
        qk_norm=True,
    )
    model = PureTransformer(cfg)

    x = torch.randn(4, 256, 192)
    out = model(x)
    assert out.shape == x.shape


def test_pure_transformer_channel_separated_attention():
    """Test with channel-separated attention."""
    cfg = PureTransformerConfig(
        input_dim=192,
        hidden_dim=192,
        depth=4,
        num_heads=6,
        attention_type="channel_separated",
        group_size=32,
    )
    model = PureTransformer(cfg)

    x = torch.randn(4, 256, 192)
    out = model(x)
    assert out.shape == x.shape


def test_pure_transformer_drop_path():
    """Drop-path should introduce stochasticity in training."""
    cfg = PureTransformerConfig(
        input_dim=64, hidden_dim=64, depth=8, num_heads=4, drop_path=0.3
    )
    model = PureTransformer(cfg)
    model.train()

    x = torch.randn(4, 128, 64)
    out1 = model(x)
    out2 = model(x)

    # With drop-path, outputs should differ
    assert not torch.allclose(out1, out2)


def test_pure_transformer_inference_deterministic():
    """Inference should be deterministic."""
    cfg = PureTransformerConfig(
        input_dim=64, hidden_dim=64, depth=8, num_heads=4, drop_path=0.3
    )
    model = PureTransformer(cfg)
    model.eval()

    x = torch.randn(4, 128, 64)
    out1 = model(x)
    out2 = model(x)

    # In eval mode, should be deterministic
    assert torch.allclose(out1, out2)


def test_pure_transformer_depth_scaling():
    """Deeper models should have linearly increasing drop-path rates."""
    cfg = PureTransformerConfig(
        input_dim=64, hidden_dim=64, depth=12, num_heads=4, drop_path=0.2
    )
    model = PureTransformer(cfg)

    # Check drop-path rates increase linearly
    drop_rates = [layer.drop_path1.drop_prob for layer in model.layers]
    expected = [0.2 * i / 11 for i in range(12)]

    for actual, exp in zip(drop_rates, expected):
        assert abs(actual - exp) < 1e-6


def test_transformer_block():
    """TransformerBlock should work standalone."""
    block = TransformerBlock(
        dim=192,
        num_heads=6,
        attention_type="standard",
        group_size=32,
        mlp_ratio=4.0,
        qk_norm=True,
        drop_path=0.1,
        dropout=0.0,
    )

    x = torch.randn(4, 256, 192)
    out = block(x)
    assert out.shape == x.shape
```

---

#### 2. Add Architecture Type Config

**File**: `src/ups/models/latent_operator.py` (MODIFY)

**Changes**:

```python
# Add to imports (line ~10)
from src.ups.models.pure_transformer import PureTransformer, PureTransformerConfig

# Modify LatentOperatorConfig (line ~43)
@dataclass
class LatentOperatorConfig:
    """Configuration for latent space operator.

    Args:
        latent_dim: Latent token dimension.
        time_embed_dim: Time embedding dimension.
        architecture_type: Type of approximator architecture.
            - "pdet_unet": U-shaped PDE-Transformer (default, current production)
            - "pdet_stack": Pure stacked transformer (new, UPT-style)
        pdet: Configuration for PDETransformer (used by both architectures).
        conditioner: Optional adaptive conditioning config.
    """
    latent_dim: int
    time_embed_dim: int = 64
    architecture_type: Literal["pdet_unet", "pdet_stack"] = "pdet_unet"
    pdet: PDETransformerConfig | PureTransformerConfig = None
    conditioner: Optional[ConditioningConfig] = None

# Modify __init__ to support both architectures (line ~50)
def __init__(self, cfg: LatentOperatorConfig):
    super().__init__()
    self.cfg = cfg

    # Validate dimension match
    if cfg.architecture_type == "pdet_unet":
        if cfg.pdet.input_dim != cfg.latent_dim:
            raise ValueError("PDETransformer input_dim must match latent_dim")
        self.core = PDETransformerBlock(cfg.pdet)
    elif cfg.architecture_type == "pdet_stack":
        if cfg.pdet.input_dim != cfg.latent_dim:
            raise ValueError("PureTransformer input_dim must match latent_dim")
        self.core = PureTransformer(cfg.pdet)
    else:
        raise ValueError(f"Unknown architecture_type: {cfg.architecture_type}")

    # Time embedding (shared across architectures)
    self.time_embed = TimeEmbedding(cfg.time_embed_dim)
    self.time_to_latent = nn.Linear(cfg.time_embed_dim, cfg.latent_dim)

    # Optional conditioning (shared across architectures)
    if cfg.conditioner is not None:
        self.conditioner = AdaLNConditioner(cfg.conditioner)
    else:
        self.conditioner = None

    # Output normalization (shared)
    self.output_norm = nn.LayerNorm(cfg.latent_dim)
```

---

### Success Criteria

#### Automated Verification:
- [x] Pure transformer unit tests pass: `pytest tests/unit/test_pure_transformer.py -v` (14/14 passed)
- [x] Modified operator tests pass: `pytest tests/unit/test_latent_operator.py -v` (6/6 passed, includes new architecture type tests)
- [ ] All existing configs validate: `python scripts/validate_config.py configs/*.yaml`
- [ ] Backward compatibility: Golden config still runs unchanged
- [ ] Type checking: `mypy src/ups/models/` (deferred - pre-existing E402 import warnings)
- [ ] Integration test: `pytest tests/integration/test_training_pipeline.py -k operator`

#### Manual Verification:
- [ ] Pure transformer trains without errors (1 epoch smoke test)
- [ ] Loss curves look reasonable (convergence similar to U-shaped)
- [ ] Memory usage acceptable (within 20% of U-shaped at same token count)
- [ ] Training time comparable or faster

**Implementation Note**: Pause here for review before Phase 3.3.

---

## Phase 3.3: Configuration Templates and Validation (Week 5)

**Goal**: Create comprehensive config templates for different token count scales.

### Changes Required

#### 1. Create UPT-Style Configs

**File**: `configs/train_burgers_upt_128tokens_pure.yaml` (NEW)

```yaml
# UPT Phase 3: Pure Transformer with 128 Tokens
# Compare against U-shaped winner from Phase 2
# Expected: Comparable or slightly better NRMSE with simpler architecture

experiment:
  name: "burgers-upt-128tokens-pure"
  description: "Pure stacked transformer, 128 tokens, standard attention"

latent:
  dim: 64           # Phase 2 winner configuration
  tokens: 128       # Phase 2 winner

operator:
  architecture_type: pdet_stack    # NEW: Pure transformer
  pdet:
    input_dim: 64                  # Must match latent.dim
    hidden_dim: 192                # 3x latent.dim
    depth: 8                       # 8 layers (medium depth)
    num_heads: 6
    attention_type: standard       # NEW: Standard attention
    qk_norm: true                  # Stabilize attention
    mlp_ratio: 4.0                 # Standard transformer expansion
    drop_path: 0.1                 # NEW: Stochastic depth
    dropout: 0.0

encoder:
  type: grid
  latent_len: 128
  latent_dim: 64
  patch_size: 4
  use_fourier_features: true
  fourier_frequencies: [1.0, 2.0]

decoder:
  latent_dim: 64
  query_dim: 3
  hidden_dim: 192
  num_layers: 3
  num_heads: 4
  mlp_hidden_dim: 128

training:
  stages:
    operator:
      epochs: 25
      batch_size: 8                # Reduced for larger model
      grad_accum_steps: 6          # Effective batch = 48
      learning_rate: 1.0e-3
      weight_decay: 0.03
      grad_clip: 1.0
      amp: true
      compile: true

      # Phase 1 losses
      lambda_inv_enc: 0.5
      lambda_inv_dec: 0.5
      lambda_spectral: 0.01
      lambda_rollout: 0.1

logging:
  wandb:
    enabled: true
    project: "universal-simulator"
    entity: "emgun-morpheus-space"
    tags: ["upt", "phase3", "pure-transformer", "128tokens"]
```

---

**File**: `configs/train_burgers_upt_256tokens_pure.yaml` (NEW)

```yaml
# UPT Phase 3: Pure Transformer with 256 Tokens
# UPT recommendation: 256-512 tokens for pure transformer
# Expected: Significant improvement over U-shaped at same token count

experiment:
  name: "burgers-upt-256tokens-pure"
  description: "Pure stacked transformer, 256 tokens (UPT recommendation)"

latent:
  dim: 192          # UPT-17M equivalent
  tokens: 256       # UPT recommendation threshold

operator:
  architecture_type: pdet_stack    # Pure transformer
  pdet:
    input_dim: 192                 # Match latent.dim
    hidden_dim: 384                # 2x latent.dim
    depth: 8                       # 8 layers (medium depth)
    num_heads: 6
    attention_type: standard       # Standard attention
    qk_norm: true                  # Stabilize for 8 layers
    mlp_ratio: 4.0                 # Standard expansion
    drop_path: 0.1                 # Regularize 8-layer network
    dropout: 0.0

encoder:
  type: grid
  latent_len: 256
  latent_dim: 192
  patch_size: 4
  use_fourier_features: true
  fourier_frequencies: [1.0, 2.0, 4.0]

decoder:
  latent_dim: 192
  query_dim: 3
  hidden_dim: 256
  num_layers: 3
  num_heads: 6
  mlp_hidden_dim: 256

training:
  stages:
    operator:
      epochs: 30
      batch_size: 6                # Larger model
      grad_accum_steps: 8          # Effective batch = 48
      learning_rate: 8.0e-4        # Slightly lower for larger model
      weight_decay: 0.03
      grad_clip: 1.0
      amp: true
      compile: true

      # Phase 1 losses
      lambda_inv_enc: 0.5
      lambda_inv_dec: 0.5
      lambda_spectral: 0.01
      lambda_rollout: 0.1

logging:
  wandb:
    enabled: true
    project: "universal-simulator"
    entity: "emgun-morpheus-space"
    tags: ["upt", "phase3", "pure-transformer", "256tokens", "upt-17m"]
```

---

**File**: `configs/train_burgers_upt_128tokens_channel_sep.yaml` (NEW)

```yaml
# UPT Phase 3: Pure Transformer with Channel-Separated Attention
# Test existing attention mechanism in pure transformer architecture

experiment:
  name: "burgers-upt-128tokens-channel-sep"
  description: "Pure transformer with channel-separated attention"

latent:
  dim: 64
  tokens: 128

operator:
  architecture_type: pdet_stack    # Pure transformer
  pdet:
    input_dim: 64
    hidden_dim: 192
    depth: 8
    num_heads: 6
    attention_type: channel_separated  # Use existing attention
    group_size: 32                     # Channel groups
    mlp_ratio: 4.0
    drop_path: 0.1
    dropout: 0.0

# Rest same as standard attention config...
```

---

#### 2. Update Config Schema

**File**: `src/ups/utils/config_loader.py` (MODIFY)

Add validation for new config options:

```python
def validate_operator_config(cfg: dict) -> None:
    """Validate operator configuration."""
    if "architecture_type" in cfg:
        arch_type = cfg["architecture_type"]
        if arch_type not in ["pdet_unet", "pdet_stack"]:
            raise ValueError(f"Unknown architecture_type: {arch_type}")

    pdet_cfg = cfg.get("pdet", {})

    # Validate attention type if specified
    if "attention_type" in pdet_cfg:
        attn_type = pdet_cfg["attention_type"]
        if attn_type not in ["standard", "channel_separated"]:
            raise ValueError(f"Unknown attention_type: {attn_type}")

        # Channel-separated requires group_size
        if attn_type == "channel_separated":
            if "group_size" not in pdet_cfg:
                raise ValueError("channel_separated attention requires group_size")

    # Validate drop_path range
    if "drop_path" in pdet_cfg:
        drop_path = pdet_cfg["drop_path"]
        if not (0.0 <= drop_path <= 0.3):
            raise ValueError(f"drop_path {drop_path} outside reasonable range [0.0, 0.3]")
```

---

#### 3. Update Documentation

**File**: `configs/README.md` (MODIFY)

Add section on Phase 3 configs:

```markdown
## Phase 3 Configurations (Architecture Simplification)

### Pure Transformer Configs

**128 Tokens** (Transition zone - test both architectures):
- `train_burgers_upt_128tokens_pure.yaml` - Pure transformer, standard attention
- `train_burgers_upt_128tokens_channel_sep.yaml` - Pure transformer, channel-separated attention
- Compare against `ablation_upt_128tokens_fixed.yaml` (U-shaped winner from Phase 2)

**256 Tokens** (UPT recommendation threshold):
- `train_burgers_upt_256tokens_pure.yaml` - Pure transformer, 256 tokens, standard attention
- Expected to significantly outperform U-shaped at same token count

**512 Tokens** (Large model):
- `train_burgers_upt_512tokens_pure.yaml` - Large model, deep network (12 layers)

### Architecture Type Selection

**When to use `pdet_unet` (U-shaped)**:
- Latent tokens: 16-128
- Limited token budget
- Multi-scale processing beneficial
- Production-tested for Burgers

**When to use `pdet_stack` (Pure transformer)**:
- Latent tokens: 256-512+
- Sufficient single-scale capacity
- Clearer latent space semantics
- Following UPT recommendations

### Attention Type Selection

**Standard Attention** (`attention_type: standard`):
- More common in literature
- Easier to understand and debug
- Recommended for new experiments
- Works well with qk_norm

**Channel-Separated Attention** (`attention_type: channel_separated`):
- Existing mechanism from Phase 1-2
- Production-tested
- Requires group_size divisibility
- May have slight memory advantage

### Drop-Path Guidelines

| Depth | Recommended drop_path |
|-------|----------------------|
| 4 layers | 0.0 (optional) |
| 8 layers | 0.1 |
| 12 layers | 0.15-0.2 |
```

---

### Success Criteria

#### Automated Verification:
- [x] All new configs validate: `python scripts/validate_config.py configs/train_burgers_upt_*.yaml` (31-32/31-32 checks passed for all 3 configs)
- [x] Schema validation catches invalid configs (added architecture_type, depth, attention_type, group_size validation)
- [ ] Dry-run estimates complete for all new configs (requires data download)
- [ ] Config catalog updated: `python scripts/config_catalog.py`

#### Manual Verification:
- [x] Documentation is clear and actionable (configs/README.md updated with Phase 3 section and architecture guidelines)
- [x] Config naming follows conventions (train_burgers_upt_<tokens>_<variant>.yaml)
- [x] All dimension constraints documented (latent_dim, hidden_dim, group_size divisibility rules in README and validation)

---

## Phase 3.4: Ablation Study and Validation (Week 6-7)

**Goal**: Run comprehensive ablation study comparing architectures and attention mechanisms.

### Ablation Matrix

| Config Name | Tokens | Architecture | Attention | Drop-Path | Notes |
|-------------|--------|--------------|-----------|-----------|-------|
| **Baselines (Phase 2)** | | | | | |
| `ablation_upt_128tokens_fixed` | 128 | U-shaped | Channel-sep | 0.0 | Phase 2 winner |
| `ablation_upt_256tokens_fixed` | 256 | U-shaped | Channel-sep | 0.0 | Phase 2 result |
| **Phase 3 Pure Transformer** | | | | | |
| `train_burgers_upt_128tokens_pure` | 128 | Pure | Standard | 0.1 | Test at winning token count |
| `train_burgers_upt_128tokens_channel_sep` | 128 | Pure | Channel-sep | 0.1 | Compare attention types |
| `train_burgers_upt_256tokens_pure` | 256 | Pure | Standard | 0.1 | UPT recommendation |
| **Depth Ablation** | | | | | |
| `train_burgers_upt_256tokens_deep` | 256 | Pure | Standard | 0.15 | 12 layers |
| **Drop-Path Ablation** | | | | | |
| `train_burgers_upt_256tokens_no_droppath` | 256 | Pure | Standard | 0.0 | Control |

### Execution Plan

**Week 6: Core Ablations**

```bash
# 1. Launch 128-token pure transformer (standard attention)
python scripts/vast_launch.py launch \
  --config configs/train_burgers_upt_128tokens_pure.yaml \
  --auto-shutdown

# 2. Launch 128-token pure transformer (channel-separated attention)
python scripts/vast_launch.py launch \
  --config configs/train_burgers_upt_128tokens_channel_sep.yaml \
  --auto-shutdown

# 3. Launch 256-token pure transformer (UPT recommendation)
python scripts/vast_launch.py launch \
  --config configs/train_burgers_upt_256tokens_pure.yaml \
  --auto-shutdown
```

**Week 7: Extended Ablations**

```bash
# 4. Deep network (12 layers)
python scripts/vast_launch.py launch \
  --config configs/train_burgers_upt_256tokens_deep.yaml \
  --auto-shutdown

# 5. Drop-path ablation (control)
python scripts/vast_launch.py launch \
  --config configs/train_burgers_upt_256tokens_no_droppath.yaml \
  --auto-shutdown
```

### Analysis Scripts

**File**: `scripts/analyze_phase3_ablation.py` (NEW)

```python
"""Analyze Phase 3 ablation study results."""
import argparse
from pathlib import Path

import wandb
import pandas as pd
import matplotlib.pyplot as plt


def fetch_phase3_runs():
    """Fetch all Phase 3 runs from WandB."""
    api = wandb.Api()
    runs = api.runs(
        "emgun-morpheus-space/universal-simulator",
        filters={"tags": {"$in": ["upt", "phase3"]}},
    )
    return runs


def compare_architectures(runs):
    """Compare U-shaped vs Pure transformer at same token counts."""
    results = []

    for run in runs:
        config = run.config
        summary = run.summary

        results.append({
            "name": run.name,
            "tokens": config.get("latent", {}).get("tokens"),
            "architecture": config.get("operator", {}).get("architecture_type"),
            "attention": config.get("operator", {}).get("pdet", {}).get("attention_type"),
            "drop_path": config.get("operator", {}).get("pdet", {}).get("drop_path", 0.0),
            "depth": config.get("operator", {}).get("pdet", {}).get("depth"),
            "nrmse_baseline": summary.get("eval/baseline_nrmse"),
            "nrmse_ttc": summary.get("eval/ttc_nrmse"),
            "training_time": summary.get("training_time_hours"),
        })

    df = pd.DataFrame(results)
    return df


def plot_architecture_comparison(df, output_dir):
    """Plot NRMSE comparison across architectures."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Group by token count
    for tokens in [128, 256]:
        subset = df[df["tokens"] == tokens]

        # Baseline NRMSE
        axes[0].bar(
            subset["name"],
            subset["nrmse_baseline"],
            label=f"{tokens} tokens",
        )

        # TTC NRMSE
        axes[1].bar(
            subset["name"],
            subset["nrmse_ttc"],
            label=f"{tokens} tokens",
        )

    axes[0].set_title("Baseline NRMSE")
    axes[0].set_ylabel("NRMSE")
    axes[0].legend()
    axes[0].tick_params(axis='x', rotation=45)

    axes[1].set_title("TTC NRMSE")
    axes[1].set_ylabel("NRMSE")
    axes[1].legend()
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / "architecture_comparison.png", dpi=150)
    print(f"Saved plot: {output_dir / 'architecture_comparison.png'}")


def generate_report(df, output_path):
    """Generate markdown report."""
    report = f"""# Phase 3 Ablation Study Results

**Generated**: {pd.Timestamp.now()}

## Summary

Total runs analyzed: {len(df)}

## Results by Token Count

### 128 Tokens (Transition Zone)

{df[df["tokens"] == 128].to_markdown()}

**Key Observations**:
- Compare Pure vs U-shaped at winning token count from Phase 2
- Compare standard vs channel-separated attention in pure architecture

### 256 Tokens (UPT Recommendation)

{df[df["tokens"] == 256].to_markdown()}

**Key Observations**:
- Pure transformer should excel at this token count
- Test drop-path effectiveness for 8-12 layer networks

## Architecture Comparison

### Best Configuration by Token Count

"""

    for tokens in [128, 256]:
        subset = df[df["tokens"] == tokens]
        best = subset.loc[subset["nrmse_ttc"].idxmin()]
        report += f"\n**{tokens} tokens**: {best['name']} (NRMSE: {best['nrmse_ttc']:.6f})\n"

    report += "\n## Recommendations\n\n"
    # Add recommendations based on results...

    with open(output_path, "w") as f:
        f.write(report)

    print(f"Saved report: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("reports/phase3"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching Phase 3 runs from WandB...")
    runs = fetch_phase3_runs()

    print(f"Found {len(runs)} runs")
    df = compare_architectures(runs)

    print("Generating plots...")
    plot_architecture_comparison(df, args.output_dir)

    print("Generating report...")
    generate_report(df, args.output_dir / "ablation_report.md")

    print("Done!")


if __name__ == "__main__":
    main()
```

---

### Success Criteria

#### Automated Verification (Infrastructure Ready):
- [x] Analysis script created: `scripts/analyze_phase3_ablation.py`
- [x] Execution guide documented: `docs/phase3_execution_guide.md`
- [x] Configs validated and ready to launch (3/3 configs pass validation)
- [ ] All ablation runs complete successfully (no crashes) - **MANUAL STEP: Run experiments**
- [ ] Operator final loss < 0.001 for all configs - **PENDING: Awaits run completion**
- [ ] No NaN/Inf gradients in any run - **PENDING: Awaits run completion**
- [ ] Checkpoints saved and loadable - **PENDING: Awaits run completion**
- [ ] Analysis script runs without errors - **PENDING: Awaits WandB data**

#### Manual Verification (Experiments Required):

**Primary Hypothesis Tests**:
1. **H1: Pure transformer matches U-shaped at 128 tokens**
   - [ ] 128-token pure NRMSE within 5% of 128-token U-shaped (Phase 2 winner: 0.0577)
   - Target: Pure NRMSE ≤ 0.0606
   - **PENDING: Run config `train_burgers_upt_128tokens_pure.yaml`**

2. **H2: Pure transformer outperforms U-shaped at 256 tokens**
   - [ ] 256-token pure NRMSE improves by ≥10% over 256-token U-shaped (Phase 2: 0.0596)
   - Target: Pure NRMSE ≤ 0.0536
   - **PENDING: Run config `train_burgers_upt_256tokens_pure.yaml`**

3. **H3: Standard attention comparable to channel-separated**
   - [ ] Standard attention within 5% NRMSE of channel-separated at same config
   - **PENDING: Run both `128tokens_pure.yaml` and `128tokens_channel_sep.yaml`**

4. **H4: Drop-path effectiveness** (deferred - not in current ablation matrix)
   - [ ] Would require additional 12-layer configs with/without drop-path

**Performance Benchmarks**:
- [ ] Training time scales reasonably with depth (not exponential)
- [ ] Memory usage within budget (fits on A100 40GB)
- [ ] Convergence speed comparable to U-shaped (epochs to threshold loss)

**Production Readiness**:
- [ ] Identify winning configuration for each token count
- [ ] Document clear guidelines for architecture selection
- [x] Execution workflow documented (see `docs/phase3_execution_guide.md`)

**Note**: Phase 3.4 infrastructure is complete. Manual experiments must be run to complete this phase.

---

## Phase 3.5: Documentation and Production Promotion (Week 8)

**Goal**: Document findings, update guidelines, and promote winning configs to production.

### Changes Required

#### 1. Update Main UPT Plan

**File**: `thoughts/shared/plans/2025-01-23-upt-inverse-losses-implementation.md` (MODIFY)

Add Phase 3 completion status with results summary.

---

#### 2. Create Architecture Selection Guide

**File**: `docs/architecture_selection_guide.md` (NEW)

```markdown
# UPS Approximator Architecture Selection Guide

This guide helps you choose the right approximator architecture based on your latent token count and PDE complexity.

## Quick Decision Tree

```
START
  ↓
Do you have 16-64 latent tokens?
  ├─ YES → Use U-shaped architecture (pdet_unet)
  │         - Multi-scale processing essential
  │         - Production-tested
  │         - Example: configs/train_burgers_golden.yaml
  │
  └─ NO → Do you have 128 tokens?
      ├─ YES → Test both architectures
      │         - U-shaped: Proven winner (Phase 2)
      │         - Pure: Simpler, may match performance
      │         - Start with U-shaped for safety
      │
      └─ NO → Do you have 256+ tokens?
          └─ YES → Use pure stacked transformer (pdet_stack)
                   - Clearer semantics
                   - Easier to scale
                   - UPT recommendation validated (Phase 3)
```

## Architecture Comparison

| Feature | U-Shaped (pdet_unet) | Pure Stacked (pdet_stack) |
|---------|---------------------|---------------------------|
| **Best Token Range** | 16-128 | 256-512+ |
| **Token Flow** | Variable (pooling/unpooling) | Fixed throughout |
| **Complexity** | Multi-scale, skip connections | Simple stacked layers |
| **Maturity** | Production (Phase 1-2) | Validated (Phase 3) |
| **Scalability** | Limited by U-shape | Easy to add depth |
| **Memory** | Efficient at low tokens | Efficient at high tokens |

## Attention Mechanism Selection

### Standard Attention
**When to use**:
- New experiments following UPT recommendations
- Pure stacked transformer (recommended pairing)
- Need standard transformer semantics
- Using qk_norm for stability (8+ layers)

**Config**:
```yaml
operator:
  pdet:
    attention_type: standard
    qk_norm: true  # Recommended for 8+ layers
```

### Channel-Separated Attention
**When to use**:
- Existing production configs (backward compatibility)
- U-shaped architecture (original pairing)
- Proven performance on your PDE
- Memory constraints (slight advantage)

**Config**:
```yaml
operator:
  pdet:
    attention_type: channel_separated
    group_size: 32  # Must divide hidden_dim
```

## Drop-Path (Stochastic Depth)

**Purpose**: Regularize deep networks (8+ layers) by randomly dropping residual branches during training.

**When to use**:
- Depth ≥ 8 layers
- Overfitting observed (large train-val gap)
- Following UPT recommendations

**Recommended values**:

| Depth | drop_path |
|-------|-----------|
| 4 layers | 0.0 (optional) |
| 8 layers | 0.1 |
| 12 layers | 0.15 |

**Config**:
```yaml
operator:
  pdet:
    depth: 8
    drop_path: 0.1
```

## Example Configurations

### Small Model (16-64 tokens, U-shaped)
```yaml
latent:
  dim: 32
  tokens: 32

operator:
  architecture_type: pdet_unet  # U-shaped
  pdet:
    input_dim: 32
    hidden_dim: 96
    depths: [1, 1, 1]  # Shallow U-net
    attention_type: channel_separated
    group_size: 12
```

### Medium Model (128 tokens, flexible)
```yaml
latent:
  dim: 64
  tokens: 128

operator:
  architecture_type: pdet_unet  # OR pdet_stack (test both)
  pdet:
    input_dim: 64
    hidden_dim: 192
    depth: 8  # If pdet_stack
    # OR depths: [2, 2, 2]  # If pdet_unet
    attention_type: standard
    qk_norm: true
    drop_path: 0.1
```

### Large Model (256+ tokens, pure transformer)
```yaml
latent:
  dim: 192
  tokens: 256

operator:
  architecture_type: pdet_stack  # Pure stacked
  pdet:
    input_dim: 192
    hidden_dim: 384
    depth: 8
    attention_type: standard
    qk_norm: true
    drop_path: 0.1
```

## Validation Checklist

Before running a new architecture:

- [ ] Dimension match: `latent.dim == operator.pdet.input_dim`
- [ ] Attention divisibility:
  - Standard: `hidden_dim % num_heads == 0`
  - Channel-separated: `hidden_dim % group_size == 0` AND `group_size % num_heads == 0`
- [ ] Drop-path range: `0.0 <= drop_path <= 0.3`
- [ ] Token count appropriate for architecture type
- [ ] Config validates: `python scripts/validate_config.py <config>`

## Performance Expectations

Based on Phase 2-3 ablation studies on Burgers 1D:

| Token Count | Architecture | NRMSE (Baseline) | NRMSE (TTC) | Improvement |
|-------------|--------------|------------------|-------------|-------------|
| 32 (baseline) | U-shaped | 0.072 | - | - |
| 128 | U-shaped | 0.058 | ~0.05 | 20% |
| 128 | Pure | [TBD Phase 3] | [TBD] | [TBD] |
| 256 | Pure | [TBD Phase 3] | [TBD] | [TBD] |

*Note: Phase 3 results to be filled in after ablation completion.*

## Troubleshooting

### Issue: "dim must be divisible by num_heads"
- **Solution**: Adjust `num_heads` to divide `hidden_dim` evenly
- Example: `hidden_dim=192` → `num_heads` must be 2, 3, 4, 6, 8, 12, etc.

### Issue: "group_size must divide dim"
- **Solution**: For channel-separated attention, choose `group_size` that divides `hidden_dim`
- Example: `hidden_dim=192` → `group_size` could be 12, 16, 24, 32, etc.

### Issue: Out of memory
- **Solution**: Reduce batch size or token count
- Consider gradient accumulation to maintain effective batch size

### Issue: Training unstable with deep network
- **Solution**: Enable qk_norm and increase drop_path
- Example: `qk_norm: true, drop_path: 0.15` for 12 layers

## References

- Phase 1-2 Implementation: `thoughts/shared/plans/2025-01-23-upt-inverse-losses-implementation.md`
- Phase 3 Implementation: `thoughts/shared/plans/2025-11-03-upt-phase3-architecture-simplification.md`
- UPT Integration Analysis: `UPT_docs/UPT_INTEGRATION_ANALYSIS.md`
- Production Workflow: `PRODUCTION_WORKFLOW.md`
```

---

#### 3. Promote Winning Configs

```bash
# After ablation results are in, promote winners:

# If 128-token pure wins:
python scripts/promote_config.py \
  experiments/phase3/train_burgers_upt_128tokens_pure.yaml \
  --production-dir configs/ \
  --rename train_burgers_upt_medium.yaml \
  --update-leaderboard

# If 256-token pure wins:
python scripts/promote_config.py \
  experiments/phase3/train_burgers_upt_256tokens_pure.yaml \
  --production-dir configs/ \
  --rename train_burgers_upt_large.yaml \
  --update-leaderboard
```

---

### Success Criteria

#### Automated Verification:
- [ ] All documentation builds without errors
- [ ] Decision tree is clear and actionable
- [ ] All code examples in docs validate
- [ ] Promoted configs pass validation

#### Manual Verification:
- [ ] Documentation reviewed by another researcher
- [ ] Architecture selection guide tested with new user
- [ ] Production configs clearly marked in `configs/README.md`
- [ ] Leaderboard updated with Phase 3 results

---

## Testing Strategy

### Unit Tests

**Coverage targets**:
- `src/ups/core/drop_path.py`: 100% (simple module)
- `src/ups/core/attention.py`: 95%
- `src/ups/models/pure_transformer.py`: 90%

**Test categories**:
1. Shape preservation tests (input/output dims)
2. Randomness tests (training vs eval mode)
3. Edge case tests (single token, odd token counts)
4. Configuration validation tests
5. Backward compatibility tests

---

### Integration Tests

**File**: `tests/integration/test_phase3_architectures.py` (NEW)

```python
"""Integration tests for Phase 3 architectures."""
import pytest
import torch
from src.ups.models.latent_operator import LatentOperator, LatentOperatorConfig
from src.ups.models.pure_transformer import PureTransformerConfig
from src.ups.core.blocks_pdet import PDETransformerConfig
from src.ups.core.latent_state import LatentState


def test_pure_transformer_in_operator():
    """Pure transformer integrates correctly with LatentOperator."""
    pure_cfg = PureTransformerConfig(
        input_dim=64,
        hidden_dim=192,
        depth=4,
        num_heads=4,
        attention_type="standard",
        drop_path=0.1,
    )

    op_cfg = LatentOperatorConfig(
        latent_dim=64,
        architecture_type="pdet_stack",
        pdet=pure_cfg,
    )

    operator = LatentOperator(op_cfg)

    # Test forward pass
    z = torch.randn(4, 128, 64)
    state = LatentState(z=z, t=0.0, cond={})

    new_state = operator(state, dt=0.01)

    assert new_state.z.shape == z.shape
    assert new_state.t == 0.01


def test_unet_still_works():
    """U-shaped architecture still works (backward compatibility)."""
    unet_cfg = PDETransformerConfig(
        input_dim=64,
        hidden_dim=192,
        depths=[1, 1, 1],
        group_size=32,
        num_heads=4,
    )

    op_cfg = LatentOperatorConfig(
        latent_dim=64,
        architecture_type="pdet_unet",
        pdet=unet_cfg,
    )

    operator = LatentOperator(op_cfg)

    z = torch.randn(4, 128, 64)
    state = LatentState(z=z, t=0.0, cond={})

    new_state = operator(state, dt=0.01)

    assert new_state.z.shape == z.shape


def test_training_loop_compatibility():
    """Pure transformer works in training loop."""
    from src.ups.training.loop_train import train_operator
    from src.ups.data.datasets import DummyDataset

    # This is a smoke test - just check it runs without errors
    # Actual training tested in ablation study
    pass  # TODO: Implement if needed
```

---

### Ablation Validation Tests

**File**: `tests/validation/test_phase3_ablation.py` (NEW)

```python
"""Validation tests for Phase 3 ablation study."""
import pytest
from pathlib import Path


def test_all_configs_valid():
    """All Phase 3 configs validate successfully."""
    configs = [
        "train_burgers_upt_128tokens_pure.yaml",
        "train_burgers_upt_128tokens_channel_sep.yaml",
        "train_burgers_upt_256tokens_pure.yaml",
    ]

    for config_name in configs:
        config_path = Path("configs") / config_name
        # Use validate_config.py
        # This test ensures configs are syntactically correct


def test_dimension_consistency():
    """All configs have consistent dimension settings."""
    # latent.dim == operator.pdet.input_dim
    pass


def test_attention_divisibility():
    """Attention heads divide dimensions correctly."""
    # hidden_dim % num_heads == 0
    pass
```

---

## Performance Considerations

### Training Time Estimates

Based on Phase 2 results and extrapolation:

| Config | Tokens | Depth | Est. Training Time (A100) | Cost @ $1.89/hr |
|--------|--------|-------|--------------------------|-----------------|
| 128 pure (standard) | 128 | 8 | ~35 min | ~$1.10 |
| 128 pure (channel-sep) | 128 | 8 | ~33 min | ~$1.04 |
| 256 pure | 256 | 8 | ~55 min | ~$1.73 |
| 256 pure deep | 256 | 12 | ~75 min | ~$2.36 |

**Factors affecting training time**:
- Token count (linear scaling)
- Depth (linear scaling)
- Attention mechanism (channel-sep ~5% faster)
- Drop-path (negligible overhead)

### Memory Usage Estimates

| Config | Tokens | Depth | Est. GPU Memory (bf16) |
|--------|--------|-------|----------------------|
| 128 pure | 128 | 8 | ~12 GB |
| 256 pure | 256 | 8 | ~20 GB |
| 256 pure deep | 256 | 12 | ~28 GB |
| 512 pure | 512 | 8 | ~35 GB |

**Memory optimization strategies**:
- Reduce batch size (use gradient accumulation)
- Enable activation checkpointing for depth > 8
- Use torch.compile for memory-efficient attention

---

## Risk Assessment

### High-Risk Items

1. **Pure transformer underperforms at 128 tokens**
   - **Risk**: May need 256+ tokens for pure architecture to shine
   - **Mitigation**: Keep U-shaped as default for 16-128 tokens
   - **Fallback**: Document that pure transformer needs 256+ tokens

2. **Training instability with deep networks (12+ layers)**
   - **Risk**: Gradient issues, slow convergence
   - **Mitigation**: QK normalization, drop-path, gradient clipping
   - **Fallback**: Limit depth to 8 layers for production

3. **Standard attention different behavior than channel-separated**
   - **Risk**: Performance gap, need retuning
   - **Mitigation**: Thorough ablation at multiple token counts
   - **Fallback**: Keep channel-separated as default

### Medium-Risk Items

1. **Memory constraints for large models**
   - **Risk**: OOM on A100 40GB
   - **Mitigation**: Gradient accumulation, smaller batch size
   - **Fallback**: Reduce token count or depth

2. **Configuration complexity**
   - **Risk**: Users confused by multiple architecture options
   - **Mitigation**: Clear documentation, decision tree
   - **Fallback**: Provide recommended configs per token count

---

## Success Metrics

### Phase 3 Overall Success Criteria

**Must Have (P0)**:
- [ ] Pure transformer implemented and tested
- [ ] Drop-path regularization working
- [ ] Standard attention option available
- [ ] All existing configs work unchanged (backward compatibility)
- [ ] At least one pure transformer config matches or beats U-shaped

**Should Have (P1)**:
- [ ] 256-token pure transformer significantly outperforms U-shaped (≥15% NRMSE)
- [ ] Drop-path improves deep network generalization (test-val gap narrows)
- [ ] Standard attention within 5% of channel-separated performance
- [ ] Documentation clear and comprehensive

**Nice to Have (P2)**:
- [ ] 128-token pure transformer beats U-shaped
- [ ] Pure transformer enables easier architecture scaling
- [ ] Zero-shot super-resolution better with pure transformer

---

## Timeline Summary

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| **3.1: Core Components** | Week 1-2 | DropPath, StandardAttention, unit tests |
| **3.2: Pure Transformer** | Week 3-4 | PureTransformer, integration, tests |
| **3.3: Configuration** | Week 5 | Config templates, validation, docs |
| **3.4: Ablation Study** | Week 6-7 | Run experiments, analyze results |
| **3.5: Documentation** | Week 8 | Selection guide, production promotion |

**Total Duration**: 8 weeks (2 months)

**Checkpoints**:
- End of Week 2: Core components ready
- End of Week 4: Pure transformer integrated
- End of Week 5: Configs and docs ready
- End of Week 7: Ablation complete
- End of Week 8: Phase 3 complete, production ready

---

## Next Actions (Week 1 - Immediate)

### Day 1-2: Implement DropPath
1. Create `src/ups/core/drop_path.py`
2. Create `tests/unit/test_drop_path.py`
3. Run tests: `pytest tests/unit/test_drop_path.py -v`
4. Validate: Ensure all tests pass

### Day 3-4: Implement StandardAttention
1. Create `src/ups/core/attention.py`
2. Create `tests/unit/test_attention.py`
3. Run tests: `pytest tests/unit/test_attention.py -v`
4. Compare outputs with channel-separated attention

### Day 5: Integration and Review
1. Run all unit tests: `pytest tests/unit/ -v`
2. Check type hints: `mypy src/ups/core/`
3. Run linter: `ruff check src/ups/core/`
4. Code review checkpoint
5. Mark Phase 3.1 complete

---

## References

### Internal Documents
- **Phase 1-2 Implementation**: `thoughts/shared/plans/2025-01-23-upt-inverse-losses-implementation.md`
- **Phase 2 Results**:
  - `reports/analysis_run1.md` (128 tokens, 0.0577 NRMSE)
  - `reports/research/run2_analysis.md` (256 tokens, 0.0596 NRMSE)
- **Current Architecture**: `src/ups/core/blocks_pdet.py`, `src/ups/models/latent_operator.py`

### UPT Documentation
- **Integration Analysis**: `UPT_docs/UPT_INTEGRATION_ANALYSIS.md`
- **Implementation Plan**: `UPT_docs/UPT_Implementation_Plan.md`
- **Architecture Playbook**: `UPT_docs/UPT_Arch_Train_Scaling.md`

### Configuration
- **Golden Config**: `configs/train_burgers_golden.yaml` (16-dim production)
- **Phase 2 Winner**: `configs/ablation_upt_128tokens_fixed.yaml` (128 tokens, U-shaped)

### External References
- **Drop-Path Paper**: Deep Networks with Stochastic Depth (arXiv:1603.09382)
- **UPT Paper**: Universal Physics Transformers (arXiv:2402.12365)

---

**End of Plan**

This implementation plan provides a comprehensive roadmap for Phase 3: Architecture Simplification. Upon completion, the UPS framework will support both U-shaped and pure stacked transformer architectures with configurable attention mechanisms and stochastic depth regularization, validated across multiple token count scales.
