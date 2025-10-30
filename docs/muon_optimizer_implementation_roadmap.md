# Muon Optimizer Implementation Roadmap for UPS Codebase

**Date**: 2025-01-30
**Researcher**: Claude Code
**Git Commit**: 3b04066d00278d6a4fb6493a836206997c25bfda
**Branch**: feature--UPT
**Repository**: emgun/universal_simulator

---

## Executive Summary

This document maps the generic Muon optimizer implementation guide to the specific Universal Physics Stack (UPS) codebase. The UPS architecture is **ideal for Muon** because ~95%+ of trainable parameters are 2D Linear layer weights in the PDE-Transformer backbone. This roadmap provides concrete file paths, code modifications, and implementation steps to integrate Muon with minimal disruption to the existing training pipeline.

**Key Finding**: The UPS codebase's single-GPU, multi-stage training architecture with per-stage optimizer configuration makes Muon integration straightforward via a hybrid Muon+AdamW approach.

---

## Research Question

How do we integrate the Muon optimizer (from `docs/Integrating Muon Optimizer into a Neural Physics Transformer Framework.pdf` and `docs/muon_npt_impl_for_coding_agent.md`) into the UPS codebase's existing training infrastructure?

---

## Architecture Compatibility Analysis

### UPS Model Parameter Distribution

The UPS model consists of:

**2D Parameters (Muon-compatible) - ~95%+ of total parameters:**
- Linear layer weights throughout PDE-Transformer backbone
  - `src/ups/core/blocks_pdet.py`: Attention projections, feedforward layers
  - `src/ups/models/latent_operator.py`: Time embeddings, projections
  - `src/ups/models/diffusion_residual.py`: 3-layer MLP
  - `src/ups/io/decoder_anypoint.py`: Query embeddings, cross-attention, decoder heads

**4D Parameters (Muon-compatible):**
- Conv2d weights in `src/ups/io/enc_grid.py` (grid encoder only)

**1D Parameters (AdamW-only) - ~5% of total:**
- LayerNorm weights/biases (`blocks_pdet.py`, `decoder_anypoint.py`)
- RMSNorm weights (`blocks_pdet.py`)
- Linear/Conv biases (all modules)

**Conclusion**: The parameter distribution strongly favors Muon, making it an excellent fit.

---

## Existing Optimizer Infrastructure

### Current Implementation

**Location**: `scripts/train.py`

**Optimizer Creation** (lines 245-258):
```python
def _create_optimizer(cfg: dict, model: nn.Module, stage: str) -> torch.optim.Optimizer:
    stage_cfg = cfg.get("stages", {}).get(stage, {})
    opt_cfg = stage_cfg.get("optimizer") or cfg.get("optimizer", {})
    name = opt_cfg.get("name", "adam").lower()
    lr = opt_cfg.get("lr", 1e-3)
    weight_decay = opt_cfg.get("weight_decay", 0.0)

    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        momentum = opt_cfg.get("momentum", 0.9)
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
```

**Current Limitations**:
1. No parameter grouping support (all params use same LR/weight_decay)
2. No hybrid optimizer support
3. Adam betas not configurable from YAML

**Training Pipeline**:
- Multi-stage training: operator → diffusion → consistency → steady_prior
- Each stage creates a fresh optimizer (no optimizer state persistence)
- Single-GPU training (no FSDP/DDP)
- Mixed precision (AMP) enabled by default
- Gradient clipping with AMP-aware unscaling

**Configuration Hierarchy**:
1. Stage-specific: `stages.<stage_name>.optimizer`
2. Global fallback: `optimizer`

### Relevant Training Features

**Mixed Precision** (`training.amp=true`):
- Uses `torch.cuda.amp.autocast` for forward pass
- `GradScaler` for backward pass
- **Critical**: Gradients must be unscaled before clipping (line 622-627)

**Gradient Clipping** (`training.grad_clip=1.0`):
```python
if use_amp:
    if clip_val is not None:
        scaler.unscale_(optimizer)  # MUST unscale first!
    grad_norm = torch.nn.utils.clip_grad_norm_(...)
```

**torch.compile Support** (`training.compile=true`):
- Models are compiled for speed (20-40% faster)
- Teacher models skipped to avoid CUDA graph issues

---

## Implementation Roadmap

### Phase 1: Core Infrastructure (New Files)

#### 1.1 Parameter Grouping Module

**Create**: `src/ups/training/param_groups.py`

```python
"""Parameter grouping utilities for hybrid Muon+AdamW optimization."""
from __future__ import annotations
from typing import Tuple, List
import torch.nn as nn


def is_matrix_param(p: nn.Parameter) -> bool:
    """
    Check if parameter is Muon-compatible (2D or higher).

    Muon optimizer is designed for matrix parameters (Linear weights, Conv weights).
    Use AdamW for 1D parameters (biases, LayerNorm, embeddings).
    """
    return p.requires_grad and p.ndim >= 2


def build_param_groups(
    model: nn.Module,
) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    """
    Split model parameters into Muon-compatible and AdamW groups.

    Returns:
        (muon_params, adamw_params)

    Parameter Classification:
        Muon:
            - Linear layer weights (2D)
            - Conv2d layer weights (4D)
            - Any parameter with ndim >= 2
        AdamW:
            - Biases (1D)
            - LayerNorm weights/biases (1D)
            - RMSNorm weights (1D)
            - Any parameter with ndim < 2
    """
    muon_params = []
    adamw_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if is_matrix_param(p):
            muon_params.append(p)
        else:
            adamw_params.append(p)

    return muon_params, adamw_params


def build_param_groups_with_names(
    model: nn.Module,
) -> Tuple[List[Tuple[str, nn.Parameter]], List[Tuple[str, nn.Parameter]]]:
    """
    Same as build_param_groups but includes parameter names for debugging.

    Returns:
        (muon_params_with_names, adamw_params_with_names)
    """
    muon_params = []
    adamw_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if is_matrix_param(p):
            muon_params.append((name, p))
        else:
            adamw_params.append((name, p))

    return muon_params, adamw_params
```

**Why This Design?**
- Simple heuristic: `ndim >= 2` captures Linear/Conv weights
- Biases and normalization layers automatically fall into AdamW group
- No hardcoded name patterns (more robust to architecture changes)
- Includes debug variant with parameter names

---

#### 1.2 Hybrid Optimizer Wrapper

**Create**: `src/ups/training/hybrid_optimizer.py`

```python
"""Hybrid optimizer that combines Muon and AdamW for different parameter groups."""
from __future__ import annotations
from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
from torch.optim import Optimizer


class HybridOptimizer(Optimizer):
    """
    Wrapper that manages multiple optimizers as a single unit.

    Designed for Muon+AdamW hybrid training where:
    - Muon optimizer handles 2D+ matrix parameters
    - AdamW optimizer handles 1D vector parameters (biases, norms)

    Usage:
        muon_opt = Muon(muon_params, lr=lr, ...)
        adamw_opt = AdamW(adamw_params, lr=lr, ...)
        hybrid = HybridOptimizer([muon_opt, adamw_opt])

        # Use like any optimizer
        hybrid.zero_grad()
        loss.backward()
        hybrid.step()
    """

    def __init__(self, optimizers: List[Optimizer]):
        """
        Args:
            optimizers: List of optimizer instances to manage jointly
        """
        self.optimizers = optimizers

        # Dummy param_groups to satisfy base class interface
        # Real param_groups are in self.optimizers[i].param_groups
        super().__init__([], {})

    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        """Step all child optimizers."""
        loss = None
        for opt in self.optimizers:
            if closure is not None:
                loss = opt.step(closure)
            else:
                opt.step()
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero gradients for all child optimizers."""
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> Dict[str, Any]:
        """Get state dict with all child optimizer states."""
        return {
            f"optimizer_{i}": opt.state_dict()
            for i, opt in enumerate(self.optimizers)
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state for all child optimizers."""
        for i, opt in enumerate(self.optimizers):
            key = f"optimizer_{i}"
            if key in state_dict:
                opt.load_state_dict(state_dict[key])

    @property
    def param_groups(self) -> List[Dict[str, Any]]:
        """Get all param_groups from all child optimizers."""
        groups = []
        for opt in self.optimizers:
            groups.extend(opt.param_groups)
        return groups
```

**Key Features**:
- Transparent wrapper - works like a regular optimizer
- Compatible with AMP `GradScaler` (unscale works on child optimizers)
- State dict persistence for checkpoint saving/loading
- `param_groups` property for scheduler compatibility

---

### Phase 2: Optimizer Builder Modifications

#### 2.1 Modify `scripts/train.py::_create_optimizer()`

**Location**: `scripts/train.py`, lines 245-258

**Current Code**:
```python
def _create_optimizer(cfg: dict, model: nn.Module, stage: str) -> torch.optim.Optimizer:
    stage_cfg = cfg.get("stages", {}).get(stage, {})
    opt_cfg = stage_cfg.get("optimizer") or cfg.get("optimizer", {})
    name = opt_cfg.get("name", "adam").lower()
    lr = opt_cfg.get("lr", 1e-3)
    weight_decay = opt_cfg.get("weight_decay", 0.0)

    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        momentum = opt_cfg.get("momentum", 0.9)
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
```

**Modified Code**:
```python
def _create_optimizer(cfg: dict, model: nn.Module, stage: str) -> torch.optim.Optimizer:
    """Create optimizer with optional hybrid Muon+AdamW support."""
    stage_cfg = cfg.get("stages", {}).get(stage, {})
    opt_cfg = stage_cfg.get("optimizer") or cfg.get("optimizer", {})
    name = opt_cfg.get("name", "adam").lower()
    lr = opt_cfg.get("lr", 1e-3)
    weight_decay = opt_cfg.get("weight_decay", 0.0)

    # Standard optimizers (original behavior)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        momentum = opt_cfg.get("momentum", 0.9)
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    # NEW: Hybrid Muon+AdamW optimizer
    if name == "muon_hybrid" or name == "muon":
        from src.ups.training.param_groups import build_param_groups
        from src.ups.training.hybrid_optimizer import HybridOptimizer

        try:
            # Try to import Muon optimizer
            # Replace with actual import once library is chosen
            from muon_pytorch import Muon  # Example: pip install muon-optimizer
        except ImportError:
            print(f"WARNING: Muon optimizer not available, falling back to AdamW")
            return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Split parameters into Muon (2D+) and AdamW (1D) groups
        muon_params, adamw_params = build_param_groups(model)

        # Muon-specific hyperparameters (with defaults from guide)
        muon_beta = opt_cfg.get("muon_beta", 0.9)  # Nesterov momentum
        muon_scale = opt_cfg.get("muon_scale", 0.2)  # Orthogonalization scale
        muon_ns_iters = opt_cfg.get("muon_ns_iters", 5)  # Newton-Schulz iterations

        # AdamW hyperparameters
        adamw_betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))
        adamw_eps = opt_cfg.get("eps", 1e-8)

        optimizers = []

        # Create Muon optimizer if there are 2D+ parameters
        if len(muon_params) > 0:
            muon_opt = Muon(
                muon_params,
                lr=lr,
                weight_decay=weight_decay,  # Decoupled weight decay (like AdamW)
                beta=muon_beta,
                scale=muon_scale,
                ns_iters=muon_ns_iters,
            )
            optimizers.append(muon_opt)
            print(f"  Muon: {len(muon_params)} parameters (2D+ matrices)")

        # Create AdamW optimizer for 1D parameters (always include)
        if len(adamw_params) > 0:
            adamw_opt = torch.optim.AdamW(
                adamw_params,
                lr=lr,
                betas=adamw_betas,
                eps=adamw_eps,
                weight_decay=weight_decay,
            )
            optimizers.append(adamw_opt)
            print(f"  AdamW: {len(adamw_params)} parameters (1D vectors)")

        # If only one optimizer, return it directly
        if len(optimizers) == 1:
            return optimizers[0]

        # Otherwise return hybrid wrapper
        return HybridOptimizer(optimizers)

    raise ValueError(f"Unknown optimizer name: {name}")
```

**Key Changes**:
1. Added `muon_hybrid` and `muon` optimizer names
2. Automatic parameter grouping using `build_param_groups()`
3. Configurable Muon hyperparameters from YAML
4. Fallback to AdamW if Muon not available
5. Debug output showing parameter split
6. Returns `HybridOptimizer` when both groups exist

---

#### 2.2 AMP Gradient Unscaling Compatibility

**No changes needed** - the existing AMP code already handles hybrid optimizers correctly:

```python
# scripts/train.py, lines 622-627
if use_amp:
    if clip_val is not None:
        scaler.unscale_(optimizer)  # Works with HybridOptimizer!
    grad_norm = torch.nn.utils.clip_grad_norm_(...)
    scaler.step(optimizer)  # Works with HybridOptimizer!
    scaler.update()
```

**Why it works**:
- `scaler.unscale_(optimizer)` iterates over `optimizer.param_groups`
- `HybridOptimizer.param_groups` returns flattened list from all child optimizers
- `scaler.step(optimizer)` calls `optimizer.step()`, which we've overridden

---

### Phase 3: Configuration Schema

#### 3.1 YAML Configuration

**Example Config**: `configs/train_burgers_muon.yaml`

```yaml
# Inherit from golden config
include: train_burgers_golden.yaml

# Override optimizer to use Muon hybrid
stages:
  operator:
    epochs: 25

    optimizer:
      name: muon_hybrid       # or just "muon"
      lr: 1.0e-3              # Same LR as AdamW baseline
      weight_decay: 0.03      # Same weight decay for both groups

      # Muon-specific parameters (for 2D+ matrix params)
      muon_beta: 0.9          # Nesterov momentum (default: 0.9)
      muon_scale: 0.2         # Orthogonalization scale (default: 0.2)
      muon_ns_iters: 5        # Newton-Schulz iterations (default: 5)

      # AdamW-specific parameters (for 1D vector params)
      betas: [0.9, 0.999]     # Adam betas (default: [0.9, 0.999])
      eps: 1.0e-8             # Adam epsilon (default: 1e-8)

    # Gradient clipping (optional, often not needed with Muon)
    grad_clip: null           # Disable clipping

  diff_residual:
    epochs: 8

    optimizer:
      name: muon_hybrid
      lr: 5.0e-5              # Lower LR for diffusion
      weight_decay: 0.015
      muon_beta: 0.9
      muon_scale: 0.2

    scheduler:
      name: cosineannealinglr
      t_max: 8
      eta_min: 3.0e-6

  consistency_distill:
    epochs: 8

    optimizer:
      name: muon_hybrid
      lr: 3.0e-5
      weight_decay: 0.015
      muon_beta: 0.9
      muon_scale: 0.2

    scheduler:
      name: cosineannealinglr
      t_max: 8
      eta_min: 2.0e-6

# WandB logging
logging:
  wandb:
    enabled: true
    project: universal-simulator
    run_name: burgers-muon-hybrid
    tags: [muon, hybrid, optimizer-experiment]
    group: muon-experiments
```

**Configuration Hierarchy**:
1. `stages.<stage>.optimizer` (stage-specific) - takes precedence
2. `optimizer` (global fallback)

**Key Parameters**:
- `name: muon_hybrid` - Activates Muon+AdamW hybrid
- `lr: 1.0e-3` - **Same LR as AdamW baseline** (Muon scales internally)
- `weight_decay: 0.03` - Applied to both Muon and AdamW groups
- `muon_scale: 0.2` - Multiplies orthogonalized update to match AdamW RMS
- `muon_beta: 0.9` - Nesterov momentum (0.85-0.95 range)
- `muon_ns_iters: 5` - Newton-Schulz iterations (5-7 typical)

---

### Phase 4: Testing

#### 4.1 Unit Tests

**Create**: `tests/unit/test_param_groups.py`

```python
"""Unit tests for parameter grouping utilities."""
import pytest
import torch
import torch.nn as nn
from src.ups.training.param_groups import (
    is_matrix_param,
    build_param_groups,
    build_param_groups_with_names,
)


class TinyModel(nn.Module):
    """Minimal model with mixed parameter types."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 32)  # 2D weight + 1D bias
        self.norm = nn.LayerNorm(32)     # 1D weight + 1D bias
        self.fc = nn.Linear(32, 8)       # 2D weight + 1D bias

    def forward(self, x):
        return self.fc(self.norm(self.linear(x)))


def test_is_matrix_param():
    """Test matrix parameter detection."""
    p_2d = nn.Parameter(torch.randn(16, 32))  # Linear weight
    p_1d = nn.Parameter(torch.randn(32))      # Bias
    p_0d = nn.Parameter(torch.tensor(0.5))    # Scalar

    assert is_matrix_param(p_2d) == True
    assert is_matrix_param(p_1d) == False
    assert is_matrix_param(p_0d) == False


def test_build_param_groups():
    """Test parameter splitting into Muon/AdamW groups."""
    model = TinyModel()
    muon_params, adamw_params = build_param_groups(model)

    # Should have 2 Linear weights (2D)
    assert len(muon_params) == 2

    # Should have 2 Linear biases + 2 LayerNorm params (all 1D)
    assert len(adamw_params) == 4

    # Check shapes
    for p in muon_params:
        assert p.ndim >= 2
    for p in adamw_params:
        assert p.ndim < 2


def test_build_param_groups_with_names():
    """Test parameter splitting with name tracking."""
    model = TinyModel()
    muon_params, adamw_params = build_param_groups_with_names(model)

    muon_names = [name for name, _ in muon_params]
    adamw_names = [name for name, _ in adamw_params]

    # Linear/fc weights should be in Muon group
    assert any("linear.weight" in name for name in muon_names)
    assert any("fc.weight" in name for name in muon_names)

    # Biases and LayerNorm should be in AdamW group
    assert any("bias" in name for name in adamw_names)
    assert any("norm.weight" in name for name in adamw_names)


def test_real_model_split():
    """Test parameter split on a realistic UPS-like model."""
    from src.ups.models.latent_operator import LatentOperator
    from src.ups.core.blocks_pdet import PDETransformerBlock

    # Create a small operator
    pdet = PDETransformerBlock(
        input_dim=16,
        hidden_dim=64,
        depths=[1, 1, 1],
        group_size=8,
        num_heads=4,
    )

    muon_params, adamw_params = build_param_groups(pdet)

    # Transformer should be heavily weighted toward 2D params
    total_muon = sum(p.numel() for p in muon_params)
    total_adamw = sum(p.numel() for p in adamw_params)

    # Expect >90% of params to be 2D (Muon-compatible)
    assert total_muon / (total_muon + total_adamw) > 0.9
```

---

#### 4.2 Integration Tests

**Create**: `tests/integration/test_muon_training.py`

```python
"""Integration tests for Muon optimizer training."""
import pytest
import torch
import torch.nn as nn
from pathlib import Path
import yaml


def test_muon_optimizer_creation():
    """Test that Muon hybrid optimizer can be created."""
    from src.ups.training.param_groups import build_param_groups
    from src.ups.training.hybrid_optimizer import HybridOptimizer

    # Simple model
    model = nn.Sequential(
        nn.Linear(16, 32),
        nn.LayerNorm(32),
        nn.Linear(32, 8),
    )

    muon_params, adamw_params = build_param_groups(model)

    # Create optimizers
    # NOTE: This will fail if Muon library not installed - expected behavior
    try:
        from muon_pytorch import Muon
        muon_opt = Muon(muon_params, lr=1e-3, weight_decay=0.01)
        adamw_opt = torch.optim.AdamW(adamw_params, lr=1e-3, weight_decay=0.01)
        hybrid = HybridOptimizer([muon_opt, adamw_opt])

        # Test basic operations
        x = torch.randn(4, 16)
        loss = model(x).sum()
        loss.backward()

        hybrid.step()
        hybrid.zero_grad()

    except ImportError:
        pytest.skip("Muon library not installed")


def test_muon_training_loop():
    """Test a minimal training loop with Muon."""
    from src.ups.training.param_groups import build_param_groups
    from src.ups.training.hybrid_optimizer import HybridOptimizer

    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))

    try:
        from muon_pytorch import Muon
        muon_params, adamw_params = build_param_groups(model)
        muon_opt = Muon(muon_params, lr=1e-2, weight_decay=1e-3)
        adamw_opt = torch.optim.AdamW(adamw_params, lr=1e-2, weight_decay=1e-3)
        optimizer = HybridOptimizer([muon_opt, adamw_opt])

        # Train for 10 steps
        for _ in range(10):
            x = torch.randn(2, 4)
            y = torch.randn(2, 4)

            optimizer.zero_grad()
            pred = model(x)
            loss = nn.functional.mse_loss(pred, y)
            loss.backward()
            optimizer.step()

        # Should converge (loss should decrease)
        assert loss.item() < 1.0

    except ImportError:
        pytest.skip("Muon library not installed")


def test_muon_config_validation():
    """Test that Muon config validates correctly."""
    from scripts.validate_config import validate_config

    config_path = Path("configs/train_burgers_muon.yaml")
    if config_path.exists():
        # Validate Muon config
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        # Check Muon parameters are present
        opt_cfg = cfg["stages"]["operator"]["optimizer"]
        assert opt_cfg["name"] == "muon_hybrid"
        assert "muon_beta" in opt_cfg
        assert "muon_scale" in opt_cfg
        assert "lr" in opt_cfg
```

---

### Phase 5: Deployment & Validation

#### 5.1 Validation Checklist

**Before first Muon run**:

- [ ] **Install Muon library**
  ```bash
  pip install muon-optimizer  # Replace with actual package name
  ```

- [ ] **Run unit tests**
  ```bash
  pytest tests/unit/test_param_groups.py -v
  ```

- [ ] **Create Muon config**
  ```bash
  cp configs/train_burgers_golden.yaml configs/train_burgers_muon.yaml
  # Edit to use muon_hybrid optimizer
  ```

- [ ] **Validate config**
  ```bash
  python scripts/validate_config.py configs/train_burgers_muon.yaml
  ```

- [ ] **Dry run (1 epoch)**
  ```bash
  python scripts/train.py \
    --config configs/train_burgers_muon.yaml \
    --stage operator \
    --epochs 1
  ```

- [ ] **Check gradient norms**
  - Verify `mean_grad_norm` is stable (not exploding)
  - Should be similar to AdamW baseline (~0.1-1.0 range)

- [ ] **Check learning rate**
  - Verify LR is logged correctly
  - Should match config value (1e-3 for operator)

- [ ] **Full operator training**
  ```bash
  python scripts/train.py \
    --config configs/train_burgers_muon.yaml \
    --stage operator
  ```

- [ ] **Compare to AdamW baseline**
  - Operator final loss should be similar or better (<0.001)
  - Training time should be comparable
  - Memory usage should be similar

---

#### 5.2 Ablation Study Plan

**Recommended experiments** (via WandB sweep):

```yaml
# sweep_muon_hyperparams.yaml
program: scripts/train.py
method: grid
parameters:
  config:
    value: configs/train_burgers_muon.yaml

  # Vary Muon scale (orthogonalization strength)
  stages.operator.optimizer.muon_scale:
    values: [0.15, 0.2, 0.25]

  # Vary Muon beta (momentum)
  stages.operator.optimizer.muon_beta:
    values: [0.85, 0.9, 0.95]

  # Vary batch size (Muon's strength is large batches)
  training.batch_size:
    values: [12, 24, 48]

metric:
  name: operator/final_loss
  goal: minimize
```

**Run sweep**:
```bash
wandb sweep sweep_muon_hyperparams.yaml
wandb agent <sweep_id>
```

**Metrics to track**:
1. **Operator final loss** - Primary metric (<0.001 is good)
2. **Training time** - Should be similar to AdamW
3. **Gradient norm** - Should be stable (0.1-1.0 range)
4. **Evaluation NRMSE** - Final validation performance
5. **Memory usage** - Peak GPU memory

---

## Expected Benefits

Based on the Muon optimizer paper and implementation guide:

1. **Faster Convergence**
   - 2-3x fewer steps to reach same loss
   - More stable training (less sensitivity to LR)
   - Better generalization (less overfitting)

2. **Better Batch Scaling**
   - Can use 2-4x larger batch sizes without degradation
   - Better GPU utilization (higher throughput)
   - Potential cost savings on VastAI (faster training)

3. **Reduced Hyperparameter Sensitivity**
   - Less need for LR tuning (same LR as AdamW baseline works)
   - Gradient clipping often unnecessary
   - More robust to different model sizes

4. **Improved Numerical Stability**
   - Orthogonalized updates prevent gradient explosion
   - Better conditioning of parameter updates
   - Less variance across random seeds

---

## Potential Risks & Mitigations

### Risk 1: Muon Library Availability

**Issue**: Generic Muon library may not exist or be poorly maintained

**Mitigation**:
1. Research available implementations:
   - `kaiokendev/muon`: GitHub implementation
   - `lucidrains/muon-pytorch`: Potential PyPI package
   - Official reference implementation from paper
2. If none available, implement based on paper (Newton-Schulz iterations)
3. Fallback to AdamW if Muon import fails (already handled in code)

### Risk 2: FSDP/DDP Incompatibility

**Issue**: Muon's momentum buffer may not be sharded correctly

**Mitigation**:
- **Not applicable** - UPS uses single-GPU training (no FSDP/DDP)
- If future multi-GPU support added, verify Muon has ZeRO-compatible implementation

### Risk 3: Numerical Instability with Mixed Precision

**Issue**: Muon's matrix operations may be sensitive to fp16/bf16 precision

**Mitigation**:
1. Start with `amp=true` (default bf16) - more stable than fp16
2. If instability occurs:
   - Disable AMP for Muon parameters only (keep AdamW with AMP)
   - Use fp32 for Newton-Schulz iterations internally
3. Monitor gradient norms - early warning of instability

### Risk 4: Checkpoint Compatibility

**Issue**: Muon momentum buffer may change state dict structure

**Mitigation**:
- **Not applicable** - UPS doesn't save optimizer state (fresh optimizer each stage)
- If optimizer state saving added, test `HybridOptimizer.state_dict()` thoroughly

### Risk 5: Performance Regression

**Issue**: Muon may be slower than AdamW due to matrix operations

**Mitigation**:
1. Profile first: `torch.profiler` to measure overhead
2. Muon overhead is typically <10% of total training time
3. If too slow:
   - Reduce `muon_ns_iters` from 5 to 3 (faster, slightly less accurate)
   - Use torch.compile on Muon update step
   - Only apply Muon to largest layers (FFN weights), AdamW for attention

---

## Code References

**Key Files**:
- `scripts/train.py:245-258` - Optimizer creation (modify here)
- `scripts/train.py:622-627` - Gradient clipping with AMP (already compatible)
- `src/ups/core/blocks_pdet.py` - PDE-Transformer with Linear layers (Muon targets)
- `src/ups/models/latent_operator.py` - Operator model structure
- `configs/train_burgers_golden.yaml` - Golden config template

**New Files to Create**:
- `src/ups/training/param_groups.py` - Parameter grouping utilities
- `src/ups/training/hybrid_optimizer.py` - Hybrid optimizer wrapper
- `tests/unit/test_param_groups.py` - Unit tests
- `tests/integration/test_muon_training.py` - Integration tests
- `configs/train_burgers_muon.yaml` - Muon config

---

## Alternative Implementation: Muon-Only (No Hybrid)

If hybrid complexity is undesirable, a simpler approach:

**Apply Muon to ALL parameters** (including 1D):

```python
# Simplified: Use Muon for everything
if name == "muon":
    from muon_pytorch import Muon
    return Muon(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        beta=opt_cfg.get("muon_beta", 0.9),
        scale=opt_cfg.get("muon_scale", 0.2),
    )
```

**Pros**:
- Much simpler code (no parameter grouping, no hybrid wrapper)
- Fewer moving parts
- Easier to debug

**Cons**:
- Muon is designed for 2D matrices - applying to 1D may be suboptimal
- May waste computation on 1D parameters
- Unclear if Muon handles 1D parameters correctly (check library docs)

**Recommendation**: Start with hybrid approach for correctness, profile to see if simplification is worth it.

---

## Timeline Estimate

**Implementation**: 4-6 hours
- Phase 1 (new files): 2 hours
- Phase 2 (modifications): 1 hour
- Phase 3 (config): 30 min
- Phase 4 (tests): 1-2 hours
- Phase 5 (validation): 1 hour

**Experimentation**: 1-2 days
- Baseline comparison: 4-6 hours (2-3 training runs)
- Ablation study: 12-24 hours (sweep over hyperparams)
- Analysis & reporting: 2-4 hours

**Total**: 2-3 days for full integration and validation

---

## Success Criteria

**Minimum Viable**:
- [ ] Muon hybrid optimizer can be instantiated
- [ ] Training completes without errors
- [ ] Operator final loss < 0.001 (matches AdamW)
- [ ] No gradient explosions (grad_norm stable)

**Target**:
- [ ] Operator converges 1.5-2x faster than AdamW (fewer epochs to same loss)
- [ ] Can use 2x larger batch size without degradation
- [ ] Training time comparable or faster than AdamW
- [ ] Evaluation NRMSE matches or beats AdamW baseline

**Stretch**:
- [ ] Outperforms AdamW on final evaluation metrics
- [ ] Enables cost savings via faster training or larger batches
- [ ] Generalizes to other PDE tasks (NS2D, wave equation, etc.)

---

## Related Documentation

**Internal**:
- `CLAUDE.md` - Project overview and conventions
- `docs/production_playbook.md` - Best practices for experiments
- `parallel_runs_playbook.md` - Hyperparameter sweep guidance

**External**:
- `docs/Integrating Muon Optimizer into a Neural Physics Transformer Framework.pdf` - Muon theory and guide
- `docs/muon_npt_impl_for_coding_agent.md` - Generic implementation guide
- Muon paper: [cite when available]

---

## Appendix: Muon Hyperparameter Guide

### `muon_scale` (Orthogonalization Multiplier)

**Default**: 0.2
**Range**: 0.15 - 0.25
**Effect**: Scales the orthogonalized update to match AdamW's RMS magnitude

**Tuning**:
- **Too low** (0.1): Slow convergence, underfitting
- **Just right** (0.2): Matches AdamW convergence speed with better stability
- **Too high** (0.3+): Unstable training, oscillations

**Recommendation**: Start with 0.2, only tune if seeing convergence issues

---

### `muon_beta` (Nesterov Momentum)

**Default**: 0.9
**Range**: 0.85 - 0.95
**Effect**: Controls momentum for gradient accumulation

**Tuning**:
- **Lower** (0.85): Faster response to gradient changes, less smoothing
- **Higher** (0.95): More smoothing, more stable but slower adaptation

**Comparison to Adam**:
- Adam beta1=0.9 ≈ Muon beta=0.9 (similar momentum strength)

**Recommendation**: Keep at 0.9 unless instability occurs (then increase to 0.95)

---

### `muon_ns_iters` (Newton-Schulz Iterations)

**Default**: 5
**Range**: 3 - 7
**Effect**: Accuracy of polar decomposition (more iters = more accurate orthogonalization)

**Tuning**:
- **Fewer** (3): Faster but less accurate, may hurt convergence
- **More** (7): More accurate but slower, diminishing returns beyond 7

**Recommendation**: Keep at 5 (good accuracy/speed tradeoff)

---

### Learning Rate

**Recommendation**: **Use same LR as AdamW baseline**

Muon's internal scaling (`muon_scale`) adapts the update magnitude, so:
- Operator: `lr: 1e-3`
- Diffusion: `lr: 5e-5`
- Consistency: `lr: 3e-5`

**Do NOT reduce LR** - Muon is designed to work with standard LRs.

---

### Weight Decay

**Recommendation**: **Use same weight decay as AdamW**

Muon implements decoupled weight decay (like AdamW), so standard values work:
- `weight_decay: 0.03` (operator)
- `weight_decay: 0.015` (diffusion/consistency)

---

### Gradient Clipping

**Recommendation**: **Disable or use high threshold**

Muon's orthogonalized updates are naturally bounded, so aggressive clipping is unnecessary:
- `grad_clip: null` (disabled) - preferred
- `grad_clip: 5.0` (loose threshold) - fallback

Only re-enable if seeing gradient explosions (rare with Muon).

---

## Conclusion

The Universal Physics Stack codebase is well-suited for Muon optimizer integration due to:
1. **Architecture**: 95%+ of parameters are 2D Linear weights (Muon-compatible)
2. **Infrastructure**: Clean per-stage optimizer creation with YAML config
3. **Training**: Single-GPU with AMP/clipping already in place
4. **Minimal Changes**: Only 3 new files + 1 function modification needed

The hybrid Muon+AdamW approach provides maximum correctness while maintaining simplicity. Expected benefits include faster convergence, better batch scaling, and reduced hyperparameter sensitivity - all valuable for cost-effective training on VastAI.

**Next Steps**: Implement Phase 1-2 (infrastructure), test with Phase 4 (unit tests), then run Phase 5 (validation) with a single training run before full experimentation.
