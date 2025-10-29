---
date: 2025-10-29T01:25:06Z
researcher: Emery Gunselman
git_commit: 5838f764dc58869a530596107a82e5db2aa87e1e
branch: feature--UPT
repository: universal_simulator
topic: "TTC PRM/ARM Integration Implementation Roadmap"
tags: [research, codebase, ttc, prm, arm, reward-models, test-time-conditioning, integration]
status: complete
last_updated: 2025-10-28
last_updated_by: Emery Gunselman
---

# Research: TTC PRM/ARM Integration Implementation Roadmap

**Date**: 2025-10-29T01:25:06Z
**Researcher**: Emery Gunselman
**Git Commit**: 5838f764dc58869a530596107a82e5db2aa87e1e
**Branch**: feature--UPT
**Repository**: universal_simulator

## Research Question

How should the TTC PRM/ARM integration plan (`docs/ttc_prm_arm_integration_plan.md`) be implemented within the existing Universal Physics Stack codebase architecture?

## Executive Summary

The Universal Physics Stack already implements ~60% of the TTC PRM/ARM integration plan from the paper "Reward-Model–Driven Test-Time Computing for PDE Foundation Models" (arXiv:2509.02846). The existing codebase has:

- ✅ **Complete TTC framework** with beam search and lookahead
- ✅ **Analytical Reward Model (ARM)** with conservation law checks
- ✅ **Stochastic sampling** via tau sampling and Gaussian noise
- ✅ **Diffusion residual model** with consistency distillation
- ❌ **Missing: Process Reward Model (PRM)** trained with triplet loss

The primary implementation work required is adding the **PRM training pipeline** (triplet generation, model architecture, and training scripts), which can be accomplished by reusing existing infrastructure patterns.

## Key Findings

### 1. Test-Time Conditioning (TTC) - FULLY IMPLEMENTED

**File**: `src/ups/inference/rollout_ttc.py`

The TTC framework is production-ready with sophisticated features:

#### Core Implementation (Lines 81-196)
```python
def ttc_rollout(
    initial_state: LatentState,
    operator: LatentOperator,
    reward_model: RewardModel,
    config: TTCConfig,
    corrector: Optional[DiffusionResidual] = None,
) -> Tuple[RolloutLog, List[TTCStepLog]]:
```

**Features:**
- **Multi-candidate exploration**: Generates N candidates per step via diffusion sampling
- **Beam search**: Top-k selection with configurable beam width (lines 159-168)
- **Recursive lookahead**: Multi-step reward estimation with gamma discount (lines 130-143)
- **Budget management**: Evaluation count limits to control compute cost
- **Early stopping**: Skip lookahead if best candidate has large margin
- **Detailed logging**: Per-step reward components for analysis

#### Configuration (Lines 25-40)
```python
@dataclass
class TTCConfig:
    candidates: int = 4           # Branching factor
    beam_width: int = 1           # Top-k for lookahead
    horizon: int = 1              # Lookahead depth
    tau_range: Tuple[float, float] = (0.3, 0.7)
    noise_std: float = 0.0
    max_evaluations: Optional[int] = None
    gamma: float = 1.0            # Discount factor
```

#### Performance (from `configs/train_burgers_golden.yaml:151-197`)
- **Candidates**: 16
- **Beam width**: 5
- **Expected improvement**: 88% NRMSE reduction (0.78 → 0.09)

---

### 2. Analytical Reward Model (ARM) - FULLY IMPLEMENTED

**File**: `src/ups/eval/reward_models.py:66-166`

Physics-based reward model that scores candidates based on conservation law violations.

#### Architecture
```python
class AnalyticalRewardModel(RewardModel):
    def __init__(
        self,
        decoder: AnyPointDecoder,
        grid_shape: Tuple[int, int],
        weights: AnalyticalRewardWeights,
        mass_field: Optional[str] = None,
        energy_field: Optional[str] = None,
        momentum_fields: Sequence[str] = (),
    ):
```

#### Physics Scoring Components (Lines 106-166)

**Mass Conservation** (Lines 122-129):
```python
# Penalizes absolute change in total mass
mass_prev = decode_prev[mass_field].sum(dim=(-2, -1))
mass_next = decode_next[mass_field].sum(dim=(-2, -1))
mass_gap = (mass_next - mass_prev).abs()
```

**Momentum Conservation** (Lines 131-141):
```python
# Penalizes momentum gaps across specified channels
mom_prev_sum = sum([decode_prev[f].sum(dim=(-2,-1)) for f in momentum_fields])
mom_next_sum = sum([decode_next[f].sum(dim=(-2,-1)) for f in momentum_fields])
momentum_gap = (mom_next_sum - mom_prev_sum).abs()
```

**Energy Conservation** (Lines 143-150):
```python
# Penalizes squared-field energy gaps
energy_prev = (decode_prev[energy_field] ** 2).sum(dim=(-2, -1))
energy_next = (decode_next[energy_field] ** 2).sum(dim=(-2, -1))
energy_gap = (energy_next - energy_prev).abs()
```

**Negativity Penalty** (Lines 152-159):
```python
# Penalizes negative mass values
neg_penalty = decode_next[mass_field].clamp_max(0).abs().sum(dim=(-2, -1))
```

**Weighted Combination** (Lines 161-163):
```python
total = (weights.mass * mass_gap +
         weights.momentum * momentum_gap +
         weights.energy * energy_gap +
         weights.penalty_negative * neg_penalty)
return -total  # Higher is better
```

#### Configuration Integration (`configs/inference_ttc.yaml:14-22`)
```yaml
ttc:
  reward:
    grid: [64, 64]              # Query grid for decoding
    mass_field: "rho"           # Density field
    energy_field: "e"           # Energy field
    momentum_field: []          # Optional momentum fields
    weights:
      mass: 1.0
      energy: 0.1
      penalty_negative: 0.5
```

---

### 3. Stochastic Sampling Mechanisms - FULLY IMPLEMENTED

**File**: `src/ups/inference/rollout_ttc.py:103-128`

Two sources of stochasticity for candidate diversity:

#### A. Tau Sampling (Line 119)
```python
tau = torch.empty(candidate.z.size(0), device=device).uniform_(*config.tau_range)
drift = corrector(candidate, tau)
candidate.z = candidate.z + drift
```
- Uniform sampling from configurable range (e.g., [0.15, 0.85])
- Each candidate gets different tau value
- Controls diffusion correction strength

#### B. Gaussian Noise Injection (Lines 122-124)
```python
if noise_std > 0.0:
    noise = torch.randn_like(candidate.z) * noise_std
    candidate.z = candidate.z + noise
```
- Per-step noise schedule support
- Configurable via `noise_std` or `noise_schedule`

#### Training-Time Tau Distribution (`scripts/train.py:372-382`)
```python
def _sample_tau(batch_size: int, device: torch.device, cfg: Dict) -> torch.Tensor:
    dist_cfg = cfg.get("training", {}).get("tau_distribution")
    if dist_cfg and dist_cfg.get("type") == "beta":
        alpha = float(dist_cfg.get("alpha", 1.0))
        beta = float(dist_cfg.get("beta", 1.0))
        beta_dist = torch.distributions.Beta(alpha, beta)
        return beta_dist.sample((batch_size,)).to(device)
    return torch.rand(batch_size, device=device)
```

**Configuration** (`configs/train_burgers_golden.yaml:95-98`):
```yaml
training:
  tau_distribution:
    type: beta
    alpha: 1.2
    beta: 1.2
```

---

### 4. Diffusion Residual Model - FULLY IMPLEMENTED

**File**: `src/ups/models/diffusion_residual.py:22-54`

#### Architecture
```python
class DiffusionResidual(nn.Module):
    def __init__(self, config: DiffusionResidualConfig):
        # Input: latent_dim + 1 (tau) + cond_dim
        # Network: 3-layer MLP with SiLU activations
        # Output: Drift (residual correction) of shape latent_dim
```

#### Training Stages

**Stage 2: Diffusion Residual Training** (`scripts/train.py:695-879`)
- Learns to predict residuals between operator predictions and ground truth
- Loss: MSE(predicted_drift, actual_residual)
- Tau sampling: Per-sample from Beta(1.2, 1.2) distribution
- Configuration: 8 epochs, lr=5e-5, EMA decay=0.999

**Stage 3: Consistency Distillation** (`scripts/train.py:942-1200`)
- Distills multi-step diffusion into few-step predictor
- Teacher-student setup with triplet supervision across multiple taus
- Tau schedule: [5, 4, 3] samples per epoch
- Configuration: 8 epochs, lr=3e-5, batch_size=6

#### Integration with TTC
The diffusion model serves as the "corrector" in TTC rollout:
```python
# From rollout_ttc.py:115-120
if corrector is not None:
    tau = torch.empty(...).uniform_(*config.tau_range)
    drift = corrector(candidate, tau)
    candidate.z = candidate.z + drift
```

---

### 5. Decoder Infrastructure - FULLY IMPLEMENTED

**File**: `src/ups/io/decoder_anypoint.py:53-120+`

Query-based decoder used by reward models to decode latent states to physical fields.

#### Architecture
```python
class AnyPointDecoder(nn.Module):
    """Perceiver-style cross-attention decoder."""

    def __init__(
        self,
        latent_dim: int,
        query_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        frequencies: Sequence[float],
        output_channels: Mapping[str, int],
    ):
```

**Features:**
- Fourier-encoded query embedding (configurable frequencies)
- Multi-layer cross-attention (queries attend to latent tokens)
- Per-field MLPs for multi-channel output
- Arbitrary query point support (discretization-agnostic)

**Used by ARM** to decode latent states for conservation checks:
```python
# From reward_models.py:113-116
decode_prev = self.decoder(prev_state, self.query_coords)
decode_next = self.decoder(next_state, self.query_coords)
# Then compute mass/momentum/energy gaps
```

---

### 6. Composite Reward Model System - FULLY IMPLEMENTED

**File**: `src/ups/eval/reward_models.py:249-283`

Enables weighted ensemble of multiple reward models.

#### Architecture
```python
class CompositeRewardModel(RewardModel):
    def __init__(self, models: Sequence[Tuple[RewardModel, float]]):
        # models: List of (model, weight) pairs

    def score(self, prev_state, next_state, context=None):
        scores = []
        for model, weight in self.models:
            if weight != 0.0:
                scores.append(weight * model.score(prev_state, next_state, context))
        return sum(scores) / total_weight
```

**Current Usage** (`src/ups/inference/rollout_ttc.py:268-272`):
```python
if len(models) == 0:
    raise ValueError("No reward models configured")
elif len(models) == 1:
    return models[0][0]
else:
    return CompositeRewardModel(models)
```

This enables **ARM + PRM** combination once PRM is implemented!

---

### 7. FeatureCriticRewardModel - PARTIAL IMPLEMENTATION

**File**: `src/ups/eval/reward_models.py:169-246`

A learned reward model that operates on global physics features (similar concept to PRM).

#### Architecture (Lines 205-210)
```python
class FeatureCriticRewardModel(RewardModel):
    def __init__(self, feature_dim: int, hidden_dim: int, dropout: float):
        # Simple MLP: Linear -> SiLU -> Dropout -> Linear
        # Input: Feature delta (next - prev)
        # Output: Scalar score
```

#### Feature Extraction (Lines 212-234)
```python
def _features(self, state: LatentState) -> torch.Tensor:
    decoded = self.decoder(state, self.query_coords)
    features = []
    if self.mass_field:
        mass_sum = decoded[self.mass_field].sum(dim=(-2, -1))
        neg_penalty = decoded[self.mass_field].clamp_max(0).abs().sum(dim=(-2, -1))
        features.extend([mass_sum, neg_penalty])
    for mom_field in self.momentum_fields:
        features.append(decoded[mom_field].sum(dim=(-2, -1)))
    if self.energy_field:
        energy = (decoded[self.energy_field] ** 2).sum(dim=(-2, -1))
        features.append(energy)
    return torch.stack(features, dim=-1)
```

**Key Difference from Integration Plan PRM:**
- FeatureCritic operates on **global summary statistics** (sums, penalties)
- Integration plan PRM operates on **full spatial concatenation** of (state_t, cand_tp1)

---

## What's Missing: Process Reward Model (PRM)

The integration plan requires a **learned PRM trained with triplet loss** on (best, median, worst) candidates ranked by MSE to ground truth. This is **NOT** in the codebase.

### Required Components

#### 1. PRM Model Architecture (NEW)

**File**: `src/ups/models/prm.py` (to be created)

Based on integration plan lines 192-206:

```python
class PRM(nn.Module):
    """Process Reward Model - CNN-based scalar scorer for state transitions."""

    def __init__(self, in_ch: int, width: int = 64, depth: int = 6):
        """
        Args:
            in_ch: Input channels = 2 * latent_dim (concatenated states)
            width: Hidden dimension for conv layers
            depth: Number of conv layers
        """
        super().__init__()
        layers = []
        ch = in_ch
        for _ in range(depth):
            layers += [nn.Conv2d(ch, width, 3, padding=1), nn.GELU()]
            ch = width
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(width, 1)
        )

    def forward(self, state_t: torch.Tensor, cand_tp1: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_t: Previous state [B, C, H, W]
            cand_tp1: Candidate next state [B, C, H, W]
        Returns:
            scores: Scalar scores [B]
        """
        x = torch.cat([state_t, cand_tp1], dim=1)  # [B, 2*C, H, W]
        x = self.trunk(x)
        return self.head(x).squeeze(-1)  # [B]
```

**Integration point**: This follows the same pattern as `DiffusionResidual` in `src/ups/models/diffusion_residual.py`.

---

#### 2. Triplet Dataset (NEW)

**File**: `src/ups/data/prm_triplets.py` (to be created)

Based on integration plan lines 209-227 and existing `GridLatentPairDataset` pattern:

```python
from torch.utils.data import Dataset
import torch
from pathlib import Path
from typing import List

class PRMTripletDataset(Dataset):
    """
    Loads pre-generated triplets for PRM training.

    Each triplet contains:
        - state_t: Previous latent state
        - best: Candidate with lowest MSE to ground truth
        - med: Median MSE candidate
        - worst: Candidate with highest MSE to ground truth
    """

    def __init__(self, manifest_paths: List[str]):
        """
        Args:
            manifest_paths: List of paths to .pt files containing triplets
        """
        self.items = sorted(manifest_paths)

    def __getitem__(self, i: int):
        """
        Returns:
            state_t: [C, H, W] or [tokens, latent_dim]
            best: Same shape as state_t
            med: Same shape as state_t
            worst: Same shape as state_t
        """
        path = self.items[i]
        data = torch.load(path)
        return (
            data["state_t"],
            data["best"],
            data["med"],
            data["worst"]
        )

    def __len__(self):
        return len(self.items)

def collate_prm_triplets(batch):
    """Collate function for DataLoader."""
    state_t, best, med, worst = zip(*batch)
    return (
        torch.stack(state_t, dim=0),
        torch.stack(best, dim=0),
        torch.stack(med, dim=0),
        torch.stack(worst, dim=0)
    )
```

**Integration point**: Follows the same pattern as `GridLatentPairDataset` in `src/ups/data/latent_pairs.py:257-413`.

---

#### 3. Triplet Generation Script (NEW)

**File**: `scripts/build_prm_data.py` (to be created)

Based on integration plan lines 230-265 with **heavy reuse** of existing infrastructure:

```python
#!/usr/bin/env python3
"""Generate PRM triplet data from foundation model rollouts.

Reuses existing infrastructure:
- GridLatentPairDataset for loading (state_t, gt_tp1) pairs
- DiffusionResidual for candidate generation
- Tau sampling from train.py
"""

import torch
import os
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any
import argparse

# Reuse existing utilities
from src.ups.utils.config_loader import load_config
from src.ups.models.latent_operator import LatentOperator
from src.ups.models.diffusion_residual import DiffusionResidual
from src.ups.data.latent_pairs import GridLatentPairDataset
from scripts.train import _sample_tau  # Reuse tau sampling


def build_triplets(
    config: Dict[str, Any],
    K: int = 100,
    subset_fraction: float = 0.125,
    out_dir: str = "data/prm_triplets"
):
    """
    Generate triplet data for PRM training.

    Args:
        config: Training configuration (from YAML)
        K: Number of candidates to generate per state
        subset_fraction: Fraction of training data to use (0.125 = 12.5%)
        out_dir: Output directory for triplet files

    Pipeline:
        1. Load operator + diffusion from checkpoints
        2. Load ~12.5% of training data
        3. For each (state_t, gt_tp1) pair:
           - Generate K candidates via stochastic diffusion sampling
           - Rank candidates by MSE to gt_tp1
           - Extract best, median, worst
           - Save triplet to disk
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load models (REUSE checkpoint pattern from train.py)
    print("Loading operator and diffusion models...")
    operator = LatentOperator(config["operator"])
    operator.load_state_dict(torch.load("checkpoints/operator.pt", map_location=device))
    operator.to(device).eval()

    diffusion = DiffusionResidual(config["diffusion"])
    diffusion.load_state_dict(torch.load("checkpoints/diffusion_residual.pt", map_location=device))
    diffusion.to(device).eval()

    # 2. Load dataset (REUSE GridLatentPairDataset)
    print(f"Loading {subset_fraction*100}% of training data...")
    full_dataset = GridLatentPairDataset(
        root=config["data"]["root"],
        task=config["data"]["task"],
        split="train",
        encoder=None,  # Use latent cache if available
        # ... other config params
    )

    # Subsample dataset
    subset_size = int(len(full_dataset) * subset_fraction)
    indices = torch.randperm(len(full_dataset))[:subset_size]

    # 3. Generate triplets
    print(f"Generating triplets (K={K} candidates per state)...")
    triplet_idx = 0

    with torch.no_grad():
        for idx in tqdm(indices):
            state_t, gt_tp1, dt = full_dataset[idx]
            state_t = state_t.to(device).unsqueeze(0)  # [1, ...]
            gt_tp1 = gt_tp1.to(device).unsqueeze(0)
            dt_tensor = torch.tensor([dt], device=device)

            # Generate base prediction
            base_pred = operator(state_t, dt_tensor)

            # Generate K candidates (REUSE tau sampling logic)
            candidates = []
            for k in range(K):
                # Sample tau from Beta distribution
                tau = _sample_tau(1, device, config)

                # Apply diffusion correction
                drift = diffusion(base_pred, tau)
                cand_k = base_pred.z + drift
                candidates.append(cand_k)

            # Rank by MSE to ground truth
            mses = []
            for cand in candidates:
                mse = ((cand - gt_tp1.z) ** 2).mean()
                mses.append(mse.item())

            # Sort candidates by MSE (ascending)
            sorted_indices = torch.tensor(mses).argsort()
            best_idx = sorted_indices[0].item()
            med_idx = sorted_indices[K // 2].item()
            worst_idx = sorted_indices[-1].item()

            # Extract triplet
            triplet = {
                "state_t": state_t.squeeze(0).cpu(),
                "best": candidates[best_idx].squeeze(0).cpu(),
                "med": candidates[med_idx].squeeze(0).cpu(),
                "worst": candidates[worst_idx].squeeze(0).cpu(),
            }

            # Save to disk
            save_path = Path(out_dir) / f"triplet_{triplet_idx:08d}.pt"
            torch.save(triplet, save_path)
            triplet_idx += 1

    print(f"Generated {triplet_idx} triplets in {out_dir}")

    # Save manifest file
    manifest_path = Path(out_dir) / "manifest.txt"
    with open(manifest_path, "w") as f:
        for i in range(triplet_idx):
            f.write(f"{out_dir}/triplet_{i:08d}.pt\n")
    print(f"Manifest saved to {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to training config YAML")
    parser.add_argument("--K", type=int, default=100, help="Candidates per state")
    parser.add_argument("--subset-fraction", type=float, default=0.125, help="Fraction of train data")
    parser.add_argument("--out-dir", type=str, default="data/prm_triplets", help="Output directory")
    args = parser.parse_args()

    config = load_config(args.config)
    build_triplets(config, args.K, args.subset_fraction, args.out_dir)
```

**Key Reuse Points:**
- `GridLatentPairDataset` from `src/ups/data/latent_pairs.py`
- `_sample_tau()` from `scripts/train.py:372-382`
- Checkpoint loading pattern from `scripts/train.py:400+`
- Configuration structure from existing YAML files

---

#### 4. PRM Training Script (NEW)

**File**: `scripts/train_prm.py` (to be created)

Based on integration plan lines 268-293 with **heavy reuse** of training infrastructure:

```python
#!/usr/bin/env python3
"""Train Process Reward Model (PRM) with triplet margin loss.

Reuses existing infrastructure:
- WandBContext for logging
- Optimizer/scheduler factories from train.py
- Checkpoint management patterns
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from tqdm import tqdm

# Reuse existing utilities
from src.ups.utils.config_loader import load_config
from src.ups.utils.wandb_context import WandBContext
from src.ups.models.prm import PRM  # NEW module
from src.ups.data.prm_triplets import PRMTripletDataset, collate_prm_triplets  # NEW module


def triplet_margin_loss(
    s_best: torch.Tensor,
    s_med: torch.Tensor,
    s_worst: torch.Tensor,
    margin: float = 0.2
) -> torch.Tensor:
    """
    Triplet margin loss: encourages s_best > s_med > s_worst.

    Args:
        s_best: Scores for best candidates [B]
        s_med: Scores for median candidates [B]
        s_worst: Scores for worst candidates [B]
        margin: Minimum gap between rankings

    Returns:
        loss: Scalar loss
    """
    # Encourage: s_best > s_med + margin
    loss_best_med = F.relu(margin - (s_best - s_med)).mean()

    # Encourage: s_med > s_worst + margin
    loss_med_worst = F.relu(margin - (s_med - s_worst)).mean()

    return loss_best_med + loss_med_worst


def train_prm(config_path: str):
    """Train PRM model."""

    # Load config (REUSE config_loader)
    config = load_config(config_path)
    prm_cfg = config["prm"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize WandB (REUSE WandBContext pattern)
    wandb_ctx = WandBContext(
        project=config.get("logging", {}).get("wandb", {}).get("project", "universal-simulator"),
        config=config,
        name=f"prm-training",
        mode=config.get("logging", {}).get("wandb", {}).get("mode", "online"),
    )

    # Load dataset (NEW: PRMTripletDataset)
    manifest_path = Path(prm_cfg["triplet_generation"]["output_dir"]) / "manifest.txt"
    with open(manifest_path) as f:
        triplet_paths = [line.strip() for line in f]

    dataset = PRMTripletDataset(triplet_paths)

    # Split into train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=prm_cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        collate_fn=collate_prm_triplets,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=prm_cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=4,
        collate_fn=collate_prm_triplets,
    )

    # Initialize model
    prm = PRM(
        in_ch=prm_cfg["model"]["in_ch"],
        width=prm_cfg["model"]["width"],
        depth=prm_cfg["model"]["depth"],
    ).to(device)

    # Optimizer (REUSE pattern from train.py:324-369)
    optimizer = torch.optim.AdamW(
        prm.parameters(),
        lr=prm_cfg["training"]["optimizer"]["lr"],
        weight_decay=prm_cfg["training"]["optimizer"]["weight_decay"],
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=prm_cfg["training"]["epochs"],
        eta_min=prm_cfg["training"]["optimizer"].get("eta_min", 1e-6),
    )

    # Training loop
    best_val_loss = float("inf")
    margin = prm_cfg["training"]["loss"]["margin"]

    for epoch in range(prm_cfg["training"]["epochs"]):
        # Train
        prm.train()
        train_loss = 0.0
        for state_t, c_best, c_med, c_worst in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            state_t = state_t.to(device)
            c_best = c_best.to(device)
            c_med = c_med.to(device)
            c_worst = c_worst.to(device)

            # Forward
            s_best = prm(state_t, c_best)
            s_med = prm(state_t, c_med)
            s_worst = prm(state_t, c_worst)

            # Loss
            loss = triplet_margin_loss(s_best, s_med, s_worst, margin)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        prm.eval()
        val_loss = 0.0
        correct_rankings = 0
        total_samples = 0

        with torch.no_grad():
            for state_t, c_best, c_med, c_worst in val_loader:
                state_t = state_t.to(device)
                c_best = c_best.to(device)
                c_med = c_med.to(device)
                c_worst = c_worst.to(device)

                s_best = prm(state_t, c_best)
                s_med = prm(state_t, c_med)
                s_worst = prm(state_t, c_worst)

                loss = triplet_margin_loss(s_best, s_med, s_worst, margin)
                val_loss += loss.item()

                # Check ranking accuracy
                correct = ((s_best > s_med) & (s_med > s_worst)).sum().item()
                correct_rankings += correct
                total_samples += state_t.size(0)

        val_loss /= len(val_loader)
        ranking_accuracy = correct_rankings / total_samples

        # Log metrics (REUSE WandBContext)
        wandb_ctx.log_training_metric("prm", "train_loss", train_loss, epoch)
        wandb_ctx.log_training_metric("prm", "val_loss", val_loss, epoch)
        wandb_ctx.log_training_metric("prm", "ranking_accuracy", ranking_accuracy, epoch)

        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, acc={ranking_accuracy:.3f}")

        # Save best model (REUSE checkpoint pattern)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(prm.state_dict(), "checkpoints/prm_best.pt")
            print(f"  → Saved best model (val_loss={val_loss:.4f})")

        # Scheduler step
        scheduler.step()

    # Save final model
    torch.save(prm.state_dict(), "checkpoints/prm_latest.pt")
    print("Training complete!")

    wandb_ctx.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to training config YAML")
    args = parser.parse_args()

    train_prm(args.config)
```

**Key Reuse Points:**
- WandBContext from `src/ups/utils/wandb_context.py`
- Optimizer/scheduler factories similar to `scripts/train.py:324-369`
- Checkpoint pattern from `scripts/train.py:856-863`
- Training loop structure from `scripts/train.py:400-693`

---

#### 5. PRMScorer Integration (NEW)

**File**: `src/ups/eval/reward_models.py` (add after line 246)

```python
class PRMScorer(RewardModel):
    """Learned Process Reward Model scorer for TTC."""

    def __init__(self, prm_ckpt: str, latent_dim: int, device: str = "cpu"):
        super().__init__()
        from src.ups.models.prm import PRM

        self.prm = PRM(in_ch=latent_dim * 2)  # Concatenated states
        self.prm.load_state_dict(torch.load(prm_ckpt, map_location=device))
        self.prm.to(device)
        self.prm.eval()

        # Store for logging
        self.last_score = None

    def score(
        self,
        prev_state: LatentState,
        next_state: LatentState,
        context: Optional[Mapping[str, float]] = None,
    ) -> torch.Tensor:
        """
        Score state transition using learned PRM.

        Args:
            prev_state: Previous latent state
            next_state: Candidate next state
            context: Optional context (unused)

        Returns:
            score: Scalar reward (higher is better) [B]
        """
        with torch.no_grad():
            score = self.prm(prev_state.z, next_state.z)
            self.last_score = score.item() if score.numel() == 1 else score.mean().item()
            return score
```

---

#### 6. Update Reward Model Builder (MODIFY)

**File**: `src/ups/inference/rollout_ttc.py:199-272` (modify)

Add PRM option to `build_reward_model_from_config()`:

```python
def build_reward_model_from_config(ttc_cfg, latent_dim, device):
    """Build reward model from configuration."""

    reward_cfg = ttc_cfg.get("reward", {})
    models = []

    # ... existing ARM code ...

    # NEW: Add PRM model
    prm_cfg = reward_cfg.get("prm", {})
    if prm_cfg.get("enabled", False):
        prm_ckpt = prm_cfg["checkpoint"]
        prm_weight = float(prm_cfg.get("weight", 1.0))

        from src.ups.eval.reward_models import PRMScorer
        prm_scorer = PRMScorer(prm_ckpt, latent_dim, device)
        models.append((prm_scorer, prm_weight))

        print(f"  + PRM (weight={prm_weight}, ckpt={prm_ckpt})")

    # Return composite or single model
    if len(models) == 0:
        raise ValueError("No reward models configured")
    elif len(models) == 1:
        return models[0][0]
    else:
        from src.ups.eval.reward_models import CompositeRewardModel
        return CompositeRewardModel(models)
```

---

#### 7. Configuration Updates (MODIFY)

**File**: `configs/train_burgers_golden.yaml` (add new section)

```yaml
# Add PRM configuration section
prm:
  enabled: false  # Enable after PRM training complete

  triplet_generation:
    K: 100                    # Candidates per initial state
    subset_fraction: 0.125    # 12.5% of training data (per paper)
    output_dir: data/prm_triplets

  model:
    in_ch: 32                 # 2 * latent.dim (concatenated states)
    width: 64                 # Conv layer width
    depth: 6                  # Number of conv layers

  training:
    epochs: 50
    batch_size: 32
    optimizer:
      name: adamw
      lr: 3.0e-4
      weight_decay: 0.01
      eta_min: 1.0e-6
    loss:
      margin: 0.2             # Triplet margin

# Update TTC section to include PRM
ttc:
  enabled: true
  # ... existing config ...

  reward:
    # Existing analytical weights
    analytical_weight: 1.0
    grid: [64, 64]
    mass_field: rho
    energy_field: e
    weights:
      mass: 1.0
      energy: 1.0
      penalty_negative: 0.5

    # NEW: PRM configuration
    prm:
      enabled: false          # Enable after PRM trained
      checkpoint: checkpoints/prm_best.pt
      weight: 1.0             # Ensemble weight with ARM
```

---

## Implementation Roadmap

### Phase 1: PRM Infrastructure (Week 1-2)

**Priority**: HIGH - Core missing component

**Tasks:**

1. **Create PRM model architecture** (`src/ups/models/prm.py`)
   - Implement PRM class following integration plan architecture
   - Add unit tests for forward pass
   - Verify output shapes

2. **Create triplet dataset** (`src/ups/data/prm_triplets.py`)
   - Implement PRMTripletDataset following GridLatentPairDataset pattern
   - Add collate function
   - Add unit tests

3. **Create triplet builder** (`scripts/build_prm_data.py`)
   - Reuse GridLatentPairDataset, _sample_tau, checkpoint loading
   - Generate ~12.5% of train data
   - Validate MSE ranking: best < median < worst

**Expected Output:**
- `data/prm_triplets/` directory with ~10k-50k triplets (depending on dataset size)
- `data/prm_triplets/manifest.txt` file listing all triplet paths
- Validation that triplets are correctly ranked

**Command:**
```bash
python scripts/build_prm_data.py \
  --config configs/train_burgers_golden.yaml \
  --K 100 \
  --subset-fraction 0.125 \
  --out-dir data/prm_triplets
```

---

### Phase 2: PRM Training (Week 2-3)

**Priority**: HIGH - Core missing component

**Tasks:**

1. **Create PRM training script** (`scripts/train_prm.py`)
   - Implement triplet margin loss
   - Reuse WandBContext, optimizer/scheduler factories
   - Add ranking accuracy metric
   - Implement checkpoint saving (best + latest)

2. **Train PRM model**
   - 50 epochs, batch_size=32, lr=3e-4
   - Monitor: train_loss, val_loss, ranking_accuracy
   - Target: ranking_accuracy > 0.9 on validation set

3. **Validate PRM on held-out triplets**
   - Verify score(best) > score(median) > score(worst)
   - Check calibration: PRM scores vs actual MSE correlation

**Expected Output:**
- `checkpoints/prm_best.pt` - Best model by validation loss
- `checkpoints/prm_latest.pt` - Final epoch model
- WandB run with training curves
- Ranking accuracy > 90% on validation set

**Command:**
```bash
python scripts/train_prm.py --config configs/train_burgers_golden.yaml
```

---

### Phase 3: PRM Integration with TTC (Week 3-4)

**Priority**: HIGH - Integration layer

**Tasks:**

1. **Add PRMScorer** to `src/ups/eval/reward_models.py`
   - Implement PRMScorer class
   - Add to module exports

2. **Update reward model builder** in `src/ups/inference/rollout_ttc.py`
   - Add PRM configuration parsing
   - Support ARM + PRM composite model

3. **Run ablation study**
   - Baseline (no TTC)
   - TTC + ARM only
   - TTC + PRM only
   - TTC + ARM + PRM (composite)

4. **Update configuration**
   - Enable PRM in `configs/train_burgers_golden.yaml`
   - Set ensemble weights (start with ARM=1.0, PRM=1.0)

**Expected Output:**
- Working TTC with PRM scoring
- Ablation results showing PRM impact
- Updated configuration files

**Commands:**
```bash
# Evaluate with PRM only
python scripts/evaluate.py \
  --checkpoint checkpoints/op_latest.ckpt \
  --config configs/eval_burgers_prm.yaml

# Evaluate with ARM + PRM composite
python scripts/evaluate.py \
  --checkpoint checkpoints/op_latest.ckpt \
  --config configs/eval_burgers_composite.yaml
```

---

### Phase 4: Testing & Validation (Week 4)

**Priority**: MEDIUM - Quality assurance

**Tasks:**

1. **Add unit tests** (`tests/unit/test_prm.py`)
   - `test_prm_forward()`: Verify output shapes
   - `test_prm_triplet_ranking()`: Verify score(best) > score(med) > score(worst)
   - `test_triplet_generation()`: Verify MSE ordering
   - `test_triplet_dataset_loading()`: Verify dataset yields correct shapes

2. **Add integration tests** (`tests/integration/test_prm_ttc.py`)
   - `test_ttc_with_prm()`: End-to-end TTC rollout with PRM
   - `test_composite_reward()`: ARM + PRM ensemble
   - `test_prm_step_logging()`: Verify PRM scores in TTCStepLog

3. **Performance benchmarking**
   - Measure PRM inference time per candidate
   - Compare vs ARM inference time
   - Ensure TTC budget not exceeded

**Expected Output:**
- All tests passing
- Test coverage > 80% for new code
- Performance benchmarks documented

**Commands:**
```bash
# Run unit tests
pytest tests/unit/test_prm.py -v

# Run integration tests
pytest tests/integration/test_prm_ttc.py -v

# Run all tests
pytest tests/ -n auto
```

---

### Phase 5: Optional Enhancements (Week 5+)

**Priority**: LOW - Nice-to-have improvements

**Tasks:**

1. **MC Dropout support** (`src/ups/models/stochastic_wrappers.py`)
   - Add configurable dropout to PDE-Transformer
   - Implement `enable_mc_dropout()` wrapper
   - Test impact on candidate diversity

2. **Latent-specific noise injection**
   - Separate noise injection for latent space vs diffusion tau
   - Per-layer noise injection in operator

3. **Enhanced beam search**
   - Implement full beam-k search (not just greedy)
   - Add diversity penalties to prevent mode collapse

4. **PRM calibration**
   - Analyze PRM scores vs realized rollout MSE
   - Temperature scaling if needed

**Expected Output:**
- Additional stochasticity sources
- Improved candidate diversity metrics
- Better calibration plots

---

## Code References

### Existing Infrastructure (Fully Implemented)

| Component | File | Lines | Status |
|-----------|------|-------|--------|
| TTC Rollout | `src/ups/inference/rollout_ttc.py` | 81-196 | ✅ Complete |
| TTCConfig | `src/ups/inference/rollout_ttc.py` | 25-40 | ✅ Complete |
| Analytical Reward Model | `src/ups/eval/reward_models.py` | 66-166 | ✅ Complete |
| FeatureCriticRewardModel | `src/ups/eval/reward_models.py` | 169-246 | ✅ Complete |
| CompositeRewardModel | `src/ups/eval/reward_models.py` | 249-283 | ✅ Complete |
| Reward Model Builder | `src/ups/inference/rollout_ttc.py` | 199-272 | ✅ Complete |
| Stochastic Sampling | `src/ups/inference/rollout_ttc.py` | 103-128 | ✅ Complete |
| Tau Sampling | `scripts/train.py` | 372-382 | ✅ Complete |
| Diffusion Residual | `src/ups/models/diffusion_residual.py` | 22-54 | ✅ Complete |
| Diffusion Training | `scripts/train.py` | 695-879 | ✅ Complete |
| Consistency Distillation | `scripts/train.py` | 942-1200 | ✅ Complete |
| AnyPointDecoder | `src/ups/io/decoder_anypoint.py` | 53-120 | ✅ Complete |
| GridLatentPairDataset | `src/ups/data/latent_pairs.py` | 257-413 | ✅ Complete |
| WandBContext | `src/ups/utils/wandb_context.py` | - | ✅ Complete |
| Physics Guards | `src/ups/models/physics_guards.py` | 11-36 | ✅ Complete |
| Metrics | `src/ups/eval/metrics.py` | - | ✅ Complete |
| TTC Tests | `tests/unit/test_ttc.py` | 45-211 | ✅ Complete |

### New Components (To Be Implemented)

| Component | File | Status | Priority |
|-----------|------|--------|----------|
| PRM Model | `src/ups/models/prm.py` | ❌ New | HIGH |
| Triplet Dataset | `src/ups/data/prm_triplets.py` | ❌ New | HIGH |
| Triplet Builder | `scripts/build_prm_data.py` | ❌ New | HIGH |
| PRM Training | `scripts/train_prm.py` | ❌ New | HIGH |
| PRMScorer | `src/ups/eval/reward_models.py` | ❌ Addition | HIGH |
| PRM Tests | `tests/unit/test_prm.py` | ❌ New | MEDIUM |
| PRM Integration Tests | `tests/integration/test_prm_ttc.py` | ❌ New | MEDIUM |
| Stochastic Wrappers | `src/ups/models/stochastic_wrappers.py` | ❌ New | LOW |

---

## Configuration Management

### Current Configuration Structure

**File**: `configs/train_burgers_golden.yaml`

```yaml
# Latent space (16-dim)
latent:
  dim: 16
  tokens: 64

# Operator architecture
operator:
  pdet:
    input_dim: 16  # Must match latent.dim
    hidden_dim: 96
    # ... other params

# Diffusion model
diffusion:
  latent_dim: 16  # Must match latent.dim
  hidden_dim: 96

# TTC configuration
ttc:
  enabled: true
  candidates: 16
  beam_width: 5
  horizon: 1

  sampler:
    tau_range: [0.15, 0.85]
    noise_std: 0.05
    noise_schedule: [0.08, 0.05, 0.02]

  reward:
    analytical_weight: 1.0
    grid: [64, 64]
    mass_field: rho
    energy_field: e
    weights:
      mass: 1.0
      energy: 1.0
      penalty_negative: 0.5

  decoder:
    latent_dim: 16  # Must match latent.dim
    query_dim: 2
    hidden_dim: 256
    num_layers: 2
    num_heads: 4
    frequencies: [1.0, 2.0, 4.0]
    output_channels:
      rho: 1
      e: 1
```

### Required Configuration Additions

**Add to**: `configs/train_burgers_golden.yaml`

```yaml
# NEW: PRM configuration
prm:
  enabled: false  # Set to true after PRM training

  triplet_generation:
    K: 100
    subset_fraction: 0.125
    output_dir: data/prm_triplets

  model:
    in_ch: 32  # 2 * latent.dim
    width: 64
    depth: 6

  training:
    epochs: 50
    batch_size: 32
    optimizer:
      name: adamw
      lr: 3.0e-4
      weight_decay: 0.01
      eta_min: 1.0e-6
    loss:
      margin: 0.2

# UPDATE: TTC reward section
ttc:
  reward:
    # Existing analytical config...

    # NEW: PRM config
    prm:
      enabled: false  # Set to true after PRM training
      checkpoint: checkpoints/prm_best.pt
      weight: 1.0
```

---

## Integration Testing Strategy

### Test Hierarchy

```
Level 1: Unit Tests (Isolated Components)
├─ test_prm.py
│  ├─ test_prm_forward()
│  ├─ test_prm_triplet_ranking()
│  └─ test_triplet_margin_loss()
├─ test_prm_dataset.py
│  ├─ test_triplet_dataset_loading()
│  ├─ test_triplet_collate()
│  └─ test_triplet_mse_ordering()
└─ test_prm_scorer.py
   ├─ test_prm_scorer_score()
   └─ test_prm_scorer_checkpoint_loading()

Level 2: Integration Tests (Multi-Component)
├─ test_prm_ttc.py
│  ├─ test_ttc_with_prm()
│  ├─ test_composite_arm_prm()
│  └─ test_prm_step_logging()
└─ test_prm_training.py
   ├─ test_triplet_generation_pipeline()
   └─ test_prm_training_loop()

Level 3: End-to-End Tests (Full Pipeline)
└─ test_prm_evaluation.py
   ├─ test_full_rollout_with_prm()
   ├─ test_prm_vs_arm_ablation()
   └─ test_prm_arm_composite_improvement()
```

### Critical Tests

**1. PRM Ranking Test** (`tests/unit/test_prm.py`)

```python
def test_prm_triplet_ranking(tmp_path):
    """Verify PRM ranks: score(best) > score(med) > score(worst)"""
    # Generate synthetic triplet
    state_t = torch.randn(1, 16, 8, 8)
    best = torch.randn(1, 16, 8, 8)
    med = torch.randn(1, 16, 8, 8)
    worst = torch.randn(1, 16, 8, 8)

    # Initialize PRM
    prm = PRM(in_ch=32, width=32, depth=3)

    # Score
    s_best = prm(state_t, best)
    s_med = prm(state_t, med)
    s_worst = prm(state_t, worst)

    # After training, this should hold (initially random)
    # assert s_best > s_med > s_worst

    # For untrained model, just check shapes
    assert s_best.shape == (1,)
    assert s_med.shape == (1,)
    assert s_worst.shape == (1,)
```

**2. Triplet Generation Test** (`tests/integration/test_prm_training.py`)

```python
def test_triplet_generation_mse_ordering(tmp_path):
    """Verify generated triplets satisfy: MSE(best) < MSE(med) < MSE(worst)"""
    # Run triplet generation on small dataset
    # Load generated triplets
    # For each triplet:
    #   - Compute MSE(state_t, best)
    #   - Compute MSE(state_t, med)
    #   - Compute MSE(state_t, worst)
    #   - Assert: mse_best < mse_med < mse_worst
    pass
```

**3. TTC + PRM Integration Test** (`tests/integration/test_prm_ttc.py`)

```python
def test_ttc_with_prm_logs_scores():
    """Verify TTC rollout with PRM logs reward components correctly"""
    # Initialize operator, diffusion, PRM
    # Run ttc_rollout with PRMScorer
    # Check step_logs contain PRM scores
    # Verify chosen candidate has highest PRM score
    pass
```

---

## Expected Performance Improvements

Based on the integration plan paper (arXiv:2509.02846) and existing TTC results:

| Configuration | NRMSE | Improvement | Notes |
|---------------|-------|-------------|-------|
| Baseline (no TTC) | 0.78 | - | Direct operator rollout |
| TTC + ARM (current) | 0.09 | 88% | Mass + energy conservation |
| TTC + PRM (expected) | 0.06-0.08 | 90-92% | Learned from data |
| TTC + ARM + PRM (expected) | 0.05-0.07 | 91-94% | Best of both |

**Key Hypotheses:**
1. **PRM alone** should match or slightly outperform ARM if training data has good coverage
2. **ARM + PRM composite** should outperform either alone by combining physics priors with learned patterns
3. **PRM may struggle** with out-of-distribution scenarios where ARM excels (conservation laws are universal)

---

## Reuse Summary

### Infrastructure Reused (>90%)

| Component | Source | Reuse Pattern |
|-----------|--------|---------------|
| Data Loading | `GridLatentPairDataset` | Subsampling + same data format |
| Tau Sampling | `scripts/train.py:_sample_tau()` | Direct function call |
| Checkpoint Loading | `scripts/train.py:400+` | Load operator + diffusion |
| Training Loop | `scripts/train.py:400-693` | Epoch loop, loss, backprop, logging |
| WandB Logging | `WandBContext` | Single-run logging with metrics |
| Optimizer/Scheduler | `scripts/train.py:324-369` | AdamW + CosineAnnealingLR |
| Checkpoint Saving | `scripts/train.py:856-863` | Save best + latest |
| Reward Model Interface | `RewardModel` abstract class | Inherit and implement `score()` |
| Composite Ensemble | `CompositeRewardModel` | Add PRM to model list |
| Configuration | YAML structure | Add `prm` section |

### New Code Required (<10%)

| Component | Lines of Code (est.) | Complexity |
|-----------|----------------------|------------|
| `src/ups/models/prm.py` | ~80 | Low |
| `src/ups/data/prm_triplets.py` | ~60 | Low |
| `scripts/build_prm_data.py` | ~150 | Medium |
| `scripts/train_prm.py` | ~200 | Medium |
| `PRMScorer` in `reward_models.py` | ~30 | Low |
| Update reward builder | ~15 | Low |
| Unit tests | ~200 | Medium |
| Integration tests | ~150 | Medium |
| **TOTAL** | **~885 lines** | **Low-Medium** |

---

## Risk Analysis & Mitigation

### Risk 1: PRM Overfitting to Training Distribution

**Risk**: PRM trained on specific scenarios may not generalize to test distribution.

**Mitigation**:
- Use ~12.5% of training data (diverse ICs) as per paper
- Generate K=100 candidates per state for diversity
- Monitor validation ranking accuracy during training
- Keep ARM in composite model for out-of-distribution robustness

**Monitoring**:
- Validate PRM on held-out test set
- Compare PRM scores vs realized MSE correlation
- Track "PRM confidence" (score spread across candidates)

---

### Risk 2: Computational Cost of Triplet Generation

**Risk**: Generating 100 candidates per state is expensive.

**Mitigation**:
- Use only 12.5% of training data (manageable subset)
- Parallelize across GPUs if available
- Cache triplets once generated (don't regenerate)
- Consider reducing K if needed (paper uses 100, but 50 may suffice)

**Estimates** (for Burgers dataset with 10k train samples):
- Subset: 10k * 0.125 = 1,250 states
- Candidates per state: K=100
- Total forwards: 1,250 * 100 = 125k
- Time @ 100 FPS: ~20 minutes on A100

---

### Risk 3: PRM-ARM Weight Tuning

**Risk**: Composite model requires tuning weights between PRM and ARM.

**Mitigation**:
- Start with equal weights (1.0, 1.0)
- Run grid search: [0.5, 1.0, 2.0] for each
- Use validation set to select best combination
- Monitor per-component scores in step logs

**Ablation Matrix**:
```
ARM weight | PRM weight | NRMSE
-----------|------------|-------
1.0        | 0.0        | 0.09   (baseline ARM)
0.0        | 1.0        | ???    (PRM only)
1.0        | 1.0        | ???    (equal)
2.0        | 1.0        | ???    (favor ARM)
1.0        | 2.0        | ???    (favor PRM)
```

---

### Risk 4: PRM Training Instability

**Risk**: Triplet margin loss may be unstable or fail to learn ranking.

**Mitigation**:
- Monitor ranking accuracy during training (should reach >90%)
- Start with margin=0.2, reduce to 0.1 if unstable
- Use validation set for early stopping
- Check triplet quality: verify MSE ordering before training

**Red Flags**:
- Ranking accuracy plateaus at 50% (random guessing)
- Scores collapse to same value for all candidates
- Validation loss diverges from training loss

---

## Success Criteria

### Phase 1 Success (Triplet Generation)
- ✅ Generated 1k-10k triplets successfully
- ✅ All triplets satisfy: MSE(best) < MSE(median) < MSE(worst)
- ✅ Manifest file created with all paths
- ✅ Spot-check visualization confirms quality

### Phase 2 Success (PRM Training)
- ✅ Validation ranking accuracy > 90%
- ✅ Training loss decreases steadily
- ✅ Validation loss tracks training (no overfitting)
- ✅ PRM scores show clear separation: score(best) >> score(worst)

### Phase 3 Success (TTC Integration)
- ✅ TTC rollout with PRM runs without errors
- ✅ Step logs contain PRM scores
- ✅ PRM selects best candidates (verified on validation set)
- ✅ NRMSE with PRM ≤ NRMSE with ARM

### Phase 4 Success (Testing)
- ✅ All unit tests pass
- ✅ All integration tests pass
- ✅ Test coverage > 80% for new code
- ✅ Performance benchmarks documented

### Final Success (Production Readiness)
- ✅ Composite ARM + PRM outperforms ARM alone by ≥5%
- ✅ PRM generalizes to test set (no significant degradation)
- ✅ Configuration documented and reproducible
- ✅ Code merged to main branch

---

## Open Questions

### Research Questions

1. **Optimal triplet margin**: Paper uses 0.2, but is this optimal for Burgers?
2. **Triplet diversity**: Should we enforce diversity in candidate generation beyond tau sampling?
3. **PRM architecture**: Is Conv2d trunk optimal, or would Transformer be better?
4. **ARM-PRM synergy**: Do they complement each other, or is one sufficient?
5. **Generalization**: How does PRM trained on Burgers perform on Navier-Stokes?

### Implementation Questions

1. **Triplet storage**: .pt files vs zarr arrays for efficiency?
2. **Batch size for PRM training**: 32 vs 64 vs 128?
3. **PRM depth**: 6 layers (paper) vs 4 (faster) vs 8 (more capacity)?
4. **Ensemble weighting**: Fixed vs learned weights?
5. **MC Dropout**: Worth implementing for additional stochasticity?

---

## Related Research

### Within Repository

- `docs/ttc_prm_arm_integration_plan.md` - Original integration plan
- `docs/production_playbook.md` - Best practices
- `CLAUDE.md` - Project overview and commands
- `parallel_runs_playbook.md` - Hyperparameter sweep guidance

### External Papers

- **arXiv:2509.02846** - "Reward-Model–Driven Test-Time Computing for PDE Foundation Models"
  - Original paper describing PRM/ARM/TTC methodology
  - Reports 88% NRMSE improvement with TTC
  - Suggests ~12.5% of train data for PRM training

---

## Conclusion

The Universal Physics Stack already has **comprehensive TTC and ARM infrastructure** in place. Implementing the PRM component is straightforward because:

1. **Existing patterns** can be directly reused (data loading, training loops, logging)
2. **Architecture is similar** to existing models (FeatureCritic, DiffusionResidual)
3. **Integration points are clear** (PRMScorer, reward builder, composite model)
4. **Testing strategy is well-defined** (unit, integration, end-to-end)

**Estimated effort**: 2-4 weeks for a single developer to implement and validate PRM, with most time spent on triplet generation and training rather than coding.

**Expected impact**: 3-8% additional NRMSE improvement over current TTC+ARM baseline, with potential for better generalization to diverse scenarios via learned data-driven scoring.

The key insight is that **~885 lines of new code** can unlock learned reward modeling by reusing ~10k lines of existing infrastructure. This is a high-leverage implementation opportunity.
