# Reward-Model–Driven Test-Time Computing (TTC) Integration Plan
**Target**: Incorporate a reward-model–guided **test-time computing (TTC)** controller into an existing PDE foundation model to improve long-horizon rollouts by sampling multiple next-step candidates and selecting the best one according to a physics/quality reward.

**Paper**: *[Reward-Model–Driven Test-Time Computing for PDE Foundation Models](https://arxiv.org/pdf/2509.02846)* (arXiv:2509.02846)

---

## Table of Contents
1. [Overview](#overview)
2. [Repository Additions](#repository-additions)
3. [Stochasticity at Test Time](#stochasticity-at-test-time)
4. [Analytical Reward Model (ARM)](#analytical-reward-model-arm)
5. [Process Reward Model (PRM)](#process-reward-model-prm)
    - [PRM Model](#prm-model)
    - [Triplet Dataset Builder](#triplet-dataset-builder)
    - [Triplet Generation Job](#triplet-generation-job)
    - [Training the PRM](#training-the-prm)
6. [TTC Greedy Inference Controller](#ttc-greedy-inference-controller)
7. [Metrics & Evaluation](#metrics--evaluation)
8. [Command-Line Flows](#command-line-flows)
9. [Configuration Snippets (Hydra-style)](#configuration-snippets-hydra-style)
10. [Integration Notes for Common Bases](#integration-notes-for-common-bases)
11. [Testing & Guardrails](#testing--guardrails)
12. [Optional: Beam Search](#optional-beam-search)
13. [Logging (e.g., W&B)](#logging-eg-wb)
14. [Pitfalls & Gotchas](#pitfalls--gotchas)
15. [Minimal Interfaces (Copy-Paste Stubs)](#minimal-interfaces-copy-paste-stubs)
16. [Quick Success Checklist](#quick-success-checklist)
17. [References](#references)

---

## 1) Overview
**Goal**: Wrap your existing model `BaseFM` with a TTC controller that at each rollout step:
1. **Generates _B_ candidates** using a **stochastic** variant of your model.
2. **Scores** each candidate with either an **Analytical Reward Model (ARM)** or a **Process Reward Model (PRM)**.
3. **Greedily selects** the top-scoring candidate as the next state and repeats for T steps.

**Key components**:
- **Stochasticity** hooks (MC-dropout, latent noise, input perturbations).
- **ARM** (physics-driven: conservation, BCs, positivity).
- **PRM** (learned scalar scorer trained with triplet loss on best/median/worst predictions).
- **TTC controller** (greedy; optional beam search later).
- **Metrics** (rollout MSE, conservation gaps, “SampleGain”).

---

## 2) Repository Additions
```
your_repo/
  configs/
    ttc.yaml                 # TTC config
    prm.yaml                 # PRM model+train config
    arm.yaml                 # weights for conservation/BC terms
  core/
    base_fm.py               # your existing model
    schedulers.py
    data.py
  stochastic_sampling/
    dropout_wrap.py
    latent_noise.py
    input_perturb.py
  rewards/
    arm.py                   # analytic: mass/momentum/energy/BC/positivity
    prm/
      model.py               # PRM net
      dataset.py             # triplet builder from FM rollouts
      train.py               # trainer (triplet margin loss)
  ttc/
    inference.py             # greedy TTC controller
  metrics/
    pde_conservation.py
    mse.py
    sample_gain.py
  cli/
    build_prm_data.py        # generates PRM triplets (~12.5% of train)
    train_prm.py
    run_ttc.py               # inference entrypoint
```

---

## 3) Stochasticity at Test Time
You need diversity across next-step candidates. Add one or more of:

**A. MC Dropout / Stochastic Depth**
```python
# stochastic_sampling/dropout_wrap.py
import torch, torch.nn as nn

def enable_mc_dropout(model: nn.Module):
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()  # keep dropout active during eval
    return model
```

**B. Latent Noise Injection**
```python
# stochastic_sampling/latent_noise.py
import torch

@torch.no_grad()
def inject_latent_noise(latent, sigma: float):
    return latent + sigma * torch.randn_like(latent)
```

**C. Input Perturbations (small)**
```python
# stochastic_sampling/input_perturb.py
import torch

def perturb_state(x, eps: float):
    return x + eps * torch.randn_like(x)
```

> Keep noise **small** and reproducible (log seeds and scales).

---

## 4) Analytical Reward Model (ARM)
Physics-based score favoring conservation, boundary compliance, and positivity. Convert penalties into a **higher-is-better** score by negation.

```python
# rewards/arm.py
from dataclasses import dataclass
import torch

@dataclass
class ARMWeights:
    w_mass: float = 1.0
    w_momentum: float = 1.0
    w_energy: float = 1.0
    w_bc: float = 0.1
    w_positivity: float = 0.1

def integral(field, dxdy):
    return (field * dxdy).sum(dim=(-2, -1))

def mass_conservation_reward(state_t, cand_tp1, dxdy):
    rho_t  = state_t[:, 0]
    rho_tp = cand_tp1[:, 0]
    return (integral(rho_tp, dxdy) - integral(rho_t, dxdy)).abs()

def momentum_conservation_reward(state_t, cand_tp1, dxdy):
    mom_t  = state_t[:, 1:3].sum(dim=1)   # replace with exact form per state layout
    mom_tp = cand_tp1[:, 1:3].sum(dim=1)
    return (integral(mom_tp, dxdy) - integral(mom_t, dxdy)).abs()

def energy_conservation_reward(state_t, cand_tp1, dxdy):
    E_t  = state_t[:, 3]
    E_tp = cand_tp1[:, 3]
    return (integral(E_tp, dxdy) - integral(E_t, dxdy)).abs()

def bc_violation_reward(cand_tp1, bc_mask):
    # penalty where Dirichlet BCs violated (user: implement projection or compare to BC target)
    return (cand_tp1[bc_mask].abs()).mean()

def compute_pressure(state, gamma: float = 1.4):
    rho = state[:, 0]
    E   = state[:, 3]
    u   = state[:, 1] / (rho + 1e-12)
    v   = state[:, 2] / (rho + 1e-12)
    kinetic = 0.5 * rho * (u*u + v*v)
    p = (gamma - 1.0) * (E - kinetic)
    return p

def positivity_reward(cand_tp1):
    rho = cand_tp1[:, 0]
    p   = compute_pressure(cand_tp1)
    return (rho.clamp_max(0).abs().mean() + p.clamp_max(0).abs().mean())

def arm_score(state_t, cand_tp1, dxdy, bc_mask, w: ARMWeights):
    r = (w.w_mass     * mass_conservation_reward(state_t, cand_tp1, dxdy) +
         w.w_momentum * momentum_conservation_reward(state_t, cand_tp1, dxdy) +
         w.w_energy   * energy_conservation_reward(state_t, cand_tp1, dxdy) +
         w.w_bc       * bc_violation_reward(cand_tp1, bc_mask) +
         w.w_positivity * positivity_reward(cand_tp1))
    return -r  # higher is better
```

> Caution: If pretraining data violates conservation, ARM may over-penalize; consider PRM or tuned weights.

---

## 5) Process Reward Model (PRM)
A **learned** scalar scorer that ingests `(state_t, candidate_{t+1})` and outputs a quality score. Train via **contrastive triplet margin loss** using triplets `(best, median, worst)` per state, where “best/median/worst” are ranked by MSE to ground truth (for PRM training only). The paper reports building triplets from ~**100 candidates per initial condition** and using **~12.5%** of the original training ICs to train PRM.

### PRM Model
```python
# rewards/prm/model.py
import torch, torch.nn as nn, torch.nn.functional as F

class PRM(nn.Module):
    def __init__(self, in_ch, width=64, depth=6):
        super().__init__()
        layers, ch = [], in_ch
        for _ in range(depth):
            layers += [nn.Conv2d(ch, width, 3, padding=1), nn.GELU()]
            ch = width
        self.trunk = nn.Sequential(*layers)
        self.head  = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(width, 1))
    def forward(self, state_t, cand_tp1):  # [B,C,H,W] each
        x = torch.cat([state_t, cand_tp1], dim=1)  # [B, 2*C, H, W]
        return self.head(self.trunk(x)).squeeze(-1)  # [B]
```

### Triplet Dataset Builder
```python
# rewards/prm/dataset.py
import torch
from torch.utils.data import Dataset

class PRMTripletSet(Dataset):
    """
    Yields (state_t, cand_best, cand_med, cand_worst).
    Implement storage as .pt/.zarr and indexing via a manifest.
    """
    def __init__(self, manifest_paths):
        self.items = manifest_paths
    def __getitem__(self, i):
        path = self.items[i]
        # user: load tensors with (state_t, best, med, worst)
        x = torch.load(path)
        return x["state_t"], x["best"], x["med"], x["worst"]
    def __len__(self): return len(self.items)
```

### Triplet Generation Job
```python
# cli/build_prm_data.py (sketch)
import torch, numpy as np, os
from tqdm import tqdm

def build_triplets(fm, data_loader, K=100, out_dir="prm_triplets"):
    os.makedirs(out_dir, exist_ok=True)
    idx = 0
    for batch in tqdm(data_loader):
        # example: (x_t, gt_tp1) pairs batched over IC/time
        x_t, gt_tp1 = batch
        B = x_t.shape[0]
        # Generate K candidates per sample
        cands = []
        for k in range(K):
            # optionally vary seeds / noise
            cand_k = fm.sample_next(x_t, seed=k)
            cands.append(cand_k)
        # Rank by MSE to gt (for PRM training only)
        mse = [((c - gt_tp1)**2).flatten(1).mean(1) for c in cands]  # list of [B]
        mse = torch.stack(mse, dim=0)  # [K, B]
        order = torch.argsort(mse, dim=0)  # ascending
        best = torch.gather(torch.stack(cands, 0), 0, order[0:1].expand(1, *cands[0].shape)).squeeze(0)
        med_idx = K // 2
        med  = torch.gather(torch.stack(cands, 0), 0, order[med_idx:med_idx+1].expand(1, *cands[0].shape)).squeeze(0)
        worst = torch.gather(torch.stack(cands, 0), 0, order[-1:].expand(1, *cands[0].shape)).squeeze(0)
        # Save triplets
        for b in range(B):
            torch.save({
                "state_t": x_t[b].cpu(),
                "best":    best[b].cpu(),
                "med":     med[b].cpu(),
                "worst":   worst[b].cpu(),
            }, os.path.join(out_dir, f"triplet_{idx:08d}.pt"))
            idx += 1
```

### Training the PRM
```python
# rewards/prm/train.py
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from .model import PRM

def triplet_margin_loss(s_best, s_med, s_worst, margin=0.2):
    loss = F.relu(margin - (s_best - s_med)).mean() + \
           F.relu(margin - (s_med  - s_worst)).mean()
    return loss

def train_prm(manifest_paths, in_ch, epochs=50, lr=3e-4, margin=0.2, bs=32):
    ds = PRMTripletSet(manifest_paths)
    dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=4)
    prm = PRM(in_ch=in_ch)
    opt = torch.optim.AdamW(prm.parameters(), lr=lr)
    for ep in range(epochs):
        prm.train()
        for x_t, c_best, c_med, c_worst in dl:
            sb = prm(x_t, c_best)
            sm = prm(x_t, c_med)
            sw = prm(x_t, c_worst)
            loss = triplet_margin_loss(sb, sm, sw, margin)
            opt.zero_grad(); loss.backward(); opt.step()
    return prm
```

---

## 6) TTC Greedy Inference Controller
Vectorize candidate generation and scoring for speed; select argmax score each step.

```python
# ttc/inference.py
from dataclasses import dataclass, field
import torch

@dataclass
class TTCConfig:
    branching_factor: int = 5
    use_prm: bool = True
    stochastic: dict = field(default_factory=lambda: {
        "dropout": True, "latent_sigma": 0.01, "input_eps": 0.0
    })

class TTCGreedy:
    def __init__(self, fm, scorer, cfg: TTCConfig):
        self.fm = fm         # BaseFM with stochastic hooks
        self.scorer = scorer # callable: score(x_t, cand_tp1) -> [B]
        self.cfg = cfg

    @torch.no_grad()
    def rollout(self, x0, T):
        x_t = x0
        traj = [x_t]
        for t in range(T):
            # Vectorized: create B perturbed copies
            B = self.cfg.branching_factor
            x_batch = torch.stack([x_t.clone() for _ in range(B)], dim=0)  # [B, C, H, W]
            # user: apply stochasticity inside fm.sample_next via seed/noise
            cands = self.fm.sample_next(x_batch, seed=None, multi=True)    # [B, C, H, W]
            scores = self.scorer(x_t.expand_as(cands), cands)              # [B]
            best_idx = torch.argmax(scores).item()
            x_t = cands[best_idx]
            traj.append(x_t)
        return torch.stack(traj, dim=1)  # [C, T+1, H, W] or [B0, T+1, C, H, W]
```

---

## 7) Metrics & Evaluation
- **Rollout MSE** at multiple horizons (e.g., 1, 4, 16, 64, …).
- **Conservation diagnostics**: mass/momentum/energy deltas over time vs baseline.
- **SampleGain** (per-IC % improvement):  
  – `100 * (MSE_baseline - MSE_ttc) / (MSE_baseline + 1e-12)`  
  Log mean/median and **% of ICs improved**.

```python
# metrics/sample_gain.py
def sample_gain(mse_baseline_ic, mse_ttc_ic):
    return 100.0 * (mse_baseline_ic - mse_ttc_ic) / (mse_baseline_ic + 1e-12)
```

---

## 8) Command-Line Flows
**A) Build PRM data (~12.5% train ICs; K≈100)**
```bash
python -m cli.build_prm_data \
  data.train_split=train_small \
  builder.K=100 builder.T_prm=4 \
  out=/data/prm_triplets
```

**B) Train PRM**
```bash
python -m cli.train_prm \
  data.root=/data/prm_triplets \
  model.width=64 model.depth=6 \
  optim.lr=3e-4 sched.cosine=True \
  loss.margin=0.2
```

**C) Run TTC inference**
```bash
python -m cli.run_ttc \
  fm.ckpt=... use_prm=true prm.ckpt=... \
  ttc.branching_factor=5 \
  stochastic.dropout=true stochastic.latent_sigma=0.01 \
  eval.rollout_T=128
```

---

## 9) Configuration Snippets (Hydra-style)
```yaml
# configs/ttc.yaml
ttc:
  branching_factor: 5
  stochastic:
    dropout: true
    latent_sigma: 0.01
    input_eps: 0.0

reward:
  type: prm      # or "arm"
  prm_ckpt: /ckpts/prm.pt

arm:
  w_mass: 1.0
  w_momentum: 1.0
  w_energy: 1.0
  w_bc: 0.1
  w_positivity: 0.1
```

---

## 10) Integration Notes for Common Bases
- **FNO/PINO/UNet/PDEformer/UPT**: expose a unified `sample_next(x_t, seed=None, multi=False)` that internally:
  - toggles **MC-dropout/stochastic depth**,
  - injects **latent/input noise**,
  - or varies **sampler seeds/temperature** (for diffusion/flow models).
- **Autoregressive cores**: `x_{t+1} = f(x_t)` single-step operator; TTC stays **outside** model weights.
- **Geometry/BCs**: pass grid/mesh metrics (`dxdy` or quadrature weights) and `bc_mask` to ARM.

---

## 11) Testing & Guardrails
**Unit tests**
- Stochastic hooks change outputs (distributional test).
- ARM terms → ~0 on a synthetic conservative pair.
- PRM sanity: `score(best) > score(med) > score(worst)` on held-out triplets.

**Ablations**
- Baseline vs ARM vs PRM.
- Branching factor sweep (e.g., B∈{3,5,7}).
- Noise ablation (no stochasticity → TTC ≈ baseline).

**Runtime guardrails**
- If PRM score variance collapses or ARM penalizes everything, **fallback** to baseline (B=1) for that step.
- Clip extreme candidates (NaNs/Inf) and resample if needed.

---

## 12) Optional: Beam Search
Replace greedy with **beam-k**: keep top-k partial rollouts; expand each with B candidates; prune by reward (and optional tie-breaker on a cheap proxy like local residual/MSE). Increases compute; start with greedy first.

---

## 13) Logging (e.g., W&B)
- Full run config (B, noise, reward type, weights).
- Rollout MSE@{1,4,16,64,...}.
- Conservation deltas vs baseline.
- **SampleGain** histogram & fraction of ICs improved.
- PRM calibration: bucket PRM scores vs realized MSE.

---

## 14) Pitfalls & Gotchas
- **Data violations**: If training data is non-conservative, ARM may degrade choices; prefer PRM or reweight ARM.
- **Stochasticity source**: Use physically meaningful diversity; too much noise leads to unstable choices.
- **Reward hacking**: PRM can overfit artifacts; mix regimes, add conservation/BC auxiliary penalties, and cross-validate.

---

## 15) Minimal Interfaces (Copy-Paste Stubs)
```python
# core/base_fm.py
import torch, torch.nn as nn
from contextlib import contextmanager

@contextmanager
def temp_seed(seed):
    if seed is None:
        yield
        return
    state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all()
    torch.manual_seed(seed)
    try:
        yield
    finally:
        torch.random.set_rng_state(state)
        torch.cuda.set_rng_state_all(cuda_state)

class BaseFM(nn.Module):
    def forward(self, x_t): ...
    @torch.no_grad()
    def sample_next(self, x_t, seed=None, multi=False):
        # x_t: [C,H,W] or [B,C,H,W] if multi=True
        with temp_seed(seed):
            # user: apply perturbations / enable MC-dropout internally
            return self.forward(x_t)

# rewards/common.py
import torch.nn as nn

class Scorer(nn.Module):
    def score(self, x_t, cand_tp1): ...

class ARMScorer(Scorer):
    def __init__(self, weights, dxdy, bc_mask): ...
    def score(self, x_t, cand_tp1): return arm_score(x_t, cand_tp1, dxdy=self.dxdy, bc_mask=self.bc_mask, w=self.weights)

class PRMScorer(Scorer):
    def __init__(self, prm_ckpt, in_ch):
        self.prm = PRM(in_ch=in_ch); self.prm.load_state_dict(torch.load(prm_ckpt)); self.prm.eval()
    def score(self, x_t, cand_tp1):
        return self.prm(x_t, cand_tp1)
```

---

## 16) Quick Success Checklist
- [ ] Add stochastic hooks; verify output diversity and reproducibility.
- [ ] Implement ARM (mass/momentum/energy/BC/positivity) and tune weights.
- [ ] Generate PRM triplets (K≈100 per IC on ~12.5% data); store efficiently.
- [ ] Train PRM with triplet margin loss; verify ranking on held-out triplets.
- [ ] Implement TTC greedy loop (vectorized B candidates per step).
- [ ] Evaluate baseline vs ARM vs PRM across B∈{3,5,7}; compute **SampleGain**.
- [ ] Log conservation metrics; confirm TTC improves them as B increases (especially with PRM).
- [ ] Add guardrails; optionally prototype beam-k.

---

## 17) References
- **arXiv:2509.02846** — *Reward-Model–Driven Test-Time Computing for PDE Foundation Models*.  
  URL: https://arxiv.org/pdf/2509.02846

---

**License**: CC-BY 4.0 (for this plan text)
