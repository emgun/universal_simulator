# Muon-in-NPT: Architectural Implementation Guide (for a Coding Agent)

This is a **hands-on, implementation-focused** plan to integrate the **Muon optimizer** into a **Neural Physics Transformer (NPT)** codebase. It assumes PyTorch, supports **hybrid Muon+AdamW** parameter groups, works with **FSDP/DDP**, and includes **configs, code stubs, tests, and checklists**.

---

## 0) Scope & Assumptions

- **Goal:** Swap in Muon (for matrix params) while keeping AdamW for 1D/sparse params, preserving LR schedules & weight decay.
- **Targets:** Grid/mesh/particle NPTs (Encoder → Processor/Transformer → Decoder), mixed precision, long rollouts.
- **Defaults (good starting points):**
  - `lr: 5e-4`, `weight_decay: 1e-2`, `beta: 0.9` (Muon's Nesterov momentum), `warmup_steps: 2000`, cosine decay to 10%.
  - **Muon scaling**: multiply orthogonalized update by ~`0.2` to match AdamW RMS (keep same LR schedule).
  - **NS iters**: 5 (Newton–Schulz) if you implement Muon yourself. Prefer a proven library implementation.

---

## 1) Repository Layout (suggested)

```
npt/
  config/
    default.yaml
  npt/
    models/
      encoder.py
      processor_transformer.py
      decoder.py
      npt_model.py
    optim/
      param_groups.py
      build_optim.py
      schedulers.py
    train/
      train_loop.py
      fsdp_setup.py
      amp_utils.py
    eval/
      rollout.py
      metrics.py
    utils/
      logging.py
      checkpoint.py
      seed.py
  scripts/
    train.py
    eval.py
  tests/
    test_param_groups.py
    test_training_step.py
    test_rollout_metrics.py
```

---

## 2) Config Schema (YAML)

```yaml
# config/default.yaml
seed: 123
device: cuda

data:
  train_path: /data/npt/train
  val_path: /data/npt/val
  batch_size: 64
  num_workers: 8

model:
  dim: 512
  n_layers: 16
  n_heads: 8
  ff_mult: 4
  # physics inputs/outputs defined per task
  in_channels: 8
  out_channels: 8

optim:
  use_muon: true                  # turn on hybrid Muon+AdamW
  muon_scale: 0.2                 # scale orthogonalized update
  muon_beta: 0.9                  # Nesterov momentum
  lr: 5.0e-4
  weight_decay: 1.0e-2
  betas: [0.9, 0.999]             # for AdamW group
  eps: 1.0e-8
  grad_clip_norm: null            # e.g., 1.0 (often unnecessary with Muon)
  ns_iters: 5                     # if self-implementing Muon

sched:
  type: cosine
  warmup_steps: 2000
  min_lr_ratio: 0.1

train:
  max_steps: 200000
  log_every: 50
  ckpt_every: 2000
  amp_dtype: bfloat16             # or float16 if your GPUs lack bf16
  fsdp: false
  fsdp_wrap_policy: transformer   # or full
  ddp: true
```

---

## 3) Parameter Grouping (Muon for matrices, AdamW for vectors)

```python
# npt/optim/param_groups.py
from __future__ import annotations
from typing import Dict, Iterable, Tuple
import torch
import torch.nn as nn

def is_matrix_param(p: torch.nn.Parameter) -> bool:
    # Heuristic: Muon is designed for 2D weight matrices (e.g., Linear/Conv weights reshaped).
    # Use Muon for params with ndim >= 2 and requires_grad=True.
    return p.requires_grad and p.ndim >= 2

def build_param_groups(model: nn.Module) -> Tuple[Iterable[nn.Parameter], Iterable[nn.Parameter]]:
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if is_matrix_param(p):
            muon_params.append(p)
        else:
            adamw_params.append(p)
    return muon_params, adamw_params
```

**Notes**
- Embeddings, biases, LayerNorm gains/biases → **AdamW**.
- Dense Linear/Conv weights (2D or reshaped) → **Muon**.
- Very tiny matrices can be left on AdamW if desired (toggle by size threshold).

---

## 4) Optimizer Builder (Hybrid Muon+AdamW)

> Prefer a vetted Muon implementation (e.g., `muon_pytorch.Muon`). If unavailable, keep a placeholder and fall back to AdamW-only.

```python
# npt/optim/build_optim.py
from __future__ import annotations
from typing import Dict, Any, Tuple
import torch
from torch.optim import AdamW
from .param_groups import build_param_groups

try:
    # Example: pip install muon-optimizer (name may differ).
    from muon_pytorch import Muon  # <-- replace with your Muon import
except Exception:
    Muon = None

def build_optimizers(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    muon_params, adamw_params = build_param_groups(model)

    # Shared LR & WD to match AdamW schedule; Muon uses internal scaling ~0.2 to match RMS.
    lr = cfg.optim.lr
    wd = cfg.optim.weight_decay

    optim_groups = []
    if cfg.optim.use_muon and Muon is not None and len(muon_params) > 0:
        muon_opt = Muon(
            muon_params,
            lr=lr,
            weight_decay=wd,     # decoupled WD inside Muon impl
            beta=cfg.optim.muon_beta,
            scale=cfg.optim.muon_scale,
            ns_iters=getattr(cfg.optim, "ns_iters", 5),
        )
        optim_groups.append(muon_opt)

    # Always include AdamW group (may be empty)
    adamw_opt = AdamW(
        adamw_params if len(adamw_params) > 0 else muon_params,
        lr=lr,
        betas=tuple(cfg.optim.betas),
        eps=cfg.optim.eps,
        weight_decay=wd,
    )
    optim_groups.append(adamw_opt)

    # If both exist, wrap into a single "OptimizerList" style helper
    if len(optim_groups) == 1:
        optimizer = optim_groups[0]
    else:
        optimizer = OptimizerGroup(optim_groups)

    # Scheduler
    scheduler = build_scheduler(optimizer, cfg)
    return optimizer, scheduler

class OptimizerGroup(torch.optim.Optimizer):
    """Thin wrapper to step/zero_grad over multiple optimizers as one."""
    def __init__(self, optimizers):
        self.optimizers = optimizers
        # dummy param to satisfy base class
        super().__init__([{'params': []}], {})

    def step(self):
        for opt in self.optimizers:
            opt.step()

    def zero_grad(self, set_to_none: bool = True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {i: opt.state_dict() for i, opt in enumerate(self.optimizers)}

    def load_state_dict(self, state):
        for i, opt in enumerate(self.optimizers):
            opt.load_state_dict(state[i])

def build_scheduler(optimizer, cfg):
    from .schedulers import WarmupCosine
    return WarmupCosine(
        optimizer,
        warmup_steps=cfg.sched.warmup_steps,
        max_steps=cfg.train.max_steps,
        min_lr_ratio=cfg.sched.min_lr_ratio,
    )
```

```python
# npt/optim/schedulers.py
import math
from torch.optim import Optimizer

class WarmupCosine:
    def __init__(self, optimizer: Optimizer, warmup_steps: int, max_steps: int, min_lr_ratio: float = 0.1):
        self.opt = optimizer
        self.warm = warmup_steps
        self.max = max_steps
        self.min_ratio = min_lr_ratio
        self._step = 0
        self._base_lrs = []
        # Support OptimizerGroup
        opts = optimizer.optimizers if hasattr(optimizer, "optimizers") else [optimizer]
        for opt in opts:
            for group in opt.param_groups:
                self._base_lrs.append(group["lr"])

    def step(self):
        self._step += 1
        t = self._step
        for idx, opt in enumerate(self._iter_opt_groups()):
            for j, group in enumerate(opt.param_groups):
                base_lr = self._base_lrs[self._param_group_index(idx, j)]
                if t <= self.warm:
                    lr = base_lr * t / max(1, self.warm)
                else:
                    progress = (t - self.warm) / max(1, self.max - self.warm)
                    cosine = 0.5 * (1 + math.cos(math.pi * progress))
                    lr = base_lr * (self.min_ratio + (1 - self.min_ratio) * cosine)
                group["lr"] = lr

    def _iter_opt_groups(self):
        return self.opt.optimizers if hasattr(self.opt, "optimizers") else [self.opt]

    def _param_group_index(self, opt_idx, group_idx):
        # flatten index (assumes order stable across steps)
        if not hasattr(self, "_offsets"):
            self._offsets = []
            count = 0
            opts = self._iter_opt_groups()
            for o in opts:
                n = len(o.param_groups)
                self._offsets.append(count)
                count += n
        return self._offsets[opt_idx] + group_idx
```

---

## 5) Mixed Precision + (Optional) Grad Clip

```python
# npt/train/amp_utils.py
import torch

def get_autocast_dtype(name: str):
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    return None  # fp32

class AMPContext:
    def __init__(self, dtype_name: str):
        self.dtype = get_autocast_dtype(dtype_name)

    def __enter__(self):
        self.ctx = torch.autocast(device_type="cuda", dtype=self.dtype) if self.dtype else nullcontext()
        return self.ctx.__enter__()

    def __exit__(self, exc_type, exc, tb):
        return self.ctx.__exit__(exc_type, exc, tb)

from contextlib import nullcontext
```

```python
# Optional gradient clipping (often unnecessary with Muon)
def clip_gradients(model, max_norm: float | None):
    if max_norm is None:
        return
    import torch.nn.utils as utils
    utils.clip_grad_norm_(model.parameters(), max_norm)
```

---

## 6) Training Loop (DDP/FSDP-ready)

```python
# npt/train/train_loop.py
from __future__ import annotations
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from npt.optim.build_optim import build_optimizers
from npt.train.amp_utils import AMPContext, clip_gradients
from npt.utils.checkpoint import save_ckpt, load_ckpt_if_any
from npt.utils.logging import Logger

def train(model, dataloader, val_loader, cfg):
    device = torch.device(cfg.device)
    model.to(device)
    logger = Logger()

    optimizer, scheduler = build_optimizers(model, cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.train.amp_dtype == "float16"))

    step = load_ckpt_if_any(model, optimizer)  # optionally resume
    model.train()

    while step < cfg.train.max_steps:
        for batch in dataloader:
            step += 1
            inputs, targets = batch["inputs"].to(device), batch["targets"].to(device)

            with AMPContext(cfg.train.amp_dtype):
                preds = model(inputs)
                loss = loss_fn(preds, targets)  # implement task loss
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            clip_gradients(model, cfg.optim.grad_clip_norm)

            if scaler.is_enabled():
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            if step % cfg.train.log_every == 0:
                logger.log_scalar("train/loss", loss.item(), step)
                # Monitor stability
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e9).item()
                logger.log_scalar("train/grad_global_norm", total_norm, step)

            if step % cfg.train.ckpt_every == 0:
                save_ckpt(step, model, optimizer, tag=f"step{step}")

            if step >= cfg.train.max_steps:
                break

        # periodic eval (rollout over validation set)
        eval_metrics = evaluate(model, val_loader, device)
        for k, v in eval_metrics.items():
            logger.log_scalar(f"val/{k}", v, step)

def loss_fn(preds, targets):
    # Example: combine MSE with physics priors if available
    return torch.nn.functional.mse_loss(preds, targets)

def evaluate(model, val_loader, device):
    model.eval()
    # Compute rel-L2, nRMSE@k, conservation gap, etc.
    # (see metrics section)
    metrics = {"rel_l2": 0.0}
    # ...
    model.train()
    return metrics
```

**FSDP**: use `npt/train/fsdp_setup.py` to wrap model & optimizers if `cfg.train.fsdp=True`. Ensure Muon implementation is **FSDP/ZeRO-safe** (use sharded states).

---

## 7) Physics Metrics & Rollout Harness

```python
# npt/eval/metrics.py
import torch

def rel_l2(pred, tgt, eps=1e-8):
    num = torch.linalg.norm(pred - tgt)
    den = torch.linalg.norm(tgt) + eps
    return (num / den).item()

def nrmse(pred, tgt, eps=1e-8):
    mse = torch.mean((pred - tgt)**2)
    var = torch.var(tgt)
    return torch.sqrt((mse + eps) / (var + eps)).item()

def conservation_gap(pred, tgt, conserved_axes=(1,2,3)):
    # Sum over spatial dims to test mass/energy conservation approximate gap
    return torch.mean(torch.abs(torch.sum(pred, dim=conserved_axes) - torch.sum(tgt, dim=conserved_axes))).item()
```

```python
# npt/eval/rollout.py
@torch.no_grad()
def rollout(model, init_state, steps: int, step_fn):
    """
    step_fn(state) -> next_state predicted by model
    """
    states = [init_state]
    s = init_state
    for _ in range(steps):
        s = step_fn(model, s)
        states.append(s)
    return states
```

---

## 8) Logging & Diagnostics

- Track: `train/loss`, `train/grad_global_norm`, `val/rel_l2`, `val/nrmse`, `val/conservation_gap`.
- Optional: **update spectral norms** (diagnostic). If Muon exposes update RMS, log it to validate ~0.2 scaling matches AdamW RMS.
- If using W&B: name runs `npt-muon-hybrid-{date}`; save configs/ckpts as artifacts.

---

## 9) Tests (minimal but critical)

```python
# tests/test_param_groups.py
import torch, torch.nn as nn
from npt.optim.param_groups import build_param_groups

def test_param_split():
    m = nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8))
    muon_params, adamw_params = build_param_groups(m)
    assert any(p.ndim >= 2 for p in muon_params)
    assert all(p.ndim < 2 for p in adamw_params)
```

```python
# tests/test_training_step.py
import torch, torch.nn as nn
from npt.optim.build_optim import build_optimizers

class Tiny(nn.Module):
    def __init__(self): 
        super().__init__(); self.l = nn.Linear(4, 4); self.b = nn.Parameter(torch.zeros(4))
    def forward(self, x): return self.l(x) + self.b

def test_step(cfg):
    m = Tiny()
    opt, sch = build_optimizers(m, cfg)
    x = torch.randn(2, 4); y = torch.randn(2, 4)
    out = m(x).sum()
    out.backward()
    opt.step(); opt.zero_grad()
    sch.step()
```

```python
# tests/test_rollout_metrics.py
import torch
from npt.eval.metrics import rel_l2, nrmse, conservation_gap

def test_metrics_shapes():
    a = torch.randn(2, 3, 16, 16)
    b = torch.randn(2, 3, 16, 16)
    assert rel_l2(a, b) >= 0.0
    assert nrmse(a, b) >= 0.0
    assert conservation_gap(a, b) >= 0.0
```

---

## 10) FSDP/Distributed Notes

- **Param groups:** Build **before** FSDP wrapping, or use FSDP policies to select params by module type (Linear/Conv → Muon).
- **Optimizer states:** Ensure Muon implementation is ZeRO/FSDP-compatible (sharded momentum buffer). If not, use DDP.
- **Mixed precision:** Prefer **bf16** where available; falls back to fp16 + grad scaler.
- **Large batches:** Increase batch size first (Muon's strength), keep LR unchanged; validate loss doesn’t degrade.

---

## 11) Ablations & Sweeps (what to vary)

- **Batch size:** ×2, ×4; expect **similar or better** loss vs AdamW at same steps.
- **Muon β:** 0.85 → 0.95 (stability-speed trade-off).
- **Muon scale:** 0.15, **0.2**, 0.25 (match AdamW RMS best).
- **Grad clip:** off vs 1.0 norm (should be rarely triggered with Muon).
- **LR schedule:** cosine vs cosine+restart (usually cosine suffices).
- **Where to apply Muon:** FFN only vs Attention+FFN vs all 2D weights.
- **Precision:** bf16 vs fp16; check for any divergence early.

Example **W&B sweep** (pseudo):
```yaml
program: scripts/train.py
method: grid
parameters:
  optim.muon_scale: {values: [0.15, 0.2, 0.25]}
  optim.muon_beta:  {values: [0.85, 0.9, 0.95]}
  data.batch_size:  {values: [64, 128, 256]}
metric:
  name: val/rel_l2
  goal: minimize
```

---

## 12) Command-line Entrypoint

```bash
python scripts/train.py --config config/default.yaml
```

**`scripts/train.py`** (sketch):
```python
import yaml, argparse, torch
from npt.utils.seed import set_seed
from npt.models.npt_model import build_model
from npt.train.train_loop import train
from npt.data.dataloader import make_loaders  # implement per dataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    cfg = SimpleNamespace(**yaml.safe_load(open(args.config)))
    set_seed(cfg.seed)
    model = build_model(cfg.model)
    train_loader, val_loader = make_loaders(cfg.data)
    train(model, train_loader, val_loader, cfg)

if __name__ == "__main__":
    from types import SimpleNamespace
    main()
```

---

## 13) Practical Checklist (coding agent)

- [ ] Split parameters: **2D → Muon**, **1D/sparse → AdamW**.
- [ ] Use **same LR & WD** as AdamW baseline; set `muon_scale≈0.2`.
- [ ] Keep **warmup+cosine** schedule; early stop if plateau.
- [ ] (Optionally) disable grad clipping; re-enable if instability appears.
- [ ] Log **grad_global_norm** and validation **rel_l2**, **nrmse**, **conservation_gap**.
- [ ] Increase **batch size**; verify no loss regression.
- [ ] Ensure **FSDP/DDP** compatibility of Muon (or use DDP).
- [ ] Save & resume **optimizer state** correctly for both groups.
- [ ] Add **unit tests** (param split, single train step, metrics).

---

## 14) Notes on Implementing Muon Yourself (only if needed)

If you cannot import a proven Muon implementation, you **can** prototype a polar-decomposition-based update via Newton–Schulz. However, numerical stability and performance are non-trivial. Prefer a maintained library. If prototyping:
- Form momentum matrix `M` (Nesterov lookahead).
- Compute polar factor `Q` ≈ `M (M^T M)^(-1/2)` via Newton–Schulz (5–7 iterations).
- Update weight with `ΔW = scale * Q`, apply **decoupled weight decay**, and maintain momentum buffer.
- Unit test on small Linear layers; verify update RMS matches AdamW with the same LR.

---

### TL;DR
- **Hybrid Muon+AdamW** with **param groups** is the safest, fastest path.
- Keep **AdamW hyperparams & schedule**; set **Muon scale ≈ 0.2** to match update RMS.
- Expect **faster convergence, higher stability**, and **large-batch wins** without extra tuning.
