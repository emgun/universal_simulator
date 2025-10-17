# 🚀 Fast-to-SOTA Training & Eval — UPS Playbook

# Fast-to-SOTA Iteration Guide (UPS)

## 1) Setup *(once, then rarely touched)*

### A. Define the scoreboard *(can’t win what you don’t measure)*
- **Core accuracy:** task metric (e.g., `nRMSE@k`, `rel-L2`, `PSNR`, `LPIPS` for fields).
- **Physics:** conservation gap (mass/charge/energy), BC violation, spectral error (energy spectrum), shock edge accuracy (IoU/precision).
- **Reliability:** calibration (ECE), OOD stress tests (geometry, BC, regime).
- **Efficiency:** wall-clock per rollout, J/step, VRAM, params.
- **Stability:** max rollout horizon before correlation < threshold.

### B. Establish tiny but faithful proxies
- **Small Train / Small Eval (10–60 min):** downsampled domains, shorter horizons, fewer PDEs—**must preserve ranking** you’d see at full scale. This is the daily loop.
- **Full Eval (overnight):** run **only** on promising candidates.

### C. Baseline zoo *(champion to beat)*
- Keep **one strong, reproducible baseline per task family** (e.g., latent operator with `PDETransformerBlock` for transients; few‑step diffusion residual; optional steady prior).
- **Freeze** data splits, preprocessing, and seeds.

### D. Guardrails
- Prefer **post-decode projections** for fair comparisons when evaluating decoded fields (use `ups.models.physics_guards.*`, e.g., Hodge for div‑free, positivity clamps).
- **Hard abort** if conservation/BC error spikes > **X×** baseline.

---

## 2) Per-run method *(the fast SOTA loop)*

### 0) Write a hypothesis
> “One of: optimizer tweak / schedule / architecture / data / scale will improve metric **M** by **≥ Δ**.”

### 1) Start small; make it win on proxy
- Train on **Small Train**; **early stop** if validation proxy hasn’t improved by **≥ 1σ** over baseline by **30–40%** of budget.
- If it wins on proxy, **lock the config** and tag it **candidate**.

### 2) Promote to Full Eval
- Run the full suite **once**. No over-iteration—**one shot per candidate**.

### 3) Check gates in order
- **Accuracy:** task metric ↑, physics gaps ≤ baseline, calibration not worse.
- **Efficiency:** time / J / VRAM within budget.  
If passes → **new champion**; else **archive with notes**.

### 4) Log decisions
- 3-line post-mortem: **what changed**, **why it helped/hurt**, **what next**.

### 5) Keep two tracks in parallel
- **Fast track:** optimizer / schedule / regularization (daily).  
- **Slow track:** architecture / data / scale (weekly), only **after** fast track stalls.

---

## 3) Rules of thumb *(defaults that are hard to beat)*

### Optimizer & schedule *(Transformers/diffusion/operators)*
- **AdamW** β = (0.9, 0.95–0.98), **weight decay** 0.01 (operators) / 0.02–0.05 (diffusion backbones).
- **LR scale:** base LR ≈ `1e-3 × (batch_tokens / 256k)`; **clip** at `3e-4–6e-4` if unstable.
- **Warmup** 3–6% steps; **cosine decay**; **EMA** of weights `0.999–0.9999` for eval.
- **Grad clip:** global norm **1.0** (operators) / **0.5–0.8** (diffusion).
- **Norms:** **RMSNorm** (Q/K esp.); **pre-norm** everywhere.
- **Precision:** **bf16**, **flash attention**, **activation checkpointing**.

### Regularization & stabilization
- Skip label/loss smoothing; prefer **EMA + grad clip + stochastic depth 0.1–0.2** for deep stacks.
- **For PDE fields:** add **edge-aware** term (TV/WENO-like) with tiny weight `1e-3–5e-3` **only** if spectral ringing appears.

### Data & batching
- **Mixed-difficulty batches:** temperature-scaled sampling over regimes (Re/Ma/BCs) to avoid “easy-mode” collapse.
- **Curriculum:** start with shorter horizons, then expand; **freeze early layers** when growing horizon for stability.

### Diffusion/flow training
- Prefer **flow matching / rectified flow / consistency** so you can **distill to 1–4 steps** at inference.
- **Guidance:** for physics-residual guidance at sampling, keep **λ ≤ 2** and **re-check calibration (ECE)**.

- Let the **operator** do most work; run a **1–2 step diffusion corrector** every **k** steps (`k=8–32`) or when physics gaps exceed a threshold. UPS exposes this via `ups.inference.rollout_transient` and TTC in `ups.inference.rollout_ttc`.
- **Tune** corrector strength for **calibration**, not just nRMSE.

---

## 4) When to change what *(decision tree)*

- **Learning curves flat early (first 10–20% steps)?**  
  → Tweak **optimizer/schedule** (LR, warmup %, EMA, clip).

- **Trains but overfits (train ↓, val ↔/↑)?**  
  → **Data** (augment regimes, mix BCs), **regularize** (stoch depth), or **add projection** post-decode.

- **Val improves but physics gaps grow?**  
  → Add/raise **projection penalties** or a **one-step residual corrector**; lower **guidance λ**.

- **Good small-proxy, bad full-eval?**  
  → **Proxy mismatch**: add one harder PDE/regime to **Small Eval** until rankings align.

- **Horizon instability only at long rollouts?**  
  → Increase **inverse-losses** (latent operators), reduce **Δt**, or add **periodic corrector** cadence.

- **OOM/too slow?**  
  → Increase **window size** (shifted windows), reduce **token dim**, enable **activation checkpointing**, shrink channels with **channel-separated tokens retained**.

- **Plateau after all that?**  
  → **Scale** (tokens/channels/data) on a small grid: ×1.4 params, ×1.5 effective tokens, ×2 data. **Re-fit LR** linearly; keep schedules identical.

---

## 5) The minimal system *(so iteration is automatic)*

### Pipelines
- **Config-driven training** with YAML includes (see `ups.utils.config_loader.load_config_with_includes`) and commit‑tied run IDs.
- Two evaluators: use the same `scripts/evaluate.py` with two configs:
  - **Small Eval** (10–60 min): smaller `latent.tokens`, higher `training.time_stride`, reduced dataset/shards.
  - **Full Eval** (overnight): full tokens/horizons and full datasets.
- **Auto-promote:** append `evaluate.py` CSV/JSON to a simple `reports/leaderboard.csv`; use `scripts/compare_runs.py` to decide champion updates.
- **Auto-promote:** pass `--leaderboard-run-id <id>` when invoking `scripts/evaluate.py` (or call `scripts/update_leaderboard.py` separately) to append rows to `reports/leaderboard.csv/html`, optionally with `--leaderboard-wandb` for W&B sync. Then use `scripts/compare_runs.py` to decide champion updates.
- **Dashboards:** rely on `scripts/evaluate.py` HTML, plus W&B logging via `ups.utils.monitoring` (enable in config). Add “red lines” thresholds for conservation/BCs when plotting.

### Sweep engine
- Start with **low-cardinality grids**, not blind BO:
  `LR × {0.5, 1.0, 1.5}`, `warmup {3%, 6%}`, `EMA {0.999, 0.9995, 0.9999}`, `grad_clip {0.5, 1.0}`.
- Size effective tokens to saturate GPU: increase `latent.tokens` or batch size until VRAM plateau.
- After a win, run a **small BO** around the winner ±20% for **12–24 trials** (W&B Sweeps or shell loops).

### Repro & hygiene
- **Seed everything** (training uses `scripts/train.py` which sets seeds; keep seeds stable across sweeps).
- **Log environment** in run metadata (CUDA/driver, GPU model) and retain config snapshots (evaluate already saves a YAML copy).
- **Unit tests** for datasets/metrics; **CI** to catch metric drift (see `pyproject.toml` and `tests/`).

### Fail-safes
- If **conservation gap > 2×** baseline → apply `physics_guards` projection on decoded fields and **re-eval Small**.
- If **ECE worsens by > 50%** → reduce TTC guidance / increase corrector cadence and **re-eval Small**.
- If **runtime > budget** → try larger attention `window_size`, reduce `latent.dim` or `latent.tokens`.

### Concrete starting config *(works surprisingly often)*
- See the block below and `configs/train_burgers_32dim.yaml` for a production template.

---

## 5) The minimal system *(so iteration is automatic)*

### Pipelines
- **Config-driven training** (YAML/JSON) with **hash-based run IDs**; every artifact (weights, logs, metrics, plots) tied to a commit hash.
- Two evaluators: **`eval_small`** (10–60 min) and **`eval_full`** (overnight) with **fixed seeds/splits**; both write to a **leaderboard**.
- **Auto-promote:** if `eval_small` beats champion by ≥ Δ **and** physics gates pass, enqueue `eval_full`.
- **Dashboards:** training curves, physics gaps, calibration, efficiency. **Red lines** for conservation/BC thresholds.

### Sweep engine
- Start with **low-cardinality grids**, not blind BO:  
  `LR × {0.5, 1.0, 1.5}`, `warmup {3%, 6%}`, `EMA {0.999, 0.9995, 0.9999}`.  
- Size **batch tokens** to saturate GPU; ensure no CPU dataloader bottlenecks.  
- After a win, run a **small BO** around the winner ±20% for **12–24 trials**.

### Repro & hygiene
- **Seed everything**; log environment (driver/CUDA) in run metadata.  
- **Unit tests** for dataset transforms & metrics; **CI** to catch metric drift.

### Fail-safes
- If **conservation gap > 2×** baseline → auto-apply **projection + re-eval**.  
- If **ECE worsens by > 50%** → reduce guidance / increase corrector cadence and **re-eval small**.  
- If **runtime > budget** → auto-try **window size ↑** or **token dim ↓** variants.

### Concrete starting config *(works surprisingly often)*
- **Backbone:** PDE-Transformer-S (shifted windows, U-shape, channel-separated tokens, AdaLN).  
- **Latent I/O:** UPT-style encoder/decoder (fixed **512–1024** latent tokens; any-point decoder).  
- **Optimizer:** AdamW (β=0.9, 0.95), wd=0.02, base LR=3e-4 @ 256k tokens, warmup 5%, cosine, EMA 0.9995, clip=1.0, bf16, flash-attn, act-ckpt.  
- **Loss:** task nRMSE + tiny edge loss (1e-3) + soft conservation penalty (1e-3).  
- **Hybrid:** operator drift each frame; diffusion corrector every **16** steps (1–2 consistency steps), λ=1.0; **projection after decode**.  
- **Batching:** temperature-balanced sampling across PDEs/BCs; curriculum on horizon (start k=4, grow to 16/32).  
- **Small Eval:** 4 PDEs × short horizon; **Full Eval:** all PDEs, long horizon, OOD geometries/BCs.

### What *not* to do
- Endless **architecture churn** before squeezing the optimizer/schedule.  
- Training **diffusion for full transients** without distillation (you’ll lose on wall-clock).  
- Comparing models **without fixed projections/physics checks**—rankings will lie.  
- Overfitting to **one metric** (nRMSE) while conservation/BCs silently degrade.

---

## 1) Objectives

- **Goal:** Iteratively beat a fixed **Champion** on accuracy **and** physics reliability within a fixed wall-clock.
- **Core Loops:**
  1) **Small Eval** (10–60 min) for rapid ranking & gating.
  2) **Full Eval** (overnight) for champion updates only.
- **Artifacts:** Reproducible runs (config hash), dashboards, and a canonical leaderboard.

---

## 2) Metrics & Gates (scoreboard)

**Task accuracy**  
- `nRMSE@k`, `relL2`, `PSNR`, `LPIPS` (as applicable)

**Physics integrity**  
- `conservation_gap` (mass/charge/energy)  
- `bc_violation` (Dirichlet/Neumann), `div_free_error`  
- `spectral_error` (energy spectrum MAPE)

**Reliability & stability**  
- `ECE` (calibration), `rollout_horizon@ρ` (Pearson ≥ 0.8)

**Efficiency**  
- `wall_clock_s/rollout`, `J_per_step` (if available), `VRAM_MB`, `params_M` (UPS evaluates and logs core metrics; extend as needed.)

**Gates (fail if any):**
- `conservation_gap` > 2× Champion → **fail**
- `bc_violation` > 1.5× Champion → **fail**
- `ECE` increased by > 50% → **fail**
- `wall_clock_s/rollout` > budget → **fail**

---

## 3) Config Schema (UPS)

UPS is fully config‑driven; see `configs/train_burgers_32dim.yaml` for a working, production‑ready template. Key sections:

```yaml
data:
  task: burgers1d            # PDEBench task or use kind: grid/mesh/particles with Zarr
  split: train
  root: data/pdebench
  patch_size: 1              # grid encoder patches

latent:
  dim: 32                    # latent feature dim
  tokens: 32                 # latent token count

operator:
  pdet:                      # PDE-Transformer backbone (src/ups/core/blocks_pdet.py)
    input_dim: 32
    hidden_dim: 96
    depths: [1, 1, 1]
    group_size: 12
    num_heads: 6

diffusion:                   # few-step residual corrector (optional)
  latent_dim: 32
  hidden_dim: 96

training:
  batch_size: 12
  time_stride: 2
  dt: 0.1
  num_workers: 8
  latent_cache_dir: data/latent_cache
  amp: true
  compile: true

stages:                      # staged training recipe
  operator:
    epochs: 25
    optimizer: {name: adamw, lr: 1.0e-3, weight_decay: 0.03}
    scheduler: {name: cosineannealinglr, t_max: 40, eta_min: 2.5e-5}
  diff_residual:
    epochs: 8

ttc:                         # test-time computing (optional)
  enabled: true
  steps: 1
  candidates: 8
  beam_width: 3
  horizon: 1
  sampler:
    tau_range: [0.15, 0.85]
    noise_std: 0.015
    noise_schedule: [0.03, 0.015, 0.005]
  reward:
    grid: [64, 64]
    mass_field: rho
    energy_field: e
    weights: {mass: 1.2, energy: 0.15, penalty_negative: 0.6}
  decoder:                    # AnyPointDecoder for reward evaluation
    latent_dim: 32
    query_dim: 2
    hidden_dim: 96
    num_layers: 3
    num_heads: 6
    frequencies: [1.0, 2.0, 4.0, 8.0]
    output_channels: {rho: 1, e: 1}
```

### 3a) Suggested Config (Reference Template)

The original reference template (strategy-oriented). Keep this as your tuning scaffold; map to UPS keys as noted below.

```yaml
run:
  seed: 1337
  device: "auto"
  mixed_precision: "bf16"
  save_every: 1000

data:
  loaders:
    batch_tokens: 256000
    shuffle: true
    num_workers: 8
  curriculum:
    start_horizon: 4
    max_horizon: 32
    grow_every_steps: 2000

model:
  io:
    encoder: "UPTEncoder"         # discretization-agnostic
    decoder: "AnyPointDecoder"    # query anywhere
    latent_tokens: 512
    latent_dim: 384
  core:
    backbone: "PDETransformer"
    shifted_windows: true
    u_shape: true
    channel_separated: true
    depth: 24
    ff_mult: 4
  conditioning:
    adaln: true
    fields: ["u","v","p"]
    params: ["Re","dt","bc_type","geometry_id"]

train:
  objective:
    type: "operator+residual"
    weights: {task: 1.0, edge: 0.001, conservation: 0.001}
  optimizer:
    name: "adamw"
    lr: 0.0003
    betas: [0.9, 0.95]
    weight_decay: 0.02
  schedule:
    warmup_ratio: 0.05
    cosine: true
  stabilization:
    grad_clip: 1.0
    ema: 0.9995
    qk_rmsnorm: true
  hybrid:
    diffusion_corrector:
      enabled: true
      cadence_steps: 16
      steps: 2             # few-step consistency/flow student
      guidance_lambda: 1.0

physics:
  projections:
    hodge_div_free: true
    positivity_clamp: true
  budgets_check: true

eval:
  small:
    time_budget_min: 60
    horizon: 8
  full:
    time_budget_hr: 8
    horizon: 32
  thresholds:
    min_delta_nrmse: -0.01  # must be <= Champion - 0.01 to be winner

logging:
  wandb: false
  tensorboard: true
  write_html_dashboard: true
```

UPS mapping notes:
- `model.io.encoder/decoder` → use `ups.io.enc_grid.GridEncoder` or graph encoders and `ups.io.decoder_anypoint.AnyPointDecoder`.
- `latent_tokens` → `latent.tokens`; `latent_dim` → `latent.dim`.
- `model.core` → `operator.pdet` (depths, group_size, num_heads map to UPS config fields).
- `conditioning.adaln` → see `ups.core.conditioning.AdaLNConditioner` if adding conditioning.
- `train.objective/weights` → map to `ups.training.losses` weights or keep as notes.
- `hybrid.diffusion_corrector` → `diffusion` block and rollout cadence via `ups.inference.rollout_transient` or TTC.

---

## 4) CLI Entry Points (UPS)

UPS ships Python scripts instead of a Makefile. Common tasks:

```bash
# Validate config and data (fast)
python scripts/validate_config.py configs/train_burgers_32dim.yaml
python scripts/validate_data.py   configs/train_burgers_32dim.yaml
python scripts/dry_run.py         configs/train_burgers_32dim.yaml --estimate-only

# Train (stage all or a specific stage)
python scripts/train.py --config configs/train_burgers_32dim.yaml --stage all

# Evaluate a checkpoint (with optional TTC)
python scripts/evaluate.py \
  --config configs/train_burgers_32dim.yaml \
  --operator checkpoints/op_latest.ckpt \
  --device cuda \
  --output-prefix reports/eval_run \
  --leaderboard-run-id run_full_$(date +%Y%m%d_%H%M%S) \
  --leaderboard-label full_eval \
  --leaderboard-wandb --leaderboard-wandb-project universal-simulator

# Small Eval (proxy config)
python scripts/evaluate.py \
  --config configs/small_eval_burgers.yaml \
  --operator checkpoints/op_latest.ckpt \
  --device cpu \
  --output-prefix reports/eval_small \
  --leaderboard-run-id run_small_$(date +%Y%m%d_%H%M%S) \
  --leaderboard-label small_eval

# Full Eval (test split, full resolution)
python scripts/evaluate.py \
  --config configs/full_eval_burgers.yaml \
  --operator checkpoints/op_latest.ckpt \
  --device cuda \
  --output-prefix reports/eval_full \
  --leaderboard-run-id run_full_$(date +%Y%m%d_%H%M%S) \
  --leaderboard-label full_eval \
  --leaderboard-wandb --leaderboard-wandb-project universal-simulator

# Analyze and compare runs
python scripts/analyze_run.py <run_id> --output reports/analysis.md
python scripts/compare_runs.py <run1> <run2> --output reports/comparison.md

# VastAI production launch (optional)
python scripts/vast_launch.py launch --config configs/train_burgers_32dim.yaml --auto-shutdown
```

---

## 5) Per-Run Procedure (agent checklist)

1. **Hydrate config**
   - Fill `$(CFG)` from a template; set `run.seed`, `data.splits`, and GPU availability.

2. **Prep data**
   - `make prep CFG=configs/baselines/pde_t_s.yaml`  
   - Verify data hashes & cached preprocessing.

3. **Small Train + Small Eval**
   - Train with reduced budgets: set `training.time_stride`, limit `data.shard_limit` (if used), and smaller `latent.tokens` as a proxy.
   - Evaluate with `scripts/evaluate.py` (produces CSV/JSON/HTML under `reports/`).
   - **Gate:** If physics gates fail (use `ups.models.physics_guards` in your analysis) → archive run.
   - If `nrmse` improves ≥ threshold and physics passes → mark **candidate**.

4. **Full Eval (candidates only)**
   - Re-run `scripts/evaluate.py` on the full dataset settings.
   - If wins on **accuracy + physics + efficiency** → record via `scripts/compare_runs.py` and update your leaderboard CSV/HTML in `reports/`.

5. **Log decisions**
   - Append a 3-line verdict to `artifacts/runs/<run_id>/notes.txt`.

---

## 6) Sweep Protocol (fast → precise)

**Stage A (coarse grid, 12–24 trials)**  
- LR × {0.5, 1.0, 1.5}, warmup {3%, 6%}, EMA {0.999, 0.9995, 0.9999}, grad clip {0.5, 1.0}.  
- Promote top-3 by **Small Eval** only.

**Stage B (local BO around winner, 12 trials)**  
- ±20% LR, cadence {8,16,32}, corrector steps {1,2}, λ {0.5, 1.0, 1.5}.  
- Full Eval for top-2.

- Increase `latent.dim` ×1.4 or `latent.tokens` ×1.5; double data.  
- Refit LR with linear scaling; keep schedules fixed.

---

## 7) Physics Guards (available utilities)

- **Post-decode projections:** use `ups.models.physics_guards.helmholtz_hodge_projection_grid` (div‑free), `positify`, and `interface_flux_projection` when analyzing decoded fields.
- **Conservation budgets:** compute per-step & cumulative deltas on decoded fields; if |Δ| > threshold → use a diffusion corrector and re‑project.
- **Horizon guard:** if correlation < 0.8 before target horizon → reduce Δt or increase corrector cadence (`ups.inference.rollout_transient`).

---

## 8) Promotion Rules (Champion update)

A candidate becomes **Champion** iff all hold:

- `nRMSE@k(candidate) ≤ nRMSE@k(champion) - 0.01`
- `conservation_gap(candidate) ≤ champion * 1.0`
- `bc_violation(candidate) ≤ champion * 1.0`
- `ECE(candidate) ≤ champion * 1.25`
- `wall_clock(candidate) ≤ budget`

Record: `leaderboard.csv` columns = `run_id, commit, config_hash, nRMSE, cons_gap, bc_violation, ECE, wall_clock, params, date`.

---

## 9) Troubleshooting Playbooks

- **Flat learning early** → increase warmup to 6%, lower LR 0.7×, ensure Q/K RMSNorm, check grad clip.  
- **Good task, bad physics** → raise projection weights slightly; add 1-step residual corrector; reduce guidance λ.  
- **Unstable long rollout** → increase inverse-loss weight (latent operator), lower Δt, corrector every 8 steps.  
- **OOM** → bigger attention window, reduce latent_dim, enable activation checkpointing.  
- **UQ miscalibrated** → temperature scaling on logits/variance; conformal interval post-hoc.

---

## 10) Templates

**Baseline (Burgers 32‑dim + hybrid)**
```bash
# Validate + train
python scripts/validate_config.py configs/train_burgers_32dim.yaml
python scripts/validate_data.py   configs/train_burgers_32dim.yaml
python scripts/train.py --config  configs/train_burgers_32dim.yaml --stage all

# Evaluate (non‑TTC or TTC per config)
python scripts/evaluate.py --config configs/train_burgers_32dim.yaml --operator checkpoints/op_latest.ckpt --device cuda
```

**Sweep (optimizer)**
```bash
# Use your sweep service (e.g., W&B Sweeps) or shell loops to vary:
#  - stages.operator.optimizer.lr
#  - stages.operator.scheduler.t_max
#  - training.grad_clip, training.ema_decay
```

**Promote winner**
```bash
# Compare and update your leaderboard under reports/
python scripts/compare_runs.py <run_id> <baseline_run_id> --output reports/comparison.md
```

---

## 11) Acceptance Criteria (Definition of Done for a PR)

- Reproducible run with **config + seed**; all artifacts saved (metrics JSON/CSV/HTML from `evaluate.py`).
- Small Eval **passes gates** and shows ≥ 1σ improvement vs. baseline.
- Full Eval meets **promotion rules** (accuracy + physics + efficiency).
- Dashboard updated; 3‑line change note committed in `reports/` or run notes.

---

## 12) Minimal Agent Tasks (bulletized)

- [ ] Validate config and data; dry‑run estimate.
- [ ] Train with GPU saturation and bf16.
- [ ] Run Small Eval; compute scoreboard; apply gates and fail‑safes.
- [ ] For winners: run Full Eval; compare vs. champion; promote if criteria met.
- [ ] Update leaderboard & HTML dashboard via `scripts/update_leaderboard.py`; archive losers with reason; propose next sweep parameters.


## 11) Acceptance Criteria (Definition of Done for a PR)

- Reproducible run with **config + seed**; all artifacts saved.  
- Small Eval **passes gates** and shows ≥ 1σ improvement.  
- Full Eval meets **promotion rules**.  
- Dashboard updated; 3-line change note committed.

---

## 12) Minimal Agent Tasks (bulletized)

- [ ] Validate dataset hashes; build `proc/` cache.  
- [ ] Spawn training with **GPU saturation** and **bf16**.  
- [ ] Run Small Eval; compute scoreboard; apply **gates**.  
- [ ] For winners: run Full Eval; **promote** if criteria met.  
- [ ] Update leaderboard & HTML dashboard.  
- [ ] Archive losers with reason; propose next sweep parameters.

---

# 📊 Optimal Full Hyperparameters (Opinionated Defaults)

> Reflects strong defaults from recent operator/diffusion/transformer results on physics data. Scale with safe ranges below.

**Target:** 2D/2.5D fields on an 80 GB H100; multi-PDE pretrain → downstream fine-tune  
**Backbone:** PDE-Transformer core, UPT I/O, hybrid operator+diffusion corrector

```yaml
# === OPTIMIZER & SCHEDULE ===
optimizer: adamw
betas: [0.9, 0.95]
weight_decay: 0.02
base_lr: 0.0003           # @ ~256k batch tokens (scale linearly w.r.t. batch_tokens)
lr_schedule: cosine
warmup_ratio: 0.05        # 5% steps
ema_decay: 0.9995
grad_clip: 1.0
precision: bf16
attn: flash                # use fused attention kernels
activation_checkpointing: true

# === DATA ===
batch_tokens: 256000      # tune to saturate GPU; prefer token budget over sample count
horizon_curriculum: [4, 8, 16, 32]
sampling_temperature: 0.7 # regime/BC sampler temp to avoid easy-mode collapse

# === I/O (UPT-style) ===
latent_tokens: 512        # fixed-length latent
latent_dim: 384
encoder_supernodes: 4096  # 2k–8k depending on domain size
decoder_query_points: 16384  # per frame; bump at eval for super-res
inverse_encode_weight: 0.5
inverse_decode_weight: 0.5

# === PDE-TRANSFORMER CORE ===
depth: 24                 # S/B/L ~ 24/32/40
model_dim: 384            # S/B/L ~ 384/512/768
mlp_mult: 4
num_heads: 8              # keep head_dim ~64
shifted_windows: true
window_size: 8            # 8–16; increase to cut memory
u_shape: true
u_stages: [1, 2, 4]       # token down/up factors
channel_separated: true   # per-physical-channel token streams
qk_rmsnorm: true
dropout: 0.0              # usually not needed with EMA + clip

# === CONDITIONING ===
adaln: true
cond_fields: ["pde_type","Re","dt","bc_type","geom_hash"]
time_encoding: learned_fourier  # for Δt/step index

# === LOSSES (operator training) ===
loss:
  task_nrmse: 1.0
  edge_tv: 0.001            # turn off if spectra look fine
  conservation_soft: 0.001  # mass/charge/energy integrals
post_decode_projection:
  hodge_div_free: true
  positivity_clamp: true
  flux_continuity: true

# === HYBRID: DIFFUSION CORRECTOR ===
corrector:
  enabled: true
  method: consistency        # or rectified_flow
  steps: 2                   # few-step student
  cadence_steps: 16
  guidance_lambda: 1.0       # physics-residual guidance at sampling

# === STEADY-STATE DIFFUSION PRIOR (separate head) ===
steady_prior:
  training_objective: flow_matching
  noise_schedule: cosine
  sampler_steps: 8           # after consistency distillation
  guidance_lambda: 1.0

# === EVAL GATES ===
gates:
  max_bc_violation_ratio: 1.0
  max_conservation_ratio: 1.0
  max_ece_ratio: 1.25
  min_corr_horizon: 0.8
```

---

## Ranges & Rules of Thumb

### Optimizer & schedule
- **LR scaling:** `lr ≈ 3e-4 × (batch_tokens / 256k)`; clamp ∈ [1.5e-4, 6e-4] if unstable.
- **Warmup:** 3–6% of steps; **EMA:** 0.999–0.9999; **clip:** 0.5–1.0.

### PDE-Transformer core
- **S:** depth 24, dim 384, heads 8  
- **B:** depth 32, dim 512, heads 8  
- **L:** depth 40, dim 768, heads 12  
- `window_size` 8–16; `u_stages` `[1,2,4]` (+`[8]` for huge domains).

### UPT I/O
- `latent_tokens` 512–1024; `latent_dim` 256–512 (384 sweet spot).  
- `supernodes` 2k–8k; `query_points` 8k–32k (increase at eval).

### Hybrid operator + diffusion corrector
- Cadence 8–32; steps 1–2; guidance λ 0.5–1.5 (tune for **calibration**).

### Steady diffusion prior
- 6–12 steps post-distillation (flow/consistency preferred to DDPM).
- Rich **AdaLN conditioning** for BCs, geometry, parameters.

### Loss weights
- `task_nrmse: 1.0`, `edge_tv: 1e-3` (toggle), `conservation_soft: 1e-3`.
- `inverse_encode/decode: 0.5/0.5` for latent stability.

---

## 3D & Large-Domain Notes
- Increase `window_size` to 12–16; add `[8]` in `u_stages`.  
- Consider dim 640 (vs 768) if VRAM bound; keep `head_dim ~64`.  
- Strong activation checkpointing; keep bf16.  
- Particles/contacts: neighbor cap k=32–64; hierarchical neighborhoods; symplectic/port-Hamiltonian updates.

---

## Curriculum & Batching
- Horizon: 4 → 8 → 16 → 32; freeze bottom ⅓ layers when stepping up first time.  
- Regime sampler temperature 0.6–0.8 across PDEs/BCs/params.  
- Mix steady/transient minibatches 1:3 if training both heads jointly.

---

## Scale Up Only After Optimizer Wins
- If plateau: multiply **one axis**: `model_dim ×1.4` or `latent_tokens ×1.5` or **data ×2**.  
- Refit LR (linear) and keep schedules fixed.

---

## TL;DR Presets

- **Single-GPU (80 GB) “S”**: depth 24, dim 384, heads 8, window 8, latent_tokens 512, latent_dim 384, LR 3e-4, warmup 5%, EMA 0.9995, cadence 16, corrector 2 steps, λ 1.0.  
- **Multi-GPU (4×80 GB) “B”**: depth 32, dim 512, heads 8, window 12, latent_tokens 768, latent_dim 512, batch_tokens 512k, LR 6e-4, warmup 3%, EMA 0.9995, cadence 16, corrector 2, λ 1.0.  
- **Large 3D “L”**: depth 40, dim 768→**consider 640** if VRAM tight, heads 12, window 16, U-stages `[1,2,4,8]`, latent_tokens 1024, latent_dim 512, strong checkpointing; cadence 8–16.
```

---

*End of document.*
