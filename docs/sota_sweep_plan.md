# SOTA Sweep Plan - Target: nRMSE < 0.035

**Date:** 2025-10-18
**Baseline:** `rerun_txxoc8a8` with nRMSE ~0.09
**Goal:** Achieve nRMSE < 0.035 with physics integrity maintained

---

## Strategy Overview

Following the **Fast-to-SOTA** and **Parallel Runs** playbooks, we execute a multi-round parallel sweep focusing on high-ROI hyperparameters. Each round learns from the previous, promoting only the best candidates to full evaluation.

### Guiding Principles

1. **Orthogonalize:** Test independent axes (optimizer vs. capacity vs. physics)
2. **Fix compute:** Keep batch tokens and evaluation consistent
3. **Low-cardinality first:** 3-5 runs per round to learn fast
4. **Promote quickly:** Top 1-2 from small-eval → full-eval → champion

---

## Round A: Optimizer & Schedule (3 runs)

**Hypothesis:** The baseline uses suboptimal LR/warmup/EMA; better optimization will yield 20-40% nRMSE reduction.

| Config | LR | Warmup | EMA | Epochs | Tags |
|--------|-------|--------|----------|--------|------|
| `sweep_round_a_lr2e4_w3.yaml` | 2e-4 | 3% | 0.9995 | 20+8+8 | Conservative LR, early warmup |
| `sweep_round_a_lr3e4_w5.yaml` | 3e-4 | 5% | 0.9995 | 20+8+8 | Balanced mid-range |
| `sweep_round_a_lr45e4_w5.yaml` | 4.5e-4 | 5% | 0.9999 | 20+8+8 | Aggressive LR, high EMA |

**Key Changes from Baseline:**
- Changed betas from `[0.9, 0.999]` → `[0.9, 0.95]` (more stable, per playbook)
- Extended epochs: operator 15→20, diff_residual 5→8, consistency 6→8
- Added explicit warmup_steps_ratio (baseline missing)
- Varied EMA decay (0.9995 vs 0.9999)

**Expected Outcome:** One config improves nRMSE by ≥0.01, passes physics gates → promote to Round B.

---

## Round B: Capacity Scaling (2 runs)

**Hypothesis:** Baseline is under-parameterized (16 tokens, 64 hidden); scaling capacity will capture finer dynamics.

| Config | Latent Tokens | Hidden Dim | Depths | Heads | Batch Size | Tags |
|--------|---------------|------------|--------|-------|------------|------|
| `sweep_round_b_capacity_up.yaml` | 32 | 96 | [1,1,1] | 6 | 10 | Width+tokens scale |
| `sweep_round_b_deeper.yaml` | 24 | 80 | [2,2,2] | 5 | 10 | Depth scale |

**Key Changes:**
- **Capacity Up:** 2× tokens (16→32), 1.5× hidden (64→96), scaled heads/groups
- **Deeper:** 2× depth ([1,1,1]→[2,2,2]), moderate width increase
- Adjusted batch size (12→10) and accum_steps (4→5) to maintain VRAM budget
- Scaled decoder hidden_dim and num_layers proportionally
- LR scaled linearly with capacity per playbook (3e-4 → 4e-4 for capacity_up)

**Expected Outcome:** Capacity increase yields 15-25% nRMSE reduction if baseline was bottlenecked; depth may help long-horizon stability.

---

## Round Aggressive: Combined SOTA Push (1 run)

**Hypothesis:** Combine best optimizer + maximum feasible capacity + extended training → target nRMSE < 0.035.

| Config | Latent Dim/Tokens | Hidden | Depths | Epochs | TTC Steps | Tags |
|--------|-------------------|--------|--------|--------|-----------|------|
| `sweep_aggressive_sota.yaml` | 48/48 | 128 | [2,2,2] | 30+12+12 | 2 | Large model, extended |

**Key Changes:**
- **Massive capacity:** dim 32→48, tokens 16→48, hidden 64→128
- **Extended training:** operator 15→30, diff 5→12, consistency 6→12
- **Enhanced TTC:** steps 1→2, candidates 8→12, beam 3→4, tighter threshold (0.35→0.3)
- **Stronger physics weighting:** gamma 1.0→1.2, mass weight 1.2→1.5
- **More frequencies:** Added 16.0 to decoder
- **Optimized regularization:** lambda_spectral 0.05→0.08, weight_decay 0.01→0.03
- Batch size reduced to 8 (VRAM), accum_steps increased to 6

**Expected Outcome:** If capacity was the primary bottleneck, this should achieve nRMSE < 0.05, potentially < 0.035 with TTC refinement.

---

## Execution Plan

### Phase 1: Launch Parallel Runs (6 instances total)

1. **Round A (3 instances):** optimizer sweep - smallest VRAM, fastest iteration
2. **Round B (2 instances):** capacity sweep - moderate VRAM
3. **Aggressive (1 instance):** large model - high VRAM, longest runtime

**Compute Allocation:**
- Round A configs: A100 40GB (cheaper, sufficient)
- Round B configs: A100 40GB or 80GB (moderate)
- Aggressive config: A100 80GB (required)

**Vast.ai Search Criteria:**
```bash
vastai search offers 'gpu_ram >= 40 reliability > 0.95 num_gpus=1 disk_space >= 64' --order 'dph_total'
# For aggressive: gpu_ram >= 80
```

### Phase 2: Monitor & Gate (6-12 hours)

- All runs log to W&B under `universal-simulator` project
- Round A group: `sota-sweep-round-a`
- Round B group: `sota-sweep-round-b`
- Aggressive group: `sota-aggressive`

**Small-Eval Gating:**
- Must improve nRMSE by ≥0.01 vs baseline (0.09 → ≤0.08)
- Conservation gap ≤1.0× baseline
- BC violation ≤1.0× baseline
- ECE ≤1.25× baseline

### Phase 3: Promote Winners (12-24 hours)

- Top 1 from Round A → full-eval
- Top 1 from Round B → full-eval
- Aggressive → full-eval (forced, even if small-eval marginal)

**Full-Eval Promotion:**
- nRMSE < 0.08 → new baseline
- nRMSE < 0.05 → strong candidate
- nRMSE < 0.035 → **SOTA achieved**

### Phase 4: Iteration (if needed)

If no config reaches < 0.035:

**Round C: Local BO around best**
- Take top config from A+B
- Vary ±20% around winner: LR, grad_clip, TTC threshold, lambda_spectral
- 6-12 trials via W&B sweep

**Round D: Architecture refinement**
- If aggressive model shows promise but not quite there:
  - Extend epochs further (40+15+15)
  - Add stochastic depth 0.1-0.2
  - Tune TTC reward weights more aggressively
  - Try dim 64, tokens 64 (if VRAM allows)

---

## Checkpoints & Artifacts

Each run produces:
- `checkpoints/{operator,diffusion}_ema.pt`
- `artifacts/runs/{run_id}/summary.json`
- `reports/leaderboard.csv` (updated)
- `reports/eval_{run_id}.{json,csv,html}`
- W&B artifacts (config, metrics, plots)

**Leaderboard Columns:**
- `run_id`, `timestamp`, `label`, `nrmse`, `conservation_gap`, `bc_violation`, `ece`, `wall_clock`, `params`, `tags`

---

## Success Criteria

### Minimum Acceptable (Baseline Improvement)
- nRMSE ≤ 0.07 (22% reduction)
- Conservation gap ≤ baseline
- Wall-clock ≤ 2× baseline

### Target (Strong SOTA)
- nRMSE ≤ 0.05 (44% reduction)
- Conservation gap ≤ 0.5× baseline
- ECE ≤ baseline

### Stretch Goal (Full SOTA)
- **nRMSE < 0.035** (61% reduction)
- Conservation gap < 0.3× baseline
- BC violation < 0.5× baseline
- ECE ≤ baseline
- Rollout horizon@ρ≥0.8 ≥ 64 steps

---

## Risk Mitigation

### If OOM on Vast.ai
- Reduce batch_size by 1-2
- Increase accum_steps proportionally
- Disable compile temporarily
- Use fp32 for latent_cache_dtype → bf16

### If Training Diverges
- Lower LR by 0.7×
- Increase warmup to 8%
- Lower grad_clip to 0.5
- Check for NaNs in TTC reward computation

### If Physics Gates Fail
- Increase lambda_spectral (0.05 → 0.1)
- Raise TTC mass/energy weights
- Apply post-decode Hodge projection
- Reduce guidance_lambda (1.0 → 0.7)

### If Runs Stall on Vast.ai
- SSH check: `vastai ssh <instance_id> 'tail -50 nohup.out'`
- Kill stuck instance: `vastai destroy instance <id>`
- Relaunch with same config + `--resume` flag (if implemented)

---

## Timeline Estimate

| Phase | Duration | Parallel | Notes |
|-------|----------|----------|-------|
| Config creation | 1 hr | - | Done |
| Instance launch | 0.5 hr | - | Automated via vast_launch.py |
| Data download | 0.5 hr | Yes | Parallel across instances |
| Training (Round A) | 6-8 hr | 3× | Shortest (small model) |
| Training (Round B) | 8-10 hr | 2× | Moderate (medium model) |
| Training (Aggressive) | 12-16 hr | 1× | Longest (large model) |
| Small-eval | 0.5 hr | All | Concurrent |
| Full-eval (top 3) | 2-3 hr | 3× | Concurrent |
| Analysis & comparison | 1 hr | - | Post-processing |
| **Total (wall-clock)** | **16-20 hr** | **6×** | End-to-end |

**Cost Estimate (Vast.ai):**
- Round A (3× A100 40GB @ $0.40/hr × 8hr) = $9.60
- Round B (2× A100 40GB @ $0.40/hr × 10hr) = $8.00
- Aggressive (1× A100 80GB @ $0.95/hr × 16hr) = $15.20
- **Total: ~$33** for full sweep

---

## Next Steps

1. ✅ **Create configs** (completed)
2. 🔄 **Search Vast.ai instances** (next)
3. 🔄 **Launch 6 parallel runs** (next)
4. ⏳ **Monitor W&B dashboards** (after launch)
5. ⏳ **Apply gates & promote** (after small-eval)
6. ⏳ **Iterate if needed** (conditional)

---

## Config Summary Table

| Config | Type | Tokens | Dim | Hidden | Depths | Heads | LR | Warmup | EMA | Epochs | VRAM Est. |
|--------|------|--------|-----|--------|--------|-------|-------|--------|----------|--------|-----------|
| `sweep_round_a_lr2e4_w3` | Opt | 16 | 32 | 64 | [1,1,1] | 4 | 2e-4 | 3% | 0.9995 | 20+8+8 | ~18GB |
| `sweep_round_a_lr3e4_w5` | Opt | 16 | 32 | 64 | [1,1,1] | 4 | 3e-4 | 5% | 0.9995 | 20+8+8 | ~18GB |
| `sweep_round_a_lr45e4_w5` | Opt | 16 | 32 | 64 | [1,1,1] | 4 | 4.5e-4 | 5% | 0.9999 | 20+8+8 | ~18GB |
| `sweep_round_b_capacity_up` | Cap | 32 | 32 | 96 | [1,1,1] | 6 | 4e-4 | 5% | 0.9995 | 20+8+8 | ~28GB |
| `sweep_round_b_deeper` | Cap | 24 | 32 | 80 | [2,2,2] | 5 | 3.5e-4 | 6% | 0.9995 | 22+8+8 | ~24GB |
| `sweep_aggressive_sota` | Agg | 48 | 48 | 128 | [2,2,2] | 8 | 3.5e-4 | 6% | 0.9997 | 30+12+12 | ~55GB |

---

*End of Plan*
