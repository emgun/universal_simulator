# Current SOTA Sweep Status

**Last Updated:** 2025-10-19
**Status:** 6 instances launched and starting up ✅

---

## Instance Overview

| Instance ID | Config | GPU | Status | Cost/hr | SSH |
|-------------|--------|-----|--------|---------|-----|
| 27006848 | rerun_txxoc8a8 (baseline test) | RTX 5880Ada | ✅ Running | $0.37 | ssh4.vast.ai:16848 |
| 27006913 | sweep_round_a_lr2e4_w3 | Q RTX 8000 | 🟡 Loading | $0.25 | ssh9.vast.ai:16912 |
| 27006920 | sweep_round_a_lr45e4_w5 | Q RTX 8000 | 🟡 Loading | $0.26 | ssh2.vast.ai:16920 |
| 27006926 | sweep_round_b_capacity_up | Q RTX 8000 | 🟡 Loading | $0.27 | ssh9.vast.ai:16926 |
| 27006930 | sweep_round_b_deeper | Q RTX 8000 | 🟡 Loading | $0.27 | ssh9.vast.ai:16930 |
| 27006938 | sweep_aggressive_sota_48gb | RTX 5880Ada | ✅ Running | $0.37 | ssh8.vast.ai:16938 |

**Total Cost:** ~$0.25-0.37/hr × 6 = **~$1.70/hr** (very cost-effective!)

---

## Config Details

### Instance 27006848 - Baseline Test
- **Config:** rerun_txxoc8a8.yaml
- **Purpose:** Verify pipeline works (baseline nRMSE ~0.09)
- **Params:** dim=32, tokens=16, hidden=64

### Instance 27006913 - Round A-1
- **Config:** sweep_round_a_lr2e4_w3.yaml
- **Hypothesis:** Lower LR with early warmup improves stability
- **Changes:** LR=2e-4 (vs 1e-3), warmup=3%, EMA=0.9995, betas=[0.9,0.95]

### Instance 27006920 - Round A-3
- **Config:** sweep_round_a_lr45e4_w5.yaml
- **Hypothesis:** Higher LR with strong EMA enables faster convergence
- **Changes:** LR=4.5e-4, warmup=5%, EMA=0.9999, betas=[0.9,0.95]

### Instance 27006926 - Round B-1 (Capacity Up)
- **Config:** sweep_round_b_capacity_up.yaml
- **Hypothesis:** Baseline is under-parameterized, more capacity reduces nRMSE
- **Changes:** tokens=32 (2×), hidden=96 (1.5×), heads=6, epochs=20+8+8

### Instance 27006930 - Round B-2 (Deeper)
- **Config:** sweep_round_b_deeper.yaml
- **Hypothesis:** Depth helps learn temporal dynamics
- **Changes:** depths=[2,2,2] (2×), tokens=24, hidden=80, grad_clip=0.8

### Instance 27006938 - Aggressive SOTA
- **Config:** sweep_aggressive_sota_48gb.yaml
- **Hypothesis:** Maximum capacity + extended training → nRMSE < 0.035
- **Changes:** dim=40, tokens=40, hidden=112, depths=[2,2,2], epochs=28+10+10

**Missing from full sweep:** sweep_round_a_lr3e4_w5.yaml (mid-range LR) - can launch separately if needed

---

## Current Issues & Solutions

### ✅ Fixed: Instance Launch Failures
- **Problem:** First 6 instances (26979xxx) had SSH connectivity issues
- **Solution:** Destroyed and relaunched with explicit offer IDs
- **Result:** All new instances launching successfully

### 🔄 Ongoing: SSH Access Timing Out
- **Problem:** SSH to instances timing out (Vast.ai infrastructure issue)
- **Impact:** Cannot manually monitor via SSH right now
- **Workaround:**
  - Instances run autonomously via onstart scripts
  - Monitor via W&B instead: https://wandb.ai/emgun-morpheus-space/universal-simulator
  - Use `vastai logs <instance_id>` as alternative

### ⏳ Expected: Startup Time
- **Normal behavior:** Instances take 2-5 minutes to go from "loading" → "running"
- **Then:** Onstart script runs (10-30 min for data download + latent cache)
- **Then:** Training begins (visible in W&B)

---

## Monitoring Strategy

Since SSH is unreliable, use these alternatives:

### 1. Vast.ai Dashboard
```bash
# Check instance status every 15-30 minutes
vastai show instances
```

### 2. Weights & Biases (PRIMARY)
- **URL:** https://wandb.ai/emgun-morpheus-space/universal-simulator
- **Look for:** New runs appearing with tags:
  - `sweep-round-a` + `optimizer-sweep`
  - `sweep-round-b` + `capacity-scale` or `depth-scale`
  - `aggressive-sota` + `extended-training`
  - `txxoc8a8-rerun` + `baseline-validation`

### 3. Vast.ai Logs (when SSH fails)
```bash
# Check onstart script output
vastai logs 27006848  # baseline
vastai logs 27006913  # round-a-1
vastai logs 27006920  # round-a-3
vastai logs 27006926  # round-b-1
vastai logs 27006930  # round-b-2
vastai logs 27006938  # aggressive
```

---

## Expected Timeline

### Phase 1: Startup (0-30 min) - CURRENT
- ✅ Instances created
- 🔄 Docker image loading
- ⏳ Onstart script executing
  - Git clone
  - pip install
  - Data download from B2
  - Latent cache precomputation

**Check:** W&B runs should appear within 30-45 minutes

### Phase 2: Training (2-14 hours)
- Operator stage (4-8 hours)
- Diffusion residual (1-3 hours)
- Consistency distillation (1-3 hours)

**Check:** W&B loss curves should show steady decrease

### Phase 3: Evaluation (0.5-2 hours)
- Small eval
- Full eval (with --force-full-eval flag)
- Leaderboard update

### Phase 4: Auto-Shutdown
- Instances power off automatically
- Final artifacts in W&B

**Total ETA:** 12-18 hours from now (complete by 2025-10-20 morning)

---

## Success Criteria

### For ANY run to be considered successful:
- ✅ Completes all training stages without OOM/divergence
- ✅ nRMSE ≤ 0.08 (improvement over 0.09 baseline)
- ✅ Physics gates pass (conservation ≤1.0×, BC violation ≤1.0×)
- ✅ Full eval metrics logged to leaderboard

### For SOTA achievement:
- 🎯 **nRMSE < 0.05** (target)
- 🎯 **nRMSE < 0.035** (stretch goal)
- 🎯 Conservation gap < 0.5× baseline
- 🎯 Long rollout horizon (≥64 steps @ ρ≥0.8)

---

## Troubleshooting

### If instance shows "exited" within first hour:
```bash
vastai logs <instance_id>  # Check what failed
```
Common causes:
- OOM during data download → Relaunch with smaller batch
- Git clone failed → Network hiccup, relaunch same config
- Missing env vars → Check `vastai show env-vars`

### If no W&B runs appear after 1 hour:
- Check `vastai logs <instance_id>` for errors
- Verify instance still "running" (not "exited")
- Possible that data download is slow (large files)

### If training diverges (loss → NaN):
- Note which config failed
- Check if LR too high or grad_clip too loose
- Will inform next iteration

---

## Next Actions

### Immediate (within 1 hour):
- ✅ Wait for all instances to reach "running" status
- ✅ Verify W&B runs start appearing
- ✅ Check first loss values are reasonable (< 1.0)

### Short-term (3-6 hours):
- Monitor W&B training curves
- Check for any early exits or OOM issues
- Verify at least 3-4 runs progressing normally

### Long-term (12-18 hours):
- Wait for completion
- Download leaderboard results
- Compare runs and identify best config
- Decide on next iteration (if needed)

---

## Cost Tracking

**Current burn rate:** ~$1.70/hr

**Estimated total:**
- Best case (all complete in 12hr): $20.40
- Expected (14hr): $23.80
- Worst case (18hr): $30.60

**Budget:** $30-35
**Status:** ✅ Within budget

---

*This document will be updated as runs progress. Check W&B for real-time metrics.*
