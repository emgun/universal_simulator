# Active SOTA Sweep Runs - Status & Monitoring

**Launch Date:** 2025-10-18
**Total Instances:** 6 RTX A6000 (48GB)
**Total Estimated Cost:** ~$24-30
**Expected Completion:** 12-18 hours (2025-10-19 morning/afternoon)

---

## Instance Status

### Round A: Optimizer Sweep (3 instances) ✅

| Instance ID | Config | LR | Warmup | EMA | Status | SSH |
|-------------|--------|-------|--------|----------|--------|-----|
| 26979106 | sweep_round_a_lr2e4_w3 | 2e-4 | 3% | 0.9995 | running | ssh1.vast.ai:19106 |
| 26979113 | sweep_round_a_lr3e4_w5 | 3e-4 | 5% | 0.9995 | running | ssh5.vast.ai:19112 |
| 26979117 | sweep_round_a_lr45e4_w5 | 4.5e-4 | 5% | 0.9999 | running | ssh9.vast.ai:19116 |

**Target:** Find optimal LR/warmup/EMA combination
**Expected Runtime:** 8-10 hours
**Expected Cost:** ~$3.50 per instance = $10.50 total

### Round B: Capacity Sweep (2 instances) 🟡

| Instance ID | Config | Tokens | Hidden | Depths | Status | SSH |
|-------------|--------|--------|--------|--------|--------|-----|
| 26979132 | sweep_round_b_capacity_up | 32 | 96 | [1,1,1] | loading | ssh1.vast.ai:19132 |
| 26979138 | sweep_round_b_deeper | 24 | 80 | [2,2,2] | loading | ssh4.vast.ai:19138 |

**Target:** Determine if capacity bottleneck exists
**Expected Runtime:** 9-11 hours
**Expected Cost:** ~$4.00 per instance = $8.00 total

### Aggressive: Large SOTA Model (1 instance) 🟡

| Instance ID | Config | Dim/Tokens | Hidden | Depths | Status | SSH |
|-------------|--------|------------|--------|--------|--------|-----|
| 26979151 | sweep_aggressive_sota_48gb | 40/40 | 112 | [2,2,2] | launching | ssh6.vast.ai:19150 |

**Target:** Maximum feasible capacity for nRMSE < 0.035
**Expected Runtime:** 14-18 hours
**Expected Cost:** ~$6.50

---

## Monitoring Commands

### Check All Instances
```bash
vastai show instances
```

### SSH into Specific Instance
```bash
# Round A-1 (LR 2e-4)
ssh -p 19106 root@ssh1.vast.ai

# Round A-2 (LR 3e-4)
ssh -p 19112 root@ssh5.vast.ai

# Round A-3 (LR 4.5e-4)
ssh -p 19116 root@ssh9.vast.ai

# Round B-1 (Capacity Up)
ssh -p 19132 root@ssh1.vast.ai

# Round B-2 (Deeper)
ssh -p 19138 root@ssh4.vast.ai

# Aggressive
ssh -p 19150 root@ssh6.vast.ai
```

### Check Training Progress
```bash
# SSH into instance, then:
tail -f nohup.out                    # Live logs
grep -E "Epoch|Loss|nrmse" nohup.out | tail -20  # Recent progress
ps aux | grep python                 # Check if training running
nvidia-smi                           # GPU utilization
```

### View Instance Logs (without SSH)
```bash
vastai logs <instance_id>
# Example:
vastai logs 26979106
```

### Stop Instance (if needed)
```bash
vastai destroy instance <instance_id>
```

---

## Weights & Biases Monitoring

**Project:** https://wandb.ai/emgun-morpheus-space/universal-simulator

**Groups:**
- `sota-sweep-round-a` - Optimizer sweep runs
- `sota-sweep-round-b` - Capacity sweep runs
- `sota-aggressive` - Large model run

**Tags to filter:**
- `sweep-round-a` + `optimizer-sweep`
- `sweep-round-b` + `capacity-scale` or `depth-scale`
- `aggressive-sota` + `extended-training`

**Key Metrics to Watch:**
- `training/loss` - Should decrease steadily
- `training/grad_norm` - Should be < 1.0 (grad_clip threshold)
- `validation/nrmse` - Target: < 0.09 (baseline), stretch: < 0.035
- `physics/conservation_gap` - Must stay ≤ 1.0× baseline
- `physics/bc_violation` - Must stay ≤ 1.0× baseline
- `calibration/ece` - Should stay ≤ 1.25× baseline

---

## Expected Timeline

### Phase 1: Data Download (0-30 min)
- Instances clone repo
- Download training data from B2
- Precompute latent caches (CPU)

**Check:** SSH and look for "Precomputing latent caches" in nohup.out

### Phase 2: Operator Training (2-8 hours)
- Main latent operator training
- Round A: ~4-6 hours
- Round B: ~5-7 hours
- Aggressive: ~8-10 hours

**Check:** W&B shows `stage=operator` and loss decreasing

### Phase 3: Diffusion Residual (1-3 hours)
- Few-step residual corrector training
- All configs: 8-12 epochs

**Check:** W&B shows `stage=diff_residual`

### Phase 4: Consistency Distillation (1-3 hours)
- Distill to 1-2 step sampler
- All configs: 8-12 epochs

**Check:** W&B shows `stage=consistency_distill`

### Phase 5: Evaluation (0.5-2 hours)
- Small eval (proxy)
- Full eval (if gates pass)
- Leaderboard update

**Check:** Look for `artifacts/runs/{run_id}/summary.json`

### Phase 6: Auto-Shutdown
- Instances power off automatically
- Check `vastai show instances` - status should be "exited"

---

## Success Criteria (per run)

### Minimum (Improvement over baseline)
- ✅ Training completes without OOM or divergence
- ✅ nRMSE ≤ 0.08 (11% improvement over 0.09 baseline)
- ✅ Conservation gap ≤ 1.0× baseline
- ✅ BC violation ≤ 1.0× baseline

### Target (Strong SOTA)
- ✅ nRMSE ≤ 0.05 (44% improvement)
- ✅ Conservation gap ≤ 0.5× baseline
- ✅ ECE ≤ 1.0× baseline

### Stretch (Full SOTA)
- ✅ **nRMSE < 0.035** (61% improvement)
- ✅ Conservation gap < 0.3× baseline
- ✅ BC violation < 0.5× baseline
- ✅ Rollout horizon@ρ≥0.8 ≥ 64 steps

---

## Troubleshooting

### Instance stuck in "loading"
- Wait 2-5 minutes (normal startup time)
- If > 10 min: `vastai destroy instance <id>` and relaunch

### Instance shows "exited" early
- Check logs: `vastai logs <instance_id>`
- Likely causes: OOM, data download failure, git clone issue
- Relaunch with same config

### Training diverges (loss → NaN)
- Check W&B for explosion point
- Likely: LR too high, grad_clip too loose
- Note which config failed for next iteration

### OOM during training
- Check nvidia-smi via SSH
- If consistent: reduce batch_size by 1-2 in config, increase accum_steps
- Aggressive config most likely to OOM

### No W&B updates after 1 hour
- SSH into instance
- Check: `echo $WANDB_API_KEY` (should be set)
- Check: `tail -100 nohup.out | grep -i wandb`
- May need to manually sync artifacts later

---

## Results Collection

When runs complete (~12-18 hours):

### 1. Pull Leaderboard
```bash
# From local machine (if leaderboard synced to W&B)
# Check W&B artifacts under project

# Or SSH into any instance and download
scp -P 19106 root@ssh1.vast.ai:/workspace/universal_simulator/reports/leaderboard.csv ./reports/
```

### 2. Compare Runs
```bash
# Locally, after runs complete
python scripts/compare_runs.py <best_run_id> <baseline_run_id> \
  --output reports/round_a_b_aggressive_comparison.md
```

### 3. Analyze Best Run
```bash
python scripts/analyze_run.py <best_run_id> \
  --output reports/best_run_analysis.md
```

### 4. Download Checkpoints (optional)
```bash
# If a run achieves target nRMSE < 0.05
scp -P 19XXX root@sshX.vast.ai:/workspace/universal_simulator/checkpoints/*.ckpt ./checkpoints/sota_candidate/
```

---

## Next Steps After Completion

### If nRMSE < 0.05 achieved:
1. ✅ **Declare success** and promote to champion
2. Run extended evaluation on more PDE families
3. Document hyperparameters in playbook
4. Create PR with winning config

### If 0.05 ≤ nRMSE < 0.08:
1. Analyze which axis helped most (optimizer vs capacity)
2. Run **Round C**: Local BO around winner
   - Vary ±20% LR, TTC threshold, lambda_spectral
   - 6-12 trials
3. Consider extending training epochs for best config

### If nRMSE ≥ 0.08 (no improvement):
1. **Debug mode:**
   - Check loss curves for plateaus
   - Verify data quality (inspect samples)
   - Check TTC reward signals
2. **Architecture exploration:**
   - Try U-shape with more stages [1,2,4,8]
   - Add stochastic depth 0.1-0.2
   - Experiment with attention window sizes

---

## Cost Tracking

**Current Run:**
- 3× RTX A6000 @ $0.42/hr × 9hr = $11.34
- 2× RTX A6000 @ $0.42/hr × 10hr = $8.40
- 1× RTX A6000 @ $0.43/hr × 16hr = $6.88
- **Total Estimated: $26.62**

**Budget:** $30
**Remaining:** ~$3.38

---

*Last Updated: 2025-10-18 (launch time)*
*Next Update: Check status in 2-3 hours*
