# Phase 3 Ablation Study - Execution Guide

**Date**: 2025-11-04
**Status**: Ready to execute
**Duration**: ~2-3 hours total (3 runs)

---

## Overview

Phase 3 tests pure stacked transformer architecture against U-shaped baseline at different token counts and with different attention mechanisms.

**Hypothesis Tests**:
1. **H1**: Pure transformer matches U-shaped at 128 tokens (within 5%)
2. **H2**: Pure transformer outperforms U-shaped at 256 tokens (≥10% improvement)
3. **H3**: Standard attention comparable to channel-separated (within 5%)

---

## Ablation Matrix

| # | Config | Tokens | Architecture | Attention | Purpose |
|---|--------|--------|--------------|-----------|---------|
| 1 | `train_burgers_upt_128tokens_pure.yaml` | 128 | pdet_stack | standard | Test pure at Phase 2 winner token count |
| 2 | `train_burgers_upt_256tokens_pure.yaml` | 256 | pdet_stack | standard | UPT recommendation (256-512 tokens) |
| 3 | `train_burgers_upt_128tokens_channel_sep.yaml` | 128 | pdet_stack | channel_separated | Compare attention mechanisms |

**Baselines (Phase 2 - already completed)**:
- 128 tokens U-shaped: NRMSE 0.0577 (from `ablation_upt_128tokens_fixed.yaml`)
- 256 tokens U-shaped: NRMSE 0.0596 (from `ablation_upt_256tokens_fixed.yaml`)

---

## Execution Steps

### Step 1: Verify Configs

```bash
# Validate all Phase 3 configs
python scripts/validate_config.py configs/train_burgers_upt_128tokens_pure.yaml
python scripts/validate_config.py configs/train_burgers_upt_256tokens_pure.yaml
python scripts/validate_config.py configs/train_burgers_upt_128tokens_channel_sep.yaml

# All should show: "✅ Config is valid and ready for training!"
```

### Step 2: Launch Experiments

**Option A: Launch all 3 in parallel** (recommended if budget allows):

```bash
# Run 1: 128 tokens pure (standard attention)
python scripts/vast_launch.py launch \
  --config configs/train_burgers_upt_128tokens_pure.yaml \
  --auto-shutdown

# Run 2: 256 tokens pure (UPT recommendation)
python scripts/vast_launch.py launch \
  --config configs/train_burgers_upt_256tokens_pure.yaml \
  --auto-shutdown

# Run 3: 128 tokens pure (channel-separated)
python scripts/vast_launch.py launch \
  --config configs/train_burgers_upt_128tokens_channel_sep.yaml \
  --auto-shutdown
```

**Option B: Launch sequentially** (lower cost):

```bash
# Launch one at a time, wait for completion
python scripts/vast_launch.py launch \
  --config configs/train_burgers_upt_128tokens_pure.yaml \
  --auto-shutdown

# Wait ~30-40 minutes for completion, then launch next...
```

### Step 3: Monitor Progress

```bash
# Check VastAI instances
vastai show instances

# Watch logs
vastai logs <instance_id> -f

# Check WandB dashboard
# https://wandb.ai/emgun-morpheus-space/universal-simulator
# Filter by tags: phase3, upt
```

### Step 4: Validate Results

After all runs complete, check that each reached expected milestones:

**Operator Stage**:
- [ ] Final loss < 0.001 (typically ~0.0002)
- [ ] No NaN/Inf gradients
- [ ] Training completed all epochs

**Diffusion Stage**:
- [ ] Loss converged (final < 0.01)
- [ ] No instabilities

**Evaluation**:
- [ ] Baseline NRMSE logged
- [ ] TTC NRMSE logged
- [ ] Physics metrics logged (conservation gap, BC violation)

### Step 5: Analyze Results

```bash
# Run analysis script
python scripts/analyze_phase3_ablation.py --output-dir reports/phase3

# This generates:
#   - reports/phase3/architecture_comparison.csv
#   - reports/phase3/architecture_comparison.png
#   - reports/phase3/ablation_report.md
```

---

## Expected Outcomes

### Success Criteria

**Primary Hypothesis Tests**:
1. **H1: Pure matches U-shaped at 128 tokens**
   - ✅ PASS if: Pure NRMSE within 5% of U-shaped (0.0577)
   - Target: Pure NRMSE ≤ 0.0606

2. **H2: Pure outperforms U-shaped at 256 tokens**
   - ✅ PASS if: Pure NRMSE ≥10% better than U-shaped (0.0596)
   - Target: Pure NRMSE ≤ 0.0536

3. **H3: Standard attention comparable to channel-separated**
   - ✅ PASS if: Standard NRMSE within 5% of channel-separated
   - Both at 128 tokens, pure architecture

**Performance Benchmarks**:
- [ ] Training time scales reasonably with token count
- [ ] Memory usage within A100 40GB budget
- [ ] Convergence speed comparable to U-shaped
- [ ] No gradient instabilities or NaN losses

---

## Troubleshooting

### Issue: Instance stuck in "loading"
```bash
vastai destroy instance <ID>
# Relaunch
```

### Issue: Training errors
```bash
# SSH into instance
vastai ssh <instance_id>

# Check logs
tail -100 /workspace/universal_simulator/logs/*.log

# Check GPU memory
nvidia-smi
```

### Issue: OOM (Out of Memory)
- Reduce batch_size in config
- Increase grad_accum_steps to maintain effective batch size

### Issue: Operator not converging
- Check learning rate (should be 8e-4 to 1e-3)
- Check inverse loss weights (should be 0.5)
- Verify dimension matching in config

---

## Cost Estimate

**Per run** (assuming A100 @ $1.89/hr):
- 128 tokens: ~35-40 min = ~$1.10-1.26
- 256 tokens: ~55-60 min = ~$1.73-1.89

**Total Phase 3 ablation**: ~$4.00-4.50

---

## Analysis Checklist

After runs complete:

- [ ] All 3 runs finished successfully (state='finished')
- [ ] Operator final loss < 0.001 for all runs
- [ ] Evaluation metrics logged for all runs
- [ ] No NaN/Inf gradients in any run
- [ ] Checkpoints saved and loadable
- [ ] Analysis script runs without errors
- [ ] Hypothesis tests evaluated (H1, H2, H3)
- [ ] Plots generated
- [ ] Report generated

---

## Post-Analysis Actions

Based on results:

### If hypotheses pass:

1. **Update leaderboard** with winning configurations
2. **Promote configs** to production:
   ```bash
   # If 256-token pure wins
   python scripts/promote_config.py \
     configs/train_burgers_upt_256tokens_pure.yaml \
     --update-leaderboard
   ```

3. **Document findings** in:
   - Phase 3 implementation plan (mark complete)
   - CLAUDE.md (update architecture guidance)
   - configs/README.md (mark as production-ready)

### If hypotheses fail:

1. **Analyze failure modes**:
   - Loss curves: Convergence issues?
   - Gradients: Instability?
   - Metrics: Systematic underperformance?

2. **Iterate if needed**:
   - Tune hyperparameters (learning rate, drop-path)
   - Adjust depth (try 12 layers)
   - Test different drop-path rates

3. **Document learnings** in reports/phase3/

---

## Next Steps (Phase 3.5)

After analysis complete:

1. Create architecture selection guide (`docs/architecture_selection_guide.md`)
2. Update main UPT plan with Phase 3 completion status
3. Promote winning configurations to production
4. Update documentation with Phase 3 findings

---

## References

- **Implementation Plan**: `thoughts/shared/plans/2025-11-03-upt-phase3-architecture-simplification.md`
- **Phase 2 Results**: `reports/research/runs_comparison.md`
- **UPT Integration Analysis**: `UPT_docs/UPT_INTEGRATION_ANALYSIS.md`
- **Configs**: `configs/train_burgers_upt_*.yaml`

---

**Questions? Issues?**

Contact: emgun
Date: 2025-11-04
