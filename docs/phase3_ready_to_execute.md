# Phase 3 Ablation Study - Ready to Execute

**Status**: ✅ All infrastructure complete, configs validated
**Date**: 2025-11-04
**Estimated Total Cost**: ~$4.00-4.50
**Estimated Total Time**: ~2-3 hours (3 runs)

---

## ✅ Completed Infrastructure

- [x] **Core Components**: DropPath, StandardSelfAttention (Phase 3.1)
- [x] **Architecture**: PureTransformer, TransformerBlock, LatentOperator integration (Phase 3.2)
- [x] **Configs**: 3 validated ablation configs ready (Phase 3.3)
- [x] **Analysis Script**: `scripts/analyze_phase3_ablation.py` created
- [x] **Execution Guide**: `docs/phase3_execution_guide.md` created
- [x] **Validation**: All configs pass 31-32 checks

---

## 🚀 Execute Experiments (Manual Step)

### Option A: Parallel Launch (Fastest, ~$4.50 total)

Launch all 3 experiments simultaneously (recommended if budget allows):

```bash
# Terminal 1: 128 tokens pure (standard attention)
python scripts/vast_launch.py launch \
  --config configs/train_burgers_upt_128tokens_pure.yaml \
  --auto-shutdown

# Terminal 2: 256 tokens pure (UPT recommendation)
python scripts/vast_launch.py launch \
  --config configs/train_burgers_upt_256tokens_pure.yaml \
  --auto-shutdown

# Terminal 3: 128 tokens pure (channel-separated)
python scripts/vast_launch.py launch \
  --config configs/train_burgers_upt_128tokens_channel_sep.yaml \
  --auto-shutdown
```

**Monitoring**:
```bash
# Check instances
vastai show instances

# Watch logs (run in each terminal)
vastai logs <instance_id> -f
```

### Option B: Sequential Launch (Lower cost, ~$4.00 total)

Launch one at a time to minimize concurrent instance costs:

```bash
# Run 1: 128 tokens pure (standard) - ~35-40 min
python scripts/vast_launch.py launch \
  --config configs/train_burgers_upt_128tokens_pure.yaml \
  --auto-shutdown

# Wait for completion, then run 2: 256 tokens pure - ~55-60 min
python scripts/vast_launch.py launch \
  --config configs/train_burgers_upt_256tokens_pure.yaml \
  --auto-shutdown

# Wait for completion, then run 3: 128 tokens channel-sep - ~35-40 min
python scripts/vast_launch.py launch \
  --config configs/train_burgers_upt_128tokens_channel_sep.yaml \
  --auto-shutdown
```

---

## 📊 After Experiments Complete

### 1. Verify Completion

Check WandB dashboard for all 3 runs:
- https://wandb.ai/emgun-morpheus-space/universal-simulator
- Filter by tags: `phase3`, `upt`
- Verify state: `finished` (not `failed` or `crashed`)

### 2. Run Analysis

```bash
python scripts/analyze_phase3_ablation.py --output-dir reports/phase3
```

**Outputs**:
- `reports/phase3/architecture_comparison.csv` - Full comparison data
- `reports/phase3/architecture_comparison.png` - Visualization plots
- `reports/phase3/ablation_report.md` - Hypothesis test results

### 3. Review Hypothesis Tests

The analysis script will automatically test:

**H1**: Pure transformer matches U-shaped at 128 tokens (within 5%)
- ✅ PASS if pure NRMSE ≤ 0.0606
- Baseline: U-shaped NRMSE = 0.0577 (Phase 2)

**H2**: Pure transformer outperforms U-shaped at 256 tokens (≥10% improvement)
- ✅ PASS if pure NRMSE ≤ 0.0536
- Baseline: U-shaped NRMSE = 0.0596 (Phase 2)

**H3**: Standard attention comparable to channel-separated (within 5%)
- ✅ PASS if standard and channel-separated NRMSEs within 5%
- Both at 128 tokens, pure architecture

---

## 🎯 Expected Outcomes

**If hypotheses pass**:
1. Update leaderboard with winning configurations
2. Promote best config to production
3. Update architecture selection guidelines
4. Proceed to Phase 3.5 (Documentation)

**If hypotheses fail**:
1. Analyze failure modes (loss curves, gradients, metrics)
2. Iterate with hyperparameter adjustments
3. Document learnings in reports/phase3/

---

## 📝 Phase 2 Baseline Results (for comparison)

| Config | Tokens | Architecture | NRMSE | Improvement |
|--------|--------|--------------|-------|-------------|
| `ablation_upt_128tokens_fixed` | 128 | U-shaped | 0.0577 | 20% |
| `ablation_upt_256tokens_fixed` | 256 | U-shaped | 0.0596 | 17% |

Phase 3 should show pure transformer matching/exceeding these results at respective token counts.

---

## 🔧 Troubleshooting

**Instance stuck in "loading"**:
```bash
vastai destroy instance <ID>
# Relaunch experiment
```

**OOM (Out of Memory)**:
- Check `nvidia-smi` memory usage
- Config already optimized for A100 40GB

**Operator not converging**:
- Verify operator final loss < 0.001
- Check WandB loss curves for anomalies

**SSH access for debugging**:
```bash
vastai ssh <instance_id>
cd /workspace/universal_simulator
tail -100 nohup.out
```

---

## 📚 Reference Documentation

- **Detailed Execution Guide**: `docs/phase3_execution_guide.md`
- **Implementation Plan**: `thoughts/shared/plans/2025-11-03-upt-phase3-architecture-simplification.md`
- **Phase 2 Results**: `reports/research/runs_comparison.md`
- **UPT Integration Analysis**: `UPT_docs/UPT_INTEGRATION_ANALYSIS.md`

---

## ✅ Pre-Launch Checklist

- [x] All 3 configs validated
- [x] VastAI credentials configured (`python scripts/vast_launch.py setup-env`)
- [x] WandB API key set in environment
- [x] Analysis script tested
- [ ] Budget approved (~$4-4.50)
- [ ] Terminal windows ready for monitoring

---

**Ready to launch!** Choose Option A (parallel) or Option B (sequential) above.
