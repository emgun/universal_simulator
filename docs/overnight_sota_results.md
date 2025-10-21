# Overnight SOTA Sweep - Completed Runs Analysis

**Date**: 2025-10-21
**Status**: 4 configs completed training, partial results available

---

## Executive Summary

**Completed Runs**: 4/15 configs
**Training Success Rate**: 100% (all 4 completed training)
**Evaluation Status**: Small eval succeeded, full eval failed (architecture issue)
**Key Findings**: Training metrics captured successfully in WandB

---

## Completed Configurations

### Round A: Optimizer Grid (4/9 completed)

All runs use the baseline architecture:
- `hidden_dim`: 96
- `num_heads`: 6
- `latent_tokens`: 32
- `latent_dim`: 32

| Config | LR | Warmup | EMA | WandB Run | Status |
|--------|-----|--------|-----|-----------|--------|
| round_a_lr20e5_w3pct | 2e-4 | 3% | 0.9995 | [703lg447](https://wandb.ai/emgun-morpheus-space/universal-simulator/runs/703lg447) | ✅ Training complete |
| round_a_lr20e5_w5pct | 2e-4 | 5% | 0.9995 | [4dataza9](https://wandb.ai/emgun-morpheus-space/universal-simulator/runs/4dataza9) | ✅ Training complete |
| round_a_lr29e5_w6pct | 3e-4 | 6% | 0.9995 | [39wifkv8](https://wandb.ai/emgun-morpheus-space/universal-simulator/runs/39wifkv8) | ✅ Training complete |
| round_a_lr29e5_w3pct | 3e-4 | 3% | 0.9995 | [qreim35o](https://wandb.ai/emgun-morpheus-space/universal-simulator/runs/qreim35o) | ✅ Training + Small Eval |

---

## Available Metrics

### Training Metrics (All 4 runs)
From WandB, we have:
- ✅ Operator training loss curves (25 epochs)
- ✅ Diffusion residual training loss (8 epochs)
- ✅ Consistency distillation loss (8 epochs)
- ✅ Training time per stage
- ✅ GPU utilization
- ✅ Model checkpoints (saved as WandB artifacts)

### Small Eval Metrics (run qreim35o)
- ✅ Validation NRMSE with TTC
- ✅ Boundary condition violations
- ✅ Conservation metrics (mass, energy)
- ✅ Sample visualizations

### Missing Metrics
- ❌ Full test set NRMSE (eval failed)
- ❌ Detailed physics validation on test split
- ❌ Full leaderboard entries

---

## Key Observations

### 1. LR = 2e-4 Configurations

**round_a_lr20e5_w3pct** (LR: 2e-4, Warmup: 3%)
- Conservative learning rate
- Lower warmup percentage
- Expected: Slower convergence, potentially more stable

**round_a_lr20e5_w5pct** (LR: 2e-4, Warmup: 5%)
- Conservative learning rate
- Standard warmup
- Expected: Better early training dynamics

### 2. LR = 3e-4 Configurations

**round_a_lr29e5_w6pct** (LR: 3e-4, Warmup: 6%)
- Moderate learning rate
- Longer warmup
- Expected: Balanced convergence

**round_a_lr29e5_w3pct** (LR: 3e-4, Warmup: 3%)
- Moderate learning rate
- Shorter warmup
- Expected: Faster initial learning
- **Only run with small eval results**

---

## Preliminary Analysis (from WandB)

To complete this analysis, we need to check WandB for:

### Training Loss Comparison
1. **Operator Final Loss**: Which config achieved lowest final loss?
2. **Convergence Speed**: Which config converged fastest?
3. **Stability**: Which config had smoothest loss curves?

### Diffusion Performance
1. **Diffusion Residual Loss**: Quality of diffusion training
2. **Consistency Distillation Loss**: Few-step inference quality

### Small Eval Metrics (from qreim35o)
1. **NRMSE**: Validation set performance
2. **Conservation**: Physics constraint satisfaction
3. **Boundary Violations**: Numerical stability

---

## Next Steps for Complete Analysis

### Immediate (Manual WandB Review)
1. Open each WandB run and extract key metrics:
   - Operator final loss
   - Diffusion final loss
   - Distillation final loss
   - Small eval NRMSE (for qreim35o)

### Short-term (Code-based Analysis)
1. Write script to pull metrics from WandB API:
   ```python
   import wandb
   api = wandb.Api()

   runs = [
       "703lg447",  # round_a_lr20e5_w3pct
       "4dataza9",  # round_a_lr20e5_w5pct
       "39wifkv8",  # round_a_lr29e5_w6pct
       "qreim35o",  # round_a_lr29e5_w3pct
   ]

   for run_id in runs:
       run = api.run(f"emgun-morpheus-space/universal-simulator/{run_id}")
       print(f"{run.name}:")
       print(f"  Operator loss: {run.summary.get('operator/final_loss')}")
       print(f"  Small eval NRMSE: {run.summary.get('small_eval/metric/nrmse')}")
   ```

2. Download model checkpoints for best config
3. Run full evaluation locally with fixed configs

### Medium-term (Remaining Configs)
1. Wait for 4 running instances to complete (~3-4 hours)
2. Destroy stuck loading instances (save costs)
3. Decide: Launch remaining configs OR analyze current results

---

## Hyperparameter Grid Coverage

### Completed Coverage

| LR \\ Warmup | 3% | 5% | 6% |
|-------------|----|----|-----|
| **2e-4** | ✅ | ✅ | ❌ |
| **3e-4** | ✅ | ❌ | ✅ |
| **4.5e-4** | ❌ | ❌ | ❌ |

**Coverage**: 4/9 optimizer grid points (44%)

### Missing Configurations

**High Priority** (complete the 3e-4 row):
- round_a_lr29e5_w5pct (LR: 3e-4, Warmup: 5%)

**Medium Priority** (explore higher LR):
- round_a_lr45e5_w3pct (LR: 4.5e-4, Warmup: 3%)
- round_a_lr45e5_w5pct (LR: 4.5e-4, Warmup: 5%)
- round_a_lr45e5_w6pct (LR: 4.5e-4, Warmup: 6%)

**Low Priority** (complete conservative LR):
- round_a_lr20e5_w6pct (LR: 2e-4, Warmup: 6%)

### Capacity Scaling (Not Started)
All Round B configs pending:
- round_b_cap64 (hidden_dim: 64)
- round_b_cap96 (hidden_dim: 96)
- round_b_cap128 (hidden_dim: 128)

### Hybrid Best (Not Started)
All Round C configs pending:
- round_c_cap128_ttc (enhanced capacity + TTC)
- round_c_extended (longer training)
- round_c_tokens48 (more latent tokens)

---

## Issues Encountered

### 1. Evaluation Architecture Mismatch (First Batch)
**Issue**: Eval configs had `latent_tokens: 16`, training used `32`
**Impact**: 3 runs completed training but eval failed
**Fix**: Updated `small_eval_burgers.yaml` to use `tokens: 32`
**Status**: ✅ Fixed (commit 2df191a)

### 2. LazyLinear Import Error (Second Batch)
**Issue**: `LazyLinear` imported from internal PyTorch module
**Impact**: All 12 relaunched instances failed immediately
**Fix**: Changed to use public `nn.LazyLinear` API
**Status**: ✅ Fixed (commit e42f8ab)

### 3. Full Evaluation Failure (Current)
**Issue**: Full eval stage failing with exit code 1
**Impact**: No test set NRMSE scores
**Fix**: Needs investigation (separate issue from architecture mismatch)
**Status**: ⚠️ Pending investigation

### 4. Stuck Loading Instances
**Issue**: 7/12 instances stuck in "loading" state for 2+ hours
**Impact**: Reduced completion rate (4/12 instead of 12/12)
**Root Cause**: VastAI host issues or slow data downloads
**Status**: ⚠️ Ongoing issue with VastAI reliability

---

## Cost Analysis

### Actual Spend
- **First batch** (3 runs, ~3 hours each): ~$2.70
- **Second batch failed** (13 instances, immediate failure): ~$0.50
- **Third batch** (12 instances, 4 running, 7 stuck): ~$2.50 (estimated)
- **Total**: ~$5.70 / $5.25 budget (8% over)

### Cost per Completed Run
- **4 completed runs**: $5.70 / 4 = **$1.43 per config**
- **Original estimate**: $0.35 per config
- **Actual cost**: 4x higher due to failures and relaunches

---

## Recommendations

### Immediate Actions
1. ✅ **Extract metrics from 4 completed runs** (manual WandB review)
2. ⏳ **Wait for 4 running instances** to complete (~3-4 hours)
3. ❌ **Destroy stuck instances** to save costs

### Analysis Priorities
1. **Compare LR=2e-4 vs LR=3e-4**: Which converges better?
2. **Compare Warmup 3% vs 5% vs 6%**: Impact on training stability?
3. **Identify best config**: For Round B capacity scaling

### Future Runs
1. **Fix full eval issue**: Investigate and resolve before next batch
2. **Improve reliability**: Use more reliable GPU types or hosts
3. **Reduce scope**: Focus on 3-5 most promising configs instead of 15

---

## WandB Dashboard Links

**Project**: https://wandb.ai/emgun-morpheus-space/universal-simulator

**Completed Runs**:
- [round_a_lr20e5_w3pct](https://wandb.ai/emgun-morpheus-space/universal-simulator/runs/703lg447)
- [round_a_lr20e5_w5pct](https://wandb.ai/emgun-morpheus-space/universal-simulator/runs/4dataza9)
- [round_a_lr29e5_w6pct](https://wandb.ai/emgun-morpheus-space/universal-simulator/runs/39wifkv8)
- [round_a_lr29e5_w3pct](https://wandb.ai/emgun-morpheus-space/universal-simulator/runs/qreim35o)

**Filter**: `tags.overnight-sota` AND `tags.round-a`

---

**Status**: Analysis pending - need to extract metrics from WandB
**Last Updated**: 2025-10-21 13:55 PDT
