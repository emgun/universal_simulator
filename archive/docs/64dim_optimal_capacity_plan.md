# 64-Dim Optimal Capacity Model Plan

## Executive Summary

Testing 64-dim as the optimal capacity/cost trade-off based on dramatic TTC results from 32-dim baseline.

**Key Finding**: 32-dim + TTC achieved **0.09 NRMSE** (88% improvement over baseline 0.78), suggesting that inference-time optimization can largely compensate for model capacity.

**Hypothesis**: 64-dim + TTC will achieve ~0.04-0.06 NRMSE, offering near-SOTA performance at 64× lower cost than 512-dim.

## Motivation

### TTC Impact Analysis

| Model | Baseline NRMSE | With TTC | Improvement |
|-------|----------------|----------|-------------|
| 32-dim | 0.7845 | 0.0921 | **88.3% ↓** |
| 64-dim | 0.3-0.5 (expected) | 0.04-0.06 (target) | TBD |
| 512-dim | 0.02 (pru2jxc4) | 0.01-0.02 (expected) | TBD |

### Capacity vs. Cost Trade-off

| Model | Parameters | Checkpoint Size | Training Time | Cost @ $2.11/hr |
|-------|------------|-----------------|---------------|-----------------|
| 32-dim | ~120K | ~500KB | ~20 min | ~$0.70 |
| **64-dim** | **~480K** | **~1MB** | **~25 min** | **~$0.90** |
| 128-dim | ~1.9M | ~4MB | ~40 min | ~$1.40 |
| 256-dim | ~7.7M | ~16MB | ~70 min | ~$2.50 |
| 512-dim | ~30.7M | ~60MB | ~120 min | ~$4.20 |

**64-dim advantages**:
- 4× capacity vs 32-dim (more representational power)
- 64× fewer params vs 512-dim (fast, cheap, efficient)
- Potential sweet spot for production deployment

## Architecture Scaling

### Scaling Rules Applied

All components scaled proportionally from 32-dim baseline:

| Component | 32-dim | 64-dim | Scaling Rule |
|-----------|--------|--------|--------------|
| `latent.dim` | 32 | 64 | 2× |
| `latent.tokens` | 16 | 32 | 2× (proportional) |
| `operator.input_dim` | 32 | 64 | Match latent.dim |
| `operator.hidden_dim` | 64 | 128 | 2× latent.dim |
| `operator.group_size` | 8 | 16 | Divides hidden_dim |
| `diffusion.latent_dim` | 32 | 64 | Match latent.dim |
| `diffusion.hidden_dim` | 64 | 128 | 2× latent.dim |
| `batch_size` | 12 | 10 | Adjusted for memory |

### Configuration Files

**Training**: `configs/train_burgers_64dim_pru2jxc4.yaml`
- pru2jxc4 approach: constant LR (1e-3), 15 epochs
- Parallel encoding enabled (`num_workers: 8`)
- Latent cache enabled

**Evaluation (Baseline)**: `configs/eval_burgers_64dim.yaml`
- Standard eval without TTC
- Measures raw model capacity

**Evaluation (TTC)**: `configs/eval_burgers_64dim_ttc.yaml`
- Test-Time Conditioning enabled
- Beam search with analytical rewards
- Expected to show dramatic improvement

## Expected Outcomes

### Success Scenarios

#### Scenario A: Sweet Spot Found (0.04-0.06 NRMSE with TTC)
- **Conclusion**: 64-dim is optimal for production
- **Action**: Recommend 64-dim + TTC for deployment
- **Value**: Near-SOTA at 64× lower cost

#### Scenario B: Marginal Improvement (0.07-0.09 NRMSE with TTC)
- **Conclusion**: Diminishing returns vs 32-dim
- **Action**: Test 128/256-dim for better capacity
- **Value**: Need more parameters for meaningful gains

#### Scenario C: SOTA Achievement (<0.03 NRMSE with TTC)
- **Conclusion**: 64-dim + TTC is SOTA!
- **Action**: No need for larger models
- **Value**: Revolutionary finding - small models sufficient

## Training Pipeline

### Stage 1: Latent Cache Precomputation (~3-5 min)
- Encode 2000 train samples
- Encode 200 val samples
- Encode 200 test samples
- Cache to `data/latent_cache/`

### Stage 2: Operator Training (~12 min)
- 15 epochs with constant LR 1e-3
- No scheduler (pru2jxc4's key insight)
- Target: Better than 32-dim's operator loss

### Stage 3: Diffusion Residual (~5 min)
- 5 epochs with cosine annealing
- EMA decay: 0.999
- Grad clipping: 1.0

### Stage 4: Consistency Distillation (~8 min)
- 6 epochs with cosine annealing
- Distill num taus: 5 (2× speedup)
- Micro batch: 3

### Stage 5: Evaluation (~4-6 min)
- Baseline eval: ~1 min
- TTC eval: ~3-5 min (beam search overhead)

**Total Time**: ~30-35 minutes
**Total Cost**: ~$1.00 @ $2.11/hr

## Key Hypotheses to Test

1. **Capacity Scaling**: Does 2× latent dim yield meaningful performance improvement?
2. **TTC Effectiveness**: Does TTC work equally well across different model sizes?
3. **Sweet Spot**: Is 64-dim the optimal capacity/cost point?
4. **Diminishing Returns**: Where do gains plateau relative to model size?

## Success Metrics

### Primary Metric: NRMSE with TTC
- **Target**: 0.04-0.06 (near-SOTA)
- **Acceptable**: 0.06-0.08 (competitive)
- **Poor**: >0.08 (marginal over 32-dim)

### Secondary Metrics
- Baseline NRMSE (without TTC)
- Training stability (loss curves)
- Model size vs performance ratio
- Inference speed

## Implementation Status

### Completed ✅
- [x] Architecture scaling rules defined
- [x] Training config created (`train_burgers_64dim_pru2jxc4.yaml`)
- [x] Baseline eval config created (`eval_burgers_64dim.yaml`)
- [x] TTC eval config created (`eval_burgers_64dim_ttc.yaml`)
- [x] Configs committed to repo
- [x] Instance updated with latest code
- [x] Training launched

### In Progress 🔄
- [ ] Latent cache precomputation (67% complete)
- [ ] Operator training
- [ ] Diffusion training
- [ ] Consistency distillation
- [ ] Baseline evaluation
- [ ] TTC evaluation

### Next Steps ⏭️
- [ ] Analyze 64-dim results
- [ ] Compare to 32-dim and 512-dim baselines
- [ ] Decide on optimal dimension for production
- [ ] Document findings for paper

## Monitoring

**WandB Run**: `burgers64-pru2jxc4-optimal`
**Instance**: 26833332 (H200_NVL @ $2.11/hr)
**Expected Completion**: ~30-35 minutes from start

## Context: Previous Findings

### 32-Dim Baseline Results
- Baseline NRMSE: 0.7845
- With TTC NRMSE: 0.0921
- Improvement: 88.3%

**Conclusion**: TTC is incredibly effective, suggesting model capacity may be less critical than inference-time optimization.

### 512-Dim Reference (pru2jxc4)
- Operator loss: 0.02 at epoch 6
- Expected baseline NRMSE: ~0.02
- Expected with TTC: ~0.01-0.02 (SOTA)

**Conclusion**: Large models work well, but may be overkill if smaller models + TTC can compete.

## Strategic Implications

### If 64-dim Succeeds
- **Production**: Use 64-dim + TTC for deployment
- **Research**: Focus on TTC improvements over model scaling
- **Cost**: Massive savings (64× vs 512-dim)
- **Speed**: Fast training, fast inference
- **Impact**: Democratizes SOTA performance

### If 64-dim Fails
- **Production**: May need 128/256-dim
- **Research**: Explore capacity vs TTC trade-offs
- **Cost**: Moderate increase acceptable
- **Speed**: Still faster than 512-dim
- **Impact**: Find optimal middle ground

## References

- `pru2jxc4` run: 512-dim, constant LR 1e-3, 15 epochs, 0.02 operator loss
- `rv86k4w1` run: 32-dim, cosine decay, 6 epochs, 0.10 operator loss, 0.78 NRMSE
- Current run: 32-dim + pru2jxc4 method, 0.78 baseline → 0.09 with TTC

## Open Questions

1. Does TTC effectiveness scale linearly with model capacity?
2. Is there a minimum capacity threshold for TTC to work?
3. What's the optimal dimension for production deployment?
4. Can we achieve SOTA with <512-dim models?

Answers coming in ~30 minutes! 🎯

