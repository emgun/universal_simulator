# Distillation Stage Optimization Guide

## Problem

The consistency distillation stage is significantly slower than other training stages due to:

1. **Multiple noise levels sampling** (`distill_num_taus`): Each sample is evaluated at N different noise levels
2. **Micro-batch accumulation** (`distill_micro_batch`): Batch is split into smaller chunks for gradient accumulation
3. **Small batch size**: GPU not fully saturated, leading to low utilization

## Mathematics

**Computational cost per epoch:**
```
iterations = num_samples / batch_size
micro_batches_per_iter = batch_size / distill_micro_batch
forward_passes = iterations × micro_batches_per_iter × distill_num_taus
```

**Example (before optimization):**
```
2000 / 6 × (6/3) × 5 = 333 × 2 × 5 = 3,330 forward passes
```

**After optimization:**
```
2000 / 12 × (12/4) × 4 = 167 × 3 × 4 = 2,004 forward passes (-40%)
```

## Applied Optimizations (v1 → v2)

| Parameter | Before | After | Impact |
|-----------|--------|-------|--------|
| `batch_size` | 6 | 12 | +100% GPU saturation |
| `distill_micro_batch` | 3 | 4 | +33% efficiency |
| `distill_num_taus` | 5 | 4 | -20% compute |
| `accum_steps` | - | 2 | Effective batch=24 |

**Expected speedup:** 2-3×  
**Expected time:** 10-15 min (down from 30-40 min)  
**GPU utilization:** 75-80% (up from 61%)  
**Quality impact:** Negligible (< 1% NRMSE change)

## Tuning Guidelines

### When to increase batch_size:
- ✅ GPU util < 70%
- ✅ GPU has memory headroom
- ✅ Want faster training

### When to reduce distill_num_taus:
- ✅ Distill stage is bottleneck
- ✅ 3-5 is standard range (4 is sweet spot)
- ⚠️ Below 3 may hurt quality

### When to increase distill_micro_batch:
- ✅ Want fewer gradient accumulation steps
- ✅ GPU memory allows
- ⚠️ Must divide batch_size evenly

### Relationship between parameters:
```python
# Effective batch size (for learning dynamics)
effective_batch = batch_size × accum_steps

# Memory usage per iteration
memory ∝ distill_micro_batch × distill_num_taus

# Iterations per epoch
iterations = num_samples / batch_size

# GPU saturation
gpu_util ∝ batch_size (up to hardware limit)
```

## Trade-offs

**Speed vs Quality:**
- More `distill_num_taus` → Better consistency modeling, slower training
- Larger `batch_size` → Better GPU util, more memory
- Larger `distill_micro_batch` → Fewer accum steps, more memory per iter

**Memory vs Speed:**
- Small `distill_micro_batch` → Less memory, more accumulation overhead
- Large `distill_micro_batch` → More memory, cleaner gradients

## Recommended Ranges

| GPU | batch_size | micro_batch | num_taus | Speedup vs baseline |
|-----|------------|-------------|----------|---------------------|
| A100 40GB | 12-16 | 4-6 | 4 | 2-3× |
| H100 80GB | 20-24 | 8-10 | 4-5 | 3-4× |
| RTX 4090 | 8-12 | 4 | 3-4 | 1.5-2× |

## Validation

After changing parameters, verify:
1. GPU utilization increased (check `nvidia-smi`)
2. Training time decreased (check WandB)
3. Final NRMSE unchanged (< 1% difference)
4. No OOM errors

## References

- Original paper: Song et al. "Consistency Models" (2023)
- Typical `num_taus` in literature: 3-7
- Our baseline: NRMSE 0.0921 with `num_taus=5`
- Our target: NRMSE < 0.095 with `num_taus=4`

