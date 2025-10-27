# UPT Optimized Training Runs

## Active Instances

### Instance 27288264 (RTX 5090 - FASTEST)
- **GPU**: RTX 5090 (203 DLP, ~2-3x faster than RTX 4090)
- **Disk**: 2TB
- **Cost**: $1.02/hr
- **Config**: `configs/train_burgers_upt_optimized.yaml`
- **Branch**: feature--UPT (commit faf7b8c)
- **Status**: Loading
- **Monitors**: fd1c49 (startup, 4min), 6ddff1 (training, 15min)

**Expected Timeline:**
- Cache build: ~45-60 min (UPT-aware cache with physical fields)
- Operator training: ~10-15 min (25 epochs with compile)
- Diffusion: ~3-5 min (8 epochs)
- Consistency: ~3-5 min (8 epochs)
- **Total: ~1-1.5 hours**

### Instance 27287960 (RTX 4090 - FAST)
- **GPU**: RTX 4090 (24GB VRAM)
- **Disk**: 2TB
- **Cost**: $0.85/hr
- **Config**: `configs/train_burgers_upt_optimized.yaml`
- **Branch**: feature--UPT (commit faf7b8c)
- **Status**: Cache encoding - 10% (52/500 samples)
- **Monitors**: 916ef3 (startup), 678b7e (training - completed)
- **Cache Progress**: 9.3s/sample = ~1.3 hours to build cache

**Expected Timeline:**
- Cache build: ~1.3 hours (UPT-aware cache with physical fields)
- Operator training: ~12-20 min (25 epochs with compile)
- Diffusion: ~4-6 min (8 epochs)
- Consistency: ~4-6 min (8 epochs)
- **Total: ~2-3 hours**

## Configuration Details

### UPT-Aware Cache (Enabled)
```yaml
training:
  latent_cache_dir: data/latent_cache
  num_workers: 4
  use_parallel_encoding: true
  pin_memory: true
  prefetch_factor: 2
```

**What's cached:**
- Latent vectors (16-dim)
- Physical fields (rho, e) for inverse losses
- Query coordinates
- Stored in float16 (~240MB total)

### torch.compile (Enabled)
```yaml
training:
  compile: true
```

**Benefits:**
- ~1.5-2x training speedup
- Requires ~100GB disk for kernel cache

### Expected Performance vs Fallback
- **Fallback** (27283596): No cache, no compile = baseline speed (~3-4 hours)
- **Optimized** (27288264, 27287960): Cache + compile = 4.5-7.5x faster training

## Disk Space Breakdown

**2TB Allocation:**
- Raw data: ~0.6GB
- UPT cache (float16): ~0.24GB
- torch.compile kernels: ~100GB
- Working space (WandB, checkpoints, temp): ~50GB
- **Total used: ~150GB**
- **Headroom: ~1850GB (16x safety margin)**

## Previous Failed Instances

All failures resolved by using 2TB disk + UPT-aware cache:

1. **27283349** - Disk full from torch.compile (64GB too small)
2. **27281504** - Disk full from legacy cache (no physical fields)
3. **27279702** - Wrong branch (feature/sota_burgers_upgrades)
4. **27262305** - Collate function mismatch (fixed in bc974c3)
5. **27252152** - Crashed in diffusion (incomplete collate fix)

## Success Criteria

### Minimum (Code Works)
- ✅ Cache builds successfully with UPT physical fields
- ✅ torch.compile succeeds without disk errors
- ✅ All training stages complete
- ✅ Auto-stop succeeds

### Target (UPT Benefits Visible)
- ✅ `L_inv_enc` and `L_inv_dec` logged and decreasing
- ✅ Final operator loss < 0.0005
- ✅ Evaluation NRMSE < 0.10
- ✅ Training completes in ~1-3 hours (vs 3-4 hours for fallback)

## WandB Tracking

**Project**: universal-simulator
**Entity**: emgun-morpheus-space
**Run names**: TBD (check logs after startup)

**Metrics to monitor:**
- `training/operator/L_inv_enc` - Inverse encoding loss
- `training/operator/L_inv_dec` - Inverse decoding loss
- `training/operator/L_forward` - Forward prediction loss
- `training/operator/loss` - Total operator loss

## Next Steps

1. Monitor cache build progress on both instances
2. Verify torch.compile doesn't fill disk
3. Compare training speed between RTX 5090 and RTX 4090
4. Update UPT_PRODUCTION_RUN.md with final results
5. Document optimized config as new golden standard if successful
