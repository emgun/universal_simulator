# DDP Crash Fix - Test Plan

**Date:** 2025-11-18
**Branch:** feature/distributed-training-ddp
**Commit:** d74607a

## What Was Fixed

✅ **Root Cause:** CUDA contamination in cache files + silent exit masking errors
✅ **Solution:** CPU verification + proper error reporting
✅ **Expected Result:** 4x speedup with multi-worker data loading

## Quick Test (Local - 5 minutes)

Test the error reporting fix locally:

```bash
# This will show you the actual error now (instead of silent crash)
python scripts/train.py \
  --config configs/train_burgers_upt_full_ddp_ramdisk.yaml \
  --stage operator

# Expected: Clear error message about CUDA contamination
# (if cache exists and has CUDA tensors)
```

## Full Test (VastAI - 30 minutes)

### Option A: Test with Existing Cache (See the Error)

```bash
# Launch instance
python scripts/vast_launch.py launch \
  --config configs/train_burgers_upt_full_ddp_ramdisk.yaml \
  --auto-shutdown

# Expected outcome:
# - Clear error message: "Cache file contains CUDA tensors!"
# - Instructions: "Delete cache and regenerate with --device cpu"
# - No more silent crashes!
```

### Option B: Fix Cache First (Recommended)

**Step 1:** SSH into a VastAI instance (or run locally if you have the data)

```bash
vastai search offers 'gpu_ram >= 48 reliability > 0.95 num_gpus=2 disk_space >= 64' --order 'dph_total'
vastai create instance <INSTANCE_ID>
vastai ssh <INSTANCE_ID>
```

**Step 2:** Clone the repo and checkout the fix

```bash
cd /workspace
git clone https://github.com/YOUR_USERNAME/universal_simulator.git
cd universal_simulator
git checkout feature/distributed-training-ddp
git pull origin feature/distributed-training-ddp
```

**Step 3:** Run the cache fix script

```bash
# Interactive mode (prompts for confirmation)
./scripts/fix_cuda_cache.sh

# Or with custom paths
./scripts/fix_cuda_cache.sh /dev/shm/latent_cache configs/train_burgers_upt_full_ddp_ramdisk.yaml
```

**Step 4:** Launch training with clean cache

```bash
python scripts/run_fast_to_sota.py \
  --train-config configs/train_burgers_upt_full_ddp_ramdisk.yaml \
  --skip-small-eval \
  --skip-full-eval
```

## Expected Results

### Before (Broken)
- ❌ Silent crash during dataset initialization
- ❌ "Training pipeline completed successfully" (false positive)
- ❌ WandB shows 0 events logged
- ❌ No error messages

### After (Fixed)
- ✅ Clear error message if CUDA contamination detected
- ✅ Instructions on how to fix (regenerate cache)
- ✅ With clean cache: Training starts successfully
- ✅ 15-20s/epoch (vs 60s baseline)
- ✅ 90%+ GPU utilization
- ✅ Multi-worker data loading working

## Performance Benchmarks to Verify

| Metric | Before | After (Expected) |
|--------|--------|------------------|
| Epoch time | 60s | 15-20s ⚡ |
| GPU util | 30-89% | 90%+ ⚡ |
| num_workers | 0 (forced) | 8 (working) ⚡ |
| Cache location | Disk | RAM disk ⚡ |

## Troubleshooting

### Error: "Cache file contains CUDA tensors!"
**Solution:** Run the fix script
```bash
./scripts/fix_cuda_cache.sh
```

### Error: "Insufficient disk space"
**Solution:** Use smaller cache or different location
```bash
# Check space
df -h /dev/shm

# Use regular disk instead of RAM disk
./scripts/fix_cuda_cache.sh data/latent_cache configs/train_burgers_upt_full_ddp.yaml
```

### Still getting crashes?
**Debug steps:**
1. Check logs for actual error (no more silent crashes!)
2. Verify cache was regenerated with CPU: `ls -lh /dev/shm/latent_cache/`
3. Check git status: `git log -1 --oneline` (should show commit d74607a)
4. Re-read investigation doc: `docs/ddp_optimization_investigation_2025-11-18.md`

## Success Criteria

✅ No more silent crashes
✅ Clear error messages when issues occur
✅ With clean cache: 15-20s/epoch
✅ 90%+ GPU utilization
✅ Multi-worker data loading functional

## Next Steps After Success

1. **Update configs** to use CPU for cache precomputation by default
2. **Add to CI/CD** to prevent CUDA cache contamination
3. **Document** in production playbook
4. **Benchmark** different num_workers values (4, 8, 16)
5. **Scale up** to larger models with confidence

## References

- Investigation: `docs/ddp_optimization_investigation_2025-11-18.md`
- Commit: d74607a
- Branch: feature/distributed-training-ddp
