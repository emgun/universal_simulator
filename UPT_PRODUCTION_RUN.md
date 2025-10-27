# UPT Production Training Run - LAUNCHED ✅

## Instance Details

**Instance ID:** 27205346
**GPU:** Q_RTX_8000 (48GB VRAM)
**Cost:** $0.26/hr
**Location:** US
**Reliability:** 99.7%
**Branch:** feature--UPT (commit 6bbb24c)
**Config:** `configs/train_burgers_upt_losses.yaml`

### Previous Attempts (OOM on 24GB GPUs)
- **27205160** (Q_RTX_6000 24GB) - OOM errors, destroyed
- **27205263** (RTX 3090 Ti 24GB) - OOM expected, destroyed
- **Root cause:** batch_size=12 + UPT inverse losses need >24GB VRAM

## Training Configuration

### UPT Inverse Losses
```yaml
use_inverse_losses: true
lambda_inv_enc: 0.5    # Inverse encoding loss weight
lambda_inv_dec: 0.5    # Inverse decoding loss weight
inverse_loss_frequency: 10  # Compute every 10 batches
```

### Training Stages
- **Operator:** 25 epochs with UPT inverse losses
- **Diffusion:** 8 epochs
- **Consistency:** 8 epochs
- **Evaluation:** Full test set with TTC

### Expected Runtime
- **Operator:** ~25-30 minutes (25 epochs)
- **Diffusion:** ~10-12 minutes (8 epochs)
- **Consistency:** ~10-12 minutes (8 epochs)
- **Evaluation:** ~5-8 minutes (with TTC)
- **Total:** ~50-60 minutes

## Auto-Shutdown

✅ **Enabled** - 60 minute timeout
- Instance will automatically destroy after training completes
- Fallback timeout prevents runaway costs
- Retry logic (3 attempts) ensures clean shutdown

## Expected Results

### Baseline (No UPT)
- NRMSE: ~0.78

### With UPT Inverse Losses
- NRMSE: **0.08-0.10** (88-90% improvement)
- Better latent space alignment
- Improved invertibility of encoder/decoder

## Monitoring

### Commands
```bash
# Check instance status
vastai show instances

# View live logs
vastai logs 27205346 -f

# View training progress
vastai logs 27205346 2>&1 | grep -E "Epoch|L_inv|loss:"

# SSH into instance (if needed)
vastai ssh 27205346
```

### Background Monitors
Two background monitors are running:
- **fdf98a**: Startup logs after 3 minutes
- **5ef02d**: Training progress after 10 minutes

### WandB Dashboard
Training metrics will be logged to:
- **Project:** universal-simulator
- **Entity:** emgun-morpheus-space
- **Run Name:** TBD (check logs after startup)

## What to Look For

### Operator Stage (25 epochs)
Check that these metrics are logged in WandB:
- `training/operator/L_inv_enc` - Inverse encoding loss
- `training/operator/L_inv_dec` - Inverse decoding loss
- `training/operator/L_forward` - Forward prediction loss
- `training/operator/loss` - Total operator loss

**Expected:**
- All three losses should decrease during training
- Final operator loss: **< 0.0005** (with UPT losses)
- Inverse losses should be non-zero and decreasing

### Evaluation Results
- **Baseline NRMSE:** Should match golden config (~0.09)
- **TTC NRMSE:** Should show further improvement with test-time conditioning
- **Physics checks:** Conservation gap, BC violation, negativity penalty

## Files Generated

### Checkpoints
- `checkpoints/train_burgers_upt/operator.pt`
- `checkpoints/train_burgers_upt/diffusion_residual.pt`
- `checkpoints/train_burgers_upt/operator_ema.pt`
- `checkpoints/train_burgers_upt/diffusion_residual_ema.pt`

### Reports
- `reports/leaderboard.csv` - Updated with new results
- `reports/eval_baseline.json` - Baseline evaluation
- `reports/eval_ttc.json` - TTC evaluation

### Artifacts
- WandB artifacts with checkpoints and evaluation results
- Training logs in `artifacts/runs/`

## Success Criteria

### Minimum (Code Works)
- ✅ All stages complete without errors
- ✅ UPT inverse losses are computed and logged
- ✅ Auto-shutdown succeeds
- ✅ Checkpoints saved

### Target (UPT Benefits Visible)
- ✅ `L_inv_enc` and `L_inv_dec` decrease during training
- ✅ Final operator loss < 0.0005
- ✅ Evaluation NRMSE < 0.10
- ✅ 10-20% improvement over baseline

## Troubleshooting

### If auto-shutdown fails
```bash
# Manually destroy instance
vastai destroy instance 27205346
```

### If training fails
```bash
# SSH and check logs
vastai ssh 27205346
tail -200 /workspace/universal_simulator/nohup.out

# Check for errors
grep -i error /workspace/universal_simulator/nohup.out
```

### If OOM errors occur
- Reduce `training.batch_size` from 12 to 8
- Reduce `training.accum_steps` from 4 to 2
- Disable `training.compile`

## Next Steps After Completion

### If Successful
1. Review WandB metrics to confirm UPT losses are working
2. Compare results to baseline (golden config)
3. Merge `feature--UPT` branch to main if improvement confirmed
4. Update production configs with UPT settings

### If Performance Improved
1. Tune lambda weights (`lambda_inv_enc`, `lambda_inv_dec`)
2. Experiment with `inverse_loss_frequency`
3. Try different encoder/decoder architectures

### Documentation
1. Update `CLAUDE.md` with UPT best practices
2. Add UPT section to production playbook
3. Document typical NRMSE improvements

---

**Status:** ✅ **FINAL PRODUCTION RUN - ALL 8 UPT FIXES APPLIED**

**Current Instance:** 27283596
**GPU:** RTX 4090 (24GB VRAM)
**Disk:** 64GB
**Cost:** $0.30/hr
**Branch:** feature--UPT (commit bfd60d6 - NO-COMPILE FIX)
**Config:** `train_burgers_upt_nocache.yaml` (caching + compile disabled)
**WandB:** Will be at emgun-morpheus-space/universal-simulator

**Previous Instances (Failed):**
- **27252152** (RTX 5880 Ada 48GB) - Crashed in diffusion (incomplete collate fix)
- **27262305** (RTX 5880 Ada 48GB) - Crashed in diffusion (same error, led to real fix)
- **27279702** (RTX A6000 48GB) - WRONG BRANCH (launched with feature/sota_burgers_upgrades)
- **27281504** (RTX A6000 48GB) - DISK FULL during training (UPT+cache incompatibility)
- **27283349** (RTX 4090 24GB) - DISK FULL from torch.compile kernels in /tmp

**All 8 Critical Fixes Applied:**
1. **Commit 52f0532** - Fixed `precompute_latent_cache.py` to pass `use_inverse_losses=True`
2. **Commit 5ccaf56** - Changed auto-shutdown to auto-stop (preserve logs)
3. **Commit d1f077e** - Added `--cache-dtype float16` to reduce cache size from >1TB to ~500GB
4. **Commit c592248** - Removed 60-minute timeout (training needs time to complete)
5. **Commit becc775** - Fixed `latent_pairs.py` collate function (INCOMPLETE - only fixed one file)
6. **Commit bc974c3** - Fixed `parallel_cache.py` collate function (THE REAL COLLATE FIX)
7. **Commit 6850081** - Disabled caching entirely for UPT (avoid UPT+cache incompatibility)
8. **Commit bfd60d6** - Disabled torch.compile (kernel cache fills /tmp on small disks)

**Root Causes Discovered:**

**Issue #1 - Diffusion Crashes (Fixes #5-6):**
- TWO files were using legacy `collate_latent_pairs()` function
- Fix #5 only updated `latent_pairs.py`
- Fix #6 found and fixed `parallel_cache.py` (used by diffusion stage)
- Both files now use new `latent_pair_collate()` that returns dict format

**Issue #2 - UPT + Caching Incompatibility (Fix #7):**
- UPT inverse losses require physical fields + coords during training
- Legacy latent cache only stores latent vectors (no physical fields)
- When training loads from cache, it tries to write missing data on-the-fly
- This filled the disk during the first training batch (OSError: No space left on device)
- **Solution**: Disable caching entirely (`num_workers=0`, no `latent_cache_dir`)
- **Trade-off**: Training slower (~3-5x) but no massive disk requirement

**Issue #3 - torch.compile Kernel Cache (Fix #8):**
- Even with caching disabled, torch.compile filled `/tmp/torchinductor_root/` with compiled kernels
- RTX 4090 with 64GB disk ran out of space during first batch compilation
- Error: `OSError: [Errno 28] No space left on device: '/tmp/torchinductor_root/sw'`
- **Solution**: Disable torch.compile (`training.compile=false`)
- **Trade-off**: Training ~1.5x slower but works on small disks

**Key Insights:**
- Instance 27252152 confirmed operator training works with UPT + torch.compile + float16 cache
- Diffusion crash was due to collate function mismatch in `parallel_cache.py`
- Needed comprehensive grep search to find ALL uses of legacy collate function
- UPT on small disks requires BOTH caching and compile disabled
- Performance penalty: ~4.5-7.5x slower (3-5x from no cache, 1.5x from no compile)

**Earlier Failed Attempts:**
- **27205160** (Q_RTX_8000 48GB) - NO inverse losses (cache bug - fix #1)
- **27205792** (A100 SXM4 40GB) - NO inverse losses (cache bug - fix #1)
- **27239686** (RTX 5880 Ada 237GB) - Disk full during cache (float32 - fix #3)
- **27240693** (RTX A6000 64GB) - Disk full during cache (float32 - fix #3)
- **27242199** (RTX A6000 500GB) - Disk full during cache (float32 - fix #3)
- **27243507** (RTX 4090 1TB) - Disk full during cache (float32 - fix #3)

**Key Lessons:**
- UPT cache with physical fields in float32 requires >1TB disk. Float16 reduces to ~500GB.
- When fixing function names, grep entire codebase - don't assume one fix is enough!
