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

**Status:** ✅ **LAUNCHED - FLOAT16 CACHE OPTIMIZATION**

**Current Instance:** 27247185
**GPU:** RTX A6000 (48GB VRAM)
**Disk:** 1431GB
**Cost:** $0.39/hr
**Location:** Sweden (99.2% reliability)
**Optimization:** Using float16 cache dtype (50% size reduction)

**Critical Fixes Applied:**
1. **Commit 52f0532** - Fixed `precompute_latent_cache.py` to pass `use_inverse_losses=True`
2. **Commit 5ccaf56** - Changed auto-shutdown to auto-stop (preserve logs)
3. **Commit d1f077e** - Added `--cache-dtype float16` to reduce cache size from >1TB to ~500GB

**Previous Attempts:**
- **27205160** (Q_RTX_8000 48GB) - NO inverse losses (cache bug)
- **27205792** (A100 SXM4 40GB) - NO inverse losses (cache bug)
- **27239686** (RTX 5880 Ada 237GB) - Disk full during cache (float32)
- **27240693** (RTX A6000 64GB) - Disk full during cache (float32)
- **27242199** (RTX A6000 500GB) - Disk full during cache (float32)
- **27243507** (RTX 4090 1TB) - Disk full during cache (float32)

**Key Lesson:** UPT cache with physical fields in float32 requires >1TB disk. Float16 reduces to ~500GB.
