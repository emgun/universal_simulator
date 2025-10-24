# UPT 1-Epoch Test Run - LAUNCHED ✅

## Instance Details

**Instance ID:** 27203163
**Config:** `configs/test_upt_all_stages_1epoch.yaml`
**GPU:** RTX 4090 (or similar)
**Auto-shutdown:** Enabled
**WandB Run:** `test-upt-all-stages-1epoch`

## Test Configuration

### What's Being Tested
- ✅ **Operator stage:** 1 epoch with UPT inverse losses
- ✅ **Diffusion stage:** 1 epoch
- ✅ **Consistency distillation:** 1 epoch
- ✅ **Evaluation:** On test set with TTC

### UPT Inverse Losses Settings
```yaml
training:
  use_inverse_losses: true
  lambda_inv_enc: 0.5  # Inverse encoding loss
  lambda_inv_dec: 0.5  # Inverse decoding loss
```

### Expected Behavior

**Operator Stage (1 epoch):**
- Loads physical fields in batches
- Computes `L_inv_enc` (ensures latent → decode → reconstruct works)
- Computes `L_inv_dec` (ensures latent → decode → re-encode works)
- Logs all loss components to WandB

**Diffusion Stage (1 epoch):**
- Trains diffusion residual model
- Uses trained operator as base

**Consistency Stage (1 epoch):**
- Distills diffusion to few-step sampler

**Evaluation:**
- Runs test set evaluation
- Reports NRMSE with and without TTC

## Expected Runtime

- **Operator:** ~2-3 minutes (1 epoch)
- **Diffusion:** ~1-2 minutes (1 epoch)
- **Consistency:** ~1-2 minutes (1 epoch)
- **Evaluation:** ~1-2 minutes
- **Total:** ~5-10 minutes

## Monitoring Commands

```bash
# Check instance status
vastai show instances

# View logs (live)
vastai logs 27203163

# View UPT-specific logs
vastai logs 27203163 2>&1 | grep -E "inverse|L_inv_enc|L_inv_dec"

# SSH into instance (if needed)
vastai ssh 27203163
```

## Expected Outputs

### WandB Logs
Should see:
- `operator/L_inv_enc` - Inverse encoding loss
- `operator/L_inv_dec` - Inverse decoding loss
- `operator/L_forward` - Forward prediction loss
- `operator/loss` - Total loss
- `diffusion_residual/loss` - Diffusion loss
- `consistency_distill/loss` - Distillation loss
- `eval/nrmse` - Final NRMSE

### Checkpoints
Created in `checkpoints/test_upt_all_stages/`:
- `operator.pt` - Trained operator with inverse losses
- `diff_latest.ckpt` - Diffusion residual
- `distill_latest.ckpt` - Consistency distilled model

## Success Criteria

### Minimum (Just Code Works)
- ✅ Training starts without errors
- ✅ Physical fields load successfully
- ✅ Inverse losses compute without NaN/inf
- ✅ All stages complete

### Ideal (UPT Benefits Visible)
- ✅ `L_inv_enc` and `L_inv_dec` decrease during training
- ✅ Operator final loss < 0.001
- ✅ Evaluation NRMSE reasonable (< 0.5)
- ✅ No crashes or OOM errors

## Monitoring Timeline

- **T+0min:** Instance starting, downloading code
- **T+1min:** Installing dependencies, downloading data
- **T+2min:** Precomputing latent cache (if not cached)
- **T+3min:** Operator training starts
- **T+5min:** Operator done, diffusion starts
- **T+7min:** Diffusion done, consistency starts
- **T+9min:** Consistency done, evaluation starts
- **T+10min:** Evaluation done, instance auto-shutdown

## Background Monitors Running

I've started background monitors that will check:
1. **Monitor 2f76e2:** Full logs after 1 minute
2. **Monitor f6f3da:** UPT-specific logs after 3 minutes

Check outputs with:
```bash
# In Claude Code, I can check these with BashOutput tool
```

## Troubleshooting

### If training fails:
```bash
# SSH into instance
vastai ssh 27203163

# Check full logs
tail -100 /workspace/universal_simulator/nohup.out

# Check for errors
grep -i error /workspace/universal_simulator/nohup.out
```

### Common issues:
1. **OOM:** Batch size too large → reduce to 4
2. **Data download fails:** Network issue → retry
3. **Import errors:** Missing dependencies → check requirements.txt

## Next Steps After Test

### If test passes:
1. ✅ Verify WandB shows inverse losses
2. ✅ Check that `L_inv_enc` and `L_inv_dec` are reasonable values
3. ✅ Compare eval NRMSE to baseline
4. Launch full 25-epoch training with same config

### If test fails:
1. Review error logs
2. Debug specific issue
3. Fix and re-launch

## Files Created

- `configs/test_upt_all_stages_1epoch.yaml` - Test config
- `.vast/onstart.sh` - Generated launch script
- `UPT_TEST_RUN_STATUS.md` - This file

## Related Documentation

- `UPT_FULL_RESTORATION_COMPLETE.md` - Implementation details
- `UPT_PHASE1.5_COMPLETE.md` - Data loading spec
- `configs/train_burgers_upt_losses.yaml` - Production config (25 epochs)

---

**Status:** ✅ Instance launched and initializing
**Next Check:** View logs in ~3 minutes to see training progress
