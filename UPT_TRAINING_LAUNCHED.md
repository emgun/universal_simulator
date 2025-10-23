# 🚀 UPT Phase 1 Training: LAUNCHED

**Launch Time**: 2025-01-23
**Status**: ✅ **RUNNING ON VASTAI**
**Instance ID**: `27192839`

---

## 📊 Training Instance Details

| Property | Value |
|----------|-------|
| **Instance ID** | 27192839 |
| **GPU** | RTX 4090 |
| **Cost** | $0.37/hr |
| **Status** | Running (9% GPU util) |
| **SSH** | ssh7.vast.ai:32838 |
| **Auto-shutdown** | ✅ Enabled (60 min timeout) |

---

## 🎯 Training Configuration

### UPT Inverse Losses: ✅ ENABLED

```yaml
training:
  use_upt_inverse_losses: true
  lambda_inv_enc: 0.5          # Inverse encoding loss weight
  lambda_inv_dec: 0.5          # Inverse decoding loss weight
  inv_enc_query_points: 2048   # Query sampling for efficiency
  inv_dec_query_points: 1024
```

### Pipeline Stages
1. **Operator** (25 epochs with UPT losses)
2. **Diffusion Residual** (8 epochs)
3. **Consistency Distillation** (8 epochs)

### Expected Timeline
- **Total Time**: ~25-30 min
  - Operator: ~15-18 min (vs 14.5 min baseline)
  - Diffusion: ~5 min
  - Distillation: ~5 min
- **Cost**: ~$0.25-0.30

---

## 🔍 Monitoring

### Check Instance Status
```bash
vastai show instances | grep 27192839
```

### View Logs
```bash
vastai logs 27192839

# Or use the monitoring script
./monitor_upt_training.sh
```

### WandB Dashboard
https://wandb.ai/emgun-morpheus-space/universal-simulator

Look for run with tags:
- `vast`
- `upt-phase1`
- `inverse-losses`
- `16dim`

---

## 📈 Expected Results

Based on UPT paper and Phase 1 implementation:

| Metric | Baseline (Golden) | Target (UPT) | Improvement |
|--------|-------------------|--------------|-------------|
| **Validation NRMSE** | 0.078 | 0.060-0.070 | **-10 to -20%** |
| **Operator Final Loss** | 0.00023 | 0.00015-0.00020 | **-15 to -30%** |
| **Correlation Time** | Baseline | Improved | **Longer rollouts** |
| **Training Time** | 14.5 min | 17-18 min | +15-20% |

### Key Metrics to Watch in WandB

**During Operator Training**:
- ✅ `loss_inv_enc` - Should decrease steadily
- ✅ `loss_inv_dec` - Should decrease steadily
- ✅ `operator/train_loss` - Should be < 0.0002 at end
- ✅ GPU utilization - Should be 80-95%

**During Evaluation**:
- ✅ `eval/nrmse` - Target: < 0.075
- ✅ `eval/ttc_nrmse` - Target: < 0.070
- ✅ `eval/correlation_time` - Should improve vs baseline

---

## 🛠️ Troubleshooting

### If Training Fails

**Check Logs**:
```bash
vastai logs 27192839 | tail -200
```

**Common Issues**:
1. **OOM Error**: Batch size too large
   - Solution: Reduce `training.batch_size` from 12 → 8
2. **UPT Loss NaN**: Inverse loss computation issue
   - Solution: Reduce `lambda_inv_enc` and `lambda_inv_dec` to 0.1
3. **Data Download Failed**: B2 connection issue
   - Solution: Restart instance manually

### Manual Intervention

**SSH Into Instance**:
```bash
# Get SSH command
vastai ssh-url 27192839

# Then use regular ssh:
# ssh -p <port> root@<host>
```

**Manual Shutdown** (if needed):
```bash
vastai destroy instance 27192839
```

---

## 📋 Onstart Script Summary

The training pipeline will:

1. ✅ Clone repo (`feature/sota_burgers_upgrades` branch)
2. ✅ Install dependencies
3. ✅ Download training data from B2
   - `burgers1d_train_000.h5`
   - `burgers1d_valid.h5`
   - `burgers1d_test.h5`
4. ✅ Precompute latent cache (saves ~3-5x training time)
5. ✅ Run full pipeline:
   ```bash
   python scripts/run_fast_to_sota.py \
     --train-config configs/train_burgers_upt_losses.yaml \
     --skip-small-eval \
     --wandb-mode online
   ```
6. ✅ Upload results to WandB
7. ✅ Auto-shutdown (via VastAI API)

---

## 🎯 Success Criteria

### Training Completion ✅
- [ ] Operator training completes (25 epochs)
- [ ] Final operator loss < 0.0002
- [ ] UPT inverse losses decrease during training
- [ ] No OOM or NaN errors
- [ ] Diffusion and consistency stages complete

### Performance Targets ✅
- [ ] Validation NRMSE < 0.075 (vs golden's 0.078)
- [ ] Operator final loss < 0.00020 (vs golden's 0.00023)
- [ ] Training time < 20 min for operator stage
- [ ] Correlation time improved vs baseline

### WandB Artifacts ✅
- [ ] Training curves uploaded
- [ ] Eval metrics logged
- [ ] Checkpoints saved
- [ ] Run tagged properly

---

## 📊 Current Status Updates

### T+0 min (Launch)
- ✅ Instance launched: 27192839
- ✅ Status: Loading → Running
- ✅ GPU: RTX 4090
- ✅ Cost: $0.37/hr

### T+2 min (Dependencies)
- ✅ Installing pip dependencies
- ✅ Building wheel for universal-physics-stack
- ✅ GPU util: 9% (setup phase)

### T+5 min (Expected: Data Download)
- ⏳ Downloading burgers1d training data from B2
- ⏳ Should see "rclone copy" in logs

### T+10 min (Expected: Latent Cache)
- ⏳ Precomputing latent cache
- ⏳ Should see "Precomputing latent caches…"

### T+15 min (Expected: Training Start)
- ⏳ Operator training begins
- ⏳ UPT losses should appear in logs
- ⏳ GPU util should jump to 80-95%

### T+30 min (Expected: Completion)
- ⏳ All stages complete
- ⏳ Auto-shutdown triggered
- ⏳ Results in WandB

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| `configs/train_burgers_upt_losses.yaml` | Training configuration |
| `.vast/onstart.sh` | Instance startup script |
| `monitor_upt_training.sh` | Monitoring helper |
| `UPT_PHASE1_COMPLETE.md` | Implementation summary |
| `UPT_INTEGRATION_ANALYSIS.md` | Gap analysis & roadmap |

---

## 🎓 What's Being Tested

### Phase 1 Implementation
This training run validates:

1. **UPT Inverse Losses**
   - `upt_inverse_encoding_loss()` - Ensures latent→decoded→original reconstruction
   - `upt_inverse_decoding_loss()` - Ensures decoded→encoded→latent reconstruction

2. **Data Pipeline**
   - Original fields loaded alongside latent pairs
   - Batching and collation working correctly
   - Memory overhead acceptable

3. **Training Integration**
   - Encoder retrieved from dataset ✅
   - Decoder created automatically ✅
   - UPT losses computed in training loop ✅
   - Gradients flow correctly ✅

4. **Performance Impact**
   - Training time overhead (+15-20% expected)
   - NRMSE improvement (-10 to -20% expected)
   - Correlation time improvement
   - No instabilities or NaN losses

---

## 🔬 Post-Training Analysis

Once training completes, run:

```bash
# Compare with golden baseline
python scripts/compare_runs.py \
  <golden_run_id> \
  <upt_run_id>

# Analyze run
python scripts/analyze_run.py <upt_run_id> \
  --output reports/upt_phase1_analysis.md

# Update leaderboard if successful
# (automatic via run_fast_to_sota.py)
```

---

## 🎊 Next Steps After Validation

### If Successful (NRMSE < 0.075)
1. ✅ Mark Phase 1 complete
2. ✅ Update production config to use UPT losses
3. ✅ Document best practices
4. ⏳ Plan Phase 2: Scale to 256 tokens

### If Needs Tuning (NRMSE 0.075-0.078)
1. Ablation study: Test different lambda weights
   - Try `lambda_inv_enc: 0.1, 0.3, 0.5, 1.0`
   - Try `lambda_inv_dec: 0.1, 0.3, 0.5, 1.0`
2. Adjust query point sampling
3. Re-run with best configuration

### If Issues Found
1. Check logs for errors
2. Verify UPT losses are decreasing
3. Check for NaN/Inf in loss values
4. Validate data pipeline changes
5. Run unit tests on failing components

---

## 💡 Useful Commands

```bash
# Monitor instance
./monitor_upt_training.sh

# Check status
vastai show instances | grep 27192839

# View logs (live)
vastai logs 27192839

# Get SSH URL
vastai ssh-url 27192839

# Manual shutdown (emergency)
vastai destroy instance 27192839

# WandB dashboard
open https://wandb.ai/emgun-morpheus-space/universal-simulator
```

---

## 📞 Support

**Questions?**
- Implementation: See `UPT_PHASE1_COMPLETE.md`
- Expected results: See `UPT_INTEGRATION_ANALYSIS.md`
- Troubleshooting: Check logs with `vastai logs 27192839`

**Issues?**
- Check WandB for training curves
- Review logs for errors
- Compare with golden baseline run

---

**Launch Status**: ✅ **RUNNING**
**Instance**: 27192839
**Expected Completion**: ~30 min from launch
**Auto-shutdown**: Enabled
**Cost**: ~$0.25-0.30 total

🎯 **Training in progress - monitor via WandB dashboard!**
