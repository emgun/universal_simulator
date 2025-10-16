# End-to-End Training & Evaluation Workflow

Complete guide to running training from scratch to final evaluation with WandB logging.

## Quick Start (TL;DR)

```bash
# 1. Launch instance
python scripts/vast_launch.py launch --gpu H200 --disk 300 \
  --overrides "TRAIN_CONFIG=configs/train_burgers_32dim_v2_fixed.yaml" \
  --auto-shutdown --b2-key-id $B2_KEY_ID --b2-app-key $B2_APP_KEY

# 2. Wait for training to complete (~60 min)

# 3. Sync WandB logs
ssh -p <PORT> root@<HOST> "cd /workspace/universal_simulator && wandb sync wandb/offline-run-*"

# 4. View results
# https://wandb.ai/emgun-morpheus-space/universal-simulator
```

## Detailed Workflow

### Phase 1: Pre-Launch Verification

#### 1.1 Test Config Locally
```bash
# Verify config loads without errors
python -c "
from ups.utils.config_loader import load_config_with_includes
cfg = load_config_with_includes('configs/train_burgers_32dim_v2_fixed.yaml')
print('✅ Config loaded')
print(f'Latent dim: {cfg[\"latent\"][\"dim\"]}')
print(f'Operator hidden_dim: {cfg[\"operator\"][\"pdet\"][\"hidden_dim\"]}')
print(f'Diffusion hidden_dim: {cfg[\"diffusion\"][\"hidden_dim\"]}')
assert cfg['operator']['pdet']['hidden_dim'] == cfg['diffusion']['hidden_dim'], 'Architecture mismatch!'
print('✅ Architectures match')
"
```

#### 1.2 Check WandB Configuration
```bash
grep -A 5 "logging:" configs/train_burgers_32dim_v2_fixed.yaml
# Should show:
#   enabled: true
#   project: universal-simulator
#   entity: emgun-morpheus-space
```

### Phase 2: Launch Instance

#### 2.1 Search for Available Instances
```bash
python scripts/vast_launch.py search "reliability > 0.98 gpu_name=H200 disk_space > 300"
```

#### 2.2 Launch Training
```bash
python scripts/vast_launch.py launch \
  --gpu H200 \
  --disk 300 \
  --image pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel \
  --overrides "TRAIN_CONFIG=configs/train_burgers_32dim_v2_fixed.yaml" \
  --auto-shutdown \
  --b2-key-id $B2_KEY_ID \
  --b2-app-key $B2_APP_KEY

# Note the instance ID (e.g., 26838185)
```

### Phase 3: Monitor Training

#### 3.1 Initial Connection (Wait ~30 seconds for startup)
```bash
# Get SSH details
vastai show instance <INSTANCE_ID> | grep ssh

# Connect
ssh -p <PORT> root@<HOST>
```

#### 3.2 Check Training Status
```bash
# Check GPU utilization
nvidia-smi

# View recent logs
tail -20 /workspace/universal_simulator/reports/training_log.jsonl

# Monitor in real-time
tail -f /workspace/universal_simulator/reports/training_log.jsonl
```

#### 3.3 Expected Timeline
```
00:00 - 00:05   Latent cache precomputation
00:05 - 00:30   Operator training (25 epochs)
00:30 - 00:38   Diffusion training (8 epochs)
00:38 - 01:12   Consistency distillation (8 epochs)
01:12 - 01:30   Evaluation (baseline + TTC)
```

### Phase 4: Verify Results

#### 4.1 Check Checkpoints
```bash
ssh -p <PORT> root@<HOST> "ls -lh /workspace/universal_simulator/checkpoints/*.pt"

# Should see:
# operator.pt             (~1MB for 32-dim with hidden_dim=96)
# diffusion_residual.pt   (~40KB)
# diffusion_residual_ema.pt (~40KB)
```

#### 4.2 Verify Architecture Match
```bash
ssh -p <PORT> root@<HOST> "python3 -c \"
import torch
op = torch.load('/workspace/universal_simulator/checkpoints/operator.pt', map_location='cpu')
diff = torch.load('/workspace/universal_simulator/checkpoints/diffusion_residual.pt', map_location='cpu')

# Check operator
op_keys = [k for k in op.keys() if 'attn_norm.weight' in k]
if op_keys:
    print(f'Operator hidden_dim: {op[op_keys[0]].shape[0]}')

# Check diffusion  
diff_keys = [k for k in diff.keys() if 'network.2.weight' in k]
if diff_keys:
    print(f'Diffusion hidden_dim: {diff[diff_keys[0]].shape[0]}')
\""
```

### Phase 5: Sync to WandB

#### 5.1 Find Offline Runs
```bash
ssh -p <PORT> root@<HOST> "ls -d /workspace/universal_simulator/wandb/offline-run-*"
```

#### 5.2 Sync All Runs
```bash
ssh -p <PORT> root@<HOST> "cd /workspace/universal_simulator && \
  for run in wandb/offline-run-*; do \
    echo \"Syncing \$run...\"; \
    wandb sync \$run --include-offline; \
  done"
```

#### 5.3 Verify on WandB
Visit: https://wandb.ai/emgun-morpheus-space/universal-simulator

Look for run with tags: `32dim, v2_fixed, end_to_end`

### Phase 6: Analyze Results

#### 6.1 View Training Metrics
```python
import wandb
api = wandb.Api()
run = api.run("emgun-morpheus-space/universal-simulator/<RUN_ID>")

# Get final metrics
print(f"Operator loss: {run.summary['operator/best_loss']}")
print(f"Diffusion loss: {run.summary['diffusion_residual/best_loss']}")
print(f"Consistency loss: {run.summary['consistency_distill/best_loss']}")
```

#### 6.2 View Evaluation Results
```bash
ssh -p <PORT> root@<HOST> "cat /workspace/universal_simulator/reports/eval_baseline.json"
ssh -p <PORT> root@<HOST> "cat /workspace/universal_simulator/reports/eval_ttc.json"
```

Expected:
- **Baseline NRMSE:** 0.4-0.5 (deterministic operator)
- **TTC NRMSE:** 0.06-0.07 (beam search optimization)

## Troubleshooting

### Issue: "size mismatch" during evaluation

**Cause:** Architecture parameters don't match between training and evaluation

**Solution:**
1. Check config has explicit `operator.pdet.hidden_dim` and `diffusion.hidden_dim`
2. Verify they match: `operator.pdet.hidden_dim == diffusion.hidden_dim`
3. Use self-contained config (no `include:`) to avoid inheritance issues

### Issue: WandB gives 401 "user is not logged in"

**Cause:** API key is invalid/expired

**Solution:** Use offline mode (already configured):
```bash
export WANDB_MODE=offline  # During training
wandb sync wandb/offline-run-*  # After training
```

### Issue: Training hangs at "compiling"

**Cause:** First epoch with `torch.compile()` takes 2-3 minutes

**Solution:** Wait patiently. After first epoch, training accelerates significantly.

### Issue: OOM during training

**Solutions:**
- Reduce `batch_size` (e.g., 12 → 8)
- Reduce `distill_num_taus` (e.g., 5 → 3)
- Reduce `num_workers` (e.g., 8 → 4)

### Issue: Evaluation takes too long (>30 min)

**Cause:** Enhanced TTC settings are very compute-intensive

**Solutions:**
- Reduce `ttc.candidates` (8 → 6)
- Reduce `ttc.beam_width` (3 → 2)
- Reduce `ttc.max_evaluations` (150 → 100)

## Best Practices

### 1. Always Test Config First
```bash
python -c "from ups.utils.config_loader import load_config_with_includes; \
           cfg = load_config_with_includes('path/to/config.yaml'); \
           print('✅ Config valid')"
```

### 2. Use Self-Contained Configs
Avoid `include:` for production configs to prevent inheritance issues.

### 3. Monitor Early Epochs
SSH in after 10 minutes to verify training started correctly:
- Check GPU utilization (should be >80% after first epoch)
- Check operator loss is decreasing
- Verify no errors in logs

### 4. Sync WandB Promptly
Sync offline runs immediately after training to avoid losing data:
```bash
wandb sync wandb/offline-run-* --include-offline
```

### 5. Keep Instances Running for Analysis
Don't use `--auto-shutdown` during development. Only use for production runs.

## Config Checklist

Before launching, verify your config has:

- [ ] `latent.dim` set correctly
- [ ] `operator.pdet.input_dim == latent.dim`
- [ ] `operator.pdet.hidden_dim` explicitly defined
- [ ] `diffusion.latent_dim == latent.dim`
- [ ] `diffusion.hidden_dim == operator.pdet.hidden_dim`
- [ ] `ttc.decoder.latent_dim == latent.dim`
- [ ] `ttc.decoder.hidden_dim == operator.pdet.hidden_dim`
- [ ] `logging.wandb.enabled: true`
- [ ] `logging.wandb.entity: emgun-morpheus-space`
- [ ] `evaluation.enabled: true`

## Summary

This workflow ensures:
✅ Config validated before training
✅ All architectures match (no size mismatches)
✅ WandB logging works (offline + sync)
✅ Evaluation runs automatically
✅ Results are reproducible
✅ Easy to troubleshoot issues

**Total time:** ~60-70 minutes for 32-dim v2 on H200
**Cost:** ~$2.00 @ $2.11/hr
**Output:** Trained checkpoints + WandB logs + evaluation metrics

