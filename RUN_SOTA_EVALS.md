# Run SOTA Evaluations - Ready to Execute

**Status:** All code ready, Vast.ai had startup issues
**Solution:** Run this script on any GPU instance (Vast.ai, Lambda Labs, RunPod, local GPU, etc.)

## What's Ready

✅ **SOTA Metrics Implemented** - nRMSE and relative L2 in evaluation code
✅ **Baseline Configs Created** - No-TTC comparison configs ready
✅ **Checkpoints Available** - Downloaded from W&B (pru2jxc4, pp0c2k31)
✅ **Evaluation Script** - Complete automation script below

## Quick Start (Any GPU Instance)

```bash
# 1. Clone repo and setup
git clone https://github.com/your-repo/universal_simulator.git
cd universal_simulator
pip install torch h5py pyyaml wandb tqdm matplotlib seaborn pydantic rclone

# 2. Set credentials
export WANDB_API_KEY='ec37eba84558733a8ef56c76e284ab530e94449b'
export B2_KEY_ID='0043616a62c8bb90000000001'
export B2_APP_KEY='K004cur7hMs3GDPixFB8FlzfCJV2PIc'

# 3. Run the evaluation script
bash /tmp/vast_sota_eval.sh
```

## Complete Evaluation Script

Save this as `run_sota_evaluations.sh`:

```bash
#!/usr/bin/env bash
set -eo pipefail

echo "===================================="
echo "SOTA Evaluation - Baseline vs TTC"
echo "===================================="

# Setup environment
export WANDB_API_KEY='ec37eba84558733a8ef56c76e284ab530e94449b'
export B2_KEY_ID='0043616a62c8bb90000000001'
export B2_APP_KEY='K004cur7hMs3GDPixFB8FlzfCJV2PIc'
export B2_BUCKET='pdebench'
export PDEBENCH_ROOT='./data/pdebench'

# Install dependencies
echo "Installing dependencies..."
pip install -q h5py pyyaml wandb tqdm matplotlib seaborn pydantic 2>&1 | grep -v "already satisfied" || true

# Download data from B2
echo "Downloading validation and test data..."
mkdir -p data/pdebench

# Configure rclone
export RCLONE_CONFIG_UPSB2_TYPE=s3
export RCLONE_CONFIG_UPSB2_PROVIDER=Other
export RCLONE_CONFIG_UPSB2_ACCESS_KEY_ID=$B2_KEY_ID
export RCLONE_CONFIG_UPSB2_SECRET_ACCESS_KEY=$B2_APP_KEY
export RCLONE_CONFIG_UPSB2_ENDPOINT=s3.us-west-004.backblazeb2.com
export RCLONE_CONFIG_UPSB2_REGION=us-west-004

rclone copy UPSB2:pdebench/pdebench/burgers1d_full_v1/burgers1d_val.h5 data/pdebench/ -P
rclone copy UPSB2:pdebench/pdebench/burgers1d_full_v1/burgers1d_test.h5 data/pdebench/ -P

# Download checkpoints
echo "Downloading checkpoints from W&B..."
mkdir -p checkpoints/scale
PYTHONPATH=src python scripts/download_checkpoints_from_wandb.py \
  --dest checkpoints/scale \
  --operator-run pru2jxc4 \
  --diffusion-run pp0c2k31

echo "✓ Setup complete"
echo ""

# Run evaluations
mkdir -p reports

echo "1/4: Baseline validation (no TTC)..."
PYTHONPATH=src python scripts/evaluate.py \
  --config configs/eval_burgers_512dim_val_baseline.yaml \
  --operator checkpoints/scale/operator.pt \
  --diffusion checkpoints/scale/diffusion_residual.pt \
  --device cuda \
  --output-prefix reports/baseline_val_notttc \
  --print-json

echo "2/4: Baseline test (no TTC)..."
PYTHONPATH=src python scripts/evaluate.py \
  --config configs/eval_burgers_512dim_test_baseline.yaml \
  --operator checkpoints/scale/operator.pt \
  --diffusion checkpoints/scale/diffusion_residual.pt \
  --device cuda \
  --output-prefix reports/baseline_test_notttc \
  --print-json

echo "3/4: TTC validation..."
PYTHONPATH=src python scripts/evaluate.py \
  --config configs/eval_burgers_512dim_ttc_val.yaml \
  --operator checkpoints/scale/operator.pt \
  --diffusion checkpoints/scale/diffusion_residual.pt \
  --device cuda \
  --output-prefix reports/ttc_val_sota \
  --print-json

echo "4/4: TTC test..."
PYTHONPATH=src python scripts/evaluate.py \
  --config configs/eval_burgers_512dim_ttc_test.yaml \
  --operator checkpoints/scale/operator.pt \
  --diffusion checkpoints/scale/diffusion_residual.pt \
  --device cuda \
  --output-prefix reports/ttc_test_sota \
  --print-json

# Analysis
echo ""
echo "===================================="
echo "SOTA COMPARISON ANALYSIS"
echo "===================================="

python3 << 'PYEOF'
import json

# Load results
files = {
    'baseline_val': 'reports/baseline_val_notttc.json',
    'baseline_test': 'reports/baseline_test_notttc.json',
    'ttc_val': 'reports/ttc_val_sota.json',
    'ttc_test': 'reports/ttc_test_sota.json'
}

results = {}
for name, path in files.items():
    with open(path) as f:
        results[name] = json.load(f)

print("\nVALIDATION SET:")
print(f"  Baseline (no TTC):")
print(f"    Relative L2: {results['baseline_val']['metrics']['rel_l2']:.6f}")
print(f"    nRMSE:       {results['baseline_val']['metrics']['nrmse']:.6f}")
print(f"  TTC:")
print(f"    Relative L2: {results['ttc_val']['metrics']['rel_l2']:.6f}")
print(f"    nRMSE:       {results['ttc_val']['metrics']['nrmse']:.6f}")

val_imp = (results['baseline_val']['metrics']['rel_l2'] - results['ttc_val']['metrics']['rel_l2']) / results['baseline_val']['metrics']['rel_l2'] * 100
print(f"  TTC Improvement: {val_imp:+.2f}%")

print("\nTEST SET:")
print(f"  Baseline (no TTC):")
print(f"    Relative L2: {results['baseline_test']['metrics']['rel_l2']:.6f}")
print(f"    nRMSE:       {results['baseline_test']['metrics']['nrmse']:.6f}")
print(f"  TTC:")
print(f"    Relative L2: {results['ttc_test']['metrics']['rel_l2']:.6f}")
print(f"    nRMSE:       {results['ttc_test']['metrics']['nrmse']:.6f}")

test_imp = (results['baseline_test']['metrics']['rel_l2'] - results['ttc_test']['metrics']['rel_l2']) / results['baseline_test']['metrics']['rel_l2'] * 100
print(f"  TTC Improvement: {test_imp:+.2f}%")

print("\nSOTA COMPARISON:")
fno = 0.0180
ours = results['ttc_test']['metrics']['rel_l2']
gap = (ours - fno) / fno * 100

print(f"  FNO-2D (SOTA 2021):  {fno:.6f}")
print(f"  Ours (TTC):          {ours:.6f}")
print(f"  Gap to SOTA:         {gap:+.2f}%")

if gap < 0:
    print(f"\n  🎉 WE BEAT SOTA by {abs(gap):.1f}%!")
elif gap < 10:
    print(f"\n  ✓ Very close to SOTA")
else:
    print(f"\n  ⚠ Room for improvement")
PYEOF

echo ""
echo "===================================="
echo "✓ All evaluations complete!"
echo "Results in reports/*.json"
echo "===================================="
```

## Manual Execution on Vast.ai

If you get a working Vast.ai instance:

```bash
# 1. Launch instance
vastai create instance <ID> --image pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime --disk 50

# 2. Wait for it to reach "running" state
vastai show instances

# 3. SSH to instance
ssh -p <PORT> root@ssh<N>.vast.ai

# 4. Setup
cd /workspace
git clone <repo-url> universal_simulator
cd universal_simulator

# 5. Run script above
bash run_sota_evaluations.sh
```

## What to Do Next

1. **Get a working GPU instance** (Vast.ai when it works, or Lambda Labs, RunPod, etc.)
2. **Run the evaluation script** - it's fully automated
3. **Compare results:**
   - Baseline vs TTC (measure TTC improvement)
   - Ours vs FNO (measure gap to SOTA)
4. **Upload to W&B:**
   ```bash
   PYTHONPATH=src python scripts/upload_artifact.py \
     sota-comparison-results \
     evaluation \
     reports/*.json \
     --project universal-simulator
   ```

## Expected Runtime

- **Data download:** ~30 seconds (314MB from B2)
- **Checkpoint download:** ~1 minute (115MB from W&B)
- **Each evaluation:** ~5-15 minutes (depends on GPU)
- **Total:** ~30-60 minutes for all 4 evaluations

## Files Already Created

✅ `configs/eval_burgers_512dim_val_baseline.yaml` - Baseline val config
✅ `configs/eval_burgers_512dim_test_baseline.yaml` - Baseline test config
✅ `src/ups/eval/pdebench_runner.py` - Updated with SOTA metrics
✅ `docs/sota_comparison_guide.md` - Complete methodology guide
✅ `scripts/add_sota_metrics.py` - Helper script
✅ Checkpoints downloaded locally: `checkpoints/scale/*.pt`

## Vast.ai Issue

**Problem:** Instances getting stuck in "created" or "loading" state
**Attempted:** Multiple instances (26798344, 26798561)
**Status:** Destroyed both, waiting for Vast.ai platform stability

**Alternative Solutions:**
1. Try Vast.ai again in a few hours
2. Use Lambda Labs GPU Cloud
3. Use RunPod
4. Use Google Colab Pro (if you have GPU access)
5. Run locally if you have a GPU

## Summary

Everything is ready to run - just need a working GPU instance. The evaluation will:
1. Download data and checkpoints
2. Run 4 evaluations (baseline val/test, TTC val/test)
3. Compare baseline vs TTC (measure TTC benefit)
4. Compare ours vs FNO-2D (measure SOTA gap)
5. Generate comprehensive reports

**All code is done - just need to execute!**
