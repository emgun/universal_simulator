# Vast.ai Setup Guide

Quick guide for running training on vast.ai GPU instances.

## Prerequisites

1. **Vast.ai account** with credits
2. **W&B API key** from https://wandb.ai/settings
3. **GitHub repo** is public (or you have access token)

## Step 1: Rent GPU

1. Go to [cloud.vast.ai](https://cloud.vast.ai)
2. Search for: RTX 5090, 4090, or 3090 with 24GB+ VRAM
3. Select **PyTorch (Vast)** template
4. Set disk space: 50GB minimum
5. Click **Rent**

## Step 2: Setup (5 minutes)

**In Jupyter Terminal** (or SSH):

```bash
cd /workspace

# Clone repo
git clone https://github.com/emgun/universal_simulator.git
cd universal_simulator

# Install dependencies
pip install -e .

# Set environment variables
export WANDB_API_KEY="your_key_from_wandb.ai/settings"
export WANDB_PROJECT="universal-simulator"
export WANDB_ENTITY="emgun-morpheus-space"  # or your username
export WANDB_DATASETS="burgers1d_subset_v1"

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## Step 3: Fetch Data & Train

```bash
cd /workspace/universal_simulator

# Fetch dataset from W&B
PYTHONPATH=src python scripts/fetch_datasets.py burgers1d_subset_v1 \
  --root data/pdebench \
  --cache artifacts/cache \
  --project universal-simulator

# Train operator (30 epochs, ~1.5 hrs on RTX 5090)
python scripts/train.py --config configs/train_pdebench_scale.yaml --stage operator

# Train diffusion (15 epochs, ~45 min)
python scripts/train.py --config configs/train_pdebench_scale.yaml --stage diff_residual

# Consistency distillation (10 epochs, ~30 min)
python scripts/train.py --config configs/train_pdebench_scale.yaml --stage consistency_distill

# Steady prior (20 epochs, ~45 min)
python scripts/train.py --config configs/train.py --config configs/train_pdebench_scale.yaml --stage steady_prior

# Evaluate
python scripts/evaluate.py \
  --config configs/eval_pdebench_scale.yaml \
  --operator checkpoints/scale/operator.pt \
  --diffusion checkpoints/scale/diffusion_residual.pt \
  --output-prefix reports/scale_eval
```

## Step 4: Monitor

**W&B Dashboard**: https://wandb.ai/YOUR_ENTITY/universal-simulator

## Step 5: Cleanup

**CRITICAL**: Destroy instance when done!

1. Go to vast.ai console
2. Click "DESTROY" on your instance
3. Verify it's stopped

## Troubleshooting

**Import errors**: Run `pip install -e .` again

**Data not found**: Check `ls data/pdebench/` shows .h5 files

**W&B auth fails**: Run `wandb login --relogin`

---

**Cost**: ~$1.40-4.00 for full run (depending on GPU)
**Time**: ~3.5 hours total

