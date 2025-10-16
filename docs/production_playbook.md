# Production Playbook

**Purpose:** Best practices, common patterns, and decision trees for production training runs.

**Audience:** ML Engineers, Researchers

**Last Updated:** 2025-10-16

---

## Quick Decision Tree

```
Need to train a model?
│
├─ Is this your first time? → Follow "First Time Setup"
├─ Production training? → Use configs/train_burgers_32dim.yaml
├─ Experimenting? → Start with dry-run, then iterate
├─ Comparing runs? → Use scripts/compare_runs.py
└─ Debugging failed run? → Check "Troubleshooting" section
```

---

## First Time Setup

### 1. Environment Setup

**Local Development:**
```bash
# Clone repository
git clone https://github.com/emgun/universal_simulator.git
cd universal_simulator

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env  # Edit with your credentials
```

**Docker (Recommended):**
```bash
# Build and start
docker-compose up --build -d

# Attach to container
docker-compose exec universal-simulator bash
```

### 2. Credentials Setup

Required environment variables in `.env`:
```bash
# WandB (for experiment tracking)
WANDB_API_KEY=your_key_here
WANDB_PROJECT=universal-simulator
WANDB_ENTITY=your_username

# B2 (for data storage)
B2_KEY_ID=your_key_id
B2_APP_KEY=your_app_key
B2_BUCKET=pdebench
B2_S3_ENDPOINT=s3.us-west-002.backblazeb2.com
B2_S3_REGION=us-west-002
```

### 3. Data Setup

**Option A: Download from B2**
```bash
# Configure rclone
rclone config  # Follow prompts

# Download training data
rclone copy B2:pdebench/full/burgers1d/burgers1d_train_000.h5 data/pdebench/
cd data/pdebench && ln -sf burgers1d_train_000.h5 burgers1d_train.h5 && cd -
```

**Option B: Download from WandB**
```bash
wandb artifact get burgers1d_full_v1 --root data/pdebench/
```

---

## Production Training Workflow

### Standard Pipeline

```bash
# 1. Validate configuration
python scripts/validate_config.py configs/train_burgers_32dim.yaml

# 2. Validate data
python scripts/validate_data.py configs/train_burgers_32dim.yaml

# 3. Dry-run (estimate cost and time)
python scripts/dry_run.py configs/train_burgers_32dim.yaml

# 4. Train (if all checks pass)
python scripts/train.py configs/train_burgers_32dim.yaml

# 5. Analyze results
python scripts/analyze_run.py <run_id> --output reports/analysis.md
```

### Expected Timeline (32-dim on A100)

| Stage | Duration | Checkpoints |
|-------|----------|-------------|
| Latent Cache | 2-3 min | latent_cache/ |
| Operator (25 epochs) | 2 min | operator.pt |
| Diffusion (8 epochs) | 1 min | diffusion_residual.pt |
| Distillation (8 epochs) | 15-20 min | diffusion_residual.pt (updated) |
| Evaluation (baseline + TTC) | 15-20 min | reports/ |
| **Total** | **~40 min** | **~$1.25 @ $1.89/hr** |

---

## Configuration Best Practices

### Choosing Latent Dimension

| Dimension | Use Case | Performance | Cost | When to Use |
|-----------|----------|-------------|------|-------------|
| 32-dim | **Production** | NRMSE ~0.09 (w/ TTC) | $0.80 | Default choice, best cost/performance |
| 64-dim | High capacity | NRMSE ~0.11 (w/ TTC) | $1.20 | Diminishing returns vs 32-dim |
| 512-dim | Research | NRMSE ~0.002 (baseline) | $5-10 | Overkill for most cases |

**Recommendation:** Start with 32-dim. Only scale up if specific use case requires it.

### Hyperparameter Guidelines

**Operator Stage:**
- LR: `1e-3` (constant, no scheduler)
- Epochs: `15-25` (25 for production)
- Weight decay: `0.02-0.03`
- Batch size: `8-12` for 32-dim

**Diffusion Stage:**
- LR: `3e-5 to 5e-5`
- Epochs: `5-8`
- Scheduler: `CosineAnnealingLR`
- EMA decay: `0.999`

**Consistency Distillation:**
- LR: `2e-5 to 3e-5`
- Epochs: `6-8`
- `distill_num_taus`: `5` (balance speed/quality)
- `distill_micro_batch`: `3-4`

### TTC Configuration

**Default (Good):**
- `candidates: 4`, `beam_width: 3`, `max_evaluations: 100`
- ~10-15% NRMSE improvement
- ~10 min evaluation

**Enhanced (Better):**
- `candidates: 8`, `beam_width: 3`, `max_evaluations: 150`
- ~15-20% NRMSE improvement
- ~15-20 min evaluation

**Aggressive (Best, but slow):**
- `candidates: 16`, `beam_width: 5`, `max_evaluations: 300`
- ~20-25% NRMSE improvement
- ~30-40 min evaluation
- **Warning:** Diminishing returns, use sparingly

---

## GPU Selection Guide

### VastAI Instance Selection

```bash
# Find cheapest suitable instance
python scripts/vast_launch.py --search-only --min-gpu-ram 24

# Recommended GPUs by use case:
# - A100 (40GB): Best cost/performance for 32-64 dim
# - H100 (80GB): Overkill for most cases, use for 512-dim
# - H200 (141GB): Only for massive batch sizes or 512-dim
```

### Cost Optimization

| GPU | $/hr | 32-dim Training | 512-dim Training | Recommendation |
|-----|------|-----------------|------------------|----------------|
| A100 40GB | $1.89 | ✅ $0.80 (~25 min) | ❌ Too small | **Best for 32-64 dim** |
| H100 80GB | $2.89 | ⚠️ $1.20 (overkill) | ✅ $5-8 | For 512-dim or parallel experiments |
| H200 141GB | $2.59 | ⚠️ $1.10 (overkill) | ✅ $4-6 | For 512-dim with large batch |

**Golden Rule:** Use cheapest GPU that fits your model + batch size.

---

## Common Patterns

### Pattern 1: Hyperparameter Sweep

```bash
# Create variations of base config
cp configs/train_burgers_32dim.yaml configs/exp_lr_sweep_1.yaml
# Edit: change LR to 5e-4

cp configs/train_burgers_32dim.yaml configs/exp_lr_sweep_2.yaml
# Edit: change LR to 2e-3

# Validate all
for config in configs/exp_lr_*.yaml; do
    python scripts/validate_config.py $config || exit 1
done

# Dry-run to estimate total cost
for config in configs/exp_lr_*.yaml; do
    python scripts/dry_run.py $config --estimate-only
done

# Launch (manually or with fleet manager)
for config in configs/exp_lr_*.yaml; do
    python scripts/vast_launch.py --config $config
done

# Compare results
python scripts/compare_runs.py run1_id run2_id run3_id --output reports/lr_sweep.md
```

### Pattern 2: Resume from Checkpoint

```bash
# Training interrupted? Just rerun - it will resume automatically
python scripts/train.py configs/train_burgers_32dim.yaml

# Checkpoints are backed up automatically before overwriting
# If you need to restore: cp checkpoints/operator.pt.backup checkpoints/operator.pt
```

### Pattern 3: Evaluate Pre-trained Model

```bash
# Download checkpoint from WandB
wandb artifact get user/project/model:v0 --root checkpoints/

# Run evaluation only
python scripts/evaluate.py --config configs/train_burgers_32dim.yaml \
    --operator checkpoints/operator.pt \
    --diffusion checkpoints/diffusion_residual.pt \
    --device cuda

# With TTC
python scripts/evaluate.py --config configs/train_burgers_32dim.yaml \
    --operator checkpoints/operator.pt \
    --diffusion checkpoints/diffusion_residual.pt \
    --device cuda \
    --ttc
```

---

## Troubleshooting

### Issue: Dimension Mismatch Errors

**Symptoms:**
```
RuntimeError: size mismatch for diffusion_residual.latent_proj.weight: 
copying a param with shape torch.Size([96, 32]) from checkpoint, 
where the shape is torch.Size([64, 32]) in current model.
```

**Cause:** Config hidden_dim doesn't match checkpoint

**Solution:**
```bash
# 1. Validate config first
python scripts/validate_config.py configs/your_config.yaml

# 2. If resuming, ensure config matches checkpoint:
python -c "import torch; ckpt = torch.load('checkpoints/diffusion_residual.pt', map_location='cpu'); print('Hidden dim:', ckpt['time_mlp.0.weight'].shape[0])"

# 3. Use self-contained configs (no include directives)
# configs/train_burgers_32dim.yaml is already self-contained
```

### Issue: Data Download Failures

**Symptoms:**
```
rclone: error reading source directory: directory not found
```

**Cause:** B2 credentials incorrect or data path wrong

**Solution:**
```bash
# 1. Verify credentials
rclone listremotes

# 2. Test connection
rclone ls B2TRAIN:pdebench/full/burgers1d/ | head

# 3. Download will auto-retry 3 times with exponential backoff
# If all retries fail, check network and credentials
```

### Issue: OOM (Out of Memory)

**Symptoms:**
```
CUDA error: out of memory
```

**Solutions:**
```bash
# 1. Reduce batch size
# Edit config: training.batch_size: 8 → 6

# 2. Disable num_workers (slower but less memory)
# Edit config: training.num_workers: 8 → 0

# 3. Use smaller latent dimension
# Edit config: latent.dim: 32 → 16

# 4. Disable compilation temporarily
# Edit config: training.compile: true → false
```

### Issue: Poor Convergence

**Symptoms:**
- Operator loss stuck > 0.001
- Diffusion loss > 0.01
- Grad norms increasing

**Diagnosis:**
```bash
# Analyze run
python scripts/analyze_run.py <run_id> --output reports/diagnosis.md

# Check recommendations section for specific fixes
```

**Common Fixes:**
- **Operator not converging:** Increase epochs (25 → 35) or reduce LR (1e-3 → 5e-4)
- **Diffusion unstable:** Add/increase gradient clipping (grad_clip: 1.0 → 0.5)
- **Grad norms exploding:** Reduce LR by 2-5x

### Issue: WandB Authentication Failures

**Symptoms:**
```
wandb.errors.AuthenticationError: API key verification failed
```

**Solution:**
```bash
# 1. Verify API key
wandb login --relogin

# 2. Check .env file
cat .env | grep WANDB

# 3. Ensure entity is correct in config
# configs/train_burgers_32dim.yaml:
#   logging.wandb.entity: your_username_here
```

---

## Performance Benchmarks

### 32-dim Burgers1D (Production Config)

**Hardware:** A100 40GB, $1.89/hr

| Stage | Steps | Time | Loss (final) | Grad Norm (final) |
|-------|-------|------|--------------|-------------------|
| Operator | 25 epochs | 2 min | 0.00023 | 0.05 |
| Diffusion | 8 epochs | 1 min | 0.0048 | 0.18 |
| Distillation | 8 epochs | 18 min | 1.2e-6 | 0.02 |
| Eval (baseline) | - | 10 min | - | - |
| Eval (TTC) | - | 15 min | - | - |

**Metrics:**
- Baseline NRMSE: 0.7845
- TTC NRMSE: 0.0921 (88.3% improvement)
- Total cost: ~$1.25

**Reference Run:** `emgun-morpheus-space/universal-simulator/rv86k4w1`

---

## Security Best Practices

1. **Never commit credentials**
   - Use `.env` files (in `.gitignore`)
   - Use environment variables on remote instances

2. **Use read-only configs in Docker**
   - Mount configs as read-only: `-v ./configs:/workspace/configs:ro`

3. **Validate configs before launch**
   - Prevents malicious config injection
   - Catches accidents early

4. **Use non-root user in containers**
   - Dockerfile already configured with `trainer` user

5. **Limit checkpoint access**
   - Don't expose checkpoint directory publicly
   - Use WandB artifacts for sharing

---

## Next Steps

- **For production use:** Follow "Production Training Workflow"
- **For experimentation:** Use "Common Patterns" as templates
- **For debugging:** Check "Troubleshooting" section
- **For operations:** See `docs/runbook.md`

---

## Additional Resources

- **Runbook:** `docs/runbook.md` - Step-by-step operational procedures
- **Docker Guide:** `DOCKER_USAGE.md` - Container setup and deployment
- **Architecture:** `docs/unified_training_eval_pipeline.md` - Pipeline details
- **Optimization:** `docs/parallel_cache_optimization.md` - Performance tuning
- **Data:** `docs/data_artifacts.md` - Dataset locations and formats

---

**Questions or Issues?** Check troubleshooting first, then open an issue on GitHub.

