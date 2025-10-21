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
# 1. Validate config & data wiring (no training)
python scripts/run_fast_to_sota.py \
  --train-config configs/train_burgers_32dim_golden.yaml \
  --small-eval-config configs/small_eval_rerun_txxoc8a8.yaml \
  --full-eval-config configs/full_eval_rerun_txxoc8a8.yaml \
  --skip-training --skip-small-eval --skip-full-eval

# 2. Dry-run (estimate cost and time)
python scripts/dry_run.py <TRAIN_CONFIG> --estimate-only

# 3. Launch production run on Vast
python scripts/vast_launch.py launch \
  --gpu <PREFERRED_GPU> \
  --config <TRAIN_CONFIG> \
  --auto-shutdown \
  --run-arg=--wandb-sync \
  --run-arg=--wandb-run-name=<WANDB_RUN_NAME> \
  --run-arg=--leaderboard-wandb \
  --run-arg=--leaderboard-wandb-project=<WANDB_PROJECT> \
  --run-arg=--leaderboard-wandb-entity=<WANDB_ENTITY>

# 4. Analyze results (after completion)
python scripts/analyze_run.py <wandb_run_id> --output reports/analysis.md
```

### Expected Timeline (example)

| Stage | Duration (typical) | Checkpoints |
|-------|--------------------|-------------|
| Latent Cache | 2-3 min | latent_cache/ |
| Operator (constant LR) | 2-3 min | operator.pt / operator_ema.pt |
| Diffusion (cosine anneal) | 1-2 min | diffusion_residual.pt |
| Consistency distill (τ schedule, tight target) | 8-12 min | diffusion_residual.pt / diffusion_residual_ema.pt |
| Evaluation (baseline + TTC) | 15-20 min | reports/, leaderboard rows |
| **Total** | **≈35 min** | **Cost scales with GPU rate** |

---

## Configuration Best Practices

### Choosing Latent Dimension

- Start with the smallest latent size that meets accuracy targets to minimize cost.
- Increase latent dimension or depth only when diagnostics show bottlenecks (e.g., persistent bias on hard regimes).
- If you scale capacity, re-tune optimizer scale (LR, batch tokens) and TTC settings to keep gates satisfied.

### Hyperparameter Guidelines

**Operator Stage:**
- LR: `1e-3` (constant, no scheduler)
- Epochs: `15-25` (25 for production)
- Weight decay: `0.02-0.03`
- Batch size: `8-12` for 32-dim

**Diffusion Stage:**
- LR: `~3e-5` (scale with batch tokens and optimizer choice)
- Epochs: `≈5`
- Scheduler: `CosineAnnealingLR` (keep enabled; reruns rely on smooth decay)
- EMA decay: `0.999`

**Consistency Distillation:**
- LR: `~2e-5`
- Epochs: `≤6` (stop once loss hits the target tolerance, e.g. `≤5e-6`)
- Scheduler: `CosineAnnealingLR`
- `distill_num_taus`: `≈5` (balance speed/quality)
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
python scripts/vast_launch.py search -g <GPU_MODEL> --max-price <BID_LIMIT>
```

### Cost Optimization
Match GPU memory to expected peak batch tokens + evaluation workloads. Use the cheapest instance that comfortably fits the job; if you need more VRAM later, adjust the search filter and rerun.

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
