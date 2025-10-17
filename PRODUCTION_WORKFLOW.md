# Production Training Workflow

**Fast, repeatable, bombproof training runs on VastAI.**

---

## Overview

This workflow uses VastAI's native PyTorch image with global environment variables for secure, reliable launches.

**Key Benefits:**
- ⚡ **Fast startup:** 3-4 minutes (git clone + deps)
- 🔒 **Bombproof:** VastAI's PyTorch image has proper CUDA/Triton setup
- 🔐 **Secure:** Credentials stored as VastAI global env-vars (not in scripts)
- ✨ **Simple:** 2 files total (vast_launch.py + run_training_pipeline.sh)
- 💯 **Reliable:** torch.compile() works natively, no workarounds

**Architecture:**
- Image: `vastai/pytorch` (PyTorch preinstalled at `/venv/main/`)
- Files: `scripts/vast_launch.py` + `scripts/run_training_pipeline.sh`
- Credentials: VastAI global env-vars (set once, use everywhere)

---

## Setup (One-Time, 2 minutes)

### Configure VastAI Global Environment Variables

Run this once to set credentials that will be injected into all future instances:

```bash
# Load credentials from .env
source .env

# Configure VastAI env-vars (one-time)
python scripts/vast_launch.py setup-env
```

This configures:
- `WANDB_API_KEY`, `WANDB_PROJECT`, `WANDB_ENTITY`
- `B2_KEY_ID`, `B2_APP_KEY`, `B2_S3_ENDPOINT`, `B2_S3_REGION`, `B2_BUCKET`

**Verify:**
```bash
vastai show env-vars
```

---

## Launching Training

### Standard Launch

```bash
python scripts/vast_launch.py launch \
  --config configs/train_burgers_32dim.yaml \
  --auto-shutdown
```

### Advanced Options

```bash
# Specific GPU type
python scripts/vast_launch.py launch \
  --config configs/train_burgers_32dim.yaml \
  --gpu H100_PCIE \
  --auto-shutdown

# Specific offer ID (fastest)
python scripts/vast_launch.py launch \
  --offer-id 12345678 \
  --config configs/train_burgers_32dim.yaml \
  --auto-shutdown

# Dry-run (test without launching)
python scripts/vast_launch.py launch \
  --config configs/train_burgers_32dim.yaml \
  --dry-run
```

### Search for Instances

```bash
# Find cheapest RTX 4090
vastai search offers 'reliability > 0.98 num_gpus=1 gpu_name=RTX_4090' --order 'dph'

# Find cheapest H100
vastai search offers 'reliability > 0.98 num_gpus=1 gpu_name=H100_PCIE' --order 'dph'
```

---

## Monitoring

### Check Status

```bash
# Show instance details
vastai show instance <ID>

# Watch logs (live)
vastai logs <ID> -f

# Check GPU utilization
vastai show instance <ID> | grep "Util"
```

### Monitor Training Progress

```bash
# Tail logs
vastai logs <ID> 2>&1 | tail -50

# Watch for errors
vastai logs <ID> 2>&1 | grep -i "error\|warning"

# Check WandB
# Look for run URL in logs, e.g.:
# wandb: 🚀 View run at https://wandb.ai/...
```

---

## What Happens During Launch

1. **Instance provisioning** (~30 sec)
   - VastAI finds GPU
   - Starts container with `vastai/pytorch` image
   
2. **Onstart script execution** (~3-4 min)
   - Install system dependencies (git, rclone, build-essential)
   - Clone repository
   - Activate VastAI's PyTorch venv (`/venv/main/`)
   - Install additional dependencies (our requirements.txt)
   - Download training data from B2 (burgers1d_train_000.h5)
   
3. **Training pipeline** (hours)
   - Precompute latent cache
   - Train operator model
   - Train diffusion residual
   - Consistency distillation
   - Evaluation (optional)
   - Auto-shutdown (if `--auto-shutdown` flag used)

---

## Troubleshooting

### Instance Stuck in "loading"

```bash
# Destroy and relaunch
vastai destroy instance <ID>
python scripts/vast_launch.py launch --config configs/train_burgers_32dim.yaml
```

### Training Errors

```bash
# SSH into instance
vastai ssh <ID>

# Check what's running
tmux ls
tmux attach -t training  # if using tmux

# Check full logs
tail -100 /workspace/universal_simulator/logs/*.log
```

### Out of Disk Space

```bash
# Launch with more disk
python scripts/vast_launch.py launch \
  --config configs/train_burgers_32dim.yaml \
  --disk 128  # 128GB instead of default 64GB
```

---

## Technical Details

### VastAI PyTorch Image

- **Image:** `vastai/pytorch`
- **PyTorch location:** `/venv/main/` (preinstalled)
- **Benefits:** Proper CUDA/Triton setup, no reinstall needed, torch.compile() works natively
- **No hacks needed:** libcuda symlinks are already configured

### File Structure

```
scripts/
├── vast_launch.py (280 lines)
│   ├── setup-env: Configure VastAI global env-vars
│   ├── launch: Generate onstart.sh + launch instance
│   └── search: Find instances
│
└── run_training_pipeline.sh (199 lines)
    ├── Validate required env-vars
    ├── Download test/val data (optional)
    ├── Precompute latent cache
    ├── Multi-stage training
    └── Evaluation (optional)
```

### Environment Variables

**Set via VastAI (`vastai create env-var`):**
- `WANDB_API_KEY`, `WANDB_PROJECT`, `WANDB_ENTITY`
- `B2_KEY_ID`, `B2_APP_KEY`, `B2_S3_ENDPOINT`, `B2_S3_REGION`

**Set by onstart.sh:**
- `TRAIN_CONFIG` (e.g., `configs/train_burgers_32dim.yaml`)
- `TRAIN_STAGE` (default: `all`)
- `RESET_CACHE` (default: `1`)

---

## Future: Docker Optimization

The Docker workflow (pre-built images with code baked in) can provide even faster iteration:
- Startup: 1-2 min (vs 3-4 min git clone)
- Code changes: 5 min (rebuild + push image)

**Archived for reference:**
- `archive/scripts/launch_production_docker.sh`
- `Dockerfile`
- `.github/workflows/docker-build.yml`

**Challenges to solve:**
- VastAI container runtime issues
- Image size optimization
- Registry authentication

**Current recommendation:** Use git clone workflow (more reliable)
```
