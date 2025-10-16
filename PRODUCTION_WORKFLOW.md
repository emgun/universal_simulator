# Production Training Workflow

**Fast, repeatable, bombproof training runs with quick iteration.**

---

## Overview

This workflow uses pre-built Docker images from GitHub Container Registry for instant, reliable launches.

**Key Benefits:**
- ⚡ **Fast startup:** 2-3 minutes (vs 5-10 with git clone)
- 🔒 **Bombproof:** All dependencies pre-installed and tested
- 🚀 **Quick iteration:** Config changes take seconds, code changes take 5 minutes
- 💰 **Cost-effective:** Less time waiting = lower costs

---

## Setup (One-Time, 5 minutes)

### 1. Enable GitHub Container Registry

The repository is already configured! GitHub Actions will automatically build Docker images when you push code changes.

### 2. Verify Environment Variables

Ensure these are set in your shell:
```bash
export WANDB_API_KEY=your_key
export WANDB_PROJECT=universal-simulator
export WANDB_ENTITY=your_username
export B2_KEY_ID=your_id
export B2_APP_KEY=your_key
export B2_S3_ENDPOINT=s3.us-west-002.backblazeb2.com
export B2_S3_REGION=us-west-002
```

---

## Iteration Workflows

### Config Changes (90% of cases) - INSTANT

**When to use:** Hyperparameter tuning, different epoch counts, TTC settings, etc.

```bash
# 1. Edit config locally
vim configs/train_burgers_32dim.yaml

# 2. Launch with updated config
./scripts/launch_production.sh train_burgers_32dim

# That's it! Uses same pre-built image, just different config.
# Time: 2-3 minutes to start training
```

### Code Changes (10% of cases) - 5 MINUTES

**When to use:** Bug fixes, new features, algorithm changes

```bash
# 1. Make code changes
vim src/ups/models/operator.py

# 2. Commit and push (triggers automatic rebuild)
git add src/ups/models/operator.py
git commit -m "fix: improve operator convergence"
git push

# 3. Wait for GitHub Action to build (~5 min)
# Watch: https://github.com/emgun/universal_simulator/actions

# 4. Launch with new image
./scripts/launch_production.sh train_burgers_32dim

# Time: 5 min build + 2-3 min to start training
```

### Dependency Changes (rare) - 10 MINUTES

**When to use:** Upgrading PyTorch, adding new packages

```bash
# 1. Update requirements.txt
vim requirements.txt

# 2. Commit and push (triggers rebuild)
git add requirements.txt
git commit -m "deps: upgrade pytorch to 2.2.0"
git push

# 3. Wait for build (~10 min)
# 4. Launch

# Time: 10 min build + 2-3 min to start training
```

---

## Production Run Commands

### Quick Launch (Recommended)

```bash
# Uses best available instance
./scripts/launch_production.sh train_burgers_32dim
```

### Manual Launch

```bash
# 1. Find instance
vastai search offers 'reliability > 0.98 num_gpus=1 gpu_name=RTX_4090 dph < 0.5' -o 'dph'

# 2. Launch
./scripts/launch_production.sh train_burgers_32dim 12345678
```

### Monitor

```bash
# Get instance ID from launch output
INSTANCE_ID=26875XXX

# Check status
vastai show instance $INSTANCE_ID

# View logs
vastai logs $INSTANCE_ID | tail -100

# Monitor continuously
watch -n 30 'vastai show instance $INSTANCE_ID'
```

---

## Docker Image Versioning

Images are automatically tagged:

- `latest` - Most recent build from main branch
- `feature-sota_burgers_upgrades` - Current feature branch
- `feature-sota_burgers_upgrades-abc1234` - Specific commit

**Use specific tags for reproducibility:**
```bash
# Use exact version
IMAGE="ghcr.io/emgun/universal_simulator:feature-sota_burgers_upgrades-abc1234"

# Or use latest
IMAGE="ghcr.io/emgun/universal_simulator:latest"
```

---

## Comparison: Old vs New Workflow

### OLD (Manual Setup)
```
1. Pull PyTorch image: 2-3 min
2. apt install packages: 1-2 min
3. git clone repo: 1-2 min
4. pip install deps: 2-3 min
5. Download data: 1-2 min
────────────────────────────
Total: 7-12 minutes to start training
```

### NEW (Production Docker)
```
1. Pull pre-built image: 1-2 min
2. Download data: 1-2 min
────────────────────────────
Total: 2-4 minutes to start training
```

**Savings: 5-8 minutes per launch = ~$0.40-0.65 @ $0.30/hr**

---

## Troubleshooting

### Image not found
```bash
# Ensure you're logged in to ghcr.io
docker login ghcr.io -u YOUR_GITHUB_USERNAME

# Or wait for GitHub Action to complete
# Check: https://github.com/emgun/universal_simulator/actions
```

### Build failed
```bash
# Check GitHub Actions logs
# Fix the issue and push again
git push
```

### Instance won't start
```bash
# Check if image exists
docker pull ghcr.io/emgun/universal_simulator:latest

# If not, trigger manual build:
# Go to: https://github.com/emgun/universal_simulator/actions
# Click "Build and Push Docker Image"
# Click "Run workflow"
```

---

## Advanced Usage

### Override Config at Runtime

```bash
# Use environment variable to specify different config
--env "TRAIN_CONFIG=configs/train_burgers_64dim.yaml"
```

### Mount Custom Config

```bash
# For rapid experimentation
--volume ./my_experiment.yaml:/app/configs/experiment.yaml
```

### Debug Mode

```bash
# SSH into running instance
vastai ssh $INSTANCE_ID

# Check what's happening
cd /app
ps aux | grep python
tail -f logs/training.log
```

---

## Best Practices

1. **Always commit config changes** - Even for experiments, keeps history
2. **Use descriptive commit messages** - Helps track which image has which features
3. **Tag important runs** - `git tag v1.0-baseline` for reproducibility
4. **Monitor builds** - Don't launch until build completes
5. **Test locally first** - Use `docker-compose up` for quick validation

---

## Quick Reference

```bash
# Launch production run
./scripts/launch_production.sh train_burgers_32dim

# Check build status
gh run list --workflow=docker-build.yml

# View latest image
docker pull ghcr.io/emgun/universal_simulator:latest
docker inspect ghcr.io/emgun/universal_simulator:latest

# Local testing
docker-compose up --build

# Stop instance
vastai stop instance $INSTANCE_ID
```

---

**Questions?** See `docs/production_playbook.md` for detailed guidance.
