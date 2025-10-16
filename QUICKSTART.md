# Quick Start Guide

**Get up and running in 5 minutes.**

---

## Production Training

**Prerequisites:** Set environment variables (one time):
```bash
export B2_KEY_ID=your_key
export B2_APP_KEY=your_app_key
export B2_S3_ENDPOINT=s3.us-west-002.backblazeb2.com
export B2_S3_REGION=us-west-002
export WANDB_API_KEY=your_key
export WANDB_ENTITY=your_entity
```

**Option A: Docker (faster, ~2-3 min startup)**
```bash
./scripts/launch_production.sh train_burgers_32dim
```

**Option B: Git clone (more reliable, ~7 min startup)**
```bash
python scripts/vast_launch.py launch \
  --overrides "TRAIN_CONFIG=configs/train_burgers_32dim.yaml TRAIN_STAGE=all" \
  --b2-key-id "$B2_KEY_ID" --b2-app-key "$B2_APP_KEY" \
  --b2-s3-endpoint "$B2_S3_ENDPOINT" --b2-s3-region "$B2_S3_REGION" \
  --wandb-api-key "$WANDB_API_KEY" --wandb-entity "$WANDB_ENTITY" \
  --auto-shutdown
```

**See:** `PRODUCTION_WORKFLOW.md` for complete guide and troubleshooting.

---

## Manual Launch (Development/Testing)

```bash
# 1. Validate config (5 seconds)
python scripts/validate_config.py configs/train_burgers_32dim.yaml

# 2. Check data (10 seconds)  
python scripts/validate_data.py configs/train_burgers_32dim.yaml

# 3. Estimate cost (10 seconds)
python scripts/dry_run.py configs/train_burgers_32dim.yaml --estimate-only

# 4. Launch training
python scripts/train.py --config configs/train_burgers_32dim.yaml --stage all

# Expected: ~25 min training, $1.25 cost, 0.09 NRMSE with TTC
```

---

## After Training Completes

```bash
# Get run ID from WandB or vast logs
RUN_ID="abc123def"

# Analyze run (30 seconds)
python scripts/analyze_run.py $RUN_ID --output reports/analysis.md

# Compare with baseline (if you have another run)
python scripts/compare_runs.py $RUN_ID baseline_run_id --output reports/comparison.md
```

---

## Common Commands

### Validation

```bash
# Full validation suite
python scripts/validate_config.py configs/train_burgers_32dim.yaml
python scripts/validate_data.py configs/train_burgers_32dim.yaml
python scripts/dry_run.py configs/train_burgers_32dim.yaml
```

### Training

```bash
# Local (for testing)
python scripts/train.py configs/train_burgers_32dim.yaml

# Remote (VastAI)
python scripts/vast_launch.py --config configs/train_burgers_32dim.yaml
```

### Monitoring

```bash
# Check instance status
python scripts/monitor_instance.sh <instance_id>

# View WandB dashboard
open https://wandb.ai/<entity>/universal-simulator
```

### Analysis

```bash
# Single run
python scripts/analyze_run.py <run_id>

# Compare runs
python scripts/compare_runs.py run1_id run2_id run3_id
```

---

## Docker Workflow

```bash
# Build and start
docker-compose up --build -d

# Enter container
docker-compose exec universal_simulator bash

# Inside container
python scripts/train.py configs/train_burgers_32dim.yaml

# Stop
docker-compose down
```

---

## Troubleshooting

| Issue | Quick Fix |
|-------|-----------|
| Dimension mismatch | Use `configs/train_burgers_32dim.yaml` (self-contained) |
| Data not found | Run `python scripts/validate_data.py` |
| Config errors | Run `python scripts/validate_config.py` |
| OOM errors | Reduce `batch_size` in config |
| Poor convergence | Check `docs/production_playbook.md` troubleshooting |

**For detailed help:** See `docs/production_playbook.md` or `docs/runbook.md`

---

## Key Files

| File | Purpose |
|------|---------|
| `configs/train_burgers_32dim.yaml` | Production-ready 32-dim config |
| `scripts/validate_config.py` | Validate config before launch |
| `scripts/validate_data.py` | Check data integrity |
| `scripts/dry_run.py` | Pre-flight test + cost estimate |
| `scripts/analyze_run.py` | Analyze training results |
| `scripts/compare_runs.py` | Compare multiple runs |
| `docs/production_playbook.md` | Best practices guide |
| `docs/runbook.md` | Operational procedures |

---

## Expected Performance (32-dim)

| Metric | Value |
|--------|-------|
| Baseline NRMSE | 0.78 |
| TTC NRMSE | 0.09 |
| Improvement | 88% |
| Training time | ~25 min |
| Cost (A100) | $1.25 |
| Operator final loss | 0.00023 |

**Reference run:** `emgun-morpheus-space/universal-simulator/rv86k4w1`

---

**Need help?** Check the playbook first: `docs/production_playbook.md`

