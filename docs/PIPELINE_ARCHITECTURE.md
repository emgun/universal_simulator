# Pipeline Architecture

**Version:** 2.0 (Streamlined)
**Last Updated:** 2025-01-22

## Overview

The Universal Simulator pipeline has been **streamlined** to eliminate redundant entry points, provide clear script organization, and implement automated experiment lifecycle management.

## Core Principles

1. **Single Orchestrator**: `run_fast_to_sota.py` is the main pipeline orchestrator
2. **Modular Design**: Orchestrator delegates to specialized scripts (`train.py`, `evaluate.py`)
3. **Clear Separation**: Development/debugging scripts vs. production orchestration
4. **Automated Lifecycle**: Experiments are automatically archived to prevent clutter

## Pipeline Flow

```
┌──────────────────────────────────────────────────────────────┐
│                    VastAI Launch                             │
│  scripts/vast_launch.py                                      │
│    ↓ (generates)                                             │
│  .vast/onstart.sh                                            │
└─────────────────────┬────────────────────────────────────────┘
                      ↓
┌──────────────────────────────────────────────────────────────┐
│              Main Pipeline Orchestrator                      │
│  scripts/run_fast_to_sota.py                                 │
│                                                               │
│  Stage 1: Validation                                         │
│    • validate_config.py                                      │
│    • validate_data.py                                        │
│    • dry_run.py                                              │
│                                                               │
│  Stage 2: Training                                           │
│    • Calls scripts/train.py (subprocess)                     │
│    • train.py runs all stages:                               │
│      - operator → diff_residual →                            │
│        consistency_distill → steady_prior                    │
│                                                               │
│  Stage 3: Small Evaluation (proxy gate)                      │
│    • Calls scripts/evaluate.py (subprocess)                  │
│    • Quick validation checkpoint                             │
│                                                               │
│  Stage 4: Full Evaluation                                    │
│    • Calls scripts/evaluate.py (subprocess)                  │
│    • Comprehensive metrics                                   │
│                                                               │
│  Stage 5: Gating & Promotion                                 │
│    • Check NRMSE thresholds                                  │
│    • Update leaderboard                                      │
│    • Optionally promote config                               │
└──────────────────────────────────────────────────────────────┘
```

## Script Organization

### Core Pipeline Scripts

| Script | Role | When to Use |
|--------|------|-------------|
| `run_fast_to_sota.py` | Main orchestrator | Production runs (local or VastAI) |
| `train.py` | Training engine | Debugging individual stages |
| `evaluate.py` | Evaluation engine | Standalone evaluation |
| `vast_launch.py` | VastAI provisioning | Cloud GPU launches |

**Key Design Note**:
- `run_fast_to_sota.py` calls `train.py` and `evaluate.py` via **subprocess**, not by importing them
- This ensures clean process isolation and proper WandB context handling
- Scripts can be used **standalone** for debugging or **orchestrated** for production

### Validation & Quality Scripts

| Script | Purpose |
|--------|---------|
| `validate_config.py` | Check config syntax, dimensions, required fields |
| `validate_data.py` | Verify dataset integrity and availability |
| `dry_run.py` | Test pipeline without execution, estimate costs |

### Experiment Lifecycle Scripts

| Script | Purpose |
|--------|---------|
| `archive_experiments.py` | Archive completed/failed experiments |
| `promote_config.py` | Promote successful experiments to production |
| `config_catalog.py` | Generate CSV catalog of all configs |

### Analysis & Utilities

| Script | Purpose |
|--------|---------|
| `analyze_run.py` | Post-run analysis and reporting |
| `compare_runs.py` | Multi-run comparison |
| `precompute_latent_cache.py` | Pre-encode training data |
| `monitor_instance.sh` | VastAI instance monitoring |

## Experiment Lifecycle Management

### Problem: Config Pile-Up

**Before**: Experimental configs accumulated in `configs/` with unclear status:
```
configs/
  train_burgers_experimental.yaml       # What experiment?
  train_burgers_experimental_v2.yaml    # Is this better?
  train_burgers_32dim_test.yaml         # Is this done?
  train_burgers_64dim_test.yaml         # Should we use this?
```

### Solution: Automated Lifecycle

**After**: Organized directory structure with automated archiving:
```
experiments/                          # Active experiments only
  2025-01-22-64dim-latent/
    config.yaml
    notes.md                          # Document hypothesis & results
    metadata.json

experiments-archive/                  # Historical record
  2025-01-15-baseline-validation/
    config.yaml
    metadata.json                     # Status: success
    notes.md
  2025-01-18-128dim-latent/
    config.yaml
    metadata.json                     # Status: failed (OOM)
    notes.md

configs/                              # Production configs only
  train_burgers_golden.yaml           # Canonical config
  train_burgers_32dim.yaml            # Production variant
```

### Lifecycle States

```
┌─────────────┐
│   Active    │  Recent runs (< 7 days) or in-progress
│  Experiment │
└──────┬──────┘
       │
       ├──────────┐
       ↓          ↓
  ┌─────────┐  ┌────────┐
  │ Success │  │ Failed │  Evaluation complete
  └────┬────┘  └───┬────┘
       │           │
       ↓           ↓
  ┌─────────┐  ┌─────────┐
  │ Promote │  │ Archive │
  │ to Prod │  │         │
  └─────────┘  └─────────┘

  ┌─────────┐
  │  Stale  │  No activity > 30 days
  │         │  → Auto-archive
  └─────────┘
```

### Workflow Example

**1. Start New Experiment**
```bash
# Create experiment directory
mkdir -p experiments/$(date +%Y-%m-%d)-64dim-latent

# Copy base config
cp configs/train_burgers_golden.yaml \
   experiments/2025-01-22-64dim-latent/config.yaml

# Edit experimental config
vim experiments/2025-01-22-64dim-latent/config.yaml
# Change: latent.dim 32 → 64
```

**2. Document Hypothesis**
```bash
cat > experiments/2025-01-22-64dim-latent/notes.md <<EOF
# Experiment: 64-dim Latent Space

## Hypothesis
Increasing latent dimension from 32 to 64 will improve NRMSE by ~20%.

## Config Changes
- latent.dim: 32 → 64
- operator.pdet.input_dim: 32 → 64
- Training time: expect 30-35 min (vs 25 min baseline)

## Results
[To be filled after run]
EOF
```

**3. Run Experiment**
```bash
# VastAI cloud run
python scripts/vast_launch.py launch \
  --config experiments/2025-01-22-64dim-latent/config.yaml \
  --auto-shutdown

# OR local run for debugging
python scripts/train.py \
  --config experiments/2025-01-22-64dim-latent/config.yaml \
  --stage operator
```

**4. Document Results**
```bash
cat >> experiments/2025-01-22-64dim-latent/notes.md <<EOF

## Results
- Baseline NRMSE: 0.78
- TTC NRMSE: 0.07 (improved from 0.09!)
- Training time: 35 min
- WandB run: abc123xyz

## Decision
✓ Promote to production - 22% improvement, worth extra compute
EOF
```

**5a. Promote Success**
```bash
python scripts/promote_config.py \
  experiments/2025-01-22-64dim-latent/config.yaml \
  --production-dir configs/ \
  --rename train_burgers_64dim.yaml \
  --update-leaderboard
```

**5b. Archive Failure/Completion**
```bash
# List experiment status
python scripts/archive_experiments.py --list-only

# Archive all non-active experiments
python scripts/archive_experiments.py --status all
```

## Entry Points Comparison

### Before Streamlining

**Confusing**: Multiple entry points, unclear relationships
```
scripts/
  run_training_pipeline.sh        # Shell orchestrator (legacy)
  run_fast_to_sota.py            # Python orchestrator (newer)
  train.py                       # Direct training (no validation)
  evaluate.py                    # Direct evaluation
  vast_launch.py                 # VastAI launcher

❌ Which one should I use?
❌ What's the difference?
❌ Are they equivalent?
```

### After Streamlining

**Clear**: Single production entry point, clear roles
```
scripts/
  run_fast_to_sota.py           # ✅ PRODUCTION ORCHESTRATOR
    ↳ Calls train.py
    ↳ Calls evaluate.py
  train.py                      # ✅ TRAINING ENGINE (standalone or orchestrated)
  evaluate.py                   # ✅ EVAL ENGINE (standalone or orchestrated)
  vast_launch.py                # ✅ CLOUD PROVISIONING (generates onstart → orchestrator)

✅ Production: run_fast_to_sota.py
✅ Debug training: train.py --stage operator
✅ Debug eval: evaluate.py --checkpoint op_latest.ckpt
✅ Cloud launch: vast_launch.py → run_fast_to_sota.py
```

## VastAI Integration

### Simplified Launch Flow

**One command to production**:
```bash
python scripts/vast_launch.py launch \
  --config configs/train_burgers_golden.yaml \
  --auto-shutdown
```

**What happens**:
1. `vast_launch.py` generates `.vast/onstart.sh`
2. Provisions VastAI instance
3. `onstart.sh` runs on instance startup:
   - Git clone repo
   - Install dependencies
   - Download data from B2
   - Call `run_fast_to_sota.py`
4. `run_fast_to_sota.py` orchestrates full pipeline
5. Instance auto-shuts down after completion

### No More Hardcoded Configs

**Before**: `.vast/onstart.sh` had hardcoded `train_burgers_golden.yaml`
**After**: Config is passed dynamically via `vast_launch.py`

## Removed Scripts

| Script | Reason | Migration |
|--------|--------|-----------|
| `run_training_pipeline.sh` | Redundant with `run_fast_to_sota.py` | Use `run_fast_to_sota.py` instead |

## Best Practices

### For Production Runs

```bash
# 1. Validate config
python scripts/validate_config.py configs/my_config.yaml

# 2. Dry-run cost estimate
python scripts/dry_run.py configs/my_config.yaml --estimate-only

# 3. Launch
python scripts/vast_launch.py launch \
  --config configs/my_config.yaml \
  --auto-shutdown
```

### For Debugging

```bash
# Debug single training stage
python scripts/train.py \
  --config configs/my_config.yaml \
  --stage operator \
  --epochs 1

# Debug evaluation
python scripts/evaluate.py \
  --checkpoint checkpoints/op_latest.ckpt \
  --config configs/eval_burgers.yaml
```

### For Experiments

```bash
# 1. Create experiment
mkdir -p experiments/$(date +%Y-%m-%d)-my-experiment
cp configs/base.yaml experiments/2025-01-22-my-experiment/config.yaml

# 2. Run and document
python scripts/vast_launch.py launch \
  --config experiments/2025-01-22-my-experiment/config.yaml \
  --auto-shutdown

# 3. Promote or archive
python scripts/promote_config.py experiments/2025-01-22-my-experiment/config.yaml
# OR
python scripts/archive_experiments.py --status all
```

### Weekly Maintenance

```bash
# Archive old experiments
python scripts/archive_experiments.py --list-only
python scripts/archive_experiments.py --status all --dry-run
python scripts/archive_experiments.py --status all

# Update config catalog
python scripts/config_catalog.py > docs/config_catalog.csv
```

## Migration Guide

### If You Have Old Experimental Configs

```bash
# Move to experiments/
mv configs/train_*_experimental.yaml \
   experiments/$(date +%Y-%m-%d)-migration/

# Document the experiment
cat > experiments/2025-01-22-migration/notes.md <<EOF
# Migrated from configs/

Original config: train_burgers_experimental.yaml
Status: [document status]
EOF
```

### If You Were Using run_training_pipeline.sh

**Replace with**:
```bash
# Old
./scripts/run_training_pipeline.sh configs/my_config.yaml

# New
python scripts/run_fast_to_sota.py \
  --train-config configs/my_config.yaml \
  --small-eval-config configs/small_eval.yaml \
  --full-eval-config configs/full_eval.yaml
```

## Directory Structure Reference

```
universal_simulator/
├── configs/                          # Production configs only
│   ├── train_burgers_golden.yaml     # Canonical config
│   └── train_burgers_32dim.yaml      # Production variant
│
├── experiments/                      # Active experiments
│   └── YYYY-MM-DD-name/
│       ├── config.yaml
│       ├── notes.md
│       └── metadata.json
│
├── experiments-archive/              # Historical experiments
│   └── YYYY-MM-DD-name/
│       ├── config.yaml
│       ├── notes.md
│       └── metadata.json
│
├── scripts/
│   ├── run_fast_to_sota.py          # Main orchestrator
│   ├── train.py                     # Training engine
│   ├── evaluate.py                  # Evaluation engine
│   ├── vast_launch.py               # VastAI provisioning
│   ├── archive_experiments.py       # Lifecycle automation
│   └── promote_config.py            # Config promotion
│
├── .vast/
│   └── onstart.sh                   # Generated startup script
│
└── docs/
    └── PIPELINE_ARCHITECTURE.md     # This file
```

## FAQ

**Q: When should I use `train.py` vs `run_fast_to_sota.py`?**
A: Use `train.py` for debugging individual training stages. Use `run_fast_to_sota.py` for production runs with full validation, evaluation, and gating.

**Q: Why does `run_fast_to_sota.py` call scripts via subprocess instead of importing them?**
A: Process isolation ensures clean WandB context management and prevents state pollution between pipeline stages.

**Q: Can I still run `evaluate.py` standalone?**
A: Yes! It's designed for standalone use: `python scripts/evaluate.py --checkpoint op_latest.ckpt --config eval.yaml`

**Q: How often should I archive experiments?**
A: Weekly is recommended. Run `python scripts/archive_experiments.py --list-only` to check status.

**Q: What if I want to resurrect an archived experiment?**
A: Copy the config back to `experiments/`: `cp experiments-archive/2025-01-15-test/config.yaml experiments/$(date +%Y-%m-%d)-retry/`

**Q: Should I commit experiments to Git?**
A: Commit configs and notes, but NOT checkpoints or artifacts (excluded via `.gitignore`).

## See Also

- `CLAUDE.md` - Full project documentation
- `experiments/README.md` - Detailed experiment workflow
- `experiments-archive/README.md` - Archive management
- `PRODUCTION_WORKFLOW.md` - VastAI production workflow
