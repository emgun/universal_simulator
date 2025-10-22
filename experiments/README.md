# Experiments Directory

This directory is for **active experimental work** - configs, notes, and artifacts that are currently being developed or tested.

## Purpose

Keep experimental work organized and separate from production configs while avoiding clutter in the main configs/ directory.

## Structure

```
experiments/
  YYYY-MM-DD-experiment-name/
    config.yaml          # Experimental config
    notes.md            # Experiment notes, hypotheses, results
    checkpoints/        # Experiment-specific checkpoints
    plots/              # Visualizations
    metadata.json       # Experiment tracking
```

## Workflow

### 1. Starting a New Experiment

```bash
# Create experiment directory
mkdir -p experiments/$(date +%Y-%m-%d)-my-experiment

# Create config
cp configs/train_burgers_golden.yaml experiments/2025-01-22-my-experiment/config.yaml

# Edit config with experimental parameters
vim experiments/2025-01-22-my-experiment/config.yaml
```

### 2. Running the Experiment

```bash
# Local run
python scripts/train.py \
  --config experiments/2025-01-22-my-experiment/config.yaml \
  --stage all

# VastAI run
python scripts/vast_launch.py launch \
  --config experiments/2025-01-22-my-experiment/config.yaml \
  --auto-shutdown
```

### 3. Documenting Results

Create a `notes.md` file in your experiment directory:

```markdown
# Experiment: 64-dim Latent Space Test

## Hypothesis
Increasing latent dimension from 32 to 64 will improve NRMSE by ~20%.

## Config Changes
- latent.dim: 32 → 64
- operator.pdet.input_dim: 32 → 64

## Results
- Baseline NRMSE: 0.78
- TTC NRMSE: 0.07 (improved from 0.09!)
- Training time: 35 min (vs 25 min baseline)

## Decision
✓ Promote to production - significant improvement worth the extra time
```

### 4. Experiment Lifecycle

**Success Path:**
```bash
# Promote successful experiment
python scripts/promote_config.py \
  experiments/2025-01-22-my-experiment/config.yaml \
  --production-dir configs/ \
  --rename train_burgers_64dim.yaml \
  --update-leaderboard
```

**Archive Path:**
```bash
# Archive completed experiments (success, failed, or stale)
python scripts/archive_experiments.py \
  --experiments-dir experiments/ \
  --status all \
  --dry-run  # Check what will be archived

# Actually archive
python scripts/archive_experiments.py --status all
```

## Best Practices

1. **One experiment per directory** - Keep related work together
2. **Date prefix naming** - `YYYY-MM-DD-description` for chronological sorting
3. **Document hypotheses** - Write down what you're testing and why
4. **Track results** - Keep notes.md updated with findings
5. **Clean up regularly** - Archive or promote experiments after completion
6. **Tag experiments** - Use WandB tags for grouping related experiments

## Automation

The experiment lifecycle is automated:

- **Active experiments** (< 7 days old) stay in `experiments/`
- **Successful experiments** can be promoted to `configs/`
- **Completed/failed experiments** (> 7 days old) auto-archive to `experiments-archive/`
- **Stale experiments** (> 30 days old) auto-archive

Run the archiver weekly:
```bash
# List experiments and their status
python scripts/archive_experiments.py --list-only

# Archive completed/stale experiments
python scripts/archive_experiments.py --status all
```

## Examples

### Hyperparameter Sweep
```
experiments/
  2025-01-22-lr-sweep/
    config_lr1e4.yaml
    config_lr5e4.yaml
    config_lr1e3.yaml
    notes.md              # Track which LR worked best
    results.csv           # Comparison table
```

### Architecture Experiment
```
experiments/
  2025-01-23-deeper-pdet/
    config.yaml           # 12 layers instead of 6
    notes.md
    ablation_study.md     # Depth vs performance
```

### Data Experiment
```
experiments/
  2025-01-24-augmentation/
    config.yaml           # New augmentation strategy
    data_samples/         # Example augmented data
    notes.md
```

## Migration

If you have experimental configs in `configs/`, migrate them:

```bash
# Move experimental config to experiments/
mv configs/train_burgers_experimental.yaml \
   experiments/$(date +%Y-%m-%d)-baseline-validation/config.yaml
```
