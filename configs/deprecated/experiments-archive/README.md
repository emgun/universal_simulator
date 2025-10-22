# Experiments Archive

This directory contains **archived experiments** - completed, failed, or stale experimental work that has been moved out of active development.

## Purpose

Keep a historical record of all experiments while keeping the active `experiments/` directory clean and focused.

## Structure

```
experiments-archive/
  YYYY-MM-DD-experiment-name/
    config.yaml          # Original experimental config
    metadata.json        # Archive metadata and classification
    artifacts/           # Run artifacts (optional, can be large)
    reports/            # Evaluation reports
    checkpoints/        # Checkpoints (optional, can be large)
```

## Archive Metadata

Each archived experiment has a `metadata.json` file:

```json
{
  "config_name": "train_burgers_experimental.yaml",
  "status": "success",
  "created_at": "2025-01-15T10:30:00",
  "archived_at": "2025-01-22T14:20:00",
  "artifacts": [
    {"path": "artifacts/runs/rv86k4w1", "size_mb": 125.3}
  ],
  "reports": [
    {"path": "reports/full_eval.json", "modified": "2025-01-22T12:00:00"}
  ],
  "checkpoints": [
    {"path": "checkpoints/op_latest.ckpt", "size_mb": 42.1}
  ]
}
```

## Experiment Status Classifications

Experiments are archived with one of these status labels:

- **success**: Completed runs that passed validation gates
- **failed**: Runs that encountered errors or failed gates
- **stale**: Experiments with no activity for > 30 days

## Archiving Process

### Automated Archiving

```bash
# List all experiments and their status
python scripts/archive_experiments.py --list-only

# Archive all non-active experiments (dry-run first)
python scripts/archive_experiments.py --status all --dry-run

# Actually archive
python scripts/archive_experiments.py --status all
```

### Selective Archiving

```bash
# Archive only successful experiments
python scripts/archive_experiments.py --status success

# Archive only failed experiments
python scripts/archive_experiments.py --status failed

# Archive only stale experiments (> 30 days old)
python scripts/archive_experiments.py --status stale
```

### Manual Archiving

If you need to manually archive an experiment:

```bash
# Create archive directory
mkdir -p experiments-archive/2025-01-22-my-experiment

# Move config and artifacts
cp experiments/2025-01-22-my-experiment/config.yaml \
   experiments-archive/2025-01-22-my-experiment/

# Create metadata
cat > experiments-archive/2025-01-22-my-experiment/metadata.json <<EOF
{
  "config_name": "my_experiment.yaml",
  "status": "failed",
  "archived_at": "$(date -Iseconds)",
  "reason": "Manual archive - OOM errors on A100"
}
EOF

# Remove from experiments/
rm -rf experiments/2025-01-22-my-experiment
```

## Retrieving Archived Experiments

To resurrect an archived experiment:

```bash
# Copy config back to experiments/
cp experiments-archive/2025-01-22-my-experiment/config.yaml \
   experiments/$(date +%Y-%m-%d)-retry-my-experiment/config.yaml

# Review metadata for context
cat experiments-archive/2025-01-22-my-experiment/metadata.json
```

## Storage Management

Archives can grow large over time. Manage storage:

```bash
# Check archive sizes
du -sh experiments-archive/*

# Remove old checkpoints (keep metadata and configs)
find experiments-archive -name "checkpoints" -type d -exec rm -rf {} +

# Compress old archives
tar -czf experiments-archive-2024.tar.gz experiments-archive/2024-*
rm -rf experiments-archive/2024-*
```

## Search and Analysis

### Find experiments by status

```bash
# List all successful experiments
grep -l '"status": "success"' experiments-archive/*/metadata.json

# List all failed experiments
grep -l '"status": "failed"' experiments-archive/*/metadata.json
```

### Find experiments by metric

```bash
# Find experiments with NRMSE < 0.10
for meta in experiments-archive/*/metadata.json; do
  nrmse=$(jq -r '.metrics.final_nrmse // "N/A"' "$meta")
  if [ "$nrmse" != "N/A" ] && (( $(echo "$nrmse < 0.10" | bc -l) )); then
    echo "$(dirname $meta): NRMSE=$nrmse"
  fi
done
```

### Generate archive report

```bash
# Count experiments by status
echo "Archive Summary:"
echo "  Success: $(grep -l '"status": "success"' experiments-archive/*/metadata.json | wc -l)"
echo "  Failed:  $(grep -l '"status": "failed"' experiments-archive/*/metadata.json | wc -l)"
echo "  Stale:   $(grep -l '"status": "stale"' experiments-archive/*/metadata.json | wc -l)"
echo "  Total:   $(ls -d experiments-archive/*/ 2>/dev/null | wc -l)"
```

## Best Practices

1. **Archive regularly** - Run archiver weekly to keep experiments/ clean
2. **Review before deleting** - Always dry-run before archiving
3. **Keep metadata** - Even if you delete checkpoints, keep metadata.json
4. **Document failures** - Add notes about why experiments failed
5. **Learn from history** - Review archived experiments before starting similar work
6. **Compress old archives** - Tar/gzip experiments older than 6 months

## Retention Policy

Suggested retention:

- **Metadata and configs**: Keep indefinitely (small size)
- **Reports and plots**: Keep for 1 year
- **Checkpoints**: Keep for 3 months (or promote to production)
- **Artifacts**: Keep for 1 month (or until archived)

## Integration with Version Control

**DO NOT commit** large artifacts or checkpoints to Git. The `.gitignore` should exclude:

```gitignore
experiments-archive/*/checkpoints/
experiments-archive/*/artifacts/
```

**DO commit** metadata and configs:

```gitignore
# In .gitignore (allow metadata)
!experiments-archive/*/metadata.json
!experiments-archive/*/config.yaml
!experiments-archive/*/notes.md
```

This allows team members to see experiment history without bloating the repo.

## Example Archive

```
experiments-archive/
  2025-01-15-baseline-validation/
    config.yaml                    # Original config (3 KB)
    metadata.json                  # Archive metadata (2 KB)
    notes.md                       # Experiment notes (5 KB)
    reports/
      full_eval.json              # Evaluation results (15 KB)
      plots/
        rollout.png               # Visualization (120 KB)

  2025-01-18-64dim-latent/
    config.yaml
    metadata.json
    artifacts/
      runs/rv86k4w1/              # Full run artifacts (125 MB)
    checkpoints/
      op_latest.ckpt              # Checkpoint (42 MB)
```
