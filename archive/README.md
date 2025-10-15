# Archive Directory

This directory contains deprecated scripts and configurations that are no longer actively used in the main pipeline but are preserved for reference.

## Contents

### `configs/` - Deprecated Configuration Files
- Old training iterations (v1, v2, smoke tests)
- Temporary evaluation variants (fixed, neutral, baseline)
- One-off experimental configs

### `scripts/` - Deprecated Scripts
- One-off debugging scripts (remote_fix_and_run.sh, restart_*.sh)
- Superseded launchers (launch_and_run_cheapest.sh)
- Old versions (resume_from_wandb.sh replaced by v2)

## Archival Criteria

Files are archived when they:
1. Are superseded by newer versions
2. Were created for one-off debugging/testing
3. Are no longer referenced in active pipelines
4. Haven't been modified in recent commits

## Recovery

If you need a file from the archive:
```bash
cp archive/configs/FILENAME configs/
# or
cp archive/scripts/FILENAME scripts/
```

## Archive Date: October 14, 2025

**Archived by:** Codebase cleanup initiative
**Main pipeline:** scripts/run_remote_scale.sh with configs/train_burgers_quality_v3.yaml
