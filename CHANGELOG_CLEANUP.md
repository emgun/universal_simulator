# Codebase Cleanup Changelog - October 14, 2025

## Summary

Major codebase cleanup initiative to establish a clean, standardized training/evaluation pipeline for the Universal Simulator project. Results uploaded to W&B, comprehensive documentation created, and deprecated files archived.

## Metrics

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Config files | 33 | 25 | -8 (-24%) |
| Shell scripts | 26 | 18 | -8 (-31%) |
| Python scripts | 21 | 20 | -1 (-5%) |
| Temp directories | 121MB | 0MB | -121MB |
| Root log files | 8 | 0 | -8 |
| Documentation | None | 2 guides | +21KB |

## Changes

### 1. W&B Artifact Upload ✅

**New Artifacts:**
- `ttc-burgers512-val-lowmem-results` - Validation TTC evaluation results
- `ttc-burgers512-test-lowmem-results` - Test TTC evaluation results

**Contents:**
- JSON metrics (MSE, MAE, RMSE)
- CSV data exports
- HTML reports
- TTC step logs (trajectory selection details)
- Reward progression plots
- Latent space visualizations
- Metadata with config details

**View at:** https://wandb.ai/emgun-morpheus-space/universal-simulator

### 2. Documentation Created ✅

**New Documents:**

1. **docs/pipeline_guide.md** (12KB)
   - Complete training/evaluation workflow guide
   - Standard configurations and environment variables
   - Common tasks and troubleshooting
   - Performance optimization tips
   - Quick start examples

2. **docs/ttc_analysis_lowmem.md** (9KB)
   - Comprehensive TTC performance analysis
   - Reward statistics and candidate diversity metrics
   - Cross-split comparison (validation vs test)
   - Memory-performance trade-off analysis
   - Recommendations for improvements

3. **archive/README.md** (1KB)
   - Documentation of archived files
   - Recovery instructions
   - Archival criteria

### 3. Files Archived ✅

**Configs Moved to `archive/configs/` (8 files):**
- `train_burgers_quality_smoke.yaml` - Smoke test variant (use flags instead)
- `train_burgers_quality_v2.yaml` - Superseded by v3
- `train_burgers_quality_v2_nocompile.yaml` - Debug variant
- `train_burgers_quality_v2_resume.yaml` - One-off resume config
- `train_burgers_quality_final.yaml` - Unclear naming vs v3
- `eval_burgers_512dim.yaml` - Superseded by TTC configs
- `eval_burgers_512dim_baseline.yaml` - Merged into main eval
- `eval_burgers_512dim_ttc_fixed.yaml` - Temp testing variant
- `eval_burgers_512dim_ttc_neutral.yaml` - Temp testing variant
- `train_baseline_pdebench.yaml` - Use train_baselines.py instead

**Scripts Moved to `archive/scripts/` (9 files):**
- `remote_fix_and_run.sh` - One-off debugging script
- `launch_and_run_cheapest.sh` - Superseded by vast_launch_eval.sh
- `restart_fast.sh` - One-off restart script
- `restart_with_precompute.sh` - One-off restart script
- `resume_from_wandb.sh` - Superseded by v2
- `remote_launch_once.sh` - One-off launch script
- `remote_hydrate_b2_once.sh` - One-off hydration script
- `remote_precompute_and_train.sh` - Redundant with run_remote_scale.sh
- `fix_checkpoint.py` - Functionality moved to download script

### 4. Files Deleted ✅

**Temporary Directories:**
- `remote_consistency_run/` - 121MB of old checkpoints
- `.smoke_tmp/` - 28KB of smoke test temp data

**Root-level Files:**
- `eval_launch.log`
- `fetch_local.log`
- `launch.log`
- `launch_ttc_eval.log`
- `monitor.log`
- `resume.log`
- `vast_eval_launch.log`
- `vast_eval_launch2.log`
- `VAST_INSTANCE_26788612.md`

### 5. Bug Fixes ✅

**src/ups/io/enc_grid.py:206-207**
- Changed `view()` to `reshape()` in Fourier feature computation
- Fixes compatibility issue with PyTorch tensor memory layout
- Smoke test now passes successfully

**Before:**
```python
sin_feat = torch.sin(angles).view(batch, -1, Hp, Wp)
cos_feat = torch.cos(angles).view(batch, -1, Hp, Wp)
```

**After:**
```python
sin_feat = torch.sin(angles).reshape(batch, -1, Hp, Wp)
cos_feat = torch.cos(angles).reshape(batch, -1, Hp, Wp)
```

### 6. .gitignore Updates ✅

**Added Patterns:**
```gitignore
# Temp/debug files
*.log
.smoke_tmp/
remote_consistency_run/
VAST_INSTANCE_*.md
eval_*.log
launch*.log
monitor.log
fetch_*.log
resume.log
```

## TTC Evaluation Results

### Key Findings

**Performance Metrics:**
- Validation MSE: 0.001216
- Test MSE: 0.001250
- Consistency: 2.8% difference between splits

**TTC Selection Quality:**
- Selection entropy: 92-96% (excellent)
- All 4 candidates actively used
- Balanced selection distribution

**Issues Identified:**
- Low candidate diversity: 0.003 reward spread (0.01% of mean)
- Memory-limited configuration: 32x32 grid vs 64x64 original
- Need baseline comparison (no TTC) to measure benefit

**Recommendations:**
1. Run baseline evaluation without TTC
2. Increase sampling diversity (noise_std: 0.01→0.05)
3. Scale up grid resolution incrementally (32→48→64)
4. Enable beam search (beam_width: 1→2)
5. Full dataset evaluation (remove max_evaluations limit)

See [docs/ttc_analysis_lowmem.md](docs/ttc_analysis_lowmem.md) for complete analysis.

## Standard Pipeline

### Quick Start

**Smoke Test:**
```bash
bash scripts/smoke_test.sh
```

**Full Training:**
```bash
TRAIN_CONFIG=configs/train_burgers_quality_v3.yaml \
bash scripts/run_remote_scale.sh
```

**Evaluation Only:**
```bash
EVAL_ONLY=1 \
OPERATOR_ARTIFACT=run-mt7rckc8-history:v0 \
DIFFUSION_ARTIFACT=run-pp0c2k31-history:v0 \
bash scripts/run_remote_scale.sh
```

See [docs/pipeline_guide.md](docs/pipeline_guide.md) for complete guide.

## Core Files Retained

### Essential Configs (25 files)
- `defaults.yaml` - Base configuration
- `train_burgers_quality_v3.yaml` - Latest training config ⭐
- `eval_burgers_512dim_ttc_val.yaml` - TTC validation eval
- `eval_burgers_512dim_ttc_test.yaml` - TTC test eval
- `eval_pdebench_scale_ttc.yaml` - Scale TTC eval
- `inference_*.yaml` - Inference configurations
- Other active training/eval configs

### Essential Scripts (38 files)

**Python Scripts (20):**
- `train.py` - Main training script ⭐
- `evaluate.py` - Evaluation script ⭐
- `train_baselines.py` - Baseline model training
- `precompute_latent_cache.py` - Cache precomputation
- `download_checkpoints_from_wandb.py` - Checkpoint management
- `upload_artifact.py` - W&B artifact uploads
- `fetch_datasets.py` - Dataset downloads
- `infer.py` - Inference script
- `vast_launch.py` - Vast.ai launcher
- Data processing scripts (convert_*, prepare_*, fetch_*)

**Shell Scripts (18):**
- `run_remote_scale.sh` - Remote orchestrator ⭐
- `vast_launch_eval.sh` - Evaluation launcher
- `fetch_datasets_b2.sh` - B2 data hydration
- `setup_remote.sh` - Remote environment setup
- `smoke_test.sh` - Quick validation test
- `test_pipeline_local.sh` - Local testing
- `monitor_instance.sh` - Instance monitoring
- `load_env.sh` - Environment loading
- Other active pipeline scripts

## Migration Guide

### Recovering Archived Files

If you need an archived file:

```bash
# Copy from archive
cp archive/configs/FILENAME configs/
# or
cp archive/scripts/FILENAME scripts/

# Make executable if needed
chmod +x scripts/FILENAME
```

### Replacing Deprecated Workflows

| Old Workflow | New Workflow |
|--------------|--------------|
| `scripts/restart_fast.sh` | `TRAIN_STAGE=<stage> scripts/run_remote_scale.sh` |
| `scripts/resume_from_wandb.sh` | Use checkpoint artifacts with `run_remote_scale.sh` |
| `configs/train_burgers_quality_v2.yaml` | `configs/train_burgers_quality_v3.yaml` |
| `configs/eval_burgers_512dim.yaml` | `configs/eval_burgers_512dim_ttc_val.yaml` |
| One-off scripts | Standard `run_remote_scale.sh` with env vars |

## Next Steps

### Immediate Actions
1. ✅ W&B sync complete
2. ✅ TTC analysis complete
3. ✅ Codebase cleanup complete
4. ✅ Documentation complete
5. ✅ Pipeline tested

### Follow-up Tasks
1. **Run baseline comparison:** Evaluate without TTC to measure benefit
2. **Test improved TTC config:** 48x48 grid, 5 candidates, beam_width=2
3. **Full dataset evaluation:** Remove max_evaluations limit
4. **Performance profiling:** Identify bottlenecks in training loop
5. **Multi-GPU support:** Enable data parallel training

### Future Improvements
1. Automatic hyperparameter tuning for TTC
2. Distributed training across multiple nodes
3. Extended benchmarks on additional PDE tasks
4. State-of-the-art baseline comparisons
5. Production deployment pipeline

## References

- **Pipeline Guide:** [docs/pipeline_guide.md](docs/pipeline_guide.md)
- **TTC Analysis:** [docs/ttc_analysis_lowmem.md](docs/ttc_analysis_lowmem.md)
- **Archive Info:** [archive/README.md](archive/README.md)
- **W&B Project:** https://wandb.ai/emgun-morpheus-space/universal-simulator
- **Main Config:** [configs/train_burgers_quality_v3.yaml](configs/train_burgers_quality_v3.yaml)

---

**Cleanup Date:** October 14, 2025
**Status:** ✅ Complete
**Smoke Test:** ✅ Passing
**Documentation:** ✅ Complete (21KB added)
**Disk Space Saved:** 121MB
