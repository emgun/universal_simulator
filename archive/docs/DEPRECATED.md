# Deprecated Files and Migration Guide

This document tracks files that have been moved to the `archive/` directory and provides migration guidance.

**Archive Location:** All deprecated files are preserved in `archive/configs/`, `archive/scripts/`, and `archive/docs/` for historical reference.

**Last Updated:** 2025-10-16

---

## Deprecated Configs

### Standalone Evaluation Configs (Superseded by Unified Pipeline)

The following evaluation configs are obsolete because evaluation is now integrated into the training pipeline via `evaluation.enabled: true` in training configs.

**Archived to `archive/configs/`:**
- `configs/eval_burgers_32dim.yaml`
- `configs/eval_burgers_32dim_ttc.yaml`
- `configs/eval_burgers_32dim_v3_hjnqclgk.yaml`
- `configs/eval_burgers_512dim.yaml`
- `configs/eval_burgers_512dim_test_baseline.yaml`
- `configs/eval_burgers_512dim_ttc_test.yaml`
- `configs/eval_burgers_512dim_ttc_val.yaml`
- `configs/eval_burgers_512dim_val_baseline.yaml`
- `configs/eval_burgers_64dim.yaml`
- `configs/eval_burgers_64dim_ttc.yaml`
- `configs/eval_pdebench.yaml`
- `configs/eval_pdebench_advection_scale.yaml`
- `configs/eval_pdebench_advection_scale_test.yaml`
- `configs/eval_pdebench_scale.yaml`
- `configs/eval_pdebench_scale_test.yaml`
- `configs/eval_pdebench_scale_test_ttc.yaml`
- `configs/eval_pdebench_scale_ttc.yaml`
- `configs/eval_pdebench_test.yaml`
- `configs/eval_pdebench_test_quick.yaml`

**Migration:** Use training configs with `evaluation.enabled: true`. The unified pipeline runs baseline and TTC evaluation automatically after training.

**Example:**
```yaml
# Old approach (deprecated)
# 1. Train: python scripts/train.py configs/train_burgers_32dim.yaml
# 2. Eval: python scripts/evaluate.py --config configs/eval_burgers_32dim.yaml

# New approach (recommended)
# Training config with evaluation enabled runs everything:
evaluation:
  enabled: true
  split: test

ttc:
  enabled: true  # Runs both baseline and TTC eval
```

---

### Experimental/Versioned Configs (Consolidated)

The following versioned configs were experimental iterations. They've been consolidated into canonical configs without version suffixes.

**Archived to `archive/configs/`:**
- `configs/train_burgers_quality_v2.yaml` → use `configs/train_burgers_32dim.yaml`
- `configs/train_burgers_quality_v3.yaml` → use `configs/train_burgers_32dim.yaml`
- `configs/train_burgers_32dim_v2_fixed.yaml` → use `configs/train_burgers_32dim.yaml`
- `configs/train_burgers_32dim_v2_improved.yaml` → use `configs/train_burgers_32dim.yaml`
- `configs/train_burgers_32dim_v2_practical.yaml` → use `configs/train_burgers_32dim.yaml`
- `configs/train_burgers_32dim_v3_aggressive.yaml` → aggressive TTC approach didn't improve results
- `configs/train_burgers_32dim_v3_rollout.yaml` → rollout loss degraded 32-dim performance
- `configs/train_burgers_512dim_v1.yaml` → use `configs/train_burgers_512dim.yaml`
- `configs/train_burgers_512dim_v2_pru2jxc4.yaml` → use `configs/train_burgers_512dim.yaml`
- `configs/train_full_scale_v3.yaml` → consolidated into dimension-specific configs

**Migration:** Use the canonical configs in `configs/`:
- `train_burgers_8dim.yaml` - minimal capacity
- `train_burgers_16dim.yaml` - fast prototyping
- `train_burgers_32dim.yaml` - cost-effective production
- `train_burgers_64dim.yaml` - optimal capacity/cost
- `train_burgers_512dim.yaml` - maximum quality

All canonical configs are:
- Fully self-contained (no `include:` directives)
- Dimension-consistent (no mismatches)
- CI-validated
- Documented with expected performance metrics

---

### Configs with Known Issues (Removed)

- `configs/train_burgers_32dim_pru2jxc4.yaml` - Had config inheritance issues
- `configs/train_burgers_64dim_pru2jxc4.yaml` - Had config inheritance issues
- `configs/train_burgers_16dim_pru2jxc4.yaml` - Had config inheritance issues
- `configs/train_burgers_8dim_pru2jxc4.yaml` - Had config inheritance issues

**Issue:** These configs used `include:` which led to dimension mismatches during training/evaluation.

**Migration:** Use the new self-contained canonical configs listed above.

---

## Archive Directory

The `archive/` directory has been moved to a separate repository: `universal_simulator_archive`.

**Removed:**
- `archive/configs/*` (8 files)
- `archive/scripts/*` (10 files)

**Reason:** These were historical experiments and one-off scripts that are no longer relevant to the production codebase.

**Access:** If you need to reference these files, they're available at:
```
git clone https://github.com/emgun/universal_simulator_archive.git
```

---

## Deprecated Scripts

### Removed Scripts - Phase 1 Cleanup (Superseded by Unified Pipeline)

**Evaluation scripts:**
- `scripts/eval_with_checkpoints.sh` - Use unified training pipeline with `evaluation.enabled: true`
- `scripts/run_eval_remote.sh` - Use unified training pipeline
- `scripts/run_eval_ttc.sh` - TTC evaluation now integrated in training pipeline
- `scripts/vast_launch_eval.sh` - Use `vast_launch.py` with evaluation-enabled configs

**Monitoring scripts:**
- `scripts/monitor_sota_instance.sh` - One-off script for specific SOTA eval runs
- `scripts/remote_sota_eval.sh` - Specific to old SOTA evaluation workflow

**Setup/environment scripts:**
- `scripts/download_training_data.sh` - Failed experiment with config-driven downloads
- `scripts/setup_remote.sh` - Use `scripts/onstart_template.py` instead
- `scripts/prepare_env.sh` - Use `scripts/onstart_template.py` instead
- `scripts/load_env.sh` - Environment variables now loaded via `.env` directly
- `scripts/fix_libcuda_symlink.sh` - Temporary workaround, no longer needed

**Old experiments:**
- `scripts/repro_pdebench.sh` - Old PDEBench reproduction script
- `scripts/repro_poc.sh` - Proof-of-concept script from early development
- `scripts/smoke_test.sh` - Use dry-run mode instead (`scripts/dry_run.py`)
- `scripts/vast_e2e.sh` - Old end-to-end test
- `scripts/resume_from_wandb_v2.sh` - Training script now has built-in resume support
- `scripts/fetch_wandb_metrics.py` - Ad-hoc metric fetcher (use WandB API or `scripts/analyze_run.py`)

### Removed Scripts - Phase 2 Aggressive Cleanup (Production Streamlining)

**Data preparation scripts (use pre-prepared datasets):**
- `scripts/convert_pdebench.py` - PDEBench format conversion
- `scripts/convert_pdebench_multimodal.py` - Multi-modal PDEBench conversion
- `scripts/download_pdebench_file.py` - Individual file downloader
- `scripts/fetch_datasets.py` - WandB dataset fetcher
- `scripts/fetch_datasets_b2.sh` - B2 dataset fetcher
- `scripts/make_val_test_splits.py` - Dataset splitting utility
- `scripts/prepare_data.py` - Data preparation pipeline
- `scripts/publish_shard_b2.py` - B2 shard publisher
- `scripts/stream_from_pdebench_metadata.py` - Streaming data processor
- `scripts/stream_shard_upload_b2.py` - Streaming B2 uploader
- `scripts/upload_artifact.py` - Generic artifact uploader

**Benchmarking/testing scripts (not part of core workflow):**
- `scripts/add_sota_metrics.py` - SOTA metric calculator
- `scripts/benchmark.py` - Benchmarking suite
- `scripts/build_full_artifacts.py` - Artifact builder
- `scripts/test_pipeline_local.sh` - Local pipeline test
- `scripts/train_baselines.py` - Baseline training variants
- `scripts/run_sota_evaluation.sh` - SOTA evaluation runner
- `scripts/infer.py` - Inference script (use evaluate.py instead)

**Monitoring variants:**
- `scripts/monitor_and_shutdown.sh` - Auto-shutdown monitor (use monitor_instance.sh)

**Utilities:**
- `scripts/download_checkpoints_from_wandb.py` - Use WandB CLI (`wandb artifact get`)

**Migration:** Core training workflow only needs 9 scripts:
1. `scripts/train.py` - Main training script
2. `scripts/evaluate.py` - Standalone evaluation
3. `scripts/validate_config.py` - Config validation
4. `scripts/vast_launch.py` - Instance launcher
5. `scripts/onstart_template.py` - Onstart script generator
6. `scripts/generate_onstart.py` - CLI for onstart generation
7. `scripts/run_remote_scale.sh` - Remote training runner (called by onstart)
8. `scripts/monitor_instance.sh` - Instance monitoring
9. `scripts/precompute_latent_cache.py` - Cache optimization

### Standalone Evaluation Script (Still Available)

- `scripts/evaluate.py` - Still exists for standalone evaluation, but most use cases now use the unified training pipeline.

**When to use standalone evaluation:**
- Evaluating pre-trained checkpoints without retraining
- Custom evaluation configurations not covered by training configs
- Debugging evaluation code in isolation
- Running evaluation on different hardware than training

**When to use unified pipeline:**
- Normal training + evaluation workflow (recommended)
- Ensures training and evaluation configs match perfectly
- Single WandB run with all metrics

---

## Deprecated Configs - Phase 2

**Experimental/task-specific configs removed:**
- `configs/benchmark_pdebench.yaml` - Benchmarking config
- `configs/coupled_toy.yaml` - Toy coupled system
- `configs/train_multi_pde.yaml` - Multi-PDE training
- `configs/train_pdebench_scale.yaml` - PDEBench scaling experiment
- `configs/train_pdebench_scale_quality.yaml` - PDEBench quality experiment
- `configs/train_pdebench_test.yaml` - PDEBench test config
- `configs/train_smoke_test.yaml` - Quick smoke test

**Remaining essential configs:**
- `configs/defaults.yaml` - Base configuration defaults
- `configs/train_pdebench.yaml` - Reference PDEBench training config
- `configs/inference_steady.yaml` - Steady-state inference
- `configs/inference_transient.yaml` - Transient inference
- `configs/inference_ttc.yaml` - TTC inference

**Note:** Dimension-specific production configs (8, 16, 32, 64, 512-dim) will be created in Phase 8 of the production infrastructure upgrade.

---

## Deprecated Documentation

### Phase 1 Removals (Moved to archive)

The following docs were moved to `archive/docs/` (now in separate repo):

- `docs/512dim_optimization_analysis.md` - Initial analysis superseded by pru2jxc4 analysis
- `docs/32dim_v2_to_sota_analysis.md` - Experimental optimization that didn't pan out
- `docs/hjnqclgk_vs_baseline_analysis.md` - Negative result documented
- `docs/SESSION_COMPLETE_SUMMARY.md` - Session-specific, not evergreen

### Phase 2 Removals (Aggressive cleanup)

**Old analyses removed:**
- `docs/512dim_optimization_analysis.md` - Superseded analysis
- `docs/512dim_pru2jxc4_analysis.md` - Specific run analysis
- `docs/64dim_optimal_capacity_plan.md` - Experiment-specific plan
- `docs/hjnqclgk_vs_baseline_analysis.md` - Negative result analysis
- `docs/baseline_metrics.md` - Old metrics documentation
- `docs/wandb_fix_summary.md` - Troubleshooting log

**Old planning docs removed:**
- `docs/MILESTONES_TODO.md` - Outdated milestone tracking
- `docs/paper_review_2402_12365.md` - Paper review notes
- `docs/universal_physics_stack_implementation_plan.md` - Old implementation plan
- `docs/scaling_plan.md` - Old scaling strategy
- `docs/ttc_integration_plan.md` - Completed integration plan
- `docs/ttc_analysis_lowmem.md` - Specific TTC analysis

**Old workflow guides removed (superseded):**
- `docs/e2e_pipeline.md` - Superseded by unified pipeline docs
- `docs/pipeline_guide.md` - Superseded by end_to_end_workflow.md
- `docs/training_run_guide.md` - Superseded by production_playbook.md
- `docs/sota_comparison_guide.md` - Superseded by next_steps_analysis.md

**Old manifests removed:**
- `docs/dataset_registry.yaml` - Outdated dataset registry
- `docs/pdebench_manifest.yaml` - Old PDEBench manifest

### Remaining Production Documentation (6 files)

- `docs/parallel_cache_optimization.md` - Performance optimization guide
- `docs/unified_training_eval_pipeline.md` - Training pipeline architecture
- `docs/onstart_scripts.md` - VastAI deployment guide
- `docs/end_to_end_workflow.md` - Operational procedures
- `docs/next_steps_analysis.md` - Research roadmap
- `docs/data_artifacts.md` - Data references and locations

**Note:** `docs/production_playbook.md` and `docs/runbook.md` will be created in Phase 10 of the production infrastructure upgrade.

---

## Breaking Changes

### Config Structure Changes

**Before (with inheritance):**
```yaml
include: train_burgers_quality_v2.yaml

latent:
  dim: 32
```

**After (self-contained):**
```yaml
# No include directive - everything explicit

data:
  task: burgers1d
  split: train
  root: data/pdebench
  # ... all data config

latent:
  dim: 32
  tokens: 32

operator:
  pdet:
    input_dim: 32  # Must match latent.dim
    hidden_dim: 96
    # ... all operator config

diffusion:
  latent_dim: 32  # Must match latent.dim
  hidden_dim: 96  # Must match operator.pdet.hidden_dim
  # ... all diffusion config

# Full config continues...
```

**Reason:** Config inheritance caused dimension mismatches and made debugging difficult. Self-contained configs are explicit and easier to validate.

---

## Deprecation Timeline

- **2025-10-16:** Phase 0 cleanup implemented
  - Removed standalone eval configs
  - Removed experimental versioned configs
  - Archived old experiments
  - Created canonical dimension-specific configs

---

## Questions?

If you're unsure about which config to use or need help migrating, see:
- `docs/production_playbook.md` for configuration best practices
- `docs/runbook.md` for step-by-step launch instructions
- Run `python scripts/validate_config.py <config>` to check any config before use

