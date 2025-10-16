#!/bin/bash
# Archive obsolete files instead of deleting them
# This preserves history while cleaning up the working directory

set -e

echo "📦 Archiving obsolete files..."
echo ""

# Create archive structure
mkdir -p archive/configs
mkdir -p archive/scripts
mkdir -p archive/docs

# ============================================================================
# CONFIGS: Archive experimental and versioned configs
# ============================================================================
echo "Archiving obsolete configs..."

# Standalone evaluation configs (superseded by unified pipeline)
mv -f configs/eval_burgers_32dim.yaml archive/configs/ 2>/dev/null || true
mv -f configs/eval_burgers_32dim_ttc.yaml archive/configs/ 2>/dev/null || true
mv -f configs/eval_burgers_32dim_v3_hjnqclgk.yaml archive/configs/ 2>/dev/null || true
mv -f configs/eval_burgers_512dim.yaml archive/configs/ 2>/dev/null || true
mv -f configs/eval_burgers_512dim_test_baseline.yaml archive/configs/ 2>/dev/null || true
mv -f configs/eval_burgers_512dim_ttc_test.yaml archive/configs/ 2>/dev/null || true
mv -f configs/eval_burgers_512dim_ttc_val.yaml archive/configs/ 2>/dev/null || true
mv -f configs/eval_burgers_512dim_val_baseline.yaml archive/configs/ 2>/dev/null || true
mv -f configs/eval_burgers_64dim.yaml archive/configs/ 2>/dev/null || true
mv -f configs/eval_burgers_64dim_ttc.yaml archive/configs/ 2>/dev/null || true
mv -f configs/eval_pdebench.yaml archive/configs/ 2>/dev/null || true
mv -f configs/eval_pdebench_advection_scale.yaml archive/configs/ 2>/dev/null || true
mv -f configs/eval_pdebench_advection_scale_test.yaml archive/configs/ 2>/dev/null || true
mv -f configs/eval_pdebench_scale.yaml archive/configs/ 2>/dev/null || true
mv -f configs/eval_pdebench_scale_test.yaml archive/configs/ 2>/dev/null || true
mv -f configs/eval_pdebench_scale_test_ttc.yaml archive/configs/ 2>/dev/null || true
mv -f configs/eval_pdebench_scale_ttc.yaml archive/configs/ 2>/dev/null || true
mv -f configs/eval_pdebench_test.yaml archive/configs/ 2>/dev/null || true
mv -f configs/eval_pdebench_test_quick.yaml archive/configs/ 2>/dev/null || true

# Experimental versioned configs
mv -f configs/train_burgers_quality_v2.yaml archive/configs/ 2>/dev/null || true
mv -f configs/train_burgers_quality_v3.yaml archive/configs/ 2>/dev/null || true
mv -f configs/train_burgers_32dim_v2_fixed.yaml archive/configs/ 2>/dev/null || true
mv -f configs/train_burgers_32dim_v2_improved.yaml archive/configs/ 2>/dev/null || true
mv -f configs/train_burgers_32dim_v2_practical.yaml archive/configs/ 2>/dev/null || true
mv -f configs/train_burgers_32dim_v3_aggressive.yaml archive/configs/ 2>/dev/null || true
mv -f configs/train_burgers_32dim_v3_rollout.yaml archive/configs/ 2>/dev/null || true
mv -f configs/train_burgers_512dim_v1.yaml archive/configs/ 2>/dev/null || true
mv -f configs/train_burgers_512dim_v2_pru2jxc4.yaml archive/configs/ 2>/dev/null || true
mv -f configs/train_full_scale_v3.yaml archive/configs/ 2>/dev/null || true

# Configs with inheritance issues
mv -f configs/train_burgers_8dim_pru2jxc4.yaml archive/configs/ 2>/dev/null || true
mv -f configs/train_burgers_16dim_pru2jxc4.yaml archive/configs/ 2>/dev/null || true
mv -f configs/train_burgers_32dim_pru2jxc4.yaml archive/configs/ 2>/dev/null || true
mv -f configs/train_burgers_64dim_pru2jxc4.yaml archive/configs/ 2>/dev/null || true

# Experimental configs - Phase 2
mv -f configs/benchmark_pdebench.yaml archive/configs/ 2>/dev/null || true
mv -f configs/coupled_toy.yaml archive/configs/ 2>/dev/null || true
mv -f configs/train_multi_pde.yaml archive/configs/ 2>/dev/null || true
mv -f configs/train_pdebench_scale.yaml archive/configs/ 2>/dev/null || true
mv -f configs/train_pdebench_scale_quality.yaml archive/configs/ 2>/dev/null || true
mv -f configs/train_pdebench_test.yaml archive/configs/ 2>/dev/null || true
mv -f configs/train_smoke_test.yaml archive/configs/ 2>/dev/null || true

# ============================================================================
# SCRIPTS: Archive obsolete scripts
# ============================================================================
echo "Archiving obsolete scripts..."

# Evaluation scripts
mv -f scripts/eval_with_checkpoints.sh archive/scripts/ 2>/dev/null || true
mv -f scripts/run_eval_remote.sh archive/scripts/ 2>/dev/null || true
mv -f scripts/run_eval_ttc.sh archive/scripts/ 2>/dev/null || true
mv -f scripts/vast_launch_eval.sh archive/scripts/ 2>/dev/null || true

# Monitoring scripts
mv -f scripts/monitor_sota_instance.sh archive/scripts/ 2>/dev/null || true
mv -f scripts/remote_sota_eval.sh archive/scripts/ 2>/dev/null || true
mv -f scripts/monitor_and_shutdown.sh archive/scripts/ 2>/dev/null || true

# Setup/environment scripts
mv -f scripts/download_training_data.sh archive/scripts/ 2>/dev/null || true
mv -f scripts/setup_remote.sh archive/scripts/ 2>/dev/null || true
mv -f scripts/prepare_env.sh archive/scripts/ 2>/dev/null || true
mv -f scripts/load_env.sh archive/scripts/ 2>/dev/null || true
mv -f scripts/fix_libcuda_symlink.sh archive/scripts/ 2>/dev/null || true

# Old experiments
mv -f scripts/repro_pdebench.sh archive/scripts/ 2>/dev/null || true
mv -f scripts/repro_poc.sh archive/scripts/ 2>/dev/null || true
mv -f scripts/smoke_test.sh archive/scripts/ 2>/dev/null || true
mv -f scripts/vast_e2e.sh archive/scripts/ 2>/dev/null || true
mv -f scripts/resume_from_wandb_v2.sh archive/scripts/ 2>/dev/null || true
mv -f scripts/fetch_wandb_metrics.py archive/scripts/ 2>/dev/null || true

# Data preparation scripts
mv -f scripts/add_sota_metrics.py archive/scripts/ 2>/dev/null || true
mv -f scripts/benchmark.py archive/scripts/ 2>/dev/null || true
mv -f scripts/build_full_artifacts.py archive/scripts/ 2>/dev/null || true
mv -f scripts/convert_pdebench.py archive/scripts/ 2>/dev/null || true
mv -f scripts/convert_pdebench_multimodal.py archive/scripts/ 2>/dev/null || true
mv -f scripts/download_pdebench_file.py archive/scripts/ 2>/dev/null || true
mv -f scripts/fetch_datasets.py archive/scripts/ 2>/dev/null || true
mv -f scripts/fetch_datasets_b2.sh archive/scripts/ 2>/dev/null || true
mv -f scripts/make_val_test_splits.py archive/scripts/ 2>/dev/null || true
mv -f scripts/prepare_data.py archive/scripts/ 2>/dev/null || true
mv -f scripts/publish_shard_b2.py archive/scripts/ 2>/dev/null || true
mv -f scripts/stream_from_pdebench_metadata.py archive/scripts/ 2>/dev/null || true
mv -f scripts/stream_shard_upload_b2.py archive/scripts/ 2>/dev/null || true
mv -f scripts/upload_artifact.py archive/scripts/ 2>/dev/null || true

# Benchmarking/testing scripts
mv -f scripts/test_pipeline_local.sh archive/scripts/ 2>/dev/null || true
mv -f scripts/train_baselines.py archive/scripts/ 2>/dev/null || true
mv -f scripts/run_sota_evaluation.sh archive/scripts/ 2>/dev/null || true
mv -f scripts/infer.py archive/scripts/ 2>/dev/null || true

# Utilities
mv -f scripts/download_checkpoints_from_wandb.py archive/scripts/ 2>/dev/null || true

# ============================================================================
# DOCS: Archive old documentation
# ============================================================================
echo "Archiving obsolete documentation..."

# Old analyses
mv -f docs/512dim_optimization_analysis.md archive/docs/ 2>/dev/null || true
mv -f docs/512dim_pru2jxc4_analysis.md archive/docs/ 2>/dev/null || true
mv -f docs/64dim_optimal_capacity_plan.md archive/docs/ 2>/dev/null || true
mv -f docs/hjnqclgk_vs_baseline_analysis.md archive/docs/ 2>/dev/null || true
mv -f docs/baseline_metrics.md archive/docs/ 2>/dev/null || true
mv -f docs/wandb_fix_summary.md archive/docs/ 2>/dev/null || true

# Old planning docs
mv -f docs/MILESTONES_TODO.md archive/docs/ 2>/dev/null || true
mv -f docs/paper_review_2402_12365.md archive/docs/ 2>/dev/null || true
mv -f docs/universal_physics_stack_implementation_plan.md archive/docs/ 2>/dev/null || true
mv -f docs/scaling_plan.md archive/docs/ 2>/dev/null || true
mv -f docs/ttc_integration_plan.md archive/docs/ 2>/dev/null || true
mv -f docs/ttc_analysis_lowmem.md archive/docs/ 2>/dev/null || true

# Old workflow guides
mv -f docs/e2e_pipeline.md archive/docs/ 2>/dev/null || true
mv -f docs/pipeline_guide.md archive/docs/ 2>/dev/null || true
mv -f docs/training_run_guide.md archive/docs/ 2>/dev/null || true
mv -f docs/sota_comparison_guide.md archive/docs/ 2>/dev/null || true

# Old manifests
mv -f docs/dataset_registry.yaml archive/docs/ 2>/dev/null || true
mv -f docs/pdebench_manifest.yaml archive/docs/ 2>/dev/null || true

# Create archive README
cat > archive/README.md << 'EOF'
# Archive Directory

This directory contains deprecated files that have been superseded by newer implementations or are no longer part of the production workflow.

## Structure

- `configs/` - Old training and evaluation configurations
- `scripts/` - Deprecated scripts and utilities
- `docs/` - Historical documentation and analyses

## Why Archive Instead of Delete?

These files are preserved for:
- Historical reference
- Understanding past experiments
- Recovering old configurations if needed
- Audit trail of development decisions

## Migration Guide

See `/DEPRECATED.md` in the project root for details on:
- What replaced each deprecated file
- How to migrate to new approaches
- When to use archived vs production files

## Last Updated

$(date +%Y-%m-%d)
EOF

echo ""
echo "✅ Archiving complete!"
echo ""
echo "Summary:"
echo "  Configs archived: $(ls archive/configs/ 2>/dev/null | wc -l | tr -d ' ')"
echo "  Scripts archived: $(ls archive/scripts/ 2>/dev/null | wc -l | tr -d ' ')"
echo "  Docs archived: $(ls archive/docs/ 2>/dev/null | wc -l | tr -d ' ')"
echo ""
echo "  Remaining configs: $(ls configs/*.yaml 2>/dev/null | wc -l | tr -d ' ')"
echo "  Remaining scripts: $(ls scripts/*.{py,sh} 2>/dev/null | wc -l | tr -d ' ')"
echo "  Remaining docs: $(ls docs/*.md 2>/dev/null | wc -l | tr -d ' ')"
echo ""
echo "📝 See DEPRECATED.md for migration guide"
echo "📦 Archived files are in archive/ directory"

