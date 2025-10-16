#!/bin/bash
# Aggressive cleanup - keep only what's used in production training workflow
set -e

echo "🧹 Aggressive cleanup - keeping only production essentials..."
echo ""

# ============================================================================
# SCRIPTS: Keep only core training workflow scripts
# ============================================================================
echo "Removing non-essential scripts..."

# Data conversion/preparation (not needed for training with pre-prepared data)
rm -f scripts/add_sota_metrics.py
rm -f scripts/benchmark.py
rm -f scripts/build_full_artifacts.py
rm -f scripts/convert_pdebench.py
rm -f scripts/convert_pdebench_multimodal.py
rm -f scripts/download_pdebench_file.py
rm -f scripts/fetch_datasets.py
rm -f scripts/fetch_datasets_b2.sh
rm -f scripts/make_val_test_splits.py
rm -f scripts/prepare_data.py
rm -f scripts/publish_shard_b2.py
rm -f scripts/stream_from_pdebench_metadata.py
rm -f scripts/stream_shard_upload_b2.py
rm -f scripts/upload_artifact.py

# One-off test/benchmark scripts
rm -f scripts/test_pipeline_local.sh
rm -f scripts/train_baselines.py
rm -f scripts/run_sota_evaluation.sh
rm -f scripts/infer.py

# Monitoring variants (keep only monitor_instance.sh)
rm -f scripts/monitor_and_shutdown.sh

# WandB checkpoint downloader (rarely used, can use WandB CLI)
rm -f scripts/download_checkpoints_from_wandb.py

# Keep cleanup script itself for now
# rm -f scripts/cleanup_obsolete.sh

# ============================================================================
# CONFIGS: Keep only defaults and inference configs
# ============================================================================
echo "Removing experimental configs..."

# Remove old PDEBench experiment configs
rm -f configs/benchmark_pdebench.yaml
rm -f configs/coupled_toy.yaml
rm -f configs/train_multi_pde.yaml
rm -f configs/train_pdebench_scale.yaml
rm -f configs/train_pdebench_scale_quality.yaml
rm -f configs/train_pdebench_test.yaml
rm -f configs/train_smoke_test.yaml

# Keep:
# - defaults.yaml (base config)
# - train_pdebench.yaml (reference for PDEBench tasks)
# - inference_*.yaml (for deployment)

# ============================================================================
# DOCS: Keep only production documentation
# ============================================================================
echo "Removing old analysis and planning docs..."

# Old analyses and comparisons
rm -f docs/512dim_optimization_analysis.md
rm -f docs/512dim_pru2jxc4_analysis.md
rm -f docs/64dim_optimal_capacity_plan.md
rm -f docs/hjnqclgk_vs_baseline_analysis.md
rm -f docs/baseline_metrics.md
rm -f docs/wandb_fix_summary.md

# Old planning/implementation docs
rm -f docs/MILESTONES_TODO.md
rm -f docs/paper_review_2402_12365.md
rm -f docs/universal_physics_stack_implementation_plan.md
rm -f docs/scaling_plan.md
rm -f docs/ttc_integration_plan.md
rm -f docs/ttc_analysis_lowmem.md

# Old workflow guides (superseded by production docs)
rm -f docs/e2e_pipeline.md
rm -f docs/pipeline_guide.md
rm -f docs/training_run_guide.md
rm -f docs/sota_comparison_guide.md

# Keep:
# - parallel_cache_optimization.md (performance guide)
# - unified_training_eval_pipeline.md (architecture)
# - onstart_scripts.md (deployment)
# - end_to_end_workflow.md (operational)
# - next_steps_analysis.md (research roadmap)
# - data_artifacts.md (data references)

# ============================================================================
# MANIFESTS: Remove old metadata files
# ============================================================================
echo "Removing old manifest files..."
rm -f docs/dataset_registry.yaml
rm -f docs/pdebench_manifest.yaml

echo ""
echo "✅ Aggressive cleanup complete!"
echo ""
echo "Remaining scripts (9 core):"
ls -1 scripts/*.py scripts/*.sh | wc -l
echo ""
echo "Remaining configs (4 essential):"
ls -1 configs/*.yaml | wc -l
echo ""
echo "Remaining docs (6 production):"
ls -1 docs/*.md 2>/dev/null | wc -l || echo "0"
echo ""
echo "📝 See DEPRECATED.md for details on removed files"

