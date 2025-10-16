#!/bin/bash
# Cleanup obsolete configs and files
# Run with: bash scripts/cleanup_obsolete.sh

set -e

echo "🧹 Cleaning up obsolete files..."
echo ""

# Create deprecated directory for tracking
mkdir -p .deprecated

# Remove standalone evaluation configs (superseded by unified pipeline)
echo "Removing standalone evaluation configs..."
rm -f configs/eval_burgers_32dim.yaml
rm -f configs/eval_burgers_32dim_ttc.yaml
rm -f configs/eval_burgers_32dim_v3_hjnqclgk.yaml
rm -f configs/eval_burgers_512dim.yaml
rm -f configs/eval_burgers_512dim_test_baseline.yaml
rm -f configs/eval_burgers_512dim_ttc_test.yaml
rm -f configs/eval_burgers_512dim_ttc_val.yaml
rm -f configs/eval_burgers_512dim_val_baseline.yaml
rm -f configs/eval_burgers_64dim.yaml
rm -f configs/eval_burgers_64dim_ttc.yaml
rm -f configs/eval_pdebench.yaml
rm -f configs/eval_pdebench_advection_scale.yaml
rm -f configs/eval_pdebench_advection_scale_test.yaml
rm -f configs/eval_pdebench_scale.yaml
rm -f configs/eval_pdebench_scale_test.yaml
rm -f configs/eval_pdebench_scale_test_ttc.yaml
rm -f configs/eval_pdebench_scale_ttc.yaml
rm -f configs/eval_pdebench_test.yaml
rm -f configs/eval_pdebench_test_quick.yaml

# Remove experimental versioned configs
echo "Removing experimental versioned configs..."
rm -f configs/train_burgers_quality_v2.yaml
rm -f configs/train_burgers_quality_v3.yaml
rm -f configs/train_burgers_32dim_v2_fixed.yaml
rm -f configs/train_burgers_32dim_v2_improved.yaml
rm -f configs/train_burgers_32dim_v2_practical.yaml
rm -f configs/train_burgers_32dim_v3_aggressive.yaml
rm -f configs/train_burgers_32dim_v3_rollout.yaml
rm -f configs/train_burgers_512dim_v1.yaml
rm -f configs/train_burgers_512dim_v2_pru2jxc4.yaml
rm -f configs/train_full_scale_v3.yaml

# Remove pru2jxc4 configs with inheritance issues
echo "Removing configs with known inheritance issues..."
rm -f configs/train_burgers_8dim_pru2jxc4.yaml
rm -f configs/train_burgers_16dim_pru2jxc4.yaml
rm -f configs/train_burgers_32dim_pru2jxc4.yaml
rm -f configs/train_burgers_64dim_pru2jxc4.yaml

# Remove archive directory (will be moved to separate repo)
echo "Removing archive directory..."
rm -rf archive/

# Remove obsolete scripts
echo "Removing obsolete scripts..."
rm -f scripts/download_training_data.sh           # Failed experiment, not used
rm -f scripts/eval_with_checkpoints.sh            # Superseded by unified pipeline
rm -f scripts/run_eval_remote.sh                  # Superseded by unified pipeline
rm -f scripts/run_eval_ttc.sh                     # Superseded by unified pipeline
rm -f scripts/monitor_sota_instance.sh            # One-off SOTA eval monitoring
rm -f scripts/remote_sota_eval.sh                 # One-off SOTA eval script
rm -f scripts/repro_pdebench.sh                   # Old reproduction script
rm -f scripts/repro_poc.sh                        # Old proof-of-concept
rm -f scripts/resume_from_wandb_v2.sh             # Superseded by built-in resume
rm -f scripts/setup_remote.sh                     # Superseded by onstart_template
rm -f scripts/smoke_test.sh                       # Old smoke test
rm -f scripts/vast_e2e.sh                         # Old end-to-end test
rm -f scripts/vast_launch_eval.sh                 # Standalone eval launcher (superseded)
rm -f scripts/fetch_wandb_metrics.py              # Ad-hoc metric fetcher

# Remove temporary/helper scripts
rm -f scripts/fix_libcuda_symlink.sh              # Temporary fix, no longer needed
rm -f scripts/load_env.sh                         # Superseded by .env loading
rm -f scripts/prepare_env.sh                      # Superseded by onstart_template

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "Summary:"
echo "  - Removed 19 standalone evaluation configs"
echo "  - Removed 10 experimental versioned training configs"
echo "  - Removed 4 configs with inheritance issues"
echo "  - Removed 17 obsolete scripts"
echo "  - Removed archive/ directory"
echo ""
echo "📝 See DEPRECATED.md for migration guide"

