#!/usr/bin/env bash
set -e

cd /workspace/universal_simulator || exit 1

# Normalize .env line endings if present
[ -f .env ] && sed -i 's/\r$//' .env || true

# Source env for WANDB/B2 credentials
set -a
[ -f .env ] && . ./.env || true
set +a

# W&B login and online mode
if command -v wandb >/dev/null 2>&1; then
  if [ -n "${WANDB_API_KEY:-}" ]; then
    wandb login --relogin "$WANDB_API_KEY" || true
    wandb online || true
  fi
fi
export WANDB_MODE=online

# Config and data paths
export TRAIN_CONFIG=${TRAIN_CONFIG:-configs/train_burgers_quality_v2.yaml}
export EVAL_CONFIG=${EVAL_CONFIG:-configs/eval_pdebench_scale_ttc.yaml}
export EVAL_TEST_CONFIG=${EVAL_TEST_CONFIG:-configs/eval_pdebench_scale_test_ttc.yaml}
export TRAIN_STAGE=${TRAIN_STAGE:-all}
export PDEBENCH_ROOT=${PDEBENCH_ROOT:-/workspace/universal_simulator/data/pdebench/burgers1d_full_v1}

# Skip resets and precompute (already done)
export RESET_CACHE=${RESET_CACHE:-0}
export PRECOMPUTE_LATENT=${PRECOMPUTE_LATENT:-0}

nohup bash scripts/run_remote_scale.sh > run_full.log 2>&1 &
sleep 2
tail -n 200 run_full.log || true






