#!/usr/bin/env bash
set -eo pipefail

cd /workspace/universal_simulator || exit 1

# Source .env
sed -i 's/\r$//' .env || true
set -a
[ -f .env ] && . ./.env
set +a

# W&B login
if [ -n "${WANDB_API_KEY:-}" ] && command -v wandb >/dev/null 2>&1; then
  printf "%s\n" "$WANDB_API_KEY" | wandb login --relogin --stdin || true
  export WANDB_MODE=online
  wandb online || true
else
  export WANDB_MODE=${WANDB_MODE:-offline}
fi

# Export all necessary variables
export TORCHINDUCTOR_DISABLE=1
export RESET_CACHE=0
export PRECOMPUTE_LATENT=1
export PDEBENCH_ROOT=/workspace/universal_simulator/data/pdebench/burgers1d_full_v1
export TRAIN_CONFIG=configs/train_burgers_quality_v2.yaml
export EVAL_CONFIG=configs/eval_pdebench_scale_ttc.yaml
export EVAL_TEST_CONFIG=configs/eval_pdebench_scale_test_ttc.yaml
export TRAIN_STAGE=all
export WANDB_DATASETS=""

echo "Starting full training pipeline with latent precomputation..."
echo "PDEBENCH_ROOT=$PDEBENCH_ROOT"
echo "PRECOMPUTE_LATENT=$PRECOMPUTE_LATENT"
echo "TRAIN_CONFIG=$TRAIN_CONFIG"

# Launch the pipeline
nohup bash scripts/run_remote_scale.sh > run_full.log 2>&1 &

echo "Pipeline launched in background. Logs: run_full.log"
echo "Waiting 10 seconds..."
sleep 10
echo "Initial log output:"
tail -n 100 run_full.log || true





