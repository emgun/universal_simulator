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
  printf "%s\n" "$WANDB_API_KEY" | wandb login --relogin 2>/dev/null || true
  export WANDB_MODE=online
  wandb online || true
else
  export WANDB_MODE=${WANDB_MODE:-offline}
fi

# Export all necessary variables
export TORCHINDUCTOR_DISABLE=1
export RESET_CACHE=0
export PRECOMPUTE_LATENT=0
export PDEBENCH_ROOT=/workspace/universal_simulator/data/pdebench/burgers1d_full_v1
export TRAIN_CONFIG=configs/train_burgers_quality_v2_fast.yaml
export TRAIN_STAGE=all
export WANDB_DATASETS=""

echo "Starting training with 15 operator epochs and checkpoints every 5 epochs..."
echo "PDEBENCH_ROOT=$PDEBENCH_ROOT"
echo "TRAIN_CONFIG=$TRAIN_CONFIG"

# Launch training directly
nohup python scripts/train.py --config "$TRAIN_CONFIG" --stage all > run_fast.log 2>&1 &

echo "Training launched. Logs: run_fast.log"
sleep 10
tail -n 100 run_fast.log || true





