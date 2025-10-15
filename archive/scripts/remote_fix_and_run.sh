#!/usr/bin/env bash
set -eo pipefail

# Robust remote fix + run script to be executed on the instance via SSH
# - Loads .env using scripts/load_env.sh if present
# - Ensures WANDB_DATASETS default and PDEBENCH_ROOT
# - Hydrates dataset via B2 if creds present, otherwise via W&B artifacts
# - Creates expected symlinks and launches the full pipeline

cd /workspace/universal_simulator || exit 1

# Normalise potential CRLF in .env
[ -f .env ] && sed -i 's/\r$//' .env || true

# Load environment if loader and .env exist
if [ -f scripts/load_env.sh ] && [ -f .env ]; then
  bash scripts/load_env.sh || true
fi

# Defaults if missing
export WANDB_DATASETS=${WANDB_DATASETS:-burgers1d_full_v1}
export PDEBENCH_ROOT=${PDEBENCH_ROOT:-/workspace/universal_simulator/data/pdebench/burgers1d_full_v1}
export TRAIN_CONFIG=${TRAIN_CONFIG:-configs/train_burgers_quality_v2.yaml}
export EVAL_CONFIG=${EVAL_CONFIG:-configs/eval_pdebench_scale_ttc.yaml}
export EVAL_TEST_CONFIG=${EVAL_TEST_CONFIG:-configs/eval_pdebench_scale_test_ttc.yaml}
export TRAIN_STAGE=${TRAIN_STAGE:-all}

# Sanitize TRAIN_CONFIG if prior env mistakenly concatenated overrides
if [ ! -f "$TRAIN_CONFIG" ] && [[ "$TRAIN_CONFIG" == *,* ]]; then
  TRAIN_CONFIG="${TRAIN_CONFIG%%,*}"
  export TRAIN_CONFIG
fi

# Non-interactive W&B login if API key present
if [ -n "${WANDB_API_KEY:-}" ] && command -v wandb >/dev/null 2>&1; then
  printf "%s\n" "$WANDB_API_KEY" | wandb login --relogin --stdin || true
  export WANDB_MODE=online
  wandb online || true
else
  export WANDB_MODE=${WANDB_MODE:-offline}
fi

# Ensure parent directory exists
mkdir -p "$(dirname "$PDEBENCH_ROOT")"

echo "Hydrating datasets: $WANDB_DATASETS"
if [ -n "${B2_KEY_ID:-}" ] && [ -n "${B2_APP_KEY:-}" ] && [ -n "${B2_BUCKET:-}" ]; then
  CLEAN_OLD_SPLITS=1 bash scripts/fetch_datasets_b2.sh "$WANDB_DATASETS" || true
else
  PYTHONPATH=src python scripts/fetch_datasets.py "$WANDB_DATASETS" \
    --root "$(dirname "$PDEBENCH_ROOT")" \
    --cache artifacts/cache \
    --project "${WANDB_PROJECT:-universal-simulator}" ${WANDB_ENTITY:+--entity "$WANDB_ENTITY"} || true
fi

# Symlink expected H5 paths if the hydrated subdir exists
if [ -d "$PDEBENCH_ROOT" ]; then
  ln -sf burgers1d_full_v1/burgers1d_train.h5 data/pdebench/burgers1d_train.h5 || true
  ln -sf burgers1d_full_v1/burgers1d_val.h5   data/pdebench/burgers1d_val.h5   || true
  ln -sf burgers1d_full_v1/burgers1d_test.h5  data/pdebench/burgers1d_test.h5  || true
fi

echo "Dataset directory contents:"
ls -l "$PDEBENCH_ROOT" || true

export PRECOMPUTE_LATENT=1
nohup bash scripts/run_remote_scale.sh > run_full.log 2>&1 &
sleep 3
tail -n 200 run_full.log || true


