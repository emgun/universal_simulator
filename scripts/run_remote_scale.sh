#!/usr/bin/env bash
set -euo pipefail

# Remote launcher for scale-quality training + TTC evaluation on remote GPU instances.
# Example:
#   WANDB_PROJECT=universal-simulator \
#   WANDB_ENTITY=myteam \
#   WANDB_DATASETS="burgers1d_subset_v1" \
#   WANDB_API_KEY=... \
#   bash scripts/run_remote_scale.sh

: "${WANDB_PROJECT:?Set WANDB_PROJECT}"
: "${WANDB_ENTITY:=}"
: "${WANDB_DATASETS:?Set WANDB_DATASETS (e.g. 'burgers1d_subset_v1')}"

TRAIN_CONFIG=${TRAIN_CONFIG:-configs/train_burgers_quality_fullstack.yaml}
TRAIN_STAGE=${TRAIN_STAGE:-all}
TRAIN_EXTRA_ARGS=${TRAIN_EXTRA_ARGS:-}
EVAL_CONFIG=${EVAL_CONFIG:-configs/eval_pdebench_scale_ttc.yaml}
EVAL_TEST_CONFIG=${EVAL_TEST_CONFIG:-configs/eval_pdebench_scale_test_ttc.yaml}
FIX_LIBCUDA=${FIX_LIBCUDA:-1}
RESET_CACHE=${RESET_CACHE:-1}
LATENT_CACHE_DIR=${LATENT_CACHE_DIR:-data/latent_cache}

WORKDIR=${WORKDIR:-$PWD}
DATA_ROOT=${DATA_ROOT:-$WORKDIR/data/pdebench}
mkdir -p "$DATA_ROOT"

# Preflight: ensure we have at least ~5GB free for hydration + caches
REQUIRED_GB=${REQUIRED_GB:-5}
AVAIL_GB=$(df -Pm "$WORKDIR" | awk 'NR==2{print int($4/1024)}')
if [ "$AVAIL_GB" -lt "$REQUIRED_GB" ]; then
  echo "Error: insufficient free space in $WORKDIR (have ${AVAIL_GB}GB, need ${REQUIRED_GB}GB)." >&2
  exit 1
fi

IFS=', ' read -r -a DATASET_ARRAY <<< "$WANDB_DATASETS"
# Prefer Backblaze B2 streaming if credentials are present; otherwise fall back to W&B artifacts
if [ -n "${B2_APP_KEY:-}" ] && [ -n "${B2_KEY_ID:-}" ] && [ -n "${B2_BUCKET:-}" ]; then
  echo "Using Backblaze B2 for dataset hydration via rclone streaming…"
  CLEAN_OLD_SPLITS=1 scripts/fetch_datasets_b2.sh "${DATASET_ARRAY[@]}"
else
  PYTHONPATH=src python scripts/fetch_datasets.py "${DATASET_ARRAY[@]}" --root "$DATA_ROOT" --cache "$WORKDIR/artifacts/cache" --project "${WANDB_PROJECT}" ${WANDB_ENTITY:+--entity "$WANDB_ENTITY"}
fi

export PDEBENCH_ROOT="$DATA_ROOT"
export WANDB_PROJECT
export WANDB_ENTITY

if [ "$FIX_LIBCUDA" -eq 1 ] && command -v bash >/dev/null; then
  if [ -x scripts/fix_libcuda_symlink.sh ]; then
    echo "Ensuring libcuda.so symlink exists…"
    bash scripts/fix_libcuda_symlink.sh || true
  fi
fi

if [ "$RESET_CACHE" -eq 1 ]; then
  echo "Resetting latent cache and checkpoints…"
  rm -rf "$LATENT_CACHE_DIR" checkpoints/scale || true
  mkdir -p "$LATENT_CACHE_DIR" checkpoints/scale
fi

echo "Running training with config: $TRAIN_CONFIG (stage=$TRAIN_STAGE)"
PYTHONPATH=src python scripts/train.py --config "$TRAIN_CONFIG" --stage "$TRAIN_STAGE" $TRAIN_EXTRA_ARGS

OP_CKPT=checkpoints/scale/operator_ema.pt
[[ -f "$OP_CKPT" ]] || OP_CKPT=checkpoints/scale/operator.pt
DIFF_CKPT=checkpoints/scale/diffusion_residual_ema.pt
[[ -f "$DIFF_CKPT" ]] || DIFF_CKPT=checkpoints/scale/diffusion_residual.pt

echo "Evaluating with config: $EVAL_CONFIG"
PYTHONPATH=src python scripts/evaluate.py --config "$EVAL_CONFIG" --operator "$OP_CKPT" --diffusion "$DIFF_CKPT" --output-prefix reports/pdebench_scale_eval --print-json

echo "Evaluating (test split) with config: $EVAL_TEST_CONFIG"
PYTHONPATH=src python scripts/evaluate.py --config "$EVAL_TEST_CONFIG" --operator "$OP_CKPT" --diffusion "$DIFF_CKPT" --output-prefix reports/pdebench_scale_eval_test --print-json

# Optional cleanup to reclaim space
if [ "${CLEANUP_AFTER_RUN:-1}" -eq 1 ]; then
  echo "Cleaning up dataset cache and temporary artifacts..."
  rm -rf "$WORKDIR/artifacts/cache" || true
  find "$DATA_ROOT" -maxdepth 1 -type f -name "*.tmp" -delete || true
fi
