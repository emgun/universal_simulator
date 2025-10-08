#!/usr/bin/env bash
set -euo pipefail

# Remote launcher for large-scale Burgers1D runs.
# Usage (example):
#   WANDB_ENTITY=myteam WANDB_PROJECT=universal-simulator \
#   WANDB_DATASET=emgun-morpheus-space/universal-simulator/burgers1d-pdebench-eval:v0 \
#   bash scripts/run_remote_scale.sh

: "${WANDB_PROJECT:?Set WANDB_PROJECT}"
: "${WANDB_ENTITY:=}"
# Space- or comma-separated dataset keys defined in docs/dataset_registry.yaml
: "${WANDB_DATASETS:?Set WANDB_DATASETS (e.g. 'burgers1d_subset_v1')}"

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

# Run all training stages in a single W&B run for better chart visualization
PYTHONPATH=src python scripts/train.py --config configs/train_pdebench_scale.yaml --stage all

OP_CKPT=checkpoints/scale/operator_ema.pt
[[ -f "$OP_CKPT" ]] || OP_CKPT=checkpoints/scale/operator.pt
DIFF_CKPT=checkpoints/scale/diffusion_residual_ema.pt
[[ -f "$DIFF_CKPT" ]] || DIFF_CKPT=checkpoints/scale/diffusion_residual.pt

PYTHONPATH=src python scripts/evaluate.py --config configs/eval_pdebench_scale.yaml --operator "$OP_CKPT" --diffusion "$DIFF_CKPT" --output-prefix reports/pdebench_scale_eval --print-json
PYTHONPATH=src python scripts/evaluate.py --config configs/eval_pdebench_scale_test.yaml --operator "$OP_CKPT" --diffusion "$DIFF_CKPT" --output-prefix reports/pdebench_scale_eval_test --print-json

# Optional cleanup to reclaim space
if [ "${CLEANUP_AFTER_RUN:-1}" -eq 1 ]; then
  echo "Cleaning up dataset cache and temporary artifacts..."
  rm -rf "$WORKDIR/artifacts/cache" || true
  find "$DATA_ROOT" -maxdepth 1 -type f -name "*.tmp" -delete || true
fi
