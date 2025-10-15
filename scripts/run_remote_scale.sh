#!/usr/bin/env bash
# Be forgiving on unset variables to allow running without full env (e.g., smoke tests)
set -eo pipefail

# Remote launcher for scale-quality training + TTC evaluation on remote GPU instances.
# Example:
#   WANDB_PROJECT=universal-simulator \
#   WANDB_ENTITY=myteam \
#   WANDB_DATASETS="burgers1d_subset_v1" \
#   WANDB_API_KEY=... \
#   bash scripts/run_remote_scale.sh

# Optional environment configuration (defaults when missing)
WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_DATASETS="${WANDB_DATASETS:-}"

# Enable W&B online mode only if we have a project and login key
if [ -n "${WANDB_API_KEY:-}" ]; then
  # Non-interactive login; ignore errors in CI-like contexts
  if command -v wandb >/dev/null 2>&1; then
    wandb login --relogin "$WANDB_API_KEY" >/dev/null 2>&1 || true
    wandb online >/dev/null 2>&1 || true
  fi
else
  export WANDB_MODE="offline"
fi

TRAIN_CONFIG=${TRAIN_CONFIG:-configs/train_burgers_quality_v2.yaml}
TRAIN_STAGE=${TRAIN_STAGE:-all}
TRAIN_EXTRA_ARGS=${TRAIN_EXTRA_ARGS:-}
EVAL_CONFIG=${EVAL_CONFIG:-configs/eval_pdebench_scale_ttc.yaml}
EVAL_TEST_CONFIG=${EVAL_TEST_CONFIG:-configs/eval_pdebench_scale_test_ttc.yaml}
FIX_LIBCUDA=${FIX_LIBCUDA:-1}
RESET_CACHE=${RESET_CACHE:-1}
LATENT_CACHE_DIR=${LATENT_CACHE_DIR:-data/latent_cache}

WORKDIR=${WORKDIR:-$PWD}
# Respect explicit PDEBENCH_ROOT when provided; otherwise default under workdir
if [ -n "${PDEBENCH_ROOT:-}" ]; then
  DATA_ROOT="$PDEBENCH_ROOT"
else
  DATA_ROOT=${DATA_ROOT:-$WORKDIR/data/pdebench}
fi
mkdir -p "$DATA_ROOT"

# Preflight: ensure we have at least ~5GB free for hydration + caches
REQUIRED_GB=${REQUIRED_GB:-5}
AVAIL_GB=$(df -Pm "$WORKDIR" | awk 'NR==2{print int($4/1024)}')
if [ "$AVAIL_GB" -lt "$REQUIRED_GB" ]; then
  echo "Error: insufficient free space in $WORKDIR (have ${AVAIL_GB}GB, need ${REQUIRED_GB}GB)." >&2
  exit 1
fi

# Hydration: if WANDB_DATASETS set, hydrate; otherwise skip and rely on existing files under DATA_ROOT
if [ -n "$WANDB_DATASETS" ]; then
  IFS=', ' read -r -a DATASET_ARRAY <<< "$WANDB_DATASETS"
  # Prefer Backblaze B2 streaming if credentials are present; otherwise fall back to W&B artifacts
  if [ -n "${B2_APP_KEY:-}" ] && [ -n "${B2_KEY_ID:-}" ] && [ -n "${B2_BUCKET:-}" ]; then
    echo "Using Backblaze B2 for dataset hydration via rclone streaming…"
    CLEAN_OLD_SPLITS=1 scripts/fetch_datasets_b2.sh "${DATASET_ARRAY[@]}"
  else
    PYTHONPATH=src python scripts/fetch_datasets.py "${DATASET_ARRAY[@]}" --root "$DATA_ROOT" --cache "$WORKDIR/artifacts/cache" --project "${WANDB_PROJECT}" ${WANDB_ENTITY:+--entity "$WANDB_ENTITY"}
  fi
else
  echo "Skipping dataset hydration (WANDB_DATASETS unset); expecting datasets under $DATA_ROOT"
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

# Sanitize TRAIN_CONFIG if it was accidentally concatenated with overrides
if [ ! -f "$TRAIN_CONFIG" ] && [[ "$TRAIN_CONFIG" == *,* ]]; then
  TRAIN_CONFIG="${TRAIN_CONFIG%%,*}"
fi

# Skip training if EVAL_ONLY=1
if [ "${EVAL_ONLY:-0}" -eq 0 ]; then
  if [ "${PRECOMPUTE_LATENT:-1}" -eq 1 ]; then
    echo "Precomputing latent caches (train/val/test)…"
    PYTHONPATH=src python scripts/precompute_latent_cache.py \
      --config "${TRAIN_CONFIG}" \
      --tasks burgers1d \
      --splits train val test \
      --root "${PDEBENCH_ROOT:-$DATA_ROOT}" \
      --cache-dir "${LATENT_CACHE_DIR}" \
      --device cuda \
      --num-workers ${PRECOMPUTE_WORKERS:-0} \
      --batch-size 4 || true
  fi

  echo "Running training with config: $TRAIN_CONFIG (stage=$TRAIN_STAGE)"
  PYTHONPATH=src python scripts/train.py --config "$TRAIN_CONFIG" --stage "$TRAIN_STAGE" $TRAIN_EXTRA_ARGS

  # Ensure scale checkpoint paths exist even when training wrote root-level files
  mkdir -p checkpoints/scale
  if [ ! -f checkpoints/scale/operator.pt ] && [ -f checkpoints/operator.pt ]; then
    cp -f checkpoints/operator.pt checkpoints/scale/operator.pt
  fi
  if [ ! -f checkpoints/scale/operator_ema.pt ] && [ -f checkpoints/operator_ema.pt ]; then
    cp -f checkpoints/operator_ema.pt checkpoints/scale/operator_ema.pt
  fi
  if [ ! -f checkpoints/scale/diffusion_residual.pt ] && [ -f checkpoints/diffusion_residual.pt ]; then
    cp -f checkpoints/diffusion_residual.pt checkpoints/scale/diffusion_residual.pt
  fi
  if [ ! -f checkpoints/scale/diffusion_residual_ema.pt ] && [ -f checkpoints/diffusion_residual_ema.pt ]; then
    cp -f checkpoints/diffusion_residual_ema.pt checkpoints/scale/diffusion_residual_ema.pt
  fi
else
  echo "EVAL_ONLY mode: Skipping training, downloading checkpoints from W&B..."
  mkdir -p checkpoints/scale

  # Download checkpoints from W&B - prefer run IDs over artifacts
  # Run IDs point to actual run files (more reliable)
  # Artifacts are used as fallback
  if [ -n "${OPERATOR_RUN:-}" ] || [ -n "${DIFFUSION_RUN:-}" ]; then
    echo "Downloading checkpoints from W&B runs..."
    echo "  Operator run: ${OPERATOR_RUN:-pru2jxc4}"
    echo "  Diffusion run: ${DIFFUSION_RUN:-pp0c2k31}"

    PYTHONPATH=src python scripts/download_checkpoints_from_wandb.py \
      --dest checkpoints/scale \
      --entity "${WANDB_ENTITY}" \
      --project "${WANDB_PROJECT}" \
      --operator-run "${OPERATOR_RUN:-pru2jxc4}" \
      --diffusion-run "${DIFFUSION_RUN:-pp0c2k31}" || {
      echo "Failed to download checkpoints from W&B runs. Exiting."
      exit 1
    }
  else
    # Fallback to artifacts
    OPERATOR_ARTIFACT="${OPERATOR_ARTIFACT:-run-mt7rckc8-history:v0}"
    DIFFUSION_ARTIFACT="${DIFFUSION_ARTIFACT:-run-pp0c2k31-history:v0}"
    CONSISTENCY_ARTIFACT="${CONSISTENCY_ARTIFACT:-run-n932efgl-history:v0}"

    echo "Downloading checkpoints from W&B artifacts..."
    echo "  Operator: ${OPERATOR_ARTIFACT}"
    echo "  Diffusion: ${DIFFUSION_ARTIFACT}"
    echo "  Consistency: ${CONSISTENCY_ARTIFACT}"

    PYTHONPATH=src python scripts/download_checkpoints_from_wandb.py \
      --dest checkpoints/scale \
      --entity "${WANDB_ENTITY}" \
      --project "${WANDB_PROJECT}" \
      --operator-artifact "${OPERATOR_ARTIFACT}" \
      --diffusion-artifact "${DIFFUSION_ARTIFACT}" \
      --consistency-artifact "${CONSISTENCY_ARTIFACT}" || {
      echo "Failed to download checkpoints from W&B artifacts. Exiting."
      exit 1
    }
  fi
fi

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
