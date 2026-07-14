#!/usr/bin/env bash
# Be forgiving on unset variables to allow running without full env (e.g., smoke tests)
set -eo pipefail

# Remote launcher for scale-quality training + TTC evaluation on remote GPU instances.
# Example:
#   WANDB_PROJECT=universal-simulator \
#   WANDB_ENTITY=myteam \
#   DATA_LOCK=/path/to/training.data.lock.json \
#   WANDB_API_KEY=... \
#   bash scripts/run_remote_scale.sh

# Optional environment configuration (defaults when missing)
WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
DATA_LOCK="${DATA_LOCK:-}"
DATA_CACHE="${DATA_CACHE:-$PWD/data/cache}"

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
RESET_CACHE=${RESET_CACHE:-0}
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

# Training consumes only bytes named by an immutable lock. A prestaged bypass is
# explicit because it forfeits automatic byte verification and provenance.
if [ -n "$DATA_LOCK" ]; then
  echo "Planning and staging locked training data…"
  PYTHONPATH=src python -m ups.data.cli plan \
    --lock "$DATA_LOCK" --cache "$DATA_CACHE" --reserve-bytes "$((REQUIRED_GB * 1024 * 1024 * 1024))"
  PYTHONPATH=src python -m ups.data.cli stage \
    --lock "$DATA_LOCK" --cache "$DATA_CACHE" --run-dir "$DATA_ROOT" \
    --report "$WORKDIR/reports/data_stage_training.json"
  PYTHONPATH=src python -m ups.data.cli verify --lock "$DATA_LOCK" --cache "$DATA_CACHE"
elif [ "${UPS_ALLOW_PRESTAGED_DATA:-0}" -ne 1 ]; then
  echo "Error: DATA_LOCK is required (or set UPS_ALLOW_PRESTAGED_DATA=1 for an explicit local-only bypass)." >&2
  exit 1
else
  echo "Using explicitly allowed prestaged data under $DATA_ROOT"
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
    echo "Precomputing latent caches (train/val only)…"
    PYTHONPATH=src python scripts/precompute_latent_cache.py \
      --config "${TRAIN_CONFIG}" \
      --tasks burgers1d \
      --splits train val \
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

  # Download checkpoints from W&B artifacts
  # Use explicit artifact paths if provided, otherwise try to auto-discover
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

OP_CKPT=checkpoints/scale/operator_ema.pt
[[ -f "$OP_CKPT" ]] || OP_CKPT=checkpoints/scale/operator.pt
DIFF_CKPT=checkpoints/scale/diffusion_residual_ema.pt
[[ -f "$DIFF_CKPT" ]] || DIFF_CKPT=checkpoints/scale/diffusion_residual.pt

echo "Evaluating with config: $EVAL_CONFIG"
PYTHONPATH=src python scripts/evaluate.py --config "$EVAL_CONFIG" --operator "$OP_CKPT" --diffusion "$DIFF_CKPT" --output-prefix reports/pdebench_scale_eval --print-json

if [ "${RUN_TEST_MEASUREMENT:-0}" -eq 1 ]; then
  : "${MEASUREMENT_DATA_LOCK:?RUN_TEST_MEASUREMENT=1 requires MEASUREMENT_DATA_LOCK}"
  echo "Staging separately authorized measurement bytes…"
  PYTHONPATH=src python -m ups.data.cli stage \
    --lock "$MEASUREMENT_DATA_LOCK" --cache "$DATA_CACHE" --run-dir "$DATA_ROOT" \
    --report "$WORKDIR/reports/data_stage_measurement.json"
  DATA_LOCK="$MEASUREMENT_DATA_LOCK" PYTHONPATH=src python scripts/evaluate.py --config "$EVAL_TEST_CONFIG" --operator "$OP_CKPT" --diffusion "$DIFF_CKPT" --output-prefix reports/pdebench_scale_eval_test --print-json
else
  echo "Skipping test measurement; set RUN_TEST_MEASUREMENT=1 with a measurement lock to authorize it."
fi

# Optional cleanup to reclaim space
if [ "${CLEANUP_AFTER_RUN:-0}" -eq 1 ] && [ -n "$DATA_LOCK" ]; then
  echo "Reporting unpinned cache objects (dry run)…"
  PYTHONPATH=src python -m ups.data.cli evict --cache "$DATA_CACHE" --lock "$DATA_LOCK"
fi
