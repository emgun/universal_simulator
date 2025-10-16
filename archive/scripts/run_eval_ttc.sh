#!/usr/bin/env bash
set -euo pipefail

# TTC Evaluation runner for remote instances
# Fetches checkpoints from W&B and runs comprehensive TTC evaluation

echo "=== TTC Evaluation Pipeline ==="

# Environment setup
WANDB_PROJECT="${WANDB_PROJECT:-universal-simulator}"
WANDB_ENTITY="${WANDB_ENTITY:-emgun-morpheus-space}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
WORKDIR="${WORKDIR:-$PWD}"
DATA_ROOT="${DATA_ROOT:-$WORKDIR/data/pdebench}"
EVAL_CONFIG="${EVAL_CONFIG:-configs/eval_pdebench_scale_ttc.yaml}"
EVAL_TEST_CONFIG="${EVAL_TEST_CONFIG:-configs/eval_pdebench_scale_test_ttc.yaml}"

# Checkpoint download settings
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/scale}"
WANDB_CHECKPOINT_RUN="${WANDB_CHECKPOINT_RUN:-}"  # Optional: specific run ID to download from

mkdir -p "$DATA_ROOT" "$CHECKPOINT_DIR" reports

# W&B login if API key is set
if [ -n "${WANDB_API_KEY:-}" ]; then
  if command -v wandb >/dev/null 2>&1; then
    wandb login --relogin "$WANDB_API_KEY" >/dev/null 2>&1 || true
    wandb online >/dev/null 2>&1 || true
  fi
else
  export WANDB_MODE="offline"
fi

export WANDB_PROJECT
export WANDB_ENTITY

# Dataset hydration (if needed)
if [ -n "${WANDB_DATASETS:-}" ]; then
  echo "Hydrating datasets: $WANDB_DATASETS"
  IFS=', ' read -r -a DATASET_ARRAY <<< "$WANDB_DATASETS"

  if [ -n "${B2_APP_KEY:-}" ] && [ -n "${B2_KEY_ID:-}" ] && [ -n "${B2_BUCKET:-}" ]; then
    echo "Using Backblaze B2 for dataset hydration..."
    CLEAN_OLD_SPLITS=1 scripts/fetch_datasets_b2.sh "${DATASET_ARRAY[@]}"
  else
    echo "Using W&B artifacts for dataset hydration..."
    PYTHONPATH=src python scripts/fetch_datasets.py "${DATASET_ARRAY[@]}" \
      --root "$DATA_ROOT" \
      --cache "$WORKDIR/artifacts/cache" \
      --project "${WANDB_PROJECT}" \
      ${WANDB_ENTITY:+--entity "$WANDB_ENTITY"}
  fi
else
  echo "WANDB_DATASETS not set, assuming datasets already present"
fi

export PDEBENCH_ROOT="$DATA_ROOT"

# Download checkpoints from W&B if not present locally
if [ ! -f "$CHECKPOINT_DIR/operator.pt" ] || [ ! -f "$CHECKPOINT_DIR/diffusion_residual.pt" ]; then
  echo "Downloading checkpoints from W&B..."

  if [ -n "$WANDB_CHECKPOINT_RUN" ]; then
    # Download from specific run
    echo "Fetching checkpoints from run: $WANDB_CHECKPOINT_RUN"
    PYTHONPATH=src python -c "
import wandb
import sys
api = wandb.Api()
try:
    run = api.run('${WANDB_ENTITY}/${WANDB_PROJECT}/${WANDB_CHECKPOINT_RUN}')
    files = run.files()
    for f in files:
        if 'operator' in f.name and f.name.endswith('.pt'):
            print(f'Downloading {f.name}...')
            f.download(root='$CHECKPOINT_DIR', replace=True)
        elif 'diffusion_residual' in f.name and f.name.endswith('.pt'):
            print(f'Downloading {f.name}...')
            f.download(root='$CHECKPOINT_DIR', replace=True)
    print('✓ Checkpoints downloaded')
except Exception as e:
    print(f'Error downloading checkpoints: {e}', file=sys.stderr)
    sys.exit(1)
"
  else
    # Find most recent run with checkpoints
    echo "Finding most recent run with checkpoints..."
    PYTHONPATH=src python -c "
import wandb
import sys
api = wandb.Api()
try:
    runs = api.runs('${WANDB_ENTITY}/${WANDB_PROJECT}',
                    filters={'tags': {'\\$in': ['training', 'quality', 'burgers1d']}},
                    order='-created_at')

    for run in runs[:5]:  # Check last 5 runs
        files = [f.name for f in run.files()]
        has_operator = any('operator' in f and f.endswith('.pt') for f in files)
        has_diffusion = any('diffusion' in f and f.endswith('.pt') for f in files)

        if has_operator and has_diffusion:
            print(f'Found checkpoints in run: {run.id} ({run.name})')
            for f in run.files():
                if ('operator' in f.name or 'diffusion_residual' in f.name) and f.name.endswith('.pt'):
                    print(f'  Downloading {f.name}...')
                    f.download(root='$CHECKPOINT_DIR', replace=True)
            print('✓ Checkpoints downloaded')
            sys.exit(0)

    print('No recent runs found with checkpoints', file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f'Error finding/downloading checkpoints: {e}', file=sys.stderr)
    sys.exit(1)
"
  fi
fi

# Prefer EMA checkpoints if available
OP_CKPT="$CHECKPOINT_DIR/operator_ema.pt"
[[ -f "$OP_CKPT" ]] || OP_CKPT="$CHECKPOINT_DIR/operator.pt"

DIFF_CKPT="$CHECKPOINT_DIR/diffusion_residual_ema.pt"
[[ -f "$DIFF_CKPT" ]] || DIFF_CKPT="$CHECKPOINT_DIR/diffusion_residual.pt"

echo "Using checkpoints:"
echo "  Operator: $OP_CKPT"
echo "  Diffusion: $DIFF_CKPT"

# Run evaluation on validation split
echo ""
echo "=== Running TTC Evaluation (Validation) ==="
PYTHONPATH=src python scripts/evaluate.py \
  --config "$EVAL_CONFIG" \
  --operator "$OP_CKPT" \
  --diffusion "$DIFF_CKPT" \
  --device cuda \
  --output-prefix reports/ttc_eval_val \
  --print-json

# Run evaluation on test split
echo ""
echo "=== Running TTC Evaluation (Test) ==="
PYTHONPATH=src python scripts/evaluate.py \
  --config "$EVAL_TEST_CONFIG" \
  --operator "$OP_CKPT" \
  --diffusion "$DIFF_CKPT" \
  --device cuda \
  --output-prefix reports/ttc_eval_test \
  --print-json

echo ""
echo "=== TTC Evaluation Complete ==="
ls -lh reports/ttc_eval_*

# Optional cleanup
if [ "${CLEANUP_AFTER_RUN:-0}" -eq 1 ]; then
  echo "Cleaning up cache..."
  rm -rf "$WORKDIR/artifacts/cache" || true
fi
