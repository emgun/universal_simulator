#!/usr/bin/env bash
set -euo pipefail

# Reproduce evaluation from immutable data locks and W&B checkpoint artifacts.
: "${WANDB_PROJECT:?Set WANDB_PROJECT}"
: "${WANDB_ENTITY:?Set WANDB_ENTITY}"
: "${DATA_LOCK:?Set DATA_LOCK to an immutable train/validation lock}"
: "${CHECKPOINT:?Set CHECKPOINT artifact path for operator.pt}"
: "${DIFFUSION_CHECKPOINT:?Set DIFFUSION_CHECKPOINT artifact path}"

WORKDIR=${WORKDIR:-$PWD}
DATA_ROOT=${DATA_ROOT:-$WORKDIR/data/pdebench}
DATA_CACHE=${DATA_CACHE:-$WORKDIR/data/cache}
mkdir -p "$DATA_ROOT" "$WORKDIR/artifacts/checkpoints"

PYTHONPATH=src python -m ups.data.cli stage \
  --lock "$DATA_LOCK" --cache "$DATA_CACHE" --run-dir "$DATA_ROOT" \
  --report "$WORKDIR/reports/repro_data_stage.json"
PYTHONPATH=src python -m ups.data.cli verify --lock "$DATA_LOCK" --cache "$DATA_CACHE"

wandb artifact get "$CHECKPOINT" --root "$WORKDIR/artifacts/checkpoints/operator"
wandb artifact get "$DIFFUSION_CHECKPOINT" --root "$WORKDIR/artifacts/checkpoints/diffusion"
OPERATOR_PATH=$(find "$WORKDIR/artifacts/checkpoints/operator" -name 'operator.pt' | head -n1)
DIFF_PATH=$(find "$WORKDIR/artifacts/checkpoints/diffusion" -name 'diffusion_residual.pt' | head -n1)

export PDEBENCH_ROOT="$DATA_ROOT"
export WANDB_PROJECT WANDB_ENTITY

PYTHONPATH=src python scripts/evaluate.py --config configs/eval_pdebench_scale.yaml \
  --operator "$OPERATOR_PATH" --diffusion "$DIFF_PATH" \
  --output-prefix reports/repro_eval --print-json

if [ "${RUN_TEST_MEASUREMENT:-0}" -eq 1 ]; then
  : "${MEASUREMENT_DATA_LOCK:?RUN_TEST_MEASUREMENT=1 requires MEASUREMENT_DATA_LOCK}"
  PYTHONPATH=src python -m ups.data.cli stage \
    --lock "$MEASUREMENT_DATA_LOCK" --cache "$DATA_CACHE" --run-dir "$DATA_ROOT" \
    --report "$WORKDIR/reports/repro_measurement_stage.json"
  DATA_LOCK="$MEASUREMENT_DATA_LOCK" PYTHONPATH=src python scripts/evaluate.py --config configs/eval_pdebench_scale_test.yaml \
    --operator "$OPERATOR_PATH" --diffusion "$DIFF_PATH" \
    --output-prefix reports/repro_eval_test --print-json
fi
