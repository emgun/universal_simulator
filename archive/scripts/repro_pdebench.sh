#!/usr/bin/env bash
set -euo pipefail

# Reproduce Burgers1D evaluation from published artifacts.
# Requires WANDB login and dataset/checkpoint artifacts available.
# Usage example:
#   WANDB_PROJECT=universal-simulator WANDB_ENTITY=emgun-morpheus-space \
#   DATASETS=burgers1d_subset_v1 CHECKPOINT=emgun-morpheus-space/universal-simulator/burgers1d-operator:latest \
#   bash scripts/repro_pdebench.sh

: "${WANDB_PROJECT:?Set WANDB_PROJECT}"
: "${WANDB_ENTITY:?Set WANDB_ENTITY}"
: "${DATASETS:?Set DATASETS (space/comma separated dataset keys from docs/dataset_registry.yaml)}"
: "${CHECKPOINT:?Set CHECKPOINT artifact path for operator.pt}"
: "${DIFFUSION_CHECKPOINT:?Set DIFFUSION_CHECKPOINT artifact path (diffusion_residual.pt)}"

WORKDIR=${WORKDIR:-$PWD}
DATA_ROOT=${DATA_ROOT:-$WORKDIR/data/pdebench}
mkdir -p "$DATA_ROOT" "$WORKDIR/artifacts/cache"

IFS=', ' read -r -a DATASET_ARRAY <<< "$DATASETS"
PYTHONPATH=src python scripts/fetch_datasets.py "${DATASET_ARRAY[@]}" --root "$DATA_ROOT" --cache "$WORKDIR/artifacts/cache" --project "$WANDB_PROJECT" --entity "$WANDB_ENTITY"

# Download checkpoints
wandb artifact get "$CHECKPOINT" --root "$WORKDIR/artifacts/checkpoints/operator"
wandb artifact get "$DIFFUSION_CHECKPOINT" --root "$WORKDIR/artifacts/checkpoints/diffusion"
OPERATOR_PATH=$(find "$WORKDIR/artifacts/checkpoints/operator" -name 'operator.pt' | head -n1)
DIFF_PATH=$(find "$WORKDIR/artifacts/checkpoints/diffusion" -name 'diffusion_residual.pt' | head -n1)

export PDEBENCH_ROOT="$DATA_ROOT"
export WANDB_PROJECT
export WANDB_ENTITY

PYTHONPATH=src python scripts/evaluate.py --config configs/eval_pdebench_scale.yaml --operator "$OPERATOR_PATH" --diffusion "$DIFF_PATH" --output-prefix reports/repro_eval --print-json
PYTHONPATH=src python scripts/evaluate.py --config configs/eval_pdebench_scale_test.yaml --operator "$OPERATOR_PATH" --diffusion "$DIFF_PATH" --output-prefix reports/repro_eval_test --print-json

echo "Reproduction artifacts written to reports/repro_eval*."
