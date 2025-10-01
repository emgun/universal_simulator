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

IFS=', ' read -r -a DATASET_ARRAY <<< "$WANDB_DATASETS"
PYTHONPATH=src python scripts/fetch_datasets.py "${DATASET_ARRAY[@]}" --root "$DATA_ROOT" --cache "$WORKDIR/artifacts/cache" --project "${WANDB_PROJECT}" ${WANDB_ENTITY:+--entity "$WANDB_ENTITY"}

export PDEBENCH_ROOT="$DATA_ROOT"
export WANDB_PROJECT
export WANDB_ENTITY

PYTHONPATH=src python scripts/train.py --config configs/train_pdebench_scale.yaml --stage operator
PYTHONPATH=src python scripts/train.py --config configs/train_pdebench_scale.yaml --stage diff_residual
PYTHONPATH=src python scripts/train.py --config configs/train_pdebench_scale.yaml --stage consistency_distill
PYTHONPATH=src python scripts/train.py --config configs/train_pdebench_scale.yaml --stage steady_prior

PYTHONPATH=src python scripts/evaluate.py --config configs/eval_pdebench_scale.yaml --operator checkpoints/scale/operator.pt --diffusion checkpoints/scale/diffusion_residual.pt --output-prefix reports/pdebench_scale_eval --print-json
PYTHONPATH=src python scripts/evaluate.py --config configs/eval_pdebench_scale_test.yaml --operator checkpoints/scale/operator.pt --diffusion checkpoints/scale/diffusion_residual.pt --output-prefix reports/pdebench_scale_eval_test --print-json
