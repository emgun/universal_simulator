#!/usr/bin/env bash
set -eo pipefail

cd /workspace/universal_simulator || exit 1

export PDEBENCH_ROOT=/workspace/universal_simulator/data/pdebench/burgers1d_full_v1
export PYTHONPATH=src

echo "Starting validation evaluation..."
python scripts/evaluate.py \
  --config configs/eval_pdebench_scale_ttc.yaml \
  --operator checkpoints/scale/operator.pt \
  --diffusion checkpoints/scale/diffusion_residual.pt

echo "=== Validation evaluation complete ==="

echo "Starting test evaluation..."
python scripts/evaluate.py \
  --config configs/eval_pdebench_scale_test_ttc.yaml \
  --operator checkpoints/scale/operator.pt \
  --diffusion checkpoints/scale/diffusion_residual.pt

echo "=== Test evaluation complete ==="

echo "Syncing W&B files..."
wandb sync --sync-all

echo "✓ All evaluations complete and synced to W&B"





