#!/usr/bin/env bash
set -eo pipefail

cd /workspace/universal_simulator || exit 1

# Kill any existing training
pkill -9 -f "python scripts" || true
sleep 2

# Precompute latent cache
echo "Starting latent cache precomputation..."
export PDEBENCH_ROOT=/workspace/universal_simulator/data/pdebench/burgers1d_full_v1
export PYTHONPATH=src

python scripts/precompute_latent_cache.py \
  --config configs/train_burgers_quality_v2_resume.yaml \
  --num-workers 0

echo "✓ Latent cache precomputed"
ls -lh data/latent_cache/

# Start consistency distillation training
echo "Starting consistency distillation training..."
nohup python scripts/train.py \
  --config configs/train_burgers_quality_v2_resume.yaml \
  --stage consistency_distill \
  > run_consistency.log 2>&1 &

echo "Training launched in background"
sleep 10
tail -n 100 run_consistency.log

