#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/train_multi_pde.yaml}

set -euo pipefail

echo "[1/6] Training latent operator"
python scripts/train.py --config "$CONFIG" --stage operator

echo "[2/6] Training diffusion residual"
python scripts/train.py --config "$CONFIG" --stage diff_residual

echo "[3/6] Consistency distillation"
python scripts/train.py --config "$CONFIG" --stage consistency_distill

echo "[4/6] Training steady prior"
python scripts/train.py --config "$CONFIG" --stage steady_prior

echo "[5/6] Training identity baseline"
python scripts/train_baselines.py --config "$CONFIG" --baseline identity

echo "[6/6] Benchmarking"
python scripts/benchmark.py --config "$CONFIG" --operator checkpoints/operator.pt --output reports/benchmark.json

echo "Exporting operator"
python export/export_latent_operator.py --config "$CONFIG" --checkpoint checkpoints/operator.pt --out-dir export

echo "Evaluating operator"
python scripts/evaluate.py --config "$CONFIG" --operator checkpoints/operator.pt --output-prefix reports/eval

echo "Repro pipeline completed"
