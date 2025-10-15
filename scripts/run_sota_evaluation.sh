#!/usr/bin/env bash
set -euo pipefail

# SOTA Evaluation Script - Run baseline + TTC evaluations with SOTA metrics
# Usage: ./scripts/run_sota_evaluation.sh

echo "=== SOTA Evaluation Pipeline ==="
echo "This script will run:"
echo "  1. Baseline (no TTC) - validation split"
echo "  2. Baseline (no TTC) - test split"
echo "  3. TTC enabled - validation split"
echo "  4. TTC enabled - test split"
echo ""

# Setup
export PYTHONPATH=src
CHECKPOINT_DIR=${CHECKPOINT_DIR:-checkpoints/scale}
REPORTS_DIR=${REPORTS_DIR:-reports/sota_eval}

mkdir -p "$REPORTS_DIR"

# Check checkpoints exist
if [ ! -f "$CHECKPOINT_DIR/operator.pt" ] || [ ! -f "$CHECKPOINT_DIR/diffusion_residual.pt" ]; then
  echo "Error: Checkpoints not found in $CHECKPOINT_DIR"
  echo "Please download checkpoints first:"
  echo "  PYTHONPATH=src python scripts/download_checkpoints_from_wandb.py --dest $CHECKPOINT_DIR"
  exit 1
fi

echo "✓ Checkpoints found"
echo ""

# 1. Baseline Validation
echo "=== [1/4] Running Baseline Evaluation (Validation Split) ==="
python scripts/evaluate.py \
  --config configs/eval_burgers_512dim_val_baseline.yaml \
  --operator "$CHECKPOINT_DIR/operator.pt" \
  --diffusion "$CHECKPOINT_DIR/diffusion_residual.pt" \
  --device cuda \
  --output-prefix "$REPORTS_DIR/baseline_val" \
  --print-json

echo "✓ Baseline validation complete"
echo ""

# 2. Baseline Test
echo "=== [2/4] Running Baseline Evaluation (Test Split) ==="
python scripts/evaluate.py \
  --config configs/eval_burgers_512dim_test_baseline.yaml \
  --operator "$CHECKPOINT_DIR/operator.pt" \
  --diffusion "$CHECKPOINT_DIR/diffusion_residual.pt" \
  --device cuda \
  --output-prefix "$REPORTS_DIR/baseline_test" \
  --print-json

echo "✓ Baseline test complete"
echo ""

# 3. TTC Validation
echo "=== [3/4] Running TTC Evaluation (Validation Split) ==="
python scripts/evaluate.py \
  --config configs/eval_burgers_512dim_ttc_val.yaml \
  --operator "$CHECKPOINT_DIR/operator.pt" \
  --diffusion "$CHECKPOINT_DIR/diffusion_residual.pt" \
  --device cuda \
  --output-prefix "$REPORTS_DIR/ttc_val" \
  --print-json

echo "✓ TTC validation complete"
echo ""

# 4. TTC Test
echo "=== [4/4] Running TTC Evaluation (Test Split) ==="
python scripts/evaluate.py \
  --config configs/eval_burgers_512dim_ttc_test.yaml \
  --operator "$CHECKPOINT_DIR/operator.pt" \
  --diffusion "$CHECKPOINT_DIR/diffusion_residual.pt" \
  --device cuda \
  --output-prefix "$REPORTS_DIR/ttc_test" \
  --print-json

echo "✓ TTC test complete"
echo ""

# Summary
echo "=== Evaluation Complete! ==="
echo ""
echo "Results saved to: $REPORTS_DIR"
ls -lh "$REPORTS_DIR"/*.json
echo ""
echo "Next steps:"
echo "  1. Compare results: python scripts/compare_sota_results.py"
echo "  2. Upload to W&B: python scripts/upload_sota_artifacts.py"
echo ""
