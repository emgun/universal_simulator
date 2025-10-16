#!/usr/bin/env bash
set -euo pipefail

echo "=== Remote SOTA Evaluation Setup and Run ==="
cd /workspace/universal_simulator

# Install dependencies if needed
if ! python -c "import torch" 2>/dev/null; then
  echo "Installing PyTorch..."
  pip install --quiet --upgrade pip
  pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu121
fi

if ! python -c "import wandb" 2>/dev/null; then
  echo "Installing project dependencies..."
  pip install --quiet -e .
fi

echo "✓ Dependencies installed"

# Set environment
export PYTHONPATH=src
export WANDB_API_KEY="ec37eba84558733a8ef56c76e284ab530e94449b"
export WANDB_PROJECT="universal-simulator"
export WANDB_ENTITY="emgun-morpheus-space"
export B2_S3_ENDPOINT="https://s3.us-west-004.backblazeb2.com"
export B2_S3_REGION="us-west-004"
export B2_KEY_ID="0043616a62c8bb90000000001"
export B2_APP_KEY="K004cur7hMs3GDPixFB8FlzfCJV2PIc"

# Download checkpoints
echo "=== Downloading Checkpoints from W&B ==="
mkdir -p checkpoints/scale

python scripts/download_checkpoints_from_wandb.py \
  --dest checkpoints/scale \
  --entity "$WANDB_ENTITY" \
  --project "$WANDB_PROJECT" || {
  echo "WARNING: Checkpoint download failed"
  echo "Trying alternative method..."

  # Manual download using wandb API
  python << 'PYEOF'
import wandb
import os

wandb.login(key=os.environ['WANDB_API_KEY'])
api = wandb.Api()

# Look for the latest run with checkpoints
runs = api.runs(f"{os.environ['WANDB_ENTITY']}/{os.environ['WANDB_PROJECT']}",
                filters={"tags": "burgers1d"})

print(f"Found {len(runs)} runs")
for run in runs[:5]:
    print(f"  - {run.name} ({run.id})")
    for artifact in run.logged_artifacts():
        if 'checkpoint' in artifact.type or 'model' in artifact.type:
            print(f"    Artifact: {artifact.name}")
PYEOF
}

# Fetch PDEBench data
echo "=== Fetching PDEBench Data ==="
export PDEBENCH_ROOT=data/pdebench
bash scripts/fetch_datasets_b2.sh || {
  echo "WARNING: B2 download failed"
  PYTHONPATH=src python scripts/fetch_datasets.py --task burgers1d || echo "Dataset fetch also failed"
}

# Check if checkpoints exist
if [ ! -f checkpoints/scale/operator.pt ] || [ ! -f checkpoints/scale/diffusion_residual.pt ]; then
  echo "ERROR: Checkpoints not found!"
  ls -la checkpoints/scale/ || echo "Directory doesn't exist"
  exit 1
fi

echo "✓ Checkpoints ready"

# Run evaluations
echo "=== Running SOTA Evaluations ==="
mkdir -p reports/sota_eval

# 1. Baseline Validation
echo "[1/4] Baseline Validation..."
python scripts/evaluate.py \
  --config configs/eval_burgers_512dim_val_baseline.yaml \
  --operator checkpoints/scale/operator.pt \
  --diffusion checkpoints/scale/diffusion_residual.pt \
  --device cuda \
  --output-prefix reports/sota_eval/baseline_val \
  --print-json

# 2. Baseline Test
echo "[2/4] Baseline Test..."
python scripts/evaluate.py \
  --config configs/eval_burgers_512dim_test_baseline.yaml \
  --operator checkpoints/scale/operator.pt \
  --diffusion checkpoints/scale/diffusion_residual.pt \
  --device cuda \
  --output-prefix reports/sota_eval/baseline_test \
  --print-json

# 3. TTC Validation
echo "[3/4] TTC Validation..."
python scripts/evaluate.py \
  --config configs/eval_burgers_512dim_ttc_val.yaml \
  --operator checkpoints/scale/operator.pt \
  --diffusion checkpoints/scale/diffusion_residual.pt \
  --device cuda \
  --output-prefix reports/sota_eval/ttc_val \
  --print-json

# 4. TTC Test
echo "[4/4] TTC Test..."
python scripts/evaluate.py \
  --config configs/eval_burgers_512dim_ttc_test.yaml \
  --operator checkpoints/scale/operator.pt \
  --diffusion checkpoints/scale/diffusion_residual.pt \
  --device cuda \
  --output-prefix reports/sota_eval/ttc_test \
  --print-json

echo "=== All Evaluations Complete! ==="
ls -lh reports/sota_eval/

# Upload to W&B
echo "=== Uploading Results to W&B ==="
python << 'UPLOAD_EOF'
import wandb
import os
import json

wandb.login(key=os.environ['WANDB_API_KEY'])
run = wandb.init(
    project=os.environ['WANDB_PROJECT'],
    entity=os.environ['WANDB_ENTITY'],
    job_type='evaluation',
    tags=['sota-comparison', 'baseline-vs-ttc', 'burgers1d'],
    name='sota-eval-baseline-ttc-comparison'
)

# Upload artifact
artifact = wandb.Artifact('sota-eval-baseline-ttc', type='evaluation')
artifact.add_dir('reports/sota_eval')

# Load and log metrics
results = {}
for split in ['val', 'test']:
    for variant in ['baseline', 'ttc']:
        json_file = f'reports/sota_eval/{variant}_{split}.json'
        try:
            with open(json_file) as f:
                data = json.load(f)
                metrics = data.get('metrics', {})
                results[f'{variant}_{split}'] = metrics
                for k, v in metrics.items():
                    run.log({f'{variant}_{split}_{k}': v})
                print(f'✓ Logged metrics from {json_file}')
        except Exception as e:
            print(f'✗ Could not load {json_file}: {e}')

# Print comparison
print("\n=== SOTA Comparison Summary ===")
print(f"Baseline Val: rel_l2={results.get('baseline_val', {}).get('rel_l2', 'N/A'):.6f}, nrmse={results.get('baseline_val', {}).get('nrmse', 'N/A'):.6f}")
print(f"TTC Val:      rel_l2={results.get('ttc_val', {}).get('rel_l2', 'N/A'):.6f}, nrmse={results.get('ttc_val', {}).get('nrmse', 'N/A'):.6f}")
print(f"Baseline Test: rel_l2={results.get('baseline_test', {}).get('rel_l2', 'N/A'):.6f}, nrmse={results.get('baseline_test', {}).get('nrmse', 'N/A'):.6f}")
print(f"TTC Test:      rel_l2={results.get('ttc_test', {}).get('rel_l2', 'N/A'):.6f}, nrmse={results.get('ttc_test', {}).get('nrmse', 'N/A'):.6f}")
print(f"\nFNO-2D SOTA: rel_l2=0.0180")

run.log_artifact(artifact)
run.finish()
print('✓ Results uploaded to W&B')
UPLOAD_EOF

echo "=== SUCCESS ===="
echo "Results available at: https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT"
