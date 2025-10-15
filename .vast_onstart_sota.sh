#!/bin/bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

echo "=== SOTA Evaluation Instance Setup ==="

# Install dependencies
apt-get update && apt-get install -y git curl jq

# Setup workspace
mkdir -p /workspace/universal_simulator
cd /workspace/universal_simulator

# Clone repo
if [ ! -d .git ]; then
  git clone https://github.com/emerygunselman/universal_simulator.git .
fi
git fetch origin
git checkout feature/sota_burgers_upgrades || git checkout -b feature/sota_burgers_upgrades origin/feature/sota_burgers_upgrades || true
git pull origin feature/sota_burgers_upgrades || true

echo "=== Installing Python Dependencies ==="
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e .

echo "=== Downloading Checkpoints from W&B ==="
export WANDB_API_KEY="ec37eba84558733a8ef56c76e284ab530e94449b"
export WANDB_PROJECT="universal-simulator"
export WANDB_ENTITY="emgun-morpheus-space"

mkdir -p checkpoints/scale

PYTHONPATH=src python scripts/download_checkpoints_from_wandb.py \
  --dest checkpoints/scale \
  --entity "$WANDB_ENTITY" \
  --project "$WANDB_PROJECT" || {
  echo "WARNING: Checkpoint download failed, will try alternative method"
  # Try downloading specific artifacts
  pip install wandb
  python -c "
import wandb
import os
api = wandb.Api()
entity = os.environ['WANDB_ENTITY']
project = os.environ['WANDB_PROJECT']
# Look for recent checkpoints
runs = api.runs(f'{entity}/{project}', filters={'tags': 'burgers1d'})
print(f'Found {len(runs)} runs with burgers1d tag')
" || echo "Manual checkpoint download also failed"
}

echo "=== Fetching PDEBench Data ==="
export PDEBENCH_ROOT=/workspace/universal_simulator/data/pdebench
export B2_S3_ENDPOINT="https://s3.us-west-004.backblazeb2.com"
export B2_S3_REGION="us-west-004"
export B2_KEY_ID="0043616a62c8bb90000000001"
export B2_APP_KEY="K004cur7hMs3GDPixFB8FlzfCJV2PIc"

bash scripts/fetch_datasets_b2.sh || {
  echo "WARNING: B2 download failed, trying alternative..."
  mkdir -p "$PDEBENCH_ROOT"
  PYTHONPATH=src python scripts/fetch_datasets.py --task burgers1d || echo "Dataset fetch failed"
}

echo "=== Running SOTA Evaluations ==="
export PYTHONPATH=src
export CHECKPOINT_DIR=checkpoints/scale
export REPORTS_DIR=reports/sota_eval

bash scripts/run_sota_evaluation.sh || {
  echo "ERROR: Evaluation script failed"
  echo "Checking what went wrong..."
  ls -la checkpoints/scale/ || echo "No checkpoints directory"
  ls -la data/pdebench/ || echo "No data directory"
  exit 1
}

echo "=== Evaluation Complete! ==="
echo "Results:"
ls -lh reports/sota_eval/

echo "=== Uploading Results to W&B ==="
# Create a summary artifact
cd reports/sota_eval
tar czf ../sota_eval_results.tar.gz *.json *.csv *.html *.png 2>/dev/null || true
cd ../..

python -c "
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
for split in ['val', 'test']:
    for variant in ['baseline', 'ttc']:
        json_file = f'reports/sota_eval/{variant}_{split}.json'
        try:
            with open(json_file) as f:
                data = json.load(f)
                metrics = data.get('metrics', {})
                for k, v in metrics.items():
                    run.log({f'{variant}_{split}_{k}': v})
                print(f'Logged metrics from {json_file}')
        except Exception as e:
            print(f'Could not load {json_file}: {e}')

run.log_artifact(artifact)
run.finish()
print('✓ Results uploaded to W&B')
" || echo "WARNING: W&B upload failed"

echo ""
echo "=== Instance will shutdown in 60 seconds ==="
sleep 60
sync
command -v poweroff >/dev/null 2>&1 && poweroff || true
