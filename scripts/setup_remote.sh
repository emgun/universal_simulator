#!/usr/bin/env bash
set -euo pipefail

# Remote environment setup script for vast.ai or other cloud GPU instances
# Usage: bash scripts/setup_remote.sh

echo "=== Universal Physics Stack - Remote Setup ==="

# 1. System info
echo "System information:"
uname -a
python --version || python3 --version
nvidia-smi || echo "NVIDIA GPU not detected"

# 2. Set working directory
WORKDIR=${WORKDIR:-$PWD}
echo "Working directory: $WORKDIR"
cd "$WORKDIR"

# 3. Install Python dependencies
echo ""
echo "=== Installing Python dependencies ==="
pip install --upgrade pip
pip install -e ".[dev]"

# 4. Verify PyTorch CUDA
echo ""
echo "=== Verifying PyTorch CUDA ==="
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Device count: {torch.cuda.device_count()}') if torch.cuda.is_available() else print('No CUDA')"

# 5. Set deterministic environment
echo ""
echo "=== Setting deterministic environment ==="
bash scripts/prepare_env.sh

# 6. Verify wandb
echo ""
echo "=== Checking W&B ==="
python -c "import wandb; print(f'wandb version: {wandb.__version__}')"
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "⚠️  WANDB_API_KEY not set. Set it or run 'wandb login' before training."
else
    echo "✓ WANDB_API_KEY is set"
fi

# 7. Create necessary directories
echo ""
echo "=== Creating directories ==="
mkdir -p data/pdebench
mkdir -p checkpoints/scale
mkdir -p reports
mkdir -p artifacts/cache

# 8. Test dataset fetch (if WANDB_DATASETS is set)
if [ -n "${WANDB_DATASETS:-}" ]; then
    echo ""
    echo "=== Testing dataset fetch ==="
    echo "Datasets to fetch: $WANDB_DATASETS"
    # Will be done by run_remote_scale.sh
else
    echo "WANDB_DATASETS not set, skipping dataset fetch test"
fi

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "1. Set W&B credentials: export WANDB_API_KEY=<your-key>"
echo "2. Set datasets: export WANDB_DATASETS='burgers1d_subset_v1'"
echo "3. Run training: bash scripts/run_remote_scale.sh"
echo ""

