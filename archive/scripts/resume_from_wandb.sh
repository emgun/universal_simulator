#!/usr/bin/env bash
set -eo pipefail

# Resume training from W&B checkpoint
cd /workspace/universal_simulator || exit 1

# Source .env
sed -i 's/\r$//' .env || true
set -a
[ -f .env ] && . ./.env
set +a

# W&B login
if [ -n "${WANDB_API_KEY:-}" ] && command -v wandb >/dev/null 2>&1; then
  printf "%s\n" "$WANDB_API_KEY" | wandb login --relogin 2>/dev/null || true
  export WANDB_MODE=online
  wandb online || true
else
  export WANDB_MODE=${WANDB_MODE:-offline}
fi

echo "Downloading diffusion checkpoint from W&B..."
PYTHONPATH=src python << 'EOPY'
import wandb
import os

# Initialize W&B
run = wandb.init(project="universal-simulator", entity="emgun-morpheus-space", job_type="download")

# Download the diffusion checkpoint artifact
artifact = run.use_artifact('emgun-morpheus-space/universal-simulator/pp0c2k31:latest', type='model')
artifact_dir = artifact.download()

# Copy checkpoints to expected location
import shutil
os.makedirs("checkpoints/scale", exist_ok=True)

# Find and copy checkpoint files
for file in os.listdir(artifact_dir):
    if file.endswith('.pt'):
        src = os.path.join(artifact_dir, file)
        # Determine destination based on filename
        if 'diffusion' in file.lower() or 'diff' in file.lower():
            dst = "checkpoints/scale/diffusion_residual.pt"
        elif 'operator' in file.lower() or 'op' in file.lower():
            dst = "checkpoints/scale/operator.pt"
        else:
            dst = f"checkpoints/scale/{file}"
        
        shutil.copy(src, dst)
        print(f"Copied {file} -> {dst}")

wandb.finish()
print("✓ Checkpoints downloaded successfully")
EOPY

# Start training from consistency distillation stage
export PDEBENCH_ROOT=/workspace/universal_simulator/data/pdebench/burgers1d_full_v1
export PRECOMPUTE_LATENT=0
export TORCHINDUCTOR_DISABLE=1

echo "Starting consistency distillation training..."
nohup python scripts/train.py --config configs/train_burgers_quality_v2_resume.yaml --stage consistency_distill > run_resume.log 2>&1 &

echo "Training launched. Logs: run_resume.log"
sleep 10
tail -n 100 run_resume.log || true





