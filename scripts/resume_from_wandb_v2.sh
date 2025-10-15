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

echo "Downloading diffusion checkpoint from W&B run pp0c2k31..."
PYTHONPATH=src python << 'EOPY'
import wandb
import os

# Initialize W&B API
api = wandb.Api()

# Get the run
run = api.run("emgun-morpheus-space/universal-simulator/pp0c2k31")

# List all files in the run
print("Files in run:")
for file in run.files():
    print(f"  {file.name}")
    
# Download checkpoint files
os.makedirs("checkpoints/scale", exist_ok=True)

downloaded = False
for file in run.files():
    if file.name.endswith('.pt'):
        # Download the file
        print(f"Downloading {file.name}...")
        file.download(root=".", replace=True)
        
        # Determine destination based on filename
        basename = os.path.basename(file.name)
        if 'diffusion' in basename.lower() or 'diff' in basename.lower():
            dst = "checkpoints/scale/diffusion_residual.pt"
        elif 'operator' in basename.lower() or 'op' in basename.lower():
            dst = "checkpoints/scale/operator.pt"
        else:
            dst = f"checkpoints/scale/{basename}"
        
        # Move to final location
        os.rename(file.name, dst)
        print(f"  Saved to {dst}")
        
        downloaded = True

if not downloaded:
    print("ERROR: No checkpoint files found in run")
    exit(1)
    
print("✓ Checkpoints downloaded successfully")
print("Downloaded files:")
import subprocess
subprocess.run(["ls", "-lh", "checkpoints/scale/"])
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

