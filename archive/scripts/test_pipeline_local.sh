#!/usr/bin/env bash
set -eo pipefail

# Local end-to-end smoke of the remote pipeline via bash
# - Creates synthetic burgers1d_full_v1 H5s
# - Points PDEBENCH_ROOT to that directory
# - Uses the tiny smoke training config
# - Runs scripts/run_remote_scale.sh with WANDB offline

WORKDIR=${WORKDIR:-$PWD}
TMP=.smoke_tmp
ROOT="$WORKDIR/$TMP"
DATA_DIR="$ROOT/data/pdebench/burgers1d_full_v1"
CFG_DIR="$ROOT/configs"
mkdir -p "$DATA_DIR" "$CFG_DIR"

# Create tiny H5 files for burgers1d train/val/test
PYTHONPATH=src python - <<'PY'
import h5py, numpy as np, os
root = os.path.join('.smoke_tmp','data','pdebench','burgers1d_full_v1')
os.makedirs(root, exist_ok=True)
def write(split):
    path = os.path.join(root, f'burgers1d_{split}.h5')
    with h5py.File(path, 'w') as f:
        f['data'] = np.random.randn(8, 16, 1).astype('float32')
for s in ('train','val','test'):
    write(s)
print('Wrote synthetic dataset under', root)
PY

# Tiny training config
cat > "$CFG_DIR/train_smoke.yaml" <<'YAML'
data:
  task: burgers1d
  split: train
  root: .smoke_tmp/data/pdebench/burgers1d_full_v1
  patch_size: 1
latent:
  dim: 16
  tokens: 8
training:
  batch_size: 4
  time_stride: 2
  num_workers: 0
  compile: false
stages:
  operator:
    epochs: 1
  diff_residual:
    epochs: 1
  consistency_distill:
    epochs: 0
  steady_prior:
    epochs: 0
logging:
  wandb:
    enabled: false
YAML

echo 'Running pipeline via run_remote_scale.sh (local synthetic dataset)…'
export WANDB_MODE=offline
export PDEBENCH_ROOT="$DATA_DIR"
export TRAIN_CONFIG="$CFG_DIR/train_smoke.yaml"
export PRECOMPUTE_LATENT=1
bash scripts/run_remote_scale.sh

echo 'OK'





