#!/usr/bin/env bash
set -eo pipefail

# Minimal end-to-end smoke test on CPU with synthetic data
# - Skips hydration; uses tiny synthetic tensors via small H5s

WORKDIR=${WORKDIR:-$PWD}
TMPDIR="$WORKDIR/.smoke_tmp"
mkdir -p "$TMPDIR/configs" "$TMPDIR/data/pdebench"

# Create tiny H5 files for burgers1d train/val/test
PYTHONPATH=src python - <<'PY'
import h5py, numpy as np, os
root = os.path.join('.smoke_tmp','data','pdebench')
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
cat > "$TMPDIR/configs/train_smoke.yaml" <<'YAML'
data:
  task: burgers1d
  split: train
  root: .smoke_tmp/data/pdebench
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

echo 'Running smoke train…'
PYTHONPATH=src python scripts/train.py --config "$TMPDIR/configs/train_smoke.yaml" --stage all

echo 'Smoke test complete.'

