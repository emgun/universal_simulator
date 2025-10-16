## Universal Simulator – Full Training Run Guide

This guide documents how to run the full training pipeline (Operator → Diffusion Residual → Consistency Distillation → Steady Prior) on PDEBench Burgers1D using Vast.ai, Backblaze B2, and Weights & Biases.

### Repository layout references
- Training entrypoint: `scripts/train.py`
- Remote launcher: `scripts/run_remote_scale.sh`
- Vast launcher: `scripts/vast_launch.py`
- Dataset loader: `src/ups/data/pdebench.py`
- Monitoring utils: `src/ups/utils/monitoring.py`
- Main train config: `configs/train_burgers_quality_v2.yaml`
- TTC eval configs: `configs/eval_pdebench_scale_ttc.yaml`, `configs/eval_pdebench_scale_test_ttc.yaml`

### 0) One-time local setup
```bash
cd /Users/emerygunselman/Code/universal_simulator
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Sanity check (CPU, synthetic data):
```bash
bash scripts/smoke_test.sh
```
This should complete Operator and Diffusion stages and write checkpoints under `checkpoints/`.

### 1) Environment variables (.env)
Create a `.env` in the repo root with:
```bash
# Weights & Biases
WANDB_API_KEY=...                     # required for online logging
WANDB_ENTITY=emgun-morpheus-space
WANDB_PROJECT=universal-simulator

# Backblaze B2 for dataset hydration (optional if datasets pre-exist)
B2_KEY_ID=...
B2_APP_KEY=...
B2_BUCKET=pdebench

# Datasets list for hydration (comma or space separated)
WANDB_DATASETS=burgers1d_full_v1

# Preferred explicit dataset root on remote
PDEBENCH_ROOT=/workspace/universal_simulator/data/pdebench/burgers1d_full_v1
```
Notes:
- If `WANDB_API_KEY` is missing, the pipeline auto-falls back to `WANDB_MODE=offline`.
- `PDEBENCH_ROOT` overrides any `data.root` in configs (no symlinks required).
- If `WANDB_DATASETS` is unset, hydration is skipped and existing files under `PDEBENCH_ROOT` are used.

### 2) What the remote launcher does
`scripts/run_remote_scale.sh` orchestrates:
- Optional dataset hydration via B2 (if B2 envs set) or W&B artifacts (if `WANDB_DATASETS` set).
- Exports `PDEBENCH_ROOT`, W&B envs (non-interactive login if key present), and runs:
  - Optional latent cache precompute (default enabled).
  - Full multi-stage training via `scripts/train.py --stage all` and your config.
  - TTC-enabled evaluations (when `EVAL_CONFIG`/`EVAL_TEST_CONFIG` provided).

Defaults (override via env):
- `TRAIN_CONFIG=configs/train_burgers_quality_v2.yaml`
- `EVAL_CONFIG=configs/eval_pdebench_scale_ttc.yaml`
- `EVAL_TEST_CONFIG=configs/eval_pdebench_scale_test_ttc.yaml`
- `PRECOMPUTE_LATENT=1`

Example manual invocation on a remote:
```bash
cd /workspace/universal_simulator
set -a; [ -f .env ] && . ./.env; set +a
env \
  TRAIN_CONFIG=configs/train_burgers_quality_v2.yaml \
  EVAL_CONFIG=configs/eval_pdebench_scale_ttc.yaml \
  EVAL_TEST_CONFIG=configs/eval_pdebench_scale_test_ttc.yaml \
  PDEBENCH_ROOT=/workspace/universal_simulator/data/pdebench/burgers1d_full_v1 \
  PRECOMPUTE_LATENT=1 \
  bash scripts/run_remote_scale.sh
```

### 3) Launch on Vast.ai (recommended)
Use the launcher which constructs a robust onstart script and passes the correct env:
```bash
python scripts/vast_launch.py \
  --offer-id 21738997 \
  --image pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime \
  --disk 200 \
  --train-config configs/train_burgers_quality_v2.yaml \
  --eval-config configs/eval_pdebench_scale_ttc.yaml \
  --eval-test-config configs/eval_pdebench_scale_test_ttc.yaml
```
This will: clone repo, install deps, source `.env`, hydrate datasets (if requested), and run `scripts/run_remote_scale.sh`.

### 4) Monitor progress
- Vast logs:
```bash
vastai logs <INSTANCE_ID> --tail 200
```
- On-instance log (if launched manually):
```bash
tail -n 200 -f run_full.log
```
- Weights & Biases: project `emgun-morpheus-space/universal-simulator` (auto-run names from config `logging.wandb`).

You should see messages for hydration, latent cache precompute, stage banners (Operator, Diffusion, etc.), and checkpoints being saved.

### 5) Outputs
- Checkpoints (under `checkpoints/`):
  - `operator.pt`, `operator_ema.pt`
  - `diffusion_residual.pt`, `diffusion_residual_ema.pt`
  - `steady_prior.pt` (if enabled)
- Reports/eval artifacts (under `reports/`):
  - `pdebench_eval*.{json,csv,html,png}`

Run TTC evals manually (if needed):
```bash
PYTHONPATH=src python scripts/evaluate.py \
  --config configs/eval_pdebench_scale_ttc.yaml \
  --operator checkpoints/operator.pt \
  --diffusion checkpoints/diffusion_residual.pt \
  --output-prefix reports/pdebench_scale_eval \
  --print-json
```

### 6) Resuming & re-running
- To resume quickly, keep `PRECOMPUTE_LATENT=0` to reuse existing latent cache.
- Skip a stage by setting its `epochs: 0` in the train config (e.g., `stages.consistency_distill.epochs: 0`).

### 7) Troubleshooting
- W&B key missing → script runs with `WANDB_MODE=offline`. Add `WANDB_API_KEY` to `.env` to enable online logging.
- Dataset not found (e.g., `data/pdebench/burgers1d_train.h5`) → set `PDEBENCH_ROOT` to the real dataset dir (e.g., `/workspace/universal_simulator/data/pdebench/burgers1d_full_v1`). Loader now prioritizes `PDEBENCH_ROOT`.
- Hydration via B2 requires `B2_KEY_ID`, `B2_APP_KEY`, `B2_BUCKET` and `WANDB_DATASETS=burgers1d_full_v1`.
- CUDA library issues → `scripts/run_remote_scale.sh` attempts to fix `libcuda.so` symlink automatically.

### 8) Exact minimal commands (copy/paste)
Local sanity check:
```bash
bash scripts/smoke_test.sh
```

Vast.ai full run with onstart:
```bash
python scripts/vast_launch.py --offer-id 21738997 --image pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime --disk 200 \
  --train-config configs/train_burgers_quality_v2.yaml \
  --eval-config configs/eval_pdebench_scale_ttc.yaml \
  --eval-test-config configs/eval_pdebench_scale_test_ttc.yaml
```

Manual on-instance run:
```bash
cd /workspace/universal_simulator
set -a; . ./.env; set +a
env TRAIN_CONFIG=configs/train_burgers_quality_v2.yaml \
    EVAL_CONFIG=configs/eval_pdebench_scale_ttc.yaml \
    EVAL_TEST_CONFIG=configs/eval_pdebench_scale_test_ttc.yaml \
    PDEBENCH_ROOT=/workspace/universal_simulator/data/pdebench/burgers1d_full_v1 \
    PRECOMPUTE_LATENT=1 \
    bash scripts/run_remote_scale.sh
```

---
This guide reflects the current scripts/configs, including robust `.env` handling, `PDEBENCH_ROOT` precedence, and non-interactive W&B behavior.






