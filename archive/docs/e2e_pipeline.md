## End-to-end Vast.ai training pipeline

This guide runs setup → dataset hydration (Backblaze B2 or W&B) → latent precompute → multi-stage training → TTC eval on a Vast.ai instance. Scripts are resilient to missing env and will fall back to offline behaviors when needed.

### Prereqs
- Vast.ai CLI (`vastai`), `jq`, `ssh`, `scp`, `nc`
- A `.env` with keys:
  - `WANDB_API_KEY`, `WANDB_PROJECT`, `WANDB_ENTITY`
  - `B2_KEY_ID`, `B2_APP_KEY`, `B2_BUCKET` (optional but preferred)
  - optional: `B2_PREFIX`, `B2_S3_ENDPOINT`, `B2_S3_REGION`

### One-command E2E

```
ENV_FILE=.env \
GPU_PREF=RTX_4090 \
FALLBACK_GPU=H200 \
IMAGE=pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime \
DISK_GB=200 \
DATASET_KEY=burgers1d_full_v1 \
TRAIN_CONFIG=configs/train_burgers_quality_v2.yaml \
EVAL_CONFIG=configs/eval_pdebench_scale_ttc.yaml \
EVAL_TEST_CONFIG=configs/eval_pdebench_scale_test_ttc.yaml \
PDEBENCH_ROOT_REMOTE=/workspace/universal_simulator/data/pdebench/burgers1d_full_v1 \
TORCHINDUCTOR_DISABLE=1 \
RESET_CACHE=0 \
PRECOMPUTE_LATENT=1 \
bash scripts/vast_e2e.sh
```

What it does:
1) Picks cheapest 1× GPU (pref then fallback), launches instance via `scripts/vast_launch.py`
2) Waits for SSH, uploads `.env` and helper scripts
3) Hydrates dataset via B2 (`scripts/remote_hydrate_b2_once.sh`), creates split symlinks
4) Precomputes latents if `PRECOMPUTE_LATENT=1`
5) Runs training + eval via `scripts/remote_launch_once.sh`
6) Tails logs continuously (`vastai logs --tail 200`)

### Scripts overview

- `scripts/vast_e2e.sh`: end-to-end orchestrator (launch → hydrate → train → eval → tail)
- `scripts/remote_hydrate_b2_once.sh`:
  - Parses `.env` safely to export `B2_*` vars
  - Runs `scripts/fetch_datasets_b2.sh` (with `--copy-links`), creates split symlinks
- `scripts/remote_launch_once.sh`:
  - Normalizes `.env`, sources it, W&B non-interactive login when API key present
  - Sets online mode and runs `scripts/run_remote_scale.sh` with provided env
- `scripts/run_remote_scale.sh` (hardened):
  - Non-interactive W&B login (`wandb login --relogin $WANDB_API_KEY`) and online fallback
  - Honors `PDEBENCH_ROOT`, `PRECOMPUTE_LATENT`, `RESET_CACHE`, sanitizes `TRAIN_CONFIG` if commas
  - Precompute latents (train/val/test), train all stages, run TTC eval and eval-test
  - Normalizes checkpoint paths for eval

### Tips
- To bypass Triton/Inductor JIT on fresh images: export `TORCHINDUCTOR_DISABLE=1`
- To continue a run without re-precomputing: set `PRECOMPUTE_LATENT=0` and `RESET_CACHE=0`
- To reduce memory pressure, pass smaller batch/latent overrides via `TRAIN_EXTRA_ARGS` env






