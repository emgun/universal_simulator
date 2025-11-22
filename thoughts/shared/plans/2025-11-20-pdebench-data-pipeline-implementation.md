# PDEBench Data Pipeline & B2 Cache Implementation Plan

## Overview
Unify PDEBench task coverage with the official dataset index, enable full-fidelity download/convert/upload to B2 with consistent casing, and make latent caches reusable in training via standard config keys and cache download support. Incorporates multi-task scaling requirements (per-task metrics, aligned task names) and robust split handling.

## Current State Analysis
- `TASK_SPECS` defines 13 tasks; multi-task configs use mismatched names, and converter defaults cover only 6 tasks (`src/ups/data/pdebench.py:60-76`, `scripts/convert_pdebench_multimodal.py:48-72`, `configs/train_pdebench_11task_ddp.yaml:19-33`).
- Remote preprocess uploads truncated datasets (`--limit 100 --samples 1000`) to `B2TRAIN:PDEbench/...` and never downloads latent caches for training; baseline config uses `cache_dir` instead of `latent_cache_dir` (`scripts/remote_preprocess_pdebench.sh:141-193`, `scripts/vast_launch.py:214-260`, `scripts/setup_vast_data.sh:52-60`, `configs/train_pdebench_2task_baseline.yaml:56-61`, `src/ups/data/latent_pairs.py:714-727`).
- Split handling: converters can synthesize val/test, but missing split handling is noisy; Lightning falls back silently to train (`src/ups/data/pdebench.py:135-195`, `src/ups/data/lightning_datamodule.py:16-45`).
- Official `pdebench_data_urls.csv` lists families beyond current patterns (Advection, Burgers, Diff_Sorp, 1D_ReacDiff, Darcy, NS_Incom, SWE, 2D_ReacDiff, 1D/2D/3D CFD).

## Desired End State
- TASK_SPECS + converter patterns cover the targeted official families; configs use aligned task names and `training.latent_cache_dir`.
- Remote preprocess downloads full splits, uses consistent bucket casing (`pdebench`), supports checksum, and optional download-only/skip-upload modes.
- Training path can download latent caches from B2 (`--cache-version`) and skip recomputation; configs consistently use `latent_cache_dir`.
- Split handling is explicit: synthesis is logged/optional; missing splits produce actionable errors; Lightning logs fallback.

### Key Discoveries
- Name mismatch → KeyErrors; cache key mismatch → caches ignored.
- B2 casing mismatch and truncation undermine reuse; caches uploaded but never downloaded.
- Official CSV provides authoritative paths for all families.

## What We're NOT Doing
- No model/training-loop changes beyond cache/split wiring.
- No new datasets beyond official PDEBench and existing mesh/particle customs.
- No curriculum implementation (document only).

## Implementation Approach
Four phases: align tasks/configs → full download/convert/upload pipeline → cache download wiring → splits/logging/documentation.

## Phase 1: Align Tasks & Config Names

### Overview
Match TASK_SPECS, converter defaults, and configs to official PDEBench families.

### Changes Required
1. Extend TASK_SPECS  
**File**: `src/ups/data/pdebench.py`  
**Changes**: Add official families (e.g., `reaction_diffusion1d`, `cfd1d_shocktube`, `cfd2d_rand`, `cfd2d_turb`, `cfd3d`) with kind="grid".

2. Converter patterns  
**File**: `scripts/convert_pdebench_multimodal.py`  
**Changes**: Add DEFAULT_TASKS entries with CSV paths: `1D/ReactionDiffusion/{split}/*.hdf5`, `1D/CFD/Train/*.hdf5`, `2D/CFD/2D_Train_Rand/*.hdf5`, `2D/CFD/2D_Train_Turb/*.hdf5`, `3D/Train/*.hdf5`; include test patterns for synthesis.

3. Config alignment  
**Files**: `configs/train_pdebench_11task_ddp.yaml`, `configs/train_pdebench_2task_baseline.yaml`, `configs/train_2task_advection_darcy_ddp.yaml`  
**Changes**: Replace task names with TASK_SPECS keys; standardize on `training.latent_cache_dir`; update comments/tags accordingly.

### Success Criteria
#### Automated
- [x] `python -c "from ups.data.pdebench import TASK_SPECS; print(list(TASK_SPECS.keys()))"` shows new tasks.
- [x] `python scripts/validate_config.py configs/train_pdebench_11task_ddp.yaml` passes.
#### Manual
- [ ] Multi-task configs instantiate loaders without KeyError when data present.

## Phase 2: Full Download/Conversion & B2 Upload

### Overview
Use the official CSV map to download full datasets, convert, and upload to B2 with consistent casing and integrity options.

### Changes Required
1. Task→path matrix + checksums  
**Files**: `scripts/remote_preprocess_pdebench.sh`, `scripts/setup_vast_data.sh`  
**Changes**: Embed the task→path map from `pdebench_data_urls.csv` (Advection, Burgers, Diff_Sorp, 1D_ReacDiff, Darcy, NS_Incom, SWE, 2D_ReacDiff, 1D/2D/3D CFD); optional md5 check against CSV hashes.

2. Remove truncation & normalize casing  
**File**: `scripts/remote_preprocess_pdebench.sh`  
**Changes**: Drop default `--limit/--samples`; use `pdebench` casing; add new tasks to conversion loop; keep mesh/particle handling intact.

3. Setup download script  
**File**: `scripts/setup_vast_data.sh`  
**Changes**: Match casing; add patterns for new tasks (train/val/test); optional `--checksum`/env to enable `rclone --checksum` or md5 compare.

4. Best-effort flags  
**File**: `scripts/remote_preprocess_pdebench.sh`  
**Changes**: Add `DOWNLOAD_ONLY`/`SKIP_UPLOAD` toggles; log when limits are overridden.

5. Docs  
**File**: `UPT_docs/PDEBench_task_coverage.md`  
**Changes**: Note full downloads, casing choice, checksum option.

6. Latent cache precompute variants  
**File**: `scripts/remote_preprocess_pdebench.sh` (post-conversion block)  
**Changes**: Add optional cache precompute/upload for multiple sizes inspired by UPT configs:  
  - Base: `latent_dim=128`, `latent_len=128` → version `upt_128d_128tok`  
  - Medium-192: `latent_dim≈192`, `tokens≈256` → version `upt_192d_256tok`  
  - Medium (UPT medium: `latent_dim≈256`, `tokens≈512`): version `upt_256d_512tok`  
  - Large (UPT large: `latent_dim≈384`, `tokens≈768`): version `upt_384d_768tok`  
Gate by env/CLI (e.g., `CACHE_PRESETS=base,medium192,medium,large`). After precompute via `scripts/precompute_latent_cache.py --tasks ... --splits train val test`, upload to `pdebench/latent_caches/<version>/<task_split>/`.

### Success Criteria
#### Automated
- [ ] Dry-run preprocess (e.g., `--tasks burgers1d advection1d`) shows no limit flags.
- [ ] `rclone ls B2TRAIN:pdebench/full/advection1d/` works.
- [ ] Optional: `rclone md5sum B2TRAIN:pdebench/full/advection1d/ | head` returns hashes.
- [ ] Cache precompute dry-run builds expected versioned dirs for selected presets.
#### Manual
- [ ] Full-size HDF5s in B2 (no 100-file cap) for sampled tasks.
- [ ] Converter runs end-to-end on a new task (e.g., 1D_ReacDiff) and uploads to B2.
- [ ] Latent caches for selected presets uploaded to B2 under `pdebench/latent_caches/<version>/...`.

## Phase 3: Cache Download & Config Consistency

### Overview
Enable reuse of latent caches stored in B2; ensure configs use the correct key.

### Changes Required
1. Cache download helper  
**File**: `scripts/setup_vast_data.sh` (or new helper)  
**Changes**: Add `--cache-version` to download `pdebench/latent_caches/<version>/<task_split>/` into `training.latent_cache_dir`; guard with B2 creds; allow skip flag.

2. Launch wiring  
**File**: `scripts/vast_launch.py`  
**Changes**: Add `--cache-version` passthrough; invoke cache downloader before training when provided; keep precompute optional.

3. Config key cleanup  
**Files**: same configs as Phase 1  
**Changes**: Ensure `training.latent_cache_dir` is used; remove `cache_dir`.

### Success Criteria
#### Automated
- [ ] Cache download dry run creates task/split dirs for a sample version.
- [ ] `python scripts/validate_config.py` passes after key changes.
#### Manual
- [ ] Remote run with `--cache-version` starts using existing caches (startup noticeably shorter, no recompute).

## Phase 4: Splits & Loader Robustness

### Overview
Make split handling explicit and user-friendly.

### Changes Required
1. Converter split synthesis logging  
**File**: `scripts/convert_pdebench_multimodal.py`  
**Changes**: Log when synthetic val/test is created; add flag/env to disable synthesis.

2. Loader warnings/errors  
**File**: `src/ups/data/pdebench.py`  
**Changes**: When split files missing, raise clear error with remediation (synth enable or fix root). Optional flag to allow synth.

3. Lightning fallback logging  
**File**: `src/ups/data/lightning_datamodule.py`  
**Changes**: Log when falling back to train split.

4. Docs update  
**File**: `UPT_docs/PDEBench_task_coverage.md`  
**Changes**: Document split expectations and synthesis behavior.

### Success Criteria
#### Automated
- [ ] Unit test for split synthesis logging/warning (optional).
#### Manual
- [ ] Converter shows synthesis messages when val/test missing; loader errors are actionable.

## Testing Strategy
- Unit: TASK_SPECS coverage; config validation; optional split synthesis logging test.
- Integration: Dry-run preprocess/download; cache download dry-run; loader instantiation for updated configs with mock/small data.

## Performance Considerations
- Use parallel transfers for cache/data download with moderate `--transfers`; avoid recompute when caches exist.
- Removing preprocess limits increases runtime/storage—keep env flags to re-enable caps for quick smoke runs.

## Migration Notes
- Backward compatible: single-task configs continue to work; new task names require data availability.
- Bucket casing change may need one-time reupload or temporary dual-path handling.

## References
- `thoughts/shared/plans/2025-11-07-pdebench-multi-dataset-scaling.md`
- `thoughts/shared/plans/2025-11-20-pdebench-data-pipeline-refresh.md`
- `UPT_docs/PDEBench_task_coverage.md`
- Official dataset index: `pdebench/data_download/pdebench_data_urls.csv`
