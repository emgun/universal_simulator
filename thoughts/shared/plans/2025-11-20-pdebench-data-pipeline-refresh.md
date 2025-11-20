# PDEBench Data Pipeline Refresh & B2 Cache Reuse

## Overview
Align PDEBench task coverage, conversion, and caching with the official dataset index while enabling robust B2 upload/download (full splits, correct casing), and making latent caches reusable in training (correct config keys, download path). Also add explicit split handling and name alignment for multi-task configs.

## Current State Analysis
- TASK_SPECS defines 13 tasks, but the 11-task config uses mismatched names; converter defaults cover only 6 tasks; only `burgers1d` data exists locally (`src/ups/data/pdebench.py:60-76`, `configs/train_pdebench_11task_ddp.yaml:19-33`, `scripts/convert_pdebench_multimodal.py:48-72`, `data/pdebench` contents).
- B2 flow: remote preprocess uploads to `B2TRAIN:PDEbench/...` with partial limits (`--limit 100 --samples 1000`), and launch scripts never download latent caches (only precompute locally). Cache dir key mismatch in baseline config (`training.latent_cache_dir` vs `cache_dir`) (`scripts/remote_preprocess_pdebench.sh:141-193`, `scripts/vast_launch.py:214-260`, `scripts/setup_vast_data.sh:52-60`, `configs/train_pdebench_2task_baseline.yaml:56-61`, `src/ups/data/latent_pairs.py:714-727`).
- Split handling: converters can synthesize val/test, but remote preprocess caps size; Lightning falls back to train for missing splits, native loader errors on missing files (`src/ups/data/lightning_datamodule.py:16-45`, `src/ups/data/pdebench.py:135-195`).
- Balanced sampling only in distributed mode; mixed-resolution batches drop coords/meta on mismatch (`src/ups/data/latent_pairs.py:810-869`, `1186-1229`).

## Desired End State
- TASK_SPECS + converter patterns cover the official CSV families we train on; multi-task configs use aligned names and ship with available data.
- Remote preprocess downloads full splits (no default truncation), uploads with consistent bucket casing, and supports optional B2 cache upload.
- Training launch path can download caches from B2 and use them when `training.latent_cache_dir` is set; configs consistently use that key.
- Split handling robust: val/test synthesized only when missing; docs/configs note expected splits; Lightning/native paths both work.
- Optional: preserve balanced sampling behavior awareness (distributed vs single GPU) and document mixed-resolution collate behavior.

### Key Discoveries
- TASK_SPECS vs config mismatch causes KeyError; cache key mismatch causes cache ignore.
- B2 path casing (`PDEbench` vs `pdebench`) may break reuse; remote preprocess truncates data.
- Caches are uploaded but never downloaded in training flow.

## What We're NOT Doing
- No architectural changes to models or training loop beyond cache/split wiring.
- No new datasets beyond PDEBench official index and existing mesh/particle customs.
- No curriculum introduction beyond documenting current absence.

## Implementation Approach
Incrementally fix naming/coverage → converter/download → caching → splits/documentation. Keep defaults backward compatible (avoid breaking existing single-task configs).

## Phase 1: Align Tasks & Config Names

### Overview
Normalize TASK_SPECS and configs to match official PDEBench families and converter patterns.

### Changes Required
1) TASK_SPECS extensions  
**File**: `src/ups/data/pdebench.py`  
**Changes**: Add entries for official CSV families used/planned (e.g., `reaction_diffusion1d`, `cfd1d_shocktube`, `cfd2d_rand`, `cfd2d_turb`, `cfd3d`). Keep kind="grid".

2) Converter patterns  
**File**: `scripts/convert_pdebench_multimodal.py`  
**Changes**: Add DEFAULT_TASKS entries for the added names with paths from CSV (e.g., `1D/ReactionDiffusion/{split}/*.hdf5`, `1D/CFD/Train/*.hdf5`, `2D/CFD/2D_Train_Rand/*.hdf5`, `2D/CFD/2D_Train_Turb/*.hdf5`, `3D/Train/*.hdf5`); include test patterns where applicable for split synthesis.

3) Configs update  
**Files**: `configs/train_pdebench_11task_ddp.yaml`, `configs/train_pdebench_2task_baseline.yaml`, `configs/train_2task_advection_darcy_ddp.yaml`  
**Changes**: Replace task names with TASK_SPECS keys; ensure `training.latent_cache_dir` is used consistently; keep tags/comments aligned.

### Success Criteria
#### Automated
- [ ] `python -c "from ups.data.pdebench import TASK_SPECS; print(list(TASK_SPECS.keys()))"` includes new tasks.
- [ ] `python scripts/validate_config.py configs/train_pdebench_11task_ddp.yaml` passes.

#### Manual
- [ ] Multi-task configs load without KeyError when pointing to available data.

## Phase 2: Full Download/Conversion (No Truncation, Correct Casing)

### Overview
Ensure remote preprocess pulls full datasets, uses consistent bucket casing, and covers new patterns.

### Changes Required
0) Explicit download matrix from official CSV  
**Files**: `scripts/remote_preprocess_pdebench.sh`, `scripts/setup_vast_data.sh`  
**Changes**: Add task→path map aligned to `pdebench_data_urls.csv` for all targeted families (Advection, Burgers, Diff_Sorp, 1D_ReacDiff, Darcy, NS_Incom, SWE, 2D_ReacDiff, 1D_CFD, 2D_CFD Rand/Turb, 3D_CFD). Use these paths for rclone download/check and converter patterns. Optional: md5 verification hook using CSV hashes.

1) Remote preprocess limits/casing  
**File**: `scripts/remote_preprocess_pdebench.sh`  
**Changes**: Remove default `--limit/--samples` caps (or gate behind env flags); normalize bucket path casing (`pdebench` consistently). Add new tasks to conversions loop (grid) and keep mesh/particles untouched.

2) Setup script casing  
**File**: `scripts/setup_vast_data.sh`  
**Changes**: Align casing with preprocess (`pdebench`), add patterns for new tasks (train/val/test) with fallback synthesis note. Add optional `--checksum` or env flag to run `rclone --checksum`/md5 comparisons.

3) Docs note  
**File**: `UPT_docs/PDEBench_task_coverage.md`  
**Changes**: Add note on default full downloads and casing choice.

4) Preprocess best-effort flags  
**File**: `scripts/remote_preprocess_pdebench.sh`  
**Changes**: Add `SKIP_UPLOAD=1` and `DOWNLOAD_ONLY=1` guards for iterative debugging; log when limits are overridden.

### Success Criteria
#### Automated
- [ ] Dry-run preprocess (with `--tasks burgers1d advection1d`) shows no limit flags in generated command.
- [ ] `rclone ls B2TRAIN:pdebench/full/advection1d/` works with chosen casing.
- [ ] Optional: `rclone md5sum B2TRAIN:pdebench/full/advection1d/ | head` returns hashes.

#### Manual
- [ ] Full-size HDF5s present in B2 for sampled tasks (no 100-file cap).
- [ ] Converter runs end-to-end on at least one new task (e.g., 1D_ReacDiff) and uploads to B2.

## Phase 3: Cache Download & Config Consistency

### Overview
Allow training to download and reuse B2 latent caches; fix config key usage.

### Changes Required
1) Training download path  
**File**: `scripts/setup_vast_data.sh` or new helper (preferred)  
**Changes**: Add optional `--cache-version` handling to download `pdebench/latent_caches/<version>/<task_split>/` into `training.latent_cache_dir`. Guard with env vars (B2 creds) and flag to skip.

2) Launch script wiring  
**File**: `scripts/vast_launch.py`  
**Changes**: Add `--cache-version` flag to launch; call cache downloader before training when provided. Keep precompute optional.

3) Config key alignment  
**Files**: same configs as Phase 1  
**Changes**: Ensure `training.latent_cache_dir` used; remove deprecated `cache_dir`.

### Success Criteria
#### Automated
- [ ] `python -c "import json, pathlib; print('ok')"` (placeholder) replaced with: invoke cache download helper in a dry run and confirm dirs created for a sample version.
- [ ] `python scripts/validate_config.py` passes after key changes.

#### Manual
- [ ] On a remote run with `--cache-version`, training starts using existing caches without recomputation (short startup).

## Phase 4: Splits & Loader Robustness

### Overview
Clarify split behavior and avoid silent fallbacks; optionally enforce warnings when val/test missing.

### Changes Required
1) Converter split synthesis guard  
**File**: `scripts/convert_pdebench_multimodal.py`  
**Changes**: Log when synthetic val/test created; allow disabling synthesis via flag/env; ensure new tasks follow same logic.

2) Loader warnings  
**File**: `src/ups/data/pdebench.py`  
**Changes**: When split files missing, raise clear error suggesting synth or correct root; optional flag to allow synth.

3) Lightning datamodule note  
**File**: `src/ups/data/lightning_datamodule.py`  
**Changes**: Log when falling back to train split.

4) Docs update  
**File**: `UPT_docs/PDEBench_task_coverage.md`  
**Changes**: Add split expectations and synthesis behavior.

### Success Criteria
#### Automated
- [ ] Unit test for split synthesis logging (optional small test in `tests/unit`).

#### Manual
- [ ] Running converter without val/test prints synthesis message; missing split error is actionable.

## Testing Strategy
- Unit: TASK_SPECS coverage; config validation on updated YAMLs; optional split synthesis logging test.
- Integration: Dry-run preprocess; dry-run cache download; loader instantiation for multi-task configs using synthetic small data (could be mocked).

## Performance Considerations
- Cache download should use parallel transfers with moderate `--transfers`; avoid recompute when caches exist.
- Removing preprocess limits increases runtime/storage; provide optional env to keep caps for quick tests.

## Migration Notes
- Keep default backward compatibility: existing single-task configs still work; new task names require data availability.
- Bucket casing change may require one-time reupload or dual-path compatibility (temporary).

## References
- `UPT_docs/PDEBench_task_coverage.md`
- `thoughts/shared/plans/2025-11-07-pdebench-multi-dataset-scaling.md`
- `thoughts/shared/plans/2025-11-06-pdebench-data-pipeline-expansion.md`
- Official dataset index: `pdebench/data_download/pdebench_data_urls.csv`
