# PDEBench Task Coverage & Data Pipeline Notes

## Task Definitions vs Configs
- Defined tasks: `src/ups/data/pdebench.py:60-76` (13 tasks, includes `darcy2d_mesh`, `particles_advect`).
- 11-task DDP config uses different names: `configs/train_pdebench_11task_ddp.yaml:19-33` (e.g., `diff_react1d`, `swe2d`, `ns2d_cond`, `comp_ns2d`, `react_diff2d`, `diff_sorp2d`, `shallow_water2d_varied`, `ns2d_turb`), which are not in `TASK_SPECS` and will KeyError until aligned or added.
- Converter defaults only cover 6 tasks: `scripts/convert_pdebench_multimodal.py:48-72`; remaining plan tasks lack patterns.

## Cache Wiring
- Loader consumes caches from `training.latent_cache_dir`: `src/ups/data/latent_pairs.py:714-727`.
- 2-task baseline uses `cache_dir`, so caches are ignored: `configs/train_pdebench_2task_baseline.yaml:56-61`.
- DDP 2-task uses correct key: `configs/train_2task_advection_darcy_ddp.yaml:92`.
- Launch path precomputes caches locally but never downloads B2 latent caches; remote runs recompute unless caches are pre-seeded: `scripts/vast_launch.py:214-260`, `scripts/setup_vast_data.sh`.

## B2 Path Casing
- Remote preprocess/download now use `B2TRAIN:pdebench/...` with consistent casing: `scripts/remote_preprocess_pdebench.sh`, `scripts/setup_vast_data.sh`.
- Plan references `pdebench`; case sensitivity previously risked reuse if bucket distinguished case.

## Preprocess Truncation
- Remote preprocessing no longer enforces truncation by default; optional `LIMIT`/`SAMPLES` env can cap for smoke runs.

## Task Balancing Scope
- Balanced sampling only active in distributed mode via `MultiTaskDistributedSampler`: `src/ups/data/latent_pairs.py:810-869`, `src/ups/data/task_samplers.py:1-104`.
- Single-GPU multi-task runs use `ConcatDataset` shuffle; `task_sampling.strategy: balanced` has no effect there.

## Mixed-Resolution Collation
- Collate keeps only first sample’s `meta` and drops `coords` when shapes differ: `src/ups/data/latent_pairs.py:1186-1229`. Mixed-resolution batches lose coords/meta for inverse/physics/query logic.

## Curriculum Gaps
- No task-level curriculum in loader or `scripts/train.py`; only rollout-length curriculum exists in `src/ups/training/loop_train.py:10-62` and is unused by the main entrypoint.

## Plan-Specific Gaps (2025-11-07 PDEBench Scaling Plan)
- Plan target: 11+ tasks with staged scaling (2-task → 4-task → 5-task → mixed modality) and proper TASK_SPECS. Current `TASK_SPECS` defines 13 tasks, but the 11-task config uses mismatched names, and only `burgers1d` data is present locally; no evidence of converted/available datasets for advection/darcy/reaction_diffusion/navier_stokes/mixed-modality Zarr (`thoughts/shared/plans/2025-11-07-pdebench-multi-dataset-scaling.md` expectations vs `data/pdebench` contents).
- Plan target: per-task metrics in WandB. Code supports it (`scripts/train.py:1120-1433` and Lightning module `src/ups/training/lightning_modules.py:330-365`), so this is aligned.
- Plan target: download and reuse precomputed latent caches from B2 to cut startup. Current flow precomputes locally and never downloads B2 caches (`scripts/vast_launch.py:214-260`, `scripts/setup_vast_data.sh`), so the “one-time cache investment” from the plan is not realized.
- Plan target: staged task curriculum (e.g., add tasks over epochs). There is no task curriculum wiring in loader or `scripts/train.py`; only rollout curriculum stub is unused.
- Plan target: full remote preprocessing of all tasks. Remote preprocess currently truncates to 100 files/1000 samples per split and uses `PDEbench` casing; mixed-modality (mesh/particles) patterns exist but are not included in defaults, so Phase 4 data prep would need manual pattern additions.
- Plan target: latent cache presets for multiple model sizes. Precompute now supports presets (128/128, 192/256, 256/512, 384/768) and uploads to versioned B2 paths when enabled.

## Official PDEBench Repo & Paper Cross-Reference
- Official dataset index (`pdebench_data_urls.csv` in repo) lists 375 entries; top-level dirs: `1D` (69), `2D` (298), `3D` (8).
- Key families and paths from the CSV:
  - `Advection`: `1D/Advection/Train/` (8 HDF5 beta sweeps)
  - `Burgers`: `1D/Burgers/Train/` (12 HDF5 nu sweeps)
  - `Diff_Sorp`: `1D/diffusion-sorption/` (1 entry)
  - `1D_ReacDiff`: `1D/ReactionDiffusion/Train|Test/` (36 entries)
  - `Darcy`: `2D/DarcyFlow/` (5 entries)
  - `NS_Incom`: `2D/NS_incom/` (274 entries across train/test)
  - `SWE`: `2D/shallow-water/` (1 entry)
  - `2D_ReacDiff`: `2D/diffusion-reaction/` (1 entry)
  - `2D_CFD`: `2D/CFD/2D_Train_Rand|2D_Train_Turb/` plus tests `2D/CFD/Test/2DShock|KH|TOV/` (17 entries)
  - `1D_CFD`: `1D/CFD/Train|Test/` (ShockTube; 12 entries)
  - `3D_CFD`: `3D/Train/` and `3D/Test/BlastWave|Turbulence/` (8 entries)
- Mapping to current `TASK_SPECS` (`src/ups/data/pdebench.py:60-76`):
  - Direct matches: `advection1d`→`1D/Advection/Train`, `burgers1d`→`1D/Burgers/Train`, `diffusion_sorption1d`→`1D/diffusion-sorption/`, `darcy2d`→`2D/DarcyFlow/`, `reaction_diffusion2d`→`2D/diffusion-reaction/`, `navier_stokes2d`→`2D/NS_incom/`, `shallow_water2d`→`2D/shallow-water/`, `compressible_ns3d` could draw from `3D/Train/` and tests.
  - Present in CSV but not in `TASK_SPECS`: `1D_ReacDiff` (1D Reaction-Diffusion), `1D_CFD`, `2D_CFD`, `3D_CFD`, and the specific shallow-water/NS variations named in the 11-task config.
  - Present in `TASK_SPECS` but not in CSV: `allen_cahn2d`, `cahn_hilliard2d` (would need alternate sources).
- Config name mismatches: `configs/train_pdebench_11task_ddp.yaml:19-33` uses names like `diff_react1d`, `swe2d`, `ns2d_cond`, `comp_ns2d`, `react_diff2d`, `diff_sorp2d`, `shallow_water2d_varied`, `ns2d_turb`, none of which align with either the CSV or current `TASK_SPECS`; they need new TASK_SPECS entries and converter patterns or renaming to existing keys.
- Split expectations: CSV provides train/test for many sets (val sometimes implicit). Our converter can synthesize val/test if missing, but remote preprocess caps to 100 files/1000 samples per split, so full official splits are not realized unless limits are removed.
