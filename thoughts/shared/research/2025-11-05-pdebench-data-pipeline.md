---
date: 2025-11-06T01:00:50Z
researcher: Codex (assistant)
git_commit: 3324bfec21c1a268d026716134033792ee1ecb0b
branch: feature--UPT
repository: universal_simulator
topic: "PDEBench Data Pipeline Integration and Scaling"
tags: [research, codebase, data-pipeline, pdebench, scaling]
status: complete
last_updated: 2025-11-06
last_updated_by: Codex (assistant)
last_updated_note: "Added follow-up research on mesh/particle dataset support and conversion coverage"
---

# Research: PDEBench Data Pipeline Integration and Scaling

**Date**: 2025-11-06T01:00:50Z  
**Researcher**: Codex (assistant)  
**Git Commit**: 3324bfec21c1a268d026716134033792ee1ecb0b  
**Branch**: feature--UPT  
**Repository**: universal_simulator

## Research Question

What data pipeline infrastructure exists today for ingesting and scaling PDEBench datasets inside UPS, and how is it structured across loading, latent encoding, caching, and downstream training/evaluation hooks?

## Summary

- PDEBench ingestion relies on `PDEBenchDataset`, which normalises HDF5 shards, supports environment overrides, and limits tasks to a fixed registry in `TASK_SPECS` (`src/ups/data/pdebench.py:23`).  
- Latent pair generation is handled by `GridLatentPairDataset`, which wraps the base datasets, performs encoder calls, materialises optional caches, and prepares conditioning tensors (`src/ups/data/latent_pairs.py:260`).  
- `_build_pdebench_dataset` and `build_latent_pair_loader` orchestrate encoder construction, multi-task mixing, latent caching, and worker configuration, including a RAM-preloaded fast path (`src/ups/data/latent_pairs.py:569`, `src/ups/data/latent_pairs.py:588`).  
- `parallel_cache` adds parallel encoding, cache preloading, and resource checks to accelerate scaling, with helper utilities consumed by both training and cache precomputation flows (`src/ups/data/parallel_cache.py:33`, `src/ups/data/parallel_cache.py:258`).  
- `scripts/precompute_latent_cache.py` provides a CLI workflow to populate caches ahead of time, leveraging the same dataset builders and parallel loader strategies (`scripts/precompute_latent_cache.py:40`).  
- Core training and evaluation entry points import the shared loader, ensuring PDEBench datasets feed both operator training and evaluation without diverging code paths (`scripts/train.py:216`, `src/ups/eval/pdebench_runner.py:27`).  
- Documentation and tests cover conversion, caching, and loader behaviour; current docs reference a multimodal converter that is absent from `scripts/`, so conversions fall back to `src/ups/data/convert_pdebench.py` and associated guidance (`docs/data_artifacts.md:13`, `scripts/convert_pdebench.py:1`).

## Component Details

### PDEBench Dataset Loader
- `src/ups/data/pdebench.py:23` — `TASK_SPECS` enumerates supported PDEBench tasks (burgers1d, advection1d, darcy2d, navier_stokes2d); expanding coverage requires augmenting this mapping and, if needed, specifying target/parameter keys.  
- `src/ups/data/pdebench.py:47` — `PDEBenchDataset` loads HDF5 shards, concatenates samples across files, applies optional normalisation, and aggregates params/BC metadata when present; it honours the `PDEBENCH_ROOT` env var to override configured roots, enabling remote execution.  
- `src/ups/data/pdebench.py:129` — `__getitem__` returns field tensors along with optional parameter and boundary-condition dictionaries expected downstream by conditioning utilities.

### Latent Pair Construction
- `src/ups/data/latent_pairs.py:162` — `make_grid_coords` synthesises normalised grid coordinates per dataset, reused for caching and encoding.  
- `src/ups/data/latent_pairs.py:260` — `GridLatentPairDataset` wraps `PDEBenchDataset`, handles cache hits/misses, performs encoder calls (`GridEncoder`), down-samples in time via `time_stride`, and prepares future targets for multi-step rollouts; inverse-loss inputs can be emitted when enabled.  
- `src/ups/data/latent_pairs.py:569` — `_build_pdebench_dataset` instantiates `PDEBenchDataset`, infers grid/channel dimensions, and constructs a `GridEncoderConfig` tied to latent length/dim requirements.  
- `src/ups/data/latent_pairs.py:588` — `build_latent_pair_loader` centralises DataLoader creation: it supports single-task or multi-task PDEBench mixes (`ConcatDataset`), config-driven caching directories (`latent_cache_dir`), `time_stride`, rollout horizon, and toggles a RAM-backed `PreloadedCacheDataset` when caches are complete and memory allows.  
- `src/ups/data/latent_pairs.py:714` — Non-PDEBench data paths (Zarr grids/meshes/particles) share the same loader entry point, offering a uniform pipeline across modalities.

### Parallel Cache & Scaling Utilities
- `src/ups/data/parallel_cache.py:33` — `RawFieldDataset` defers encoding to the main process while DataLoader workers fetch raw fields, enabling safe multi-worker usage with CUDA encoders.  
- `src/ups/data/parallel_cache.py:90` — `PreloadedCacheDataset` pulls cached latents into RAM to minimise I/O during large training runs, with guard rails for cache completeness.  
- `src/ups/data/parallel_cache.py:258` — `build_parallel_latent_loader` wires the raw dataset, GPU encoding collate_fn, and worker-friendly DataLoader parameters, giving 4–8× speedups over the single-worker fallback.  
- `src/ups/data/parallel_cache.py:313` — Helper utilities (`check_cache_complete`, `estimate_cache_size_mb`, `check_sufficient_ram`) inform whether to switch to RAM-preloaded mode, and are consumed by `build_latent_pair_loader`.

### Cache Precomputation Pipeline
- `scripts/precompute_latent_cache.py:40` — `_instantiate_dataset` mirrors training config assembly when generating caches, allowing reuse of task lists, latent dimensions, and rollout horizons.  
- `scripts/precompute_latent_cache.py:83` — `_iter_dataset` optionally leverages `build_parallel_latent_loader` to populate caches quickly, falling back to legacy iteration if parallel encoding fails; progress bars use `tqdm` when available.  
- `scripts/precompute_latent_cache.py:196` — `compute_cache_hash` hashes latent config parameters to detect when cached latents are stale relative to architectural changes.

### Training & Evaluation Entry Points
- `scripts/train.py:216` — `dataset_loader` enforces presence of `data.task` / `data.kind` and delegates to `build_latent_pair_loader`, so all stages consume the same pipeline.  
- `src/ups/eval/pdebench_runner.py:27` — Baseline evaluation loads raw fields directly; `evaluate_latent_operator` reuses the latent pair loader (with worker count forced to zero) to feed operator + diffusion evaluation, ensuring parity with training data preprocessing.  
- `tests/unit/test_train_pdebench_loader.py:35` — Test suite covers PDEBench grid ingestion, conditioning broadcast, and Zarr fallbacks, validating shapes and conditioning metadata expected by training loops.

### Data Conversion & Documentation
- `scripts/convert_pdebench.py:1` — Consolidates raw PDEBench shards into uniform HDF5 datasets, normalising tensor rank and dtype for downstream loaders.  
- `docs/data_artifacts.md:13` — Playbook documents the conversion → caching → packaging flow, including references to the (currently missing) `scripts/convert_pdebench_multimodal.py`; the documented fallback CLI uses `scripts/convert_pdebench.py`.  
- `README.md:55` — Top-level workflow reiterates conversion, hydration, and remote execution steps that rely on the same dataset root and caching conventions.

## Architecture Documentation

The data stack is layered: raw HDF5 ingestion (`PDEBenchDataset`) → latent encoding / conditioning (`GridLatentPairDataset`) → DataLoader orchestration (`build_latent_pair_loader`) → optional acceleration (`parallel_cache`, RAM preload) → downstream consumers (training/evaluation scripts). Configuration knobs for scaling—latent cache roots, worker counts, stride, rollout horizons—are consistently read from `training.*` YAML fields, and CLI utilities mirror these defaults for a cohesive workflow.

## Historical Context (thoughts/)

- `docs/data_artifacts.md:13` — Establishes the intended raw-to-artifact conversion and caching process, highlighting where future multimodal converters would slot in once available.

## Related Research

- `thoughts/shared/research/2025-10-28-checkpoint-resume-system.md` — Covers checkpoint orchestration interacting with the same training stages that rely on the latent pair loader.

## Open Questions

- The repository references `scripts/convert_pdebench_multimodal.py`, but no such script exists in `scripts/`; confirming whether it lives elsewhere or needs restoration would clarify multimodal ingestion coverage.

## Follow-up Research 2025-11-06T02:31:47Z

- `src/ups/data/datasets.py:133` defines `MeshZarrDataset`, which reads Zarr groups (default `mesh_poisson`), caches CSR Laplacians via `scipy.sparse`, and exposes mesh samples with connectivity, boundary metadata, and cached Laplacian handles for downstream operators.
- `src/ups/data/datasets.py:197` provides `ParticleZarrDataset`, loading per-sample positions/velocities plus neighbour graphs (CSR + edge list) and packaging them with periodic BC metadata; samples return both trajectory fields and adjacency data needed for encoding.
- `src/ups/data/latent_pairs.py:472` implements `GraphLatentPairDataset`, which iterates mesh/particle samples, enforces consistent time dimensions, encodes each step through `MeshParticleEncoder`, and assembles latent pairs with optional rollout horizons and conditioning derived from params/BCs.
- `src/ups/io/enc_mesh_particle.py:12` documents `MeshParticleEncoder` and its config: residual message passing layers, optional coordinate concatenation, supernode pooling, and Perceiver-style token reduction to reach the configured latent length/dim.
- `src/ups/data/latent_pairs.py:714` routes `build_latent_pair_loader` to mesh/particle handling when `data.kind` is set, constructing either `MeshZarrDataset` or `ParticleZarrDataset`, deriving encoder configs, and returning a DataLoader backed by `GraphLatentPairDataset`.
- `tests/unit/test_mesh_loader.py:20` and `tests/unit/test_particles_dataset.py:16` rely on `scripts.prepare_data.Config` to synthesize Zarr stores for mesh and particle regressions, verifying node counts, Laplacian properties, and neighbour consistency—yet the repository currently lacks a `scripts/prepare_data.py`, so reproducing these datasets depends on restoring that script or providing an equivalent.

### Updated Open Questions
- The missing `scripts/prepare_data.py` (referenced by mesh/particle tests) and `scripts/convert_pdebench_multimodal.py` leave unclear how mesh/particle Zarr datasets are generated today; locating or reintroducing these utilities is necessary to fully cover non-grid PDEBench datatypes.
