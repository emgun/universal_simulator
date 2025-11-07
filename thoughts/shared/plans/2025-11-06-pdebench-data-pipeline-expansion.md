---
date: 2025-11-06T02:44:22Z
planner: Codex (assistant)
status: proposed
topic: "PDEBench Data Pipeline Expansion"
related_documents:
  - thoughts/shared/research/2025-11-05-pdebench-data-pipeline.md
---

# Goal

Expand PDEBench coverage across grids, meshes, and particles so UPS can ingest, cache, and train on every dataset variant with consistent tooling and documentation. Restore missing conversion utilities, extend task specifications, enable scaling optimizations, and validate the end-to-end pipeline.

# Success Criteria (Overall)

- All referenced conversion and preprocessing scripts exist, run without errors, and generate artifacts for grid, mesh, and particle PDEBench datasets.
- `build_latent_pair_loader` successfully handles mixed PDEBench + graph datasets using restored configs.
- Automated test suite covers dataset generation, loader behavior, and caching for all datum types.
- Documentation and dataset registry instructions match the implemented workflow.

# Phase 1 – Restore Conversion & Generation Tooling

## Tasks
- Recreate `scripts/convert_pdebench_multimodal.py` with streaming converters for grid, mesh, and particle PDEBench tasks (CLI parity with documented usage).
- Reintroduce `scripts/prepare_data.py` featuring a dataclass `Config`, dataset-specific generators, and split metadata so unit tests can synthesize mesh/particle fixtures.
- Update `docs/data_artifacts.md` and `README.md` to reflect the restored scripts and their options.

## Primary Files
- `scripts/convert_pdebench_multimodal.py`
- `scripts/prepare_data.py`
- `docs/data_artifacts.md`
- `README.md`

## Success Criteria
### Automated Verification
- [x] `python scripts/convert_pdebench_multimodal.py --help` runs without errors.
- [x] `python scripts/prepare_data.py --help` runs without errors.
- [x] Smoke conversion: `PYTHONPATH=src python scripts/convert_pdebench_multimodal.py burgers1d --root <tmp> --out <tmp>` completes and produces an HDF5 file.
### Manual Verification
- [x] Spot-check generated grid, mesh, and particle artifacts (inspect shape, metadata keys) to confirm expected structure.
- [x] Ensure updated docs accurately describe the CLI usage.

# Phase 2 – Extend Dataset Specifications & Routing

## Tasks
- Expand `TASK_SPECS` and related helpers in `src/ups/data/pdebench.py` to include all PDEBench tasks (grid + mesh/particle); define field/target/param/BC keys.
- Add routing logic or configuration patterns so mesh/particle tasks invoke the appropriate Zarr loaders while grid tasks use HDF5.
- Provide schema validation helpers or constants to cross-check dataset layouts.

## Primary Files
- `src/ups/data/pdebench.py`
- `src/ups/data/latent_pairs.py`
- `scripts/validate_data.py`
- New/updated configs under `configs/`

## Success Criteria
### Automated Verification
- [x] `python scripts/validate_data.py --data-root <tmp> --task <new_task> --split train` validates representative tasks (grid + mesh/particle).
- [x] Unit tests covering expanded task specs pass (`pytest tests/unit/test_pdebench.py` and new/updated cases).
### Manual Verification
- [x] Confirm newly defined tasks match upstream PDEBench naming and schema.

# Phase 3 – Loader, Caching, and Scaling Enhancements

## Tasks
- Ensure `build_latent_pair_loader` supports mixed task lists containing grids, meshes, and particles; update conditioning/collate logic if required.
- Extend `scripts/precompute_latent_cache.py` to precompute latents for mesh/particle datasets with parallel encoding and RAM-preload checks.
- Add cache validation utilities for graph latents in `src/ups/data/parallel_cache.py` and document memory expectations.

## Primary Files
- `src/ups/data/latent_pairs.py`
- `src/ups/data/parallel_cache.py`
- `scripts/precompute_latent_cache.py`
- `configs/*` (latent cache settings)

## Success Criteria
### Automated Verification
- [x] `python scripts/precompute_latent_cache.py --tasks burgers1d darcy2d_mesh particles_advect --splits train --cache-dir <tmp>` completes successfully.
- [x] End-to-end loader smoke test: `pytest tests/unit/test_train_pdebench_loader.py` (with new mesh/particle scenarios) passes.
### Manual Verification
- [x] Inspect cache directories to verify tensor shapes (latent len/dim) and conditioning tensors for each dataset type.
- [ ] Confirm GPU utilization improvements via parallel encoding during a short training dry-run.

# Phase 4 – Testing, Documentation, and Registry Updates

## Tasks
- Expand regression tests to cover dataset generation, loader behavior, and caching for all supported datatypes (grid/mesh/particle).
- Update fast-to-SOTA playbook and dataset registry documentation to match the new workflow.
- Record artifact metadata (names, resolutions, physics parameters) for each dataset variant in the registry.

## Primary Files
- `tests/unit/test_mesh_loader.py`
- `tests/unit/test_particles_dataset.py`
- `tests/unit/test_train_pdebench_loader.py`
- `docs/fast_to_sota_playbook.md`
- `docs/dataset_registry.yaml` (or equivalent)

## Success Criteria
### Automated Verification
- [x] `pytest tests/unit/test_mesh_loader.py tests/unit/test_particles_dataset.py tests/unit/test_train_pdebench_loader.py` passes.
- [ ] Lint/format checks succeed (`make lint` or equivalent).
### Manual Verification
- [ ] Validate documentation updates with stakeholders to ensure operational accuracy.
- [ ] Confirm registry entries align with generated artifacts and remote runners.
