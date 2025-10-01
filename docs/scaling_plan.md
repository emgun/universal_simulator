# Universal PDEBench Scaling Plan

This roadmap generalises the current Burgers1D prototype into a broad, multi-domain programme capable of ingesting every PDEBench modality (grid, mesh, particle) as well as external physics datasets. Each phase widens coverage, increases capacity, and keeps the pipeline reproducible so we can scale toward state-of-the-art performance across diverse systems.

---

## Phase 0 — Artifact & Dataset Registry

1. **Dataset Packaging**
   - Produce versioned tarballs for **every** dataset slice using `scripts/convert_pdebench.py` (or modality-specific converters):
     - `burgers1d_subset_v1`, `burgers1d_full_vx` (grid 1D).
     - `advection1d_subset`, `burgers2d_subset`, `navier_stokes2d_subset` (grid 2D/3D).
     - `darcy2d_mesh_subset`, `navier_stokes_mesh_subset` (mesh).
     - `particles_advect_subset`, `granular_contacts_subset` (particle systems).
   - For non-PDEBench data, define a consistent schema → convert → package (e.g., `solar_flare_subset`, `climate_reanalysis_subset`).
   - Upload each tarball via `scripts/upload_artifact.py` with rich metadata: domain, discretisation, resolution, parameter ranges, normalisation.
   - Maintain a registry file (`docs/dataset_registry.yaml`) mapping artifact names to config snippets.
   - Use `scripts/fetch_datasets.py <dataset_keys>` to hydrate a local `DATA_ROOT` on any machine; the helper reads the registry and downloads the matching artifacts (with optional caching).
   - When packaging new datasets, record the artifact ID + metadata in the registry to keep fetch scripts synchronised.
   - Step-by-step instructions live in `docs/data_artifacts.md` (conversion, tarballing, upload, hydration).

2. **Config Synchronization**
   - Confirm `configs/train_pdebench_scale.yaml` and `configs/eval_pdebench_scale.yaml` are artifact-driven using `${PDEBENCH_ROOT}`.
   - Add a Hydra override cheat-sheet (e.g., `configs/scale_overrides.md`) with knobs for latent size, batch size, precision modes, stage epochs.

3. **Remote Harness Smoke Test**
   - On a remote GPU box, run `WANDB_DATASETS="burgers1d_subset_v1" bash scripts/run_remote_scale.sh` to validate end-to-end:
     - Dataset fetch via registry helper → staged training → evaluation → W&B metrics/plots.
     - Confirm AMP stability and gradient clipping settings are appropriate.

---

## Phase 1 — Grid Dataset Expansion & Capacity Ramp

Goal: exceed baselines on multiple grid tasks (Burgers/Advection/Navier–Stokes) while keeping compute manageable.

1. **Data**
   - Stage at least three grid datasets (e.g., Burgers1D, Advection1D, Navier–Stokes2D) using artifacts.
   - Maintain consistent normalisation strategy; document per-dataset stats in the registry.

2. **Model Updates**
   - Increase latent dimension/tokens (e.g., 128×64) and confirm encoder can reconstruct fields (sanity check via `GridEncoder.reconstruct`).
   - Evaluate patch size and stem width; consider `patch_size=1 or 2` to avoid excessive downsampling.

3. **Training**
   - Activate EMA, gradient clipping, optional mixed precision (already supported in scale configs).
   - Stage schedule baseline: 30 epochs operator; 15 diffusion; 10 distill; 20 steady prior.
   - Alternate tasks per batch or per epoch (Hydra multi-run or custom sampler) to encourage shared representations.
   - Monitor throughput & GPU utilisation via W&B system metrics.

4. **Validation**
   - Run both latent evaluation and decoded-field inspection:
     - Add a script to decode `reports/pdebench_scale_eval_preview.npz` back to grid space and save PNG sequences.
   - Compare metrics vs baselines per task; iterate until latent RMSE beats baselines for all included datasets.

---

## Phase 2 — Mesh, Particle, and Irregular Domains

Goal: extend encoders/operators to handle unstructured meshes, particle systems, and hybrid domains.

1. **Data Handling**
   - Package mesh datasets (e.g., Darcy2D, Navier–Stokes mesh) and particle datasets (particles advection, granular flows) as artifacts.
   - Ensure conversion scripts precompute connectivity/neighbor caches to avoid recomputation on remote nodes.

2. **Encoder Enhancements**
   - Scale `MeshParticleEncoder` hidden dim and message-passing steps; benchmark identity reconstruction.
   - Add optional positional encodings / geometric features for complex meshes (curvature, boundary tags).

3. **Training Strategy**
   - Mix grid + mesh + particle mini-batches using task-aware sampling weights.
   - Introduce domain-specific conditioning tokens (e.g., Darcy permeability, particle radius).
   - Validate gradient stability with AMP; fall back to bf16 or mixed precision per modality as needed.

4. **Evaluation**
   - Extend `scripts/evaluate.py` to generate modality-specific visualisations (mesh plots, particle scatter sequences).
   - Track physics-integrity metrics (mass conservation, divergence norms) for each domain.

---

## Phase 3 — Full PDEBench + External Datasets

Goal: train on complete PDEBench suite and incorporate external datasets for generalisation.

1. **Data Orchestration**
   - Stream large raw shards directly into artifact writers (avoid local duplication).
   - For each dataset, define train/val/test splits in registry; include parameter sweeps (viscosity, Reynolds number, forcing).
   - Integrate external datasets (climate, astrophysics, robotics) via the same schema to prevent overfitting to PDEBench alone.

2. **Model/Training Enhancements**
   - Expand latent capacity (e.g., 256–512 dim, 128–256 tokens) and deeper PDE-Transformer stacks.
   - Incorporate conditioning for all relevant parameters (Re, Pe, permeability, source terms, particle radii).
   - Employ gradient accumulation or data-parallel multi-GPU training to sustain large batch sizes.
   - Use learning-rate warmup + cosine schedule (already configured) and consider per-task adaptive weighting.

3. **Regularization & Losses**
   - Activate spectral, conservation, and boundary-condition losses; calibrate weights per modality.
   - Use random time-window cropping, noise perturbations, and data augmentation (rotations, flips for isotropic grids).

4. **Evaluation & Reporting**
   - Evaluate per dataset and aggregate global metrics (macro-averaged RMSE, conservation scores).
  - Maintain a leaderboard comparing UPS vs baselines (FNO, UNet, DiT) for each modality.
  - Track energy budgets, residuals, calibration across tasks; log qualitative visualisations to W&B.

---

## Phase 4 — State-of-the-Art Benchmarking & Reporting

1. **Benchmarks**
   - Run `scripts/benchmark.py` to compare UPS vs FNO/DiT baselines on the same dataset splits.
   - Document compute cost (GPU hours), model size, throughput.

2. **Ablations**
   - Use `configs/train_pdebench_scale.yaml` as base; disable components (diffusion, spectral loss, conditioning) to quantify gains.

3. **Artifacts & Reproducibility**
   - Publish final checkpoints + evaluation reports as W&B artifacts per dataset and global aggregate.
   - Write reproduction scripts (`scripts/repro_pdebench.sh`, `scripts/repro_multimodal.sh`) pulling artifacts and regenerating figures.
   - Document the end-to-end artifact workflow in the README (see "Artifact Workflow" section) so collaborators can reproduce results easily.

4. **Documentation**
   - Update README / docs with scaling results, instructions for remote training, and summary tables.

---

## Operational Safeguards
- Implement restartable training (checkpoint every N epochs; scripts should resume when checkpoint exists).
- Track disk usage and prune old tarballs to prevent local storage exhaustion.
- Document estimated GPU hours per dataset/phases to plan cloud budgets.
- Before large launches, dry-run configs with `--stage operator --epochs 1` to catch obvious misconfigurations.
 - Maintain a dependency/env lockfile for remote runs; pin torch/cuda versions and verify before scale launches.

---

By following these phases, we can expand from the current prototype to a universal, multi-domain physics modelling stack while maintaining visibility into performance improvements and resource usage.
