# Aggressive Training & Scaling Roadmap

Goal: Reach state-of-the-art performance across PDEBench (and extensions) quickly by scaling model capacity and dataset breadth in tandem (“Chinchilla style”) while keeping the workflow artifact-first and cloud friendly (Vast.ai).

---

## Phase 0 — Bootstrapped Prototype (status ✅)
- **Data**: Burgers1D subset (20 trajectories) stored as W&B artifact `burgers1d_subset_v1`.
- **Model**: Latent dim 32, tokens 16; full four-stage pipeline proven end-to-end.
- **Outcome**: Infrastructure validated, metrics/logging/debug cycle in place.

Key Actions already completed:
1. Streaming converters (`scripts/convert_pdebench_multimodal.py`) for grid/mesh/particle modes.
2. Artifact workflow (tar + `scripts/upload_artifact.py`, registry `docs/dataset_registry.yaml`, hydration via `scripts/fetch_datasets.py`).
3. Remote runner (`scripts/run_remote_scale.sh`) and repro scripts for artifacts/checkpoints.

---

## Phase 1 — Grid Scaling Sprint
**Objective**: Outperform linear baseline on Burgers1D + Advection1D while expanding model capacity.

1. **Dataset ramp**
   - Convert & upload `burgers1d_mini_v1` (~200 trajectories) and `advection1d_subset_v1` (matching size).
   - Update registry with artifact IDs and split filenames.

2. **Training configuration**
   - Config base: `configs/train_pdebench_scale.yaml` with overrides:
     - `latent.dim=128`, `latent.tokens=64`.
     - `training.batch_size=32`, `training.amp=true`, `training.ema_decay=0.999`, `training.accum_steps=1`, `training.distill_micro_batch=16`.
     - Cosine LR for operator; ReduceLROnPlateau or cosine for other stages.
     - EMA on operator/diffusion is enabled; evaluation prefers EMA checkpoints.

3. **Run plan (Vast.ai single GPU)**
   1. `WANDB_DATASETS="burgers1d_mini_v1" bash scripts/run_remote_scale.sh`.
   2. `WANDB_DATASETS="advection1d_subset_v1" bash scripts/run_remote_scale.sh`.
   3. Combined: set `data.task=[burgers1d,advection1d]` in `configs/train_pdebench_scale.yaml` (or provide a runtime override) and rerun the script.

4. **Evaluation**
   - `scripts/evaluate.py` + `scripts/fetch_wandb_metrics.py` to confirm RMSE improvement vs baseline.
   - Decode sample rollouts via `scripts/repro_pdebench.sh` to visually confirm gains.

5. **Compute scaling**
   - Keep total tokens × data pairs roughly constant: double data ⇒ double latent dims (Chinchilla). Track training FLOPs via `wandb.summary` and `reports/wandb_*.csv`.

---

## Phase 2 — Multi-Viscosity & Navier–Stokes
**Objective**: Train on 2–3 grid tasks simultaneously and push latent dim ≈256.

1. **Artifacts**
   - Upload `navier_stokes2d_subset_v1` (e.g., 128² grids, 100 trajectories).
   - Keep each tarball ≤10 GB for faster hydration.

2. **Model scaling**
   - Overrides: `latent.dim=256`, `latent.tokens=128`, PDE-T depths `[2,2,2]`, `training.batch_size=24`, `training.ema_decay=0.9995`, `training.accum_steps=2`, `training.distill_micro_batch=12`.
   - AMP remains on; accumulation activated if GPU <24 GB.

3. **Training**
   - Multi-task mixing supported directly: set `data.task=[burgers1d,advection1d,navier_stokes2d]`.
   - Alternatively hydrate multiple artifacts via `WANDB_DATASETS` and point `data.root` to `PDEBENCH_ROOT` (hydrated automatically by the runner).
   - Stage durations: `operator.epochs=40`, `diff_residual.epochs=20`, `consistency_distill.epochs=12`, `steady_prior` optional unless steady data included.

4. **Monitoring**
   - Log per-task metrics (`eval/task_name/mse`) via evaluation overrides (create new config group per dataset).
   - Use W&B sweep for latent dim / depth / LR to find best trade-off quickly.

5. **Compute**
   - For fast turnaround, use RTX 5090/4090 32 GB instances; expect ~4–6 hours per full run.

---

## Phase 3 — Mesh & Particle Integration
**Objective**: Fold in Darcy meshes and particle advection while keeping training stable.

1. **Artifacts**
   - Convert `darcy2d_mesh_subset_v1` and `particles_advect_subset_v1` (Zarr stores). Ensure neighbor caches included.

2. **Encoder adjustments**
   - Increase `MeshParticleEncoder` hidden dims (e.g., 256) and message passing steps (≥3).
   - Validate reconstruction via unit tests (`tests/unit/test_enc_mesh_particle.py`).

3. **Training strategy**
   - Multi-task mix: grid+mesh+particles. Use per-task weighting if one dominates.
   - Conditioning: ensure viscosity/permeability/radius fields are in `LatentState.cond`.
   - If VRAM tight, consider two-phase training: (a) grid pretrain, (b) mesh+particle fine-tune.

4. **Evaluation**
   - Extend eval configs for mesh (`configs/eval_darcy_mesh.yaml`) and particle tasks.
   - Log mesh visualization/particle scatter as W&B media.

5. **Compute scaling**
   - Potentially move to dual-GPU (DDP) or higher VRAM nodes (H100 80 GB) if single GPUs bottleneck.

---

## Phase 4 — Full PDEBench + External Datasets
**Objective**: Train on the full 3.6 TiB corpus and additional physics datasets (climate, astrophysics) while maintaining Chinchilla-style efficiency.

1. **Data orchestration**
   - Artifact registry holds all major slices; hydrate subsets per run to keep scratch usage manageable.
   - For complete runs, use cloud object store (S3/GCS) as secondary distribution if W&B limits hit.

2. **Model capacity**
   - Latent dim 384–512, tokens 160–192, PDE-T depth 3–4 blocks.
   - Consider MoE or factorized attention if scaling beyond single GPU.

3. **Training**
   - Multi-stage curriculum: start with small tasks, progressively mix in harder domains (curriculum scheduling via Hydra).
   - Use gradient checkpointing, mixed precision, activation offloading as needed.

4. **Evaluation & Benchmarks**
   - Build W&B reports tracking RMSE/MAE per dataset versus published baselines (FNO, AFNO, DiT).
   - Automate `scripts/benchmark.py` comparisons post-training.
   - Publish checkpoints + metrics as artifacts; generate HTML/PDF reports for each suite.

5. **Hyperparameter sweeps**
   - Launch W&B sweeps to optimize (latent_dim, PDE-T depth, LR schedule) on subsets before committing expensive full runs.

---

## Operational Best Practices
- **Artifacts only**: keep raw PDEBench shards off developer machines; fetch slices via registry.
- **Remote workers (Vast.ai)**: use `scripts/load_env.sh` + `WANDB_DATASETS=... bash scripts/run_remote_scale.sh`; tear down on completion.
- **Checkpointing**: stage checkpoints every few epochs to resume after preemptions.
- **Monitoring**: W&B dashboards for training curves, eval metrics, GPU utilization. Use `scripts/fetch_wandb_metrics.py` for offline analysis.
- **Resource planning**: convert once (on a machine with ≥4 TB storage), upload artifacts, then rely on sub-10 GB downloads per run.

---

## Immediate Next Steps
1. Convert and upload `burgers1d_mini_v1` + `advection1d_subset_v1`; update registry.
2. Launch Phase 1 run on Vast.ai (RTX 5090 or equivalent) with scaled config.
3. Evaluate with new logging fixes; compare against baseline.
4. Prepare Navier–Stokes subset conversion for Phase 2.
5. Draft W&B dashboard template for multi-task tracking.

With this plan, we scale capacity and data in lockstep, maintain reproducibility, and use cloud GPUs efficiently, pushing toward SOTA results rapidly.
