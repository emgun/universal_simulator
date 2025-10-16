# Universal Physics Stack — SOTA Milestones & TODO (Checkable)

This is the authoritative, checkable TODO/milestone plan derived from `universal_physics_stack_implementation_plan_for_a_coding_agent.md`.

Use it as a living checklist during implementation. Each item is actionable, references concrete files/APIs, and includes a clear Definition of Done (DoD). Mark items with [x] once complete. Keep commits small and tied to checklist items.

Legend:
- [ ] pending • [~] partial • [x] done • (→) dependency • (DoD) definition of done • (Cmd) command to run • (File) path reference

Global conventions:
- Python ≥ 3.10; PyTorch ≥ 2.3; prefer bf16 mixed precision; deterministic seeds set.
- Type hints everywhere; tensor shapes noted in docstrings (e.g., `B×T×D`).
- Config via Hydra/OmegaConf only; no hardcoded hyperparameters.
- Log to Weights & Biases if configured; otherwise local MLflow or CSV.
- Tests in `tests/unit` and `tests/integration`; keep them fast and deterministic.

---

## M0 — Bootstrap & Repository Scaffold

- [x] M0.1: Initialize repository skeleton (File)
  - (Cmd) `git init && mkdir -p src/{core,io,models,training,data,inference,eval,discovery,active} tests/{unit,integration} scripts configs reports export .github/workflows`
  - (DoD) `git status` shows created dirs; default branch initialized.

- [x] M0.2: Create `pyproject.toml` and pin dependencies (File: `pyproject.toml`)
  - Include: `torch>=2.3`, `einops`, `xarray`, `zarr`, `h5py`, `hydra-core`, `omegaconf`, `wandb`, `mlflow`, `trimesh`, `pykeops`, `pyvista`, `netCDF4`, `sympy`, `pysr`, `pysindy`, `scipy`, `pandas`, `pytest`, `pytest-xdist`, `ruff`, `black`, `isort`, `mypy`, `types-PyYAML`, `matplotlib`, `seaborn`.
  - Add `[tool.ruff]`, `[tool.black]`, `[tool.isort]` basic configuration.
  - (Cmd) `pip install -e .`
  - (DoD) `python -c "import torch,einops,hydra"` succeeds.

- [x] M0.3: Pre-commit and code style (File: `.pre-commit-config.yaml`)
  - Hooks: ruff, black, isort, end-of-file-fixer, trailing-whitespace.
  - (Cmd) `pre-commit install && pre-commit run --all-files`
  - (DoD) All hooks pass on empty scaffold.

- [x] M0.4: CI workflow (File: `.github/workflows/ci.yaml`)
  - Jobs: lint (ruff/black --check), unit tests `pytest -q tests/unit`.
  - Cache pip; run on `push` and `pull_request`.
  - (DoD) CI green on first push.

- [x] M0.5: Environment bootstrap script (File: `scripts/prepare_env.sh`)
  - Export deterministic flags; pin CUDA libraries; create `.env` with `PYTORCH_ENABLE_MPS_FALLBACK=1` if macOS.
  - (Cmd) `bash scripts/prepare_env.sh`
  - (DoD) `python - <<'PY'\nimport torch; torch.use_deterministic_algorithms(True); print('OK')\nPY` prints OK.

- [x] M0.6: Project README (File: `README.md`)
  - Include quickstart, unified runbook, folder layout, data prep instructions.
  - (DoD) New contributor can run M1–M2 without extra guidance.

---

## M1 — Data Schema & Non‑Dimensionalization

- [x] M1.1: Implement physics sample schema (File: `src/ups/data/schemas.py`)
  - Create `TypedDict Sample` with keys: `kind, coords, connect, fields, bc, params, geom, time, dt, meta`.
  - Add `validate_sample(sample) -> None` raising informative errors.
  - (Tests) `tests/unit/test_schema.py::test_required_fields` and type checks.
  - (DoD) Validation catches missing/invalid keys; shape checks enforced.

- [x] M1.2: Non-dimensionalization utilities (File: `src/ups/discovery/nondim.py`)
  - Implement `to_pi_units(sample) -> Sample`, `from_pi_units(sample) -> Sample`.
  - Store `sample.meta['scale']` dict containing per-field and per-parameter scales.
  - Provide `pi_constants_from_units(units: dict) -> dict` and reversible transforms.
  - (Tests) `tests/unit/test_nondim.py::test_roundtrip_error` < 1e-6.
  - (DoD) Round-trip exact for synthetic scalings; handles vector/tensor fields.

- [x] M1.3: Metadata table scaffold (File: `data/metadata.parquet`)
  - Schema: `id:str, split:{train,val,test}, pde:str, bc:str, geom:str, units_json:str`.
  - Script function to build: `scripts/prepare_data.py --write-metadata`.
  - (DoD) 80/10/10 split within ±1% and persisted.

---

## M2 — Minimal PoC Datasets

- [x] M2.1: Grid PDE trio generator (File: `scripts/prepare_data.py`)
  - Implement CLI with Hydra: `+dataset=poc_trio out=data/poc_trio.zarr`.
  - Generate Diffusion2D, Burgers2D, Kolmogorov2D on 256×256 grids; store Zarr chunks `(C,H,W)` with Blosc.
  - Save example visualization `reports/samples_grid.png`.
  - (Tests) `tests/unit/test_zarr_reader.py` loads first batch < 1s.
  - (DoD) Read latency < 1s/batch; chunking and compression verified.

- [x] M2.2: Mesh Poisson dataset (File: `data/mesh_poisson.zarr` via `scripts/prepare_data.py`)
  - Generate meshes (2.5k–20k nodes), coords, edges, cotan Laplacian cache.
  - (Tests) `tests/unit/test_mesh_loader.py` verifies shapes and Laplacian PSD.
  - (DoD) Peak RAM < 2GB; neighbor ops ready for M3.

- [x] M2.3: Particle advection dataset (File: `data/particles_advect.zarr`)
  - 10k–100k particles with velocities; build radius-graph caches (pykeops/backend fallback).
  - (DoD) Neighbor build < 100 ms/10k; file size < 1GB/100k.

---

## M3 — Encoders & Any‑Point Decoder

- [x] M3.1: GridEncoder (File: `src/ups/io/enc_grid.py`)
  - PixelUnshuffle `p=4`, per-channel streams, local conv stem, optional Fourier coord features.
  - Provide `forward(fields, coords, params, bc, geom) -> LatentState.z` sized `latent.len×latent.dim`.
  - Add identity reconstruction head for self-checks.
  - (Tests) Identity MSE < 1e-3 with reconstruction head.
  - (DoD) Throughput matches spec; stable gradients over 1k steps.

- [x] M3.2: Mesh/Particle encoder (File: `src/ups/io/enc_mesh_particle.py`)
  - 3× message passing (radius-graph) → supernode pooling (`supernodes≈2048`) → Perceiver pooling to `latent.len` tokens.
  - Save inverse maps for `L_inv_enc`.
  - (Tests) Runtime < 50 ms for 10k nodes; identity MSE < 1e-3.
  - (DoD) Works for both mesh and particles modes.

- [x] M3.3: Any‑Point Decoder (File: `src/ups/io/decoder_anypoint.py`)
  - Perceiver-style cross-attn `decode(points[M,d], latent, cond) -> fields[M,C]`.
  - Batched, supports irregular outputs and mesh transfer.
  - (Tests) Harmonic field rel‑err < 1e-3 at unseen coordinates.
  - (DoD) Memory scales with queries; numerically stable.

---

## M4 — PDE‑Transformer Core

- [x] M4.1: Shifted-window ops (File: `src/ups/core/shifted_window.py`)
  - `partition_windows`, `merge_windows`, log-spaced rel-pos bias; `shift=True/False`.
  - (Tests) Windowing unit tests; O(T) scaling ~linear for fixed window.
  - (DoD) Correct reconstruction across boundaries with shifts.

- [x] M4.2: PDE‑T blocks (File: `src/ups/core/blocks_pdet.py`)
  - U-shape down/up with skip connections; channel-separated tokens; axial channel attention bridges; RMSNorm(Q,K).
  - (Tests) 1k forward/backward with random inputs → no NaNs; ≥ UNet baseline throughput.
  - (DoD) Meets stability and throughput targets.

- [x] M4.3: Conditioning (File: `src/ups/core/conditioning.py`)
  - AdaLN conditioning from PDE id, params, BCs, geometry into per-block scale/shift/gate.
  - (Tests) Ablation shows ≥5% nRMSE gain on PoC.
  - (DoD) Conditioning plumbed across encoders, operator, and decoder.

---

## M5 — Latent Operator (Fast Transients)

- [x] M5.1: Latent state container (File: `src/ups/core/latent_state.py`)
  - `@dataclass LatentState{ z:T×D, t:scalar|None, cond:dict }` with `.to(device)`, `.detach_clone()` helpers.
  - Serialization/deserialization to dict for checkpoints.
  - (Tests) Round-trip serde and device moves.
  - (DoD) API stable and used across modules.

- [x] M5.2: Latent operator module (File: `src/ups/models/latent_operator.py`)
  - `forward(state, dt) -> state'` using PDE‑T stack; time-step embedding; residual connection.
  - (Tests) One-step avg nRMSE ≤ 0.06 on PoC trio after brief train.
  - (DoD) Learning curves stable; no exploding gradients.

- [x] M5.3: Training losses (File: `src/ups/training/losses.py`)
  - Add `L_inv_enc`, `L_inv_dec`, `L_one_step`, `L_rollout`, `L_spec`, `L_cons`, `L_tv_edge`.
  - (Tests) Unit tests for shapes and invariants; spectral loss bands computed correctly.
  - (DoD) 100‑step latent rollout stable; energy drift < 1e‑3/step.

- [x] M5.4: Training loop & curriculum (File: `src/ups/training/loop_train.py`)
  - Rollout curriculum; mixing datasets & BCs; EMA and gradient clipping integration.
  - (Artifacts) `reports/op_rollout_metrics.html` with nRMSE@10, spectra, budgets.
  - (DoD) nRMSE@10 ≤ 0.40; spectral mid/high < 0.15 on PoC.

---

## M6 — Diffusion Residual (Few‑Step Corrector)

- [x] M6.1: Flow-matching vector field (File: `src/ups/models/diffusion_residual.py`)
  - `s_phi(z, τ, cond)` with residual-guidance based on decoded PDE residuals.
  - (DoD) 4‑step corrector reduces rollout nRMSE ≥20% on Kolmogorov vs operator-only.

- [x] M6.2: Consistency distillation (File: `src/ups/training/consistency_distill.py`)
  - Train 1–2 step student to match multi‑step teacher.
  - (DoD) ≥85% of teacher’s error reduction; runtime overhead ≤1.5× operator.

- [x] M6.3: Gated predictor–corrector inference (File: `src/ups/inference/rollout_transient.py`)
  - Trigger by fixed k or by gates in `eval/gates.py`; logs decisions.
  - (DoD) Overall error strictly decreases vs operator-only; gate logs saved.

---

## M7 — Steady‑State Latent Prior

- [x] M7.1: Conditional steady prior (File: `src/ups/models/steady_prior.py`)
  - Diffusion/flow over latent for BVP solutions (6–8 steps; configurable; distilled if needed).
  - (DoD) Residual norm ≤ solver tol on ≥95% PoC cases.

- [x] M7.2: Calibration utilities (File: `src/ups/eval/calibration.py`)
  - Reliability diagrams + temperature scaling integration.
  - (DoD) ECE ≤ 0.05 on steady tasks.

---

## M8 — Physics Guards & Projections

- [x] M8.1: Helmholtz–Hodge projection (File: `src/ups/models/physics_guards.py`)
  - FFT-based grid and cotan‑Laplacian mesh implementations.
  - (DoD) Divergence L2 ↓ ≥99%; runtime < 5 ms @256².

- [x] M8.2: Positivity clamps & bounds (File: `src/ups/models/physics_guards.py`)
  - Log‑exp clamps for {ρ, κ, ε}; bound-preserving updates.
  - (DoD) Zero negatives; ≤1% accuracy loss.

- [x] M8.3: Interface flux continuity (File: `src/ups/models/physics_guards.py`)
  - Small QP/projection to enforce conservation across domains.
  - (DoD) Global energy error < 1e‑3 over 100 steps on toy.

---

## M9 — Multiphysics Factor Graph

- [x] M9.1: API for domain nodes & port edges (File: `src/ups/models/multiphysics_factor_graph.py`)
  - Message passing K iterations with residual termination.
  - (DoD) Converges < K to residual < tol on synthetic case.

- [x] M9.2: Coupled demo & config (Files: `configs/coupled_toy.yaml`, `scripts/infer.py`)
  - Run fluid–thermal demo; save report `reports/coupled_demo.html`.
  - (DoD) Budgets within thresholds; report generated.

---

## M10 — Particles & Contacts

- [x] M10.1: Hierarchical neighbor search (File: `src/ups/models/particles_contacts.py`)
  - Grid/tree acceleration; batched radii; GPU-friendly.
  - (DoD) 100k particles step without OOM; neighbor build < 30 ms.

- [x] M10.2: Symplectic integrator + constraints (File: `src/ups/models/particles_contacts.py`)
  - Velocity Verlet; PBD/PGS constraints for contacts/joints.
  - (DoD) Energy drift < 1% over 1k steps on elastic collision.

---

## M11 — Data Assimilation & Safe Control

- [x] M11.1: Latent EnKF / 4D‑Var (File: `src/ups/inference/da_latent.py`)
  - Use decoder as H operator; ensemble management utilities.
  - (DoD) Posterior error ≥25% lower than prior with sparse sensors.

- [x] M11.2: Safe control in latent (File: `src/ups/inference/control_safe.py`)
  - MPC loop with control barrier functions (CBF) on decoded fields.
  - (DoD) Zero safety violations; ≤10% cost overhead.

---

## M12 — Benchmarks, Ablations, and SOTA Gate

- [x] M12.1: Baseline trainers (File: `scripts/train_baselines.py`)
  - Train FNO, UNet, DiT under identical data/compute settings.
  - (DoD) Baseline checkpoints saved for PoC trio.

- [x] M12.2: Unified evaluation script (File: `scripts/evaluate.py`)
  - Metrics: nRMSE@10, spectral error, conservation gaps, BC violation, ECE, tokens/sec.
  - (DoD — SOTA PoC) Our model ≥20% better nRMSE@10 and ≥2× tokens/sec; ECE ≤ 0.05; conservation/BC gaps ≤ baseline.

- [x] M12.3: Ablation report (File: `reports/ablations.html`)
  - Ablate channel‑sep, inverse losses, corrector.
  - (DoD) Each ablation worsens ≥5% on ≥1 metric.

- [x] M12.4: PDEBench evaluation suite
  - Add PDEBench dataset adapters (Burgers, Navier–Stokes, Darcy) with caching, loaders, and splits.
  - Metrics: MAE/MSE/L2/rRMSE, physics residual checks, conservation/invariant scores, n-step stability.
  - Evaluation runners: classical solver baseline, operator-only, operator+diffusion comparisons.
  - CLI: `benchmark --dataset pdebench_* --suite sim`; report JSON+plots (error vs horizon, calibration).
  - Config toggles for dataset/task selection and rollout horizon.
  - Reference checklist: datasets Burgers1D/Advection1D/Darcy2D/GrayScott2D/NavierStokes2D; install hints; ensure applicability across scales.

---

## M13 — Packaging, Export, and Repro

- [x] M13.1: One‑click repro pipeline (File: `scripts/repro_poc.sh`)
  - Data → train → eval; honor configs and seeds.
  - (DoD) Fresh machine reproduces core PoC within tolerances.

- [x] M13.2: Export operators (Dir: `export/`)
  - TorchScript/ONNX for operator+decoder; optional TensorRT path.
  - (DoD) Eager vs exported max diff < 1e‑4; improved GPU latency.

---

## Cross‑Cutting: Configs, CLI, Logging, Metrics, Tests

- [x] Configs (Files: `configs/defaults.yaml`, `configs/train_multi_pde.yaml`, `configs/inference_transient.yaml`, `configs/inference_steady.yaml`)
  - Defaults for latent dims, PDE‑T stacks, encoders, diffusion, steady prior, physics guards, coupling, training, logging.
  - (DoD) All scripts run via Hydra with overrides only.

- [x] CLIs (Files: `scripts/train.py`, `scripts/infer.py`, `scripts/evaluate.py`)
  - `train.py` stages: `operator`, `diff_residual`, `consistency_distill`, `steady_prior`.
  - `infer.py` modes: `transient`, `steady`.
  - `evaluate.py` produces HTML/PDF report and CSV.
  - (DoD) Commands from runbook succeed end‑to‑end on PoC.

- [x] Logging & reports (Files: `src/ups/eval/metrics.py`, `src/ups/eval/gates.py`, `src/ups/eval/reports.py`)
  - Per‑step budgets, residual histograms, spectra, calibration plots, gate activations; decision traces persisted.
  - (DoD) Reports reproducible and human‑readable.

- [x] Tests & acceptance criteria (Dir: `tests/`)
  - Unit: tensor shapes, window shifts, projection correctness (manufactured solutions), conservation on simple cases.
  - Integration: cross‑mesh transfer, predictor–corrector stability, steady prior residual ≤ ε, DA improves error.
  - (DoD) All green locally and in CI; acceptance gates met.

- [ ] Verify that simulation framework works from the smallest scales (quantum mechanics/High energy physics) to the largest scales (cosmic web/GR)

---

## Unified Runbook (Quick Commands)

1. `bash scripts/prepare_env.sh && pre-commit run --all-files`
2. `python scripts/prepare_data.py +dataset=poc_trio out=data/poc_trio.zarr`
3. `python scripts/train.py +config=train_multi_pde.yaml stage=operator`
4. `python scripts/train.py +config=train_multi_pde.yaml stage=diff_residual`
5. `python scripts/train.py +config=train_multi_pde.yaml stage=consistency_distill`
6. `python scripts/train.py +config=train_multi_pde.yaml stage=steady_prior`
7. `python scripts/evaluate.py ckpt=checkpoints/op_latest.ckpt mode=transient`
8. `python scripts/evaluate.py ckpt=checkpoints/steady_latest.ckpt mode=steady`
9. `python scripts/infer.py mode=transient input=examples/state.nc --save`
10. `python scripts/infer.py mode=steady bc=examples/bc.json --save`

---

## Notes & Risks

- Diffusion runtime: keep corrector sparse; distill to 1–2 steps.
- BC violations: strengthen clamp/projections; refine decoder cross‑attn.
- Geometry shift: augment encoders; increase supernodes; finetune encoders.
- Contact instability: reduce dt; stronger constraint projection; symplectic integrator.

---

## SOTA Drive — Remaining Work

- [x] **Real Data Ingestion**
  - Integrate PDEBench HDF5 pipelines; expand `PDEBenchDataset` to stream Burgers, Darcy, Navier–Stokes.
  - Add batching/transforms for mesh/particle datasets (neighbor caches, BC/param metadata).
  - (DoD) Encoders/latents operate on full PDEBench samples; synthetic fallback removed.

- [x] **Training Routines on Benchmark Data**
  - Replace synthetic latent dataset in `scripts/train.py` with real loader + data augmentations.
  - Implement stage-wise curriculum (operator → diffusion → distill → steady) with checkpoints, scheduler, early stopping.
  - (DoD) Training curves logged; reproducible reruns; metrics trending toward benchmark baselines.

- [x] **Evaluation & Reporting Revamp**
  - Flesh out `scripts/evaluate.py` to load checkpoints, run rollouts, compute metrics, and write HTML/CSV reports.
  - Generate reliability charts, spectral plots, conservation tables.
  - (DoD) Evaluation commands produce a full report for both transient and steady tasks.

- [x] **Benchmark & Baseline Comparisons**
  - Implement real baseline training (`train_baselines.py`) for FNO/UNet/DiT using shared data loaders.
  - Run PDEBench suite (`scripts/benchmark.py`) comparing baselines vs UPS model; record metrics in `reports/`.
  - (DoD) Document ≥20% improvement over baseline per SOTA gate.

- [x] **Packaging & Deployment**
  - Implement TorchScript/ONNX export in `export/export_latent_operator.py`; validate diff <1e-4 vs eager.
  - Provide Docker/conda environment, README quickstart, and `scripts/repro_poc.sh` end-to-end run (data→train→eval).
  - (DoD) Fresh machine reproduces results within tolerance; exported models available for inference.

- [x] **Monitoring & QA**
  - Add structured logging (JSON) capturing metrics per batch; integrate with W&B/MLflow.
  - Expand unit/integration tests: acceptance criteria (energy drift, BC enforcement, DA improvement).
  - (DoD) CI runs data-free checks + light integration; release candidate satisfies acceptance suite.

---

## Scaling Checklist (Artifact-Driven SOTA Push)

### Phase 1 — Grid Sprint (Burgers + Advection)
- [ ] Publish mid-size artifacts
  - (Cmd) `python scripts/convert_pdebench_multimodal.py burgers1d --limit 5 --samples 200`
  - (Cmd) `python scripts/upload_artifact.py burgers1d_mini_v1 dataset artifacts/burgers1d_mini_v1.tar.gz`
  - Update `docs/dataset_registry.yaml` with new artifact ID & splits.
- [ ] Train 128-dim model on Burgers subset
  - (Cmd) `WANDB_DATASETS="burgers1d_mini_v1" bash scripts/run_remote_scale.sh`
  - (DoD) Operator loss < baseline; W&B run logged with metrics & eval.
- [ ] Train on Advection subset and combined mix
  - (Cmd) `WANDB_DATASETS="advection1d_subset_v1" bash scripts/run_remote_scale.sh`
  - (DoD) Mix config evaluated; metrics stored, decoded previews checked in.

### Phase 2 — Multi-Viscosity + Navier–Stokes
- [ ] Convert & upload `navier_stokes2d_subset_v1`
- [ ] Launch scaled run with `latent.dim=256`, tokens 128, PDE-T depth [2,2,2]
  - Override with `configs/scale_overrides.md` settings.
- [ ] Evaluate multi-task performance; log per-task RMSE in W&B summary.
- [ ] Sweep latent dim / depth (W&B sweep) to find sweet spot.

### Phase 3 — Mesh & Particle Integration
- [ ] Convert `darcy2d_mesh_subset_v1` (Zarr) and upload artifact
- [ ] Convert `particles_advect_subset_v1` and upload artifact
- [ ] Train grid+mesh+particle mix with adjusted encoders
  - Ensure mesh/particle reconstruction tests pass.
- [ ] Extend evaluation configs to generate mesh/particle visuals; confirm W&B assets present.

### Phase 4 — Full PDEBench + Extensions
- [ ] Plan full dataset storage (≈3.6 TiB) in object store; document access.
- [ ] Stage full-grid training run with latent dim ≥384 and deep PDE-T
- [ ] Introduce external datasets (e.g., climate) as new artifacts & configs
- [ ] Compile SOTA benchmark report comparing against FNO/UNet/DiT baselines.

### Continuous Ops
- [ ] Maintain artifact registry (`docs/dataset_registry.yaml`) with version history
- [ ] Keep bootstrap scripts ready for Vast.ai (`scripts/load_env.sh`, `run_remote_scale.sh`)
- [ ] After each major run, download metrics via `scripts/fetch_wandb_metrics.py` and update reports.
