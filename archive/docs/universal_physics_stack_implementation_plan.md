# Universal Physics Stack — Implementation Plan for a Coding Agent

A production-grade blueprint to implement a **unified simulator** that combines:
- **Discretization-agnostic latent I/O** (UPT-style encoder + any-point decoder)
- **PDE-Transformer compute core** (shifted-window attention + U-shape + channel-separated tokens)
- **Latent operator drift** for fast transients + **few-step diffusion residual** for uncertainty/correction
- **Steady-state latent diffusion prior**
- **Physics guards** (projections, bound-preserving updates) and **multiphysics factor-graph coupling**
- **Particles/contacts** with symplectic updates
- **Latent-space data assimilation (DA)** and **safe control**
- **Symbolic discovery** and **active learning**

This plan is written for an autonomous coding agent. It specifies folder layout, component APIs, data schemas, training & inference pipelines, evaluation gates, and acceptance tests.

---

## 0) Scope & Operating Principles
- **Single latent interface.** All discretizations map to a fixed-length latent `z ∈ R^{T×D}`. All heavy compute occurs in latent.
- **Separation of concerns.** Encode/Decode ⟂ Evolve ⟂ Correct ⟂ Enforce.
- **Conservation-first gates.** Every rollout step logs mass/energy/charge budgets and BC violations; corrective projections are triggered by thresholds.
- **Few-step generative corrections.** Diffusion/flow components are distilled to 1–2 steps for runtime practicality.
- **Unit-aware, non-dimensional training.** All fields and parameters pass through π-group transformations.

---

## 1) Repository Layout
```
repo/
  pyproject.toml
  README.md
  configs/
    defaults.yaml
    train_multi_pde.yaml
    inference_transient.yaml
    inference_steady.yaml
  src/
    core/
      latent_state.py
      conditioning.py
      blocks_pdet.py
      shifted_window.py
      ushape.py
      channel_tokens.py
    io/
      enc_grid.py
      enc_mesh_particle.py
      decoder_anypoint.py
      queries.py
    models/
      latent_operator.py
      diffusion_residual.py
      steady_prior.py
      physics_guards.py
      multiphysics_factor_graph.py
      particles_contacts.py
    training/
      losses.py
      flow_matching.py
      consistency_distill.py
      curricula.py
      loop_train.py
      ema_clip.py
    data/
      schemas.py
      datasets.py
      transforms.py
      collate.py
    inference/
      rollout_transient.py
      solve_steady.py
      da_latent.py
      control_safe.py
    eval/
      metrics.py
      calibration.py
      gates.py
      reports.py
    discovery/
      nondim.py
      symbolic_fit.py
    active/
      acquisition.py
      mf_calibration.py
  tests/
    unit/
    integration/
  scripts/
    prepare_data.py
    train.py
    evaluate.py
    infer.py
```

---

## 2) Environment & Dependencies
- Python ≥ 3.10, PyTorch ≥ 2.3, CUDA where available; JAX optional for PINO comparisons
- `einops`, `torchvision`, `pyg` (optional for mesh/graph), `trimesh`, `pykeops` (neighbor ops), `h5py`, `zarr`, `pyvista` (viz)
- `hydra`/`omegaconf` for configs; `wandb`/`mlflow` for logging
- `sympy`, `pysr`, `pysindy` for discovery; `netCDF4`/`xarray` for grids

**Action:** `scripts/prepare_env.sh` to pin versions and set deterministic seeds.

---

## 3) Configuration System (Hydra/OmegaConf)
Example `configs/defaults.yaml`:
```yaml
seed: 17
precision: bf16
accelerator: gpu
max_batch_tokens: 1.5e6

latent:
  len: 512           # number of tokens
  dim: 256           # token dimension
  time_dim: 64

pdet:
  depths: [2,2,4,2]  # U-shape depths per scale
  win: 8             # shifted-window size
  shift: true
  channel_sep: true
  adaln: true

encoders:
  grid:
    patch: 4
    pixel_unshuffle: true
  mesh_particle:
    supernodes: 2048
    mp_layers: 3

operator:
  steps: 1
  inverse_losses: {enc_w: 0.1, dec_w: 0.1}

diffusion:
  type: flow_matching
  steps_train: 30
  steps_infer: 2
  guidance: {residual_w: 0.3}

steady_prior:
  steps_infer: 8

physics:
  projections: {helmholtz: true, positivity: [rho, kappa]}
  bc_clamp: true
  budgets: {mass: true, energy: true, charge: false}

mp_coupling:
  iters: 3
  tol: 1e-4

training:
  optimizer: adamw
  lr: 3e-4
  wd: 1e-2
  sched: cosine
  ema: {beta: 0.999}
  clip_ema: {enabled: true}
  losses:
    one_step: 1.0
    rollout: 0.25
    spectral: 0.1
    conservation: 0.1
    tv_edge: 0.05

logging:
  wandb: true
```

---

## 4) Data Schemas & Dataloaders

### 4.1 Physics Sample Schema (`data/schemas.py`)
```python
class Sample(TypedDict):
    # discretization
    kind: Literal['grid','mesh','particles']
    coords: Float[Tensor, 'N d']         # xyz per point/node/particle
    connect: Optional[Int[Tensor, 'E 2']]  # edges for mesh/graph
    fields: Dict[str, Float[Tensor, 'N C_f']]  # e.g., {'u':..., 'v':..., 'p':...}
    bc: Dict[str, Any]                    # boundary descriptors (Dirichlet/Neumann/periodic)
    params: Dict[str, float]              # PDE parameters (Re, Ma, ν, κ, ε_r, ...)
    geom: Optional[Dict[str, Any]]        # geometry descriptors (SDFs, tags)
    time: Float[Tensor, '']               # scalar t
    dt: Float[Tensor, '']                 # Δt
    meta: Dict[str, Any]                  # identifiers, domain extents, units
```
Storage formats: **Zarr/HDF5** with chunking per field; Parquet for metadata tables.

### 4.2 Non-dimensionalization (`discovery/nondim.py`)
- Read unit tags, compute π-groups; store scaling constants per-sample.
- Inverse-transform utilities for post-processing/export.

### 4.3 Dataloaders (`data/datasets.py`)
- Grid datasets: load `xarray`/NetCDF; optional down/up-sampling.
- Mesh/particles: build neighbor lists (radius/kNN) with `pykeops`; cache supernode pooling maps.
- Compose transforms: random rotations (SO(2/3) if appropriate), mirror symmetries, random Δt, BC jitter, noise.

---

## 5) Core Abstractions & APIs

### 5.1 Latent State (`core/latent_state.py`)
```python
@dataclass
class LatentState:
    z: Float[Tensor, 'T D']       # tokens
    t: Optional[Float[Tensor, '']] = None  # time scalar
    cond: Dict[str, Tensor] = field(default_factory=dict)  # conditioning embeddings
```

### 5.2 Conditioning (`core/conditioning.py`)
- Map PDE type, BC classes, parameters, geometry tokens to embeddings.
- `AdaLN` module: returns scale, shift, gate for each block.

### 5.3 Encoders
- **GridEncoder (`io/enc_grid.py`)**
  - PixelUnshuffle → patch tokens; channel-separated streams; local conv stem.
  - Optional Fourier features of coordinates.
- **MeshParticleEncoder (`io/enc_mesh_particle.py`)**
  - Message passing over radius-graph (3 layers) → supernode pooling (k-means or FPS+agg) → Perceiver pooling to `latent.len` tokens.
  - Save inverse maps for inverse-encode loss.

### 5.4 PDE-Transformer Blocks (`src/core/blocks_pdet.py`)
- U-shape with down/up scales; shifted-window attention per scale (`shifted_window.py`).
- Channel-separated tokens (one stream per physical variable) + axial channel attention bridges.
- `AdaLN` conditioning hooks.

### 5.5 Any-Point Decoder (`io/decoder_anypoint.py`)
- Inputs: query positions `Q: [M, d]`, time, cond, latent `z`.
- Perceiver cross-attention: queries attend to latent → MLP head per variable.
- Batchable; supports irregular outputs and mesh transfer.

### 5.6 Latent Operator (`models/latent_operator.py`)
- Stack of PDE-T blocks; interface:
```python
class LatentOperator(nn.Module):
    def forward(self, state: LatentState, dt: Tensor) -> LatentState:
        # returns next latent; uses AdaLN-cond on dt/params/BC
```
- **Inverse losses**: heads to reconstruct (a) inputs from latent, (b) latent from decoded outputs.

### 5.7 Diffusion Residual (Flow Matching / Consistency) (`models/diffusion_residual.py`)
- Train a vector field `s_phi(z, t, cond)` for flow matching; export a 1–2 step **corrector**.
- Guidance term: PDE residual of decoded field as extra force.

### 5.8 Steady-State Latent Prior (`models/steady_prior.py`)
- Conditional flow/diffusion over latent `z* | BC, params`; supports 6–12 step sampling; distill to ~4–8 when possible.

### 5.9 Physics Guards (`models/physics_guards.py`)
- **Helmholtz–Hodge projection** on grids (FFT-based) and meshes (Poisson solve via cotan-Laplacian).
- **Positivity clamping** (log-exp) for variables with bounds.
- **Interface flux matching** with small QP or projected gradient for conservation across domains.

### 5.10 Multiphysics Factor Graph (`models/multiphysics_factor_graph.py`)
- Nodes = domain models; edges = interfaces with port variables (fluxes, potentials).
- Message passing K iterations; stop when residual < tol or K reached.

### 5.11 Particles & Contacts (`models/particles_contacts.py`)
- Hierarchical neighbor search; learned forces with **symplectic** integrator (velocity Verlet).
- Constraint projection (PBD/PGS) for contacts/joints per step.

---

## 6) Training Pipelines

### 6.1 Multi-PDE Pretraining (operator backbone)
- Mix grids/meshes/particles batches.
- Losses: one-step velocity (nRMSE), rollout N-step (teacher-forced), spectral loss, conservation penalties, inverse-encode/dec.
- Curriculum: start with small Δt, periodic BCs; add harder BCs and shocks; introduce mesh/particle heterogeneity by epoch.

### 6.2 Flow-Matching for Diffusion Residual
- Sample `(z, z_target)` from operator rollouts or from teacher simulators.
- Train flow `s_phi` to carry `z`→`z_target` across synthetic time τ∈[0,1].
- Consistency distill to 1–2 evaluator steps; add residual-guidance on decoded PDE residuals.

### 6.3 Steady-State Prior Training
- Dataset of BVP solutions (various domains/BCs/params).
- Conditional flow/diffusion; guidance with residual norm; occasional projection step during training to stabilize.

### 6.4 Mixed-precision, EMA, Gradient Clipping
- Use bf16; EMA of weights; EMA-based gradient clipping to avoid DiT spikes.

### 6.5 Checkpointing & Resumption
- Save model, optimizer, EMA, scheduler, scaler; keep top-K by validation **conservation gap** and **BC violation** (primary), then nRMSE.

---

## 7) Inference Modes

### 7.1 Transient Rollout (Predictor–Corrector)
1. Encode state → `z_t`.
2. **Predictor**: `z_pred = f_theta(z_t, dt)`.
3. **Corrector** (every k steps or if gate triggers): apply 1–2 steps of `s_phi` with residual-guidance.
4. Decode on-demand with any-point queries; apply physics projections.
5. Update budgets; if thresholds exceeded, escalate corrector strength or hand off to classical solver.

### 7.2 Steady-State Solve
1. Encode BCs/params into cond.
2. Sample latent from **steady prior** (4–8 steps); decode.
3. Apply projections; run a short residual-corrector if needed.

### 7.3 Latent DA (EnKF/4D-Var)
- Maintain ensemble in latent; update with sensor queries through decoder; propagate with operator + occasional corrector.

### 7.4 Safe Control
- MPC in latent; **control barrier function** (CBF) checks on decoded fields; reject/clip actions that violate safety.

---

## 8) Evaluation, Metrics, and Gates
- **nRMSE**, **MAE**, **spectral error** (per-band), **SSIM** (if images), **calibration (ECE)** for UQ.
- **Conservation gaps** (mass/energy/charge), **BC violation** (L2 over boundary), **divergence norms**.
- **Runtime/Joule per step**.

**Gates:** If conservation or BC violation exceeds thresholds → projection + corrector; if still failing → fallback solver.

---

## 9) Losses (definitions)
- `L_one_step = ||x_{t+dt} - \hat{x}_{t+dt}||^2`
- `L_rollout = mean_{i≤N} ||x_{t+i dt} - \hat{x}_{t+i dt}||^2`
- `L_spec = Σ_b w_b ||FFT_b(x) - FFT_b(\hat{x})||^2`
- `L_cons = α_mass|ΔM| + α_energy|ΔE| + ...`
- `L_tv_edge` for shocks/discontinuities
- `L_inv_enc`, `L_inv_dec` for inverse-encode/decode
- Diffusion/flow matching with consistency loss for few-step student

---

## 10) Pseudocode Snippets

### 10.1 Shifted-Window Attention (sketch)
```python
def swin_attn(x, win=8, shift=True):
    x = partition_windows(x, win, shift)
    q,k,v = linear(x)
    attn = softmax(q @ k.transpose(-2,-1) / sqrt(d) + relpos_bias())
    y = attn @ v
    return merge_windows(linear_out(y), win, shift)
```

### 10.2 Any-Point Decoder
```python
def decode_any(points, latent, cond):
    q = posenc(points)
    for _ in range(L):
        q = cross_attend(q, latent, cond)  # perceiver-style
        q = mlp(q)
    return heads(q)  # per-variable
```

### 10.3 Operator + Few-Step Corrector
```python
def step(z, dt, cond):
    z_pred = f_theta(z, dt, cond)
    if gate_triggered():
        z_corr = z_pred
        for _ in range(steps):
            z_corr = z_corr + eps * s_phi(z_corr, cond, guide=residual)
        return z_corr
    return z_pred
```

### 10.4 Helmholtz–Hodge Projection (grid)
```python
def hhp_grid(u):
    phi = solve_poisson(div(u))
    return u - grad(phi)
```

### 10.5 Multiphysics Factor Graph (message passing)
```python
def couple(domains, interfaces):
    for _ in range(K):
        for e in interfaces:
            f = compute_flux(e.left, e.right)
            e.left.apply_flux(f)
            e.right.apply_flux(-f)
        if global_residual() < tol: break
```

### 10.6 Latent EnKF
```python
def enkf_update(ensemble, obs, H):
    Z = stack([z for z in ensemble])
    Y = H(decode(Z))
    K = cov(Z,Y) @ pinv(cov(Y,Y)+R)
    return [z + K @ (obs - y) for z,y in zip(Z,Y)]
```

---

## 11) CLI & API
- `scripts/train.py +configs/train_multi_pde.yaml`
- `scripts/infer.py mode=transient input=...` → saves rollout + metrics
- `scripts/infer.py mode=steady bc=...` → saves field + budgets
- `scripts/evaluate.py checkpoint=...` → full report (PDF/HTML)

---

## 12) Tests & Acceptance Criteria
**Unit tests:** tensor shapes, window shifts, projection correctness (manufactured solutions), conservation on simple cases.

**Integration tests:**
- Cross-mesh transfer: encode (mesh A) → decode (mesh B) with ≤ε error on a harmonic field.
- Predictor–corrector stability: long horizon on Burgers/Kolmogorov with bounded energy drift after projections.
- Steady-state: BC → steady prior → projection achieves ≤ε residual.
- DA: latent EnKF reduces error vs no-DA baseline on sparse sensors.

**Acceptance:** Pass gates on conservation/BC/spectral error on held-out regimes; calibration ECE ≤ target; runtime within configured budget.

---

## 13) Logging & Reporting
- Per-step budgets; residual histograms; spectra; calibration plots; gate activations.
- Save **decision trace** when gates trigger projections/correctors.

---

## 14) Risks & Mitigations
- **Diffusion too slow** → distill to 1–2 steps; schedule corrector sparsely.
- **BC violations** → increase clamp/projection weight; refine decoder.
- **Geometry shift** → augment encoders; enlarge supernode count; finetune encoders only.
- **Contact instability** → reduce time step; stronger constraint projection; symplectic integrator.

---

## 15) Extension Hooks
- Plug-in residual solvers (FFT, multigrid, AMG).
- Add symmetry/equivariance wrappers (E(n)-equivariant attention) when needed.
- Export ONNX/TensorRT inference graph for latent operator.

---

## 16) Glossary (selected)
- **Any-point decoder**: Query-based decoder that predicts fields at arbitrary coordinates.
- **Flow matching / consistency**: Training schemes to learn ODE flows enabling few-step sampling.
- **Factor graph**: Graphical model representing multiphysics subsystems and their interface constraints.
- **Projection**: Post-processing step enforcing constraints (e.g., divergence-free, bounds, flux continuity).

---

## 17) Interleaved Milestones + Checklist (Authoritative, Explicit Steps)
> Rewritten with **numbered, non-collapsible steps** so the agent can execute line-by-line. Each step includes **Where / What / Command / DoD**. Mark each item with ✅ once complete.

---

### M0 — Bootstrap & Repository Scaffold
**M0.1** Where: `repo/` — What: initialize repo, git, and basic files.  
Command: `git init && mkdir -p src tests scripts configs data export reports .github/workflows`  
DoD: `git status` shows tracked dirs; main branch created.

**M0.2** Where: `pyproject.toml` — What: pin deps (`torch>=2.3`, `einops`, `xarray`, `zarr`, `trimesh`, `pykeops`, `h5py`, `wandb`, `hydra-core`, `ruamel.yaml`, `pytest`, `ruff`, `black`).  
Command: create file; run `pip install -e .`  
DoD: `python -c "import torch,einops"` succeeds.

**M0.3** Where: `.pre-commit-config.yaml` — What: ruff/black/isort hooks.  
Command: `pre-commit install`  
DoD: `pre-commit run --all-files` passes.

**M0.4** Where: `.github/workflows/ci.yaml` — What: CI to run `pytest -q tests/unit`.  
DoD: CI green on first push.

**M0.5** Where: `scripts/prepare_env.sh` — What: deterministic flags and pinned installs.  
Command: `bash scripts/prepare_env.sh`  
DoD: `python - <<'PY'
import torch; torch.use_deterministic_algorithms(True); print('OK')
PY` prints OK.

**M0.6** Where: `README.md` — What: quickstart with commands to run M1–M3.  
DoD: Human can bootstrap without assistance.

---

### M1 — Data Schema & Non‑Dimensionalization
**M1.1** Where: `src/data/schemas.py` — What: implement `Sample` TypedDict with fields: `kind, coords, connect, fields, bc, params, geom, time, dt, meta`.  
DoD: `tests/unit/test_schema.py::test_required_fields` passes.

**M1.2** Where: `src/discovery/nondim.py` — What: implement `to_pi_units(sample)`, `from_pi_units(sample)`; carry π-groups and scaling in `sample.meta['scale']`.  
DoD: `tests/unit/test_nondim.py::test_roundtrip_error < 1e-6`.

**M1.3** Where: `data/metadata.parquet` — What: write table with columns `id, split, pde, bc, geom, units_json`.  
DoD: 80/10/10 split respected ±1%.

---

### M2 — Minimal PoC Datasets
**M2.1** Where: `scripts/prepare_data.py` — What: generator for **Diffusion2D, Burgers2D, Kolmogorov2D** on 256×256 grids; write Zarr chunks `(C, H, W)` with Blosc.  
Command: `python scripts/prepare_data.py +dataset=poc_trio out=data/poc_trio.zarr`  
DoD: `reports/samples_grid.png` saved; loader latency < 1s/batch.

**M2.2** Where: `data/mesh_poisson.zarr` — What: unstructured Poisson meshes (2.5k–20k nodes); store coords, edges, cotan Laplacian.  
DoD: `tests/unit/test_mesh_loader.py` passes; peak RAM < 2GB.

**M2.3** Where: `data/particles_advect.zarr` — What: 10k–100k particles with velocities; neighbor caches (radius graph).  
DoD: neighbor build < 100 ms/10k; file size < 1GB/100k.

---

### M3 — Encoders & Any‑Point Decoder
**M3.1** Where: `src/io/enc_grid.py` — What: `GridEncoder` (PixelUnshuffle p=4, per-channel streams, optional Fourier coord feats).  
DoD: identity reconstruction head MSE < 1e-3.

**M3.2** Where: `src/io/enc_mesh_particle.py` — What: 3× message passing → supernode pooling (`supernodes=2048`) → Perceiver pooling to `latent.len`.  
DoD: runtime < 50 ms for 10k nodes; identity MSE < 1e-3.

**M3.3** Where: `src/io/decoder_anypoint.py` — What: perceiver-style cross-attn decoder `decode(points, latent, cond)`.  
DoD: harmonic field rel‑err < 1e-3 at unseen coordinates.

---

### M4 — PDE‑Transformer Core
**M4.1** Where: `src/core/shifted_window.py` — What: `partition_windows/merge_windows`, shifted windows, log-spaced rel-pos bias.  
DoD: windowing unit tests pass; O(T) scaling ~linear for fixed window.

**M4.2** Where: `src/core/blocks_pdet.py` — What: U-shape down/up with skips; **channel-separated tokens** + axial channel attention; RMSNorm(Q,K).  
DoD: 1k forward/backward steps with random input → no NaNs; throughput ≥ UNet baseline.

**M4.3** Where: `src/core/conditioning.py` — What: AdaLN conditioning from PDE id, params, BCs, geometry into per-block scale/shift/gate.  
DoD: ablation shows ≥5% nRMSE gain over no-conditioning.

---

### M5 — Latent Operator (Fast Transients)
**M5.1** Where: `src/core/latent_state.py` — What: `LatentState{ z:T×D, t:scalar, cond:dict }`.  
DoD: serialization/deserialization round-trip works.

**M5.2** Where: `src/models/latent_operator.py` — What: `forward(state, dt)→state'` using PDE‑T stack with residual connection; time-step embedding.  
DoD: one-step avg nRMSE ≤ 0.06 on PoC trio.

**M5.3** Where: `src/training/losses.py` — What: add `L_inv_enc` & `L_inv_dec`; plug heads into enc/dec.  
DoD: 100‑step latent rollout stable; energy drift < 1e‑3/step.

**M5.4** Where: `src/training/loop_train.py` — What: rollout curriculum (increase N, mix PDEs/BCs).  
DoD: `reports/op_rollout_metrics.html` shows nRMSE@10 ≤ 0.40; spectral mid/high < 0.15.

---

### M6 — Diffusion Residual (Few‑Step Corrector)
**M6.1** Where: `src/models/diffusion_residual.py` — What: flow-matching vector field `s_φ(z, τ, cond)`; residual-guidance via PDE residual of decoded fields.  
DoD: 4‑step corrector reduces rollout nRMSE ≥20% on Kolmogorov vs operator-only.

**M6.2** Where: `src/training/consistency_distill.py` — What: train 1–2 step student to match multi-step teacher.  
DoD: retains ≥85% of teacher’s error reduction; runtime overhead ≤1.5× operator.

**M6.3** Where: `src/inference/rollout_transient.py` — What: gated predictor–corrector (trigger every k steps or by `eval/gates.py`).  
DoD: overall error strictly decreases vs operator-only; gate logs saved.

---

### M7 — Steady‑State Latent Prior
**M7.1** Where: `src/models/steady_prior.py` — What: conditional diffusion/flow over latent for BVP solutions (6–8 steps).  
DoD: residual norm ≤ solver tol on ≥95% PoC cases.

**M7.2** Where: `src/eval/calibration.py` — What: reliability diagrams + temperature scaling.  
DoD: ECE ≤ 0.05 on steady tasks.

---

### M8 — Physics Guards & Projections
**M8.1** Where: `src/models/physics_guards.py` — What: Helmholtz–Hodge projection; FFT-based (grid) and cotan-Laplacian (mesh).  
DoD: divergence L2 ↓ ≥99%; runtime < 5 ms @256².

**M8.2** Where: same — What: positivity clamps (log-exp) for {ρ, κ, ε}.  
DoD: zero negatives; accuracy loss ≤1%.

**M8.3** Where: `src/models/multiphysics_factor_graph.py` — What: interface flux continuity via small QP/projection.  
DoD: global energy error < 1e-3 over 100 steps on 2-domain toy.

---

### M9 — Multiphysics Factor Graph
**M9.1** Where: `src/models/multiphysics_factor_graph.py` — What: API for Domain nodes & Port edges; message passing K iters.  
DoD: converges < K to residual < tol.

**M9.2** Where: `configs/coupled_toy.yaml` & `scripts/infer.py` — What: run coupled demo (fluid–thermal).  
DoD: budgets within thresholds; report saved to `reports/coupled_demo.html`.

---

### M10 — Particles & Contacts
**M10.1** Where: `src/models/particles_contacts.py` — What: hierarchical neighbors (grid/tree), batched radii.  
DoD: 100k particles step without OOM; neighbor build < 30 ms.

**M10.2** Where: same — What: symplectic integrator (velocity Verlet) + PBD/PGS constraints.  
DoD: energy drift < 1% over 1k steps on elastic collision.

---

### M11 — Data Assimilation & Safe Control
**M11.1** Where: `src/inference/da_latent.py` — What: latent EnKF/4D-Var using decoder as H operator.  
DoD: posterior error ≥25% lower than prior with sparse sensors.

**M11.2** Where: `src/inference/control_safe.py` — What: MPC with control-barrier functions on decoded fields.  
DoD: zero safety violations; ≤10% cost overhead.

---

### M12 — Benchmarks, Ablations, and SOTA Gate
**M12.1** Where: `scripts/train_baselines.py` — What: train FNO, UNet, DiT under identical data/compute.  
DoD: baseline checkpoints saved.

**M12.2** Where: `scripts/evaluate.py` — What: compare metrics (nRMSE@10, spectral error, conservation gaps, BC violation, ECE, tokens/sec).  
DoD (SOTA PoC): our model ≥20% better nRMSE@10 and ≥2× tokens/sec; ECE ≤ 0.05; conservation/BC gaps ≤ baseline.

**M12.3** Where: `reports/ablations.html` — What: ablate (i) channel-sep, (ii) inverse losses, (iii) corrector.  
DoD: each ablation worsens ≥5% on ≥1 metric.

---

### M13 — Packaging, Export, and Repro
**M13.1** Where: `scripts/repro_poc.sh` — What: one-click pipeline (data→train→eval).  
DoD: fresh machine reproduces within tolerances.

**M13.2** Where: `export/` — What: TorchScript/ONNX export for operator+decoder; optional TensorRT.  
DoD: eager vs exported max diff < 1e-4; latency improved on GPU.

---

### Unified Runbook (Agent Commands)
1) `bash scripts/prepare_env.sh && pre-commit run --all-files`  
2) `python scripts/prepare_data.py +dataset=poc_trio out=data/poc_trio.zarr`  
3) `python scripts/train.py +config=train_multi_pde.yaml stage=operator`  
4) `python scripts/train.py +config=train_multi_pde.yaml stage=diff_residual`  
5) `python scripts/train.py +config=train_multi_pde.yaml stage=consistency_distill`  
6) `python scripts/train.py +config=train_multi_pde.yaml stage=steady_prior`  
7) `python scripts/evaluate.py ckpt=checkpoints/op_latest.ckpt mode=transient`  
8) `python scripts/evaluate.py ckpt=checkpoints/steady_latest.ckpt mode=steady`  
9) `python scripts/infer.py mode=transient input=examples/state.nc --save`  
10) `python scripts/infer.py mode=steady bc=examples/bc.json --save`

