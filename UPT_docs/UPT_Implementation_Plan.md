
# Universal Physics Transformers (UPT) — Implementation Plan for a Coding Agent

> A step‑by‑step, production‑ready plan to reproduce, extend, and evaluate UPT across steady, transient, and Lagrangian settings, aligned with the official paper and codebase.

---

## 0) References used in this plan
- UPT paper (arXiv:2402.12365) — latent rollout training procedure, encoder/approximator/decoder roles, and pseudo‑code: 【108751339527844†L570-L600】【108751339527844†L519-L529】【108751339527844†L3745-L3890】【108751339527844†L3891-L3974】
- Paper experiments & model config highlights (transient flow section): 【108751339527844†L728-L746】【108751339527844†L807-L812】
- GitHub repo overview/entry points: README & setup docs: 【509792645433194†L9-L21】【536006903008693†L6-L27】【536006903008693†L101-L113】
- Dataset generation & preprocessing (ShapeNet‑Car, Transient Flow, Lagrangian): 【166661166415557†L1-L13】【166661166415557†L41-L59】【166661166415557†L80-L87】【166661166415557†L96-L104】
- Key code modules for encoder/decoder: `encoders/cfd_gnn_pool_transformer_perceiver.py`, `decoders/cfd_transformer_perceiver.py`: 【632897299919862†L15-L27】【632897299919862†L83-L91】【897538948798651†L52-L60】【897538948798651†L84-L92】

> Keep these citations if you copy sections elsewhere; they map to the exact lines examined.

---

## 1) Goals & scope
- **Reproduce** core UPT results (steady ShapeNet‑Car; transient OpenFOAM pipe; Lagrangian datasets) with faithful configs.
- **Package** a modular training stack that supports **latent rollouts** and **inverse encoding/decoding losses** for fast inference and stable long‑horizon rollout【108751339527844†L570-L600】.
- **Extend** to new domains by swapping encoders/decoders while keeping the **approximator** (latent transformer) unchanged【108751339527844†L519-L529】.
- **Instrument** with robust experiment tracking, checkpointing, and eval dashboards (W&B or local).

**Success criteria**
- Rollout stability (correlation time; error drift) on transient flow comparable to paper【108751339527844†L728-L746】.
- Latent‑space efficiency: ≤1k latent tokens (typical 256–512) with competitive error.
- End‑to‑end scripts reproducible from one YAML per run.

---

## 2) Environment & repo layout (automatable)
1. **Create env (Conda)**  
   ```bash
   conda env create --file src/environment_linux.yml --name upt
   ```
   If PyTorch/pyg wheels mismatch, install per setup notes【536006903008693†L19-L27】.
2. **Static paths**: copy and edit `template_static_config_github.yaml` → `static_config.yaml` (datasets, logs, W&B mode)【536006903008693†L34-L58】.
3. **W&B config** (optional): place `wandb_configs/cvsim.yaml`, edit entity/project; all provided YAMLs default to `wandb: cvsim`【536006903008693†L68-L74】.
4. **SLURM (optional)**: create `sbatch_config.yaml` and `template_sbatch_nodes.sh`【536006903008693†L76-L85】.
5. **Run entrypoints**:  
   - Single run: `python main_train.py --devices 0,1 --hp yamls/stage2/l16_mae.yaml`【536006903008693†L109-L113】  
   - Many runs: `python main_run_folder.py --devices 0 --folder yamls_run`【536006903008693†L124-L129】  
   - SLURM queue: `python main_sbatch.py --time 24:00:00 --nodes 4 --hp yamls/stage3/l16_mae.yaml`【536006903008693†L121-L123】

**Agent task:** verify CUDA, torch, and pyg install; render `static_config.yaml` from a template with your paths; test `--devices` discovery.

---

## 3) Data pipeline (download/generate → preprocess → stats → shards)

### 3.1 ShapeNet‑Car (steady)
- **Download & unpack**: commands in `SETUP_DATA.md`【166661166415557†L1-L13】.
- **Preprocess**:  
  ```bash
  python data/shapenetcar/preprocess.py --src <SRC> --dst <DST>
  ```
  【166661166415557†L35-L39】

### 3.2 Transient Flow (OpenFOAM)
- **Requirements**: OpenFOAM v2306, MPI, Python deps【166661166415557†L48-L59】.
- **Generate**:  
  ```bash
  cd data/transientflow
  python generateCase.py n_objects n_cases n_cores empty_case_dir target_dataset_dir working_dir
  ```
  【166661166415557†L64-L76】
- **Preprocess to `.th` + fp16** & **compute normalization stats**:  
  ```bash
  python data/transientflow/cfddataset_from_openfoam.py --src <OPENFOAM_OUT> --dst <POSTPROC> --num_workers 50
  python data/transientflow/cfddataset_norm.py --root <POSTPROC> --q 0.25 --exclude_last <N_VAL+N_TEST> --num_workers 50
  ```
  【166661166415557†L80-L94】

### 3.3 Lagrangian datasets
- Auto-downloaded to `data/lagrangian_dataset` (ensure dir exists)【166661166415557†L96-L104】.

**Agent tasks (automate):**
- Validate dataset presence; if missing, run the exact commands above.
- Materialize **train/val/test** JSONs with file lists and cached stats.
- Create **small proxies** (1–2% samples) for quick iteration.

---

## 4) Model system design (UPT blocks)

### 4.1 Encoder (mesh/point input → supernodes → latent tokens)
- Build radius/edge graph, pool to **ns supernodes**, encode with **GNN + transformer**, then **perceiver pooling** to **n_latent tokens**【108751339527844†L3820-L3888】.  
- Reference implementation: `CfdGnnPoolTransformerPerceiver` with GNN pooling → prenorm transformer → perceiver pooling【632897299919862†L45-L56】【632897299919862†L68-L87】.

**Config knobs**
- `num_supernodes` (e.g., 512–2048), `num_latent_tokens` (256–1024), `enc_dim`, `enc_depth`, `enc_num_attn_heads`.

### 4.2 Approximator (latent transformer)
- Propagate latent forward in time with a **stack of transformer blocks** (∆t per step, applied repeatedly for rollouts)【108751339527844†L519-L529】【108751339527844†L3918-L3924】.
- Keep **n_latent** fixed; this is where speedup comes from (vs. token count ∝ mesh points)【108751339527844†L605-L612】.

**Config knobs**
- `latent_dim` (e.g., 192–512), `depth` (4–12), heads (3–8), drop‑path, rotary/pos‑enc.

### 4.3 Decoder (latent → queries at arbitrary positions)
- Encode **query positions** with MLP + positional enc and cross‑attend to latent tokens【108751339527844†L3961-L3974】.  
- Reference implementation: `CfdTransformerPerceiver` (perceiver attends latent to queries)【897538948798651†L106-L114】.

**Config knobs**
- `perc_dim`, `perc_num_attn_heads`, optional clamp, optional conditioning tokens.

---

## 5) Training objective & latent rollout

### 5.1 Forward losses
- **Prediction loss** at queried points/times (e.g., MSE/MAE).

### 5.2 Inverse reconstruction losses (critical for latent rollout)
- **Inverse encoding**: reconstruct inputs from encoded latent via decoder at sampled input positions.  
- **Inverse decoding**: reconstruct latent from decoder outputs at sampled positions, forcing a disentangled E/A/D partition【108751339527844†L581-L600】.

> This enables **latent rollout** at inference: encode **u₀** → step approximator in latent space → decode at desired times/locations【108751339527844†L572-L580】.

### 5.3 Regularization
- Latent norm penalty; feature clamping (log clamp exists in decoder)【897538948798651†L115-L120】.
- Optional physics priors (divergence/continuity penalties) and spectral losses (for flows).

---

## 6) Run scaffolding & automation

### 6.1 YAML schema (examples exist in `src/yamls/*`)
- Dataset: paths, normalization, sampling of **query** points.
- Model: encoder/approximator/decoder section with dims/depth/heads/tokens.
- Trainer: optimizer (AdamW), schedule (cosine/warmup), AMP, grad‑accum, EMA.
- Logging: W&B mode/project, eval cadence, checkpoint path.

### 6.2 CLI wrappers (agent‑friendly)
- `scripts/prepare_data.sh` — idempotent data prep for each dataset.
- `scripts/run_small.sh` — tiny proxy runs (≤30 min) to validate plumbing.
- `scripts/run_full.sh` — full configs for each benchmark.
- `scripts/sweep.py` — hyperparam sweeps (grid/Random/Optuna).

### 6.3 Repro checkpoints
- Save every N epochs + best‑val; store full config, git SHA, and normalization stats with the checkpoint.

---

## 7) Hyperparameters & recommended grids

**Encoder**
- `num_supernodes`: 512, 1024, 2048 (transient results mention 2048)【108751339527844†L807-L812】
- `num_latent_tokens`: 256, 512
- `enc_dim`: 192, 256, 384; `enc_depth`: 4, 8; heads: 4, 6

**Approximator**
- `latent_dim`: 192 (UPT‑17M) / 384 (UPT‑68M)【108751339527844†L2892-L2899】
- `depth`: 4, 8, 12; heads: 3–8; drop‑path: 0–0.2

**Decoder**
- `perc_dim`: match `latent_dim` or ±25%
- `perc_heads`: 4–8

**Training**
- batch size: fit VRAM (AMP on); cosine LR with warmup; wd: 0.01; EMA 0.999
- query samples per batch: 2–8k points (balance compute vs. signal)

---

## 8) Evaluation suite
- **Steady (ShapeNet‑Car)**: MSE on pressure over surface points; runtime & memory vs. baselines【108751339527844†L712-L726】.
- **Transient (OpenFOAM)**: MSE and **correlation time**; discretization convergence vs. #input/output points (robust generalization)【108751339527844†L790-L803】.
- **Lagrangian**: vector field metrics, kinetic/energy drift, trajectory error.

**Rollout protocols**
- Encode initial state, latent step for T steps, decode at each step; plot error drift and calibration.

---

## 9) Minimal code stubs (to implement if starting fresh)

> These mirror the paper pseudo‑code blocks and the reference modules.

```python
# encoder: points → supernodes → latent tokens  【108751339527844†L3820-L3888】
class Encoder(nn.Module):
    def __init__(self, gnn_dim, enc_dim, perc_dim, ns, nlatent, depth, heads):
        ...
    def forward(self, feats, pos, edges=None, batch_idx=None, cond=None):
        # 1) pool to supernodes (GNN) → (B, ns, gnn_dim)
        # 2) transformer blocks on supernodes → (B, ns, enc_dim)
        # 3) perceiver pooling to latent tokens → (B, nlatent, perc_dim)
        return latent

# approximator: latent → latent  【108751339527844†L3918-L3924】
class Approximator(nn.Module):
    def __init__(self, latent_dim, depth, heads):
        ...
    def forward(self, z_t, cond=None):
        # transformer stack on latent tokens
        return z_t1

# decoder: latent + query positions → field  【108751339527844†L3961-L3974】
class Decoder(nn.Module):
    def __init__(self, latent_dim, perc_dim, heads):
        ...
    def forward(self, z, query_pos, cond=None):
        # pos-MLP → q; cross-attend q to z; predict field
        return u_hat
```

---

## 10) Training loop (with inverse losses & latent rollout)

1. **Forward prediction**: sample `(u_t, x_query, t_query)` → encode `u_t` to `z_t` → decode → `L_pred`.
2. **Inverse encoding**: decode selected `z_t` at `k` input positions, reconstruct `u_t` → `L_inv_enc`【108751339527844†L585-L597】.
3. **Inverse decoding**: from decoder outputs at `k'` positions, reconstruct `z_t` → `L_inv_dec`【108751339527844†L592-L600】.
4. **Latent step**: `z_{t+∆t} = A(z_t)`; optionally multiple steps per batch for curriculum.
5. **Total loss**: `L = L_pred + λ1*L_inv_enc + λ2*L_inv_dec + λ_reg`.
6. **EMA** update, AMP scaler, grad clip.
7. **Eval**: periodic long rollouts; save best/baseline‑matched checkpoints.

---

## 11) Agent playbook (checklist)

- [ ] Resolve env; print torch/cuda/pyg versions; run a 1‑batch dry‑run.
- [ ] Prepare datasets; compute normalization; create small proxy splits.
- [ ] Instantiate encoder/approximator/decoder per YAML; assert tensor shapes.
- [ ] Implement inverse losses; verify they decrease on proxies.
- [ ] Tune token counts (`ns`, `n_latent`) to meet VRAM/time targets.
- [ ] Validate rollout stability (correlation time) on transient flows.
- [ ] Run sweeps (depth/heads/dim) on proxies; lock best; scale up.
- [ ] Save & tag checkpoints; push metrics to dashboard; export plots.

---

## 12) Notes on extensions
- **Conditioning**: incorporate Reynolds/inflow speed as conditioning tokens; reference Dit* blocks in the codebase【632897299919862†L57-L66】.
- **Static tokens**: optional static geometry tokens (see commented hooks in encoder)【632897299919862†L103-L107】.
- **Physics priors**: divergence penalties, spectral regularization, boundary losses.
- **Continuous time**: replace fixed‑step transformer with ODE/RNN in latent; learn ∆t embedding.
- **Multi‑domain**: swap encoders/decoders; keep approximator constant.

---

## 13) Reproduction targets (sanity)
- **ShapeNet‑Car**: competitive MSE with small latent token counts; track cost vs. performance【108751339527844†L712-L726】.
- **Transient flow**: improved MSE/correlation time versus baselines; robustness to unseen #points【108751339527844†L790-L803】.

---

*End of plan.*
