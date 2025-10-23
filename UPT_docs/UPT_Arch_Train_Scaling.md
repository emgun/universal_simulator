
# UPT — Architecture, Hyperparameters, Training & Scaling Playbook

> A focused, coding‑agent‑ready guide for implementing **Universal Physics Transformers (UPT)** with an emphasis on the **model architecture**, **tunable hyperparameters**, **training procedure (incl. latent rollout)**, and **scaling strategies**. Citations point to the exact lines in the paper and official repo that this plan is based on.

---

## 1) Mental model of the architecture

**UPT = Encoder → Approximator → Decoder**:

- **Encoder**: consumes irregular meshes/point clouds, builds a graph, **pools to supernodes**, processes with transformer blocks, and **perceiver‑pools** into a **small, fixed number of latent tokens**【108751339527844†L3820-L3888】.  
  • The transient‑flow reference uses **ns = 2048 supernodes** and **n_latent = 512 tokens**【108751339527844†L807-L812】.  
  • Repo encoder combines **GNN pooling → transformer → perceiver pooling** (`CfdGnnPoolTransformerPerceiver`)【632897299919862†L45-L56】【632897299919862†L68-L87】.

- **Approximator**: a **transformer in latent space** that steps the compressed state forward by Δt; apply repeatedly for rollouts【108751339527844†L519-L529】. Pseudo‑code shows `latent_t+1 = approximator(latent_t)`【108751339527844†L3918-L3924】.

- **Decoder**: point‑wise evaluation at **arbitrary query positions**; encode positions with an MLP + positional embedding and **cross‑attend to latent tokens**【108751339527844†L3961-L3974】.  
  • Repo decoder (`CfdTransformerPerceiver`) projects latent → perceiver cross‑attends → predicts channels【897538948798651†L106-L114】.

---

## 2) Reference sizes & capacity tiers

- The appendix lists **approximator hidden dims** and **block counts** for small/medium/large models (e.g., **128/192/384** for UPT‑8M/‑17M/‑68M)【108751339527844†L2892-L2899】.  
- Practical takeaway: scale **latent_dim** and **depth** in the approximator; scale **ns** and **n_latent** in the encoder/decoder to trade off compute vs. fidelity.

---

## 3) Hyperparameters that matter most

> Prefer **coarse-to-fine sweeps**. Start with small proxies; lock trends; then scale.

### 3.1 Latent tokens & supernodes (geometry compression)
| Knob | Recommended values | Effect |
|---|---|---|
| `num_supernodes (ns)` | 512, **1024**, **2048** | Larger ns → better capture of geometry; ↑ encoder cost【108751339527844†L807-L812】 |
| `n_latent_tokens` | 256, **512**, 768 | Larger n_latent → more capacity at decoder/approximator boundary |

### 3.2 Approximator (latent transformer)
| Knob | Recommended values | Effect |
|---|---|---|
| `latent_dim` | 192, **256**, **384** | Core capacity of time propagation【108751339527844†L2892-L2899】 |
| `depth` | 4, 8, **12** | Longer context in latent evolution |
| `num_heads` | 4, **6**, 8 | Attend across latent tokens; ↑ cost ~heads |
| `drop_path` | 0.0–0.2 | Stabilize deep stacks |

### 3.3 Encoder/Decoder stacks
| Knob | Recommended values | Effect |
|---|---|---|
| `enc_dim` | 192–384 | Width over supernodes |
| `enc_depth` | 4–8 | More refinement pre‑perceiver |
| `perc_dim` | match latent_dim (±25%) | Smooth interface with latent |
| `perc_heads` | 4–8 | Query/kv mixing capacity |
| `clamp` (decoder) | off / small log‑clamp | Tame outliers【897538948798651†L115-L120】 |

**Conditioning** (optional): inject parameters (e.g., inflow/Re) via conditional blocks in encoder/decoder (see Dit* hooks)【632897299919862†L57-L66】.

---

## 4) Training procedure (with latent rollout)

### 4.1 Core losses
1. **Prediction loss** at query points: MSE/MAE over fields.  
2. **Inverse encoding/decoding losses** (critical):  
   - Invert **encoding** by decoding latent at input positions to reconstruct signals.  
   - Invert **decoding** by reconstructing latent from decoder outputs at sampled positions.  
   This **isolates responsibilities** of E/A/D and enables **latent‑only rollouts** at test time【108751339527844†L581-L600】.

**Why it matters**: You can step only in latent space during inference (fast), then decode where you want【108751339527844†L572-L580】.

### 4.2 Batch construction
- Sample **input nodes** (for encoding) and **query nodes** (for the loss).  
- For transient data, sample **short segments** (t, t+Δt, …) for curriculum; extend horizon as training stabilizes.

### 4.3 Optimizer & schedule
- **AdamW** (β1=0.9, β2=0.999, wd≈1e‑2), **cosine** LR with warmup (couple hundred to a few thousand steps).  
- **AMP**, gradient clipping (1.0), optional **EMA** (0.999) for stable eval.  
- Accumulate gradients to fit larger token counts.

### 4.4 Validation and rollouts
- Periodically run **long rollouts** in latent space; decode at each step; track **MSE** and **correlation time** stability curves【108751339527844†L790-L803】.

---

## 5) Scaling strategies

### 5.1 What to scale first
1. **Approximator capacity**: increase `latent_dim` then `depth`.  
2. **Interface bandwidth**: increase `n_latent_tokens`.  
3. **Geometry capture**: increase `num_supernodes`.  
4. **Decoder heads/dim** if queries are complex (many points/BC variety).

### 5.2 Staged scaling plan
- **Stage S0 (proxy)**: tiny dataset slice; ns=512, n_latent=256, latent_dim=192, depth=4. Purpose: **plumbing** + loss sanity.  
- **Stage S1 (mid‑scale)**: ns=1024, n_latent=512, latent_dim=256, depth=8. Add inverse losses; begin longer rollouts.  
- **Stage S2 (full)**: ns=2048, n_latent=512–768, latent_dim=384, depth=8–12. Turn on drop‑path; increase query points.  
- **Stage S3 (robustness)**: train on mixed discretizations/BCs; evaluate unseen point counts (as in discretization tests)【108751339527844†L796-L803】.

### 5.3 Efficiency tips
- **Token‑aware batching**: cap `B × (ns + n_latent) × dims` to fit VRAM.  
- **Query sub‑sampling**: train with 2–8k queries per batch; at test time, decode dense grids.  
- **Activation checkpointing** in the approximator when depth ≥ 8.  
- **Compile** (PyTorch 2.x) and fused optimizers where available.  
- **Mixed precision** end‑to‑end (fp16 or bf16).

### 5.4 When to change what
- **If long‑horizon error grows fast** → increase **approximator depth/dim**; add drop‑path; strengthen inverse losses.  
- **If steady error high but rollouts look stable** → increase **decoder perc_dim/heads** and **n_latent_tokens**.  
- **If geometry‑specific artifacts** → increase **num_supernodes**; add local message‑passing depth pre‑perceiver.  
- **If compute‑bound** → reduce queries per step, use curriculum on horizon, keep approximator width, and sacrifice ns first.

---

## 6) Minimal configs to try (by budget)

> Use these as starting points; tune around them.

- **Small (~8–12M params)**: `ns=512`, `n_latent=256`, `latent_dim=192`, `depth=4`, `heads=4`, `enc_dim=192`, `enc_depth=4`, `perc_dim≈latent_dim`.  
- **Medium (~17–30M)**: `ns=1024`, `n_latent=512`, `latent_dim=256`, `depth=8`, `heads=6`, `enc_dim=256`, `enc_depth=6`.  
- **Large (~68M)**: `ns=2048`, `n_latent=512–768`, `latent_dim=384`, `depth=8–12`, `heads=6–8`, `enc_dim=384`, `enc_depth=8`【108751339527844†L2892-L2899】【108751339527844†L807-L812】.

---

## 7) Checklists for the coding agent

**Architecture**
- [ ] Build encoder = GNN pooling → transformer blocks → perceiver pooling【632897299919862†L45-L56】【632897299919862†L68-L87】.  
- [ ] Build approximator = latent transformer (repeatable Δt step)【108751339527844†L519-L529】.  
- [ ] Build decoder = pos‑MLP for queries + perceiver cross‑attention【108751339527844†L3961-L3974】.

**Hyperparameters**
- [ ] Define grids: `{ns, n_latent, latent_dim, depth, heads, enc_depth, perc_dim}`.  
- [ ] Add conditioning tokens if needed (Dit* paths)【632897299919862†L57-L66】.

**Training**
- [ ] Implement `L_pred + λ1 L_inv_enc + λ2 L_inv_dec + L_reg`【108751339527844†L581-L600】.  
- [ ] Curriculum: increase horizon, queries per step.  
- [ ] Eval: long rollouts, MSE + correlation time【108751339527844†L790-L803】.

**Scaling**
- [ ] Stage S0→S3 plan executed; promote only if stability targets met.  
- [ ] Track VRAM budget vs. `(ns, n_latent, dims, depth)`; add checkpointing when needed.  

---

## 8) Pointers to the exact repo modules

- Encoder: `src/models/encoders/cfd_gnn_pool_transformer_perceiver.py`【632897299919862†L15-L27】【632897299919862†L83-L91】  
- Decoder: `src/models/decoders/cfd_transformer_perceiver.py`【897538948798651†L52-L60】【897538948798651†L84-L92】

---

### Appendix: Paper pseudo‑code anchors
- **Encoder** (graph → supernodes → latent): 【108751339527844†L3820-L3888】  
- **Approximator** (latent step): 【108751339527844†L3918-L3924】  
- **Decoder** (query positions → field): 【108751339527844†L3961-L3974】
