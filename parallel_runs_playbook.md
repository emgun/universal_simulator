# Parallel Runs Playbook — High-ROI Hyperparams to Vary

> Run many jobs in parallel? Vary *few, high‑leverage knobs per run* so you learn fast, avoid confounds, and keep compute comparable.

---

## Principles
- **Orthogonalize:** group knobs that hit different failure modes (opt/schedule vs uncertainty vs physics vs capacity).
- **Fix compute:** keep **batch_tokens, horizon, and eval suite** identical so comparisons are fair; if a change raises memory, compensate elsewhere.
- **Low-cardinality first:** 3–4 values per knob; 12–24 runs per round is a sweet spot.
- **Avoid full grids:** use **fractional factorial** or **Latin hypercube** to capture interactions cheaply.
- **Promote quickly:** best small-eval result → full eval, then narrow locally.

---

## Tiered Knobs to Vary

### Tier 1 — Cheap & High Impact (run every round)
**A. Optimizer/Schedule (3×3 grid)**
- **LR:** {2e-4, **3e-4**, 4.5e-4} (scale linearly with batch_tokens)
- **Warmup:** {3%, **5%**, 6%}
- **EMA:** {0.999, **0.9995**, 0.9999}  
*Why:* Often 60–80% of attainable gains without touching model size.

**B. Hybrid Diffusion Corrector (uncertainty/stability)**
- **Cadence:** {8, **16**, 32}
- **Steps:** {1, **2**}
- **Guidance λ:** {0.5, **1.0**, 1.5}  
*Why:* Tunes calibration & long-horizon stability with low runtime tax.

**C. Physics Projections (weight ablation)**
- **Projection weight:** {0.5×, **1×**, 2×}  
*Why:* Prevents “metric wins” that break conservation/BCs.

**D. Inverse-Loss Weights (latent stability)**
- **inverse_encode / inverse_decode:** {(0.25,0.25), **(0.5,0.5)**, (0.75,0.75)}  
*Why:* Stabilizes latent-only rollouts; right weight depends on latent size.

---

### Tier 2 — Medium Cost, High Signal
**E. Capacity Scale (change one axis)**
- **model_dim:** {320, **384**, 448} **OR**
- **depth:** {20, **24**, 28}  
*Why:* Smooth accuracy/compute trade; observe local scaling law.

**F. Attention Window (memory ↔ context)**
- **window_size:** {8, **12**, 16}  
*Why:* Big VRAM lever with small accuracy impact.

**G. Latent Size (UPT)**
- **latent_tokens:** {384, **512**, 768}  
*Why:* Controls information bottleneck across discretizations.

---

### Tier 3 — Data & Curriculum
**H. Horizon Curriculum**
- **max_horizon:** {16, **32**, 48}
- **freeze_bottom_layers_on_expand:** {false, **true**}  
*Why:* Stabilizes long rollouts.

**I. Regime Sampler Temperature**
- **temp:** {0.6, **0.7**, 0.8}  
*Why:* Prevents collapse to easy regimes; improves OOD.

---

## What *Not* to Vary Together (until later)
- Positional encodings, normalization type, **channel_separated** toggle — change one at a time.
- Multiple architectural toggles simultaneously (e.g., turn off U-shape **and** change windowing).

---

## Suggested Simultaneous-Run Bundles

### Round A — *Make it train beautifully* (12 runs)
- **Optimizer/Schedule 3×3** = 9 runs (LR × Warmup), EMA fixed at 0.9995.
- **Corrector cadence:** add 3 runs fixing best LR/Warmup with cadence {8,16,32}.  
→ Pick top‑2 to full eval.

### Round B — *Stability & UQ* (12 runs)
- Fix best from Round A. Sweep **guidance λ** {0.5,1.0,1.5} × **steps** {1,2} × **cadence** {16}. (6 runs)
- Sweep **inverse losses** {(0.25,0.25), (0.5,0.5), (0.75,0.75)} (3 runs)
- **Projection weight** {0.5×,1×,2×} (3 runs)  
→ Keep best calibration (ECE) subject to physics gates.

### Round C — *Capacity & Memory* (12–15 runs)
- **model_dim** {320,384,448} × **window_size** {8,12,16} = 9 runs (use 6 via fractional factorial if tight).
- **latent_tokens** {384,512,768} (3 runs), adjusting batch_tokens to keep VRAM fixed.  
→ Choose best nRMSE per Joule/step; confirm physics gates.

### Round D — *Curriculum & OOD* (9–12 runs)
- **max_horizon** {16,32,48} × **freeze_bottom_layers_on_expand** {false,true}.
- **sampler temp** {0.6,0.7,0.8} (Latin hypercube with horizon).  
→ Look for long-horizon stability and OOD geometry wins.

---

## Interaction Notes
- **LR × EMA:** higher EMA tolerates slightly higher LR but can hide instabilities—watch train/val gap.
- **window_size × model_dim:** bigger windows help small dims more (context bottleneck); for large dims, windows mostly save memory.
- **latent_tokens × inverse-losses:** more latent tokens → often reduce inverse-loss weights.
- **guidance λ × projection weight:** raising both can over‑constrain → tune one at a time.
- **horizon × cadence:** as horizon grows, decrease cadence (use corrector more often) until physics gaps stabilize.

---

## Quick Sweep Templates

**Coarse Optimizer Grid (9 runs)**
```
LR:         [2e-4, 3e-4, 4.5e-4]
Warmup:     [0.03, 0.05, 0.06]
EMA:        [0.9995]      # fixed here
```

**Corrector Sweep (6 runs)**
```
Cadence:    [8, 16, 32]   # pick 2 if tight: [8, 16]
Steps:      [1, 2]
Guidance λ: [1.0]         # fix first, then open to {0.5,1.0,1.5}
```

**Capacity × Memory (6–9 runs)**
```
model_dim:  [320, 384, 448]      # OR depth: [20, 24, 28]
window:     [8, 12, 16]
```

**Latent Size (3 runs)**
```
latent_tokens: [384, 512, 768]   # adjust batch_tokens to keep VRAM fixed
```

**Curriculum/OOD (4–6 runs)**
```
max_horizon: [16, 32, 48]
freeze_on_expand: [true, false]
sampler_temp: [0.6, 0.7, 0.8]    # factor with fewer combos via LHS
```

---

## Guardrails for Parallel Runs
- **Keep batch_tokens constant** (or re-fit LR if you change it).
- Rank by **task + physics gates + calibration + Joules/step**.
- **Early-stop** if small-eval delta < 1σ vs baseline by 30–40% budget.
- **Promote** only top‑k (k=2–3) per round to full eval.

---

## TL;DR
Vary in parallel: **(1) LR/Warmup/EMA**, **(2) corrector cadence/steps/λ**, **(3) window_size & model_dim (not depth simultaneously)**, **(4) latent_tokens**, **(5) horizon/temperature**. Use fractional factorial/LHS, keep compute fixed, and pick winners by **task + physics gates + calibration + Joules/step** to converge fast.
