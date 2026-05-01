# Cutting-Edge Architecture Research for UPS

Date: 2026-04-07

## Goal

Identify the most promising next-step architectures and systems for turning UPS from a competent latent PDE surrogate into a more general PDE foundation model.

This document is not a generic literature survey. It maps the current UPS codebase to recent primary-source work and recommends a concrete implementation and experiment order.

## Current UPS Snapshot

The current repo is strongest at:

- latent-token evolution with a residual operator in [`src/ups/models/latent_operator.py`](../src/ups/models/latent_operator.py)
- lightweight metadata conditioning through AdaLN in [`src/ups/core/conditioning.py`](../src/ups/core/conditioning.py)
- discretization-aware latent pair loading in [`src/ups/data/latent_pairs.py`](../src/ups/data/latent_pairs.py)
- any-point decoding in [`src/ups/io/decoder_anypoint.py`](../src/ups/io/decoder_anypoint.py)
- optional diffusion/corrector logic and TTC wrappers in [`src/ups/models/diffusion_residual.py`](../src/ups/models/diffusion_residual.py) and [`src/ups/inference/rollout_ttc.py`](../src/ups/inference/rollout_ttc.py)

The current repo is weakest at:

- true foundation-model style pretraining across diverse PDE families
- equation-aware conditioning beyond flat parameter / BC features
- decoded physical-space rollout metrics as the main training and promotion target
- continuous-time learning on irregular time grids
- robust generalization to novel geometries and highly variable boundary conditions
- systematic transfer evaluation and sim-to-real evaluation

In practical terms, UPS is still much closer to a one-step latent operator with attached extras than to a general PDE foundation model.

## Research Signals That Matter Most

### 1. Poseidon

Source: [Poseidon: Efficient Foundation Models for PDEs](https://arxiv.org/abs/2405.19101)

Why it matters:

- strong evidence that pretraining on a small set of well-chosen PDE families can generalize to unseen downstream PDEs
- uses a multiscale operator transformer backbone rather than a purely generic transformer
- explicitly exploits the semigroup property for time-dependent PDEs to create more training signal
- continuous-in-time behavior is helped by time-conditioned layer norms

UPS fit:

- very high
- UPS already has a latent operator and conditioning hooks
- the largest missing ingredient is the semigroup-style pretraining regime and transfer setup

### 2. PDEformer-2

Source: [PDEformer-2: A Versatile Foundation Model for Two-Dimensional Partial Differential Equations](https://arxiv.org/abs/2507.15409)

Why it matters:

- feeds the PDE form itself as input through a computational graph representation
- handles varying symbolic forms, domain shapes, boundary conditions, variable counts, and time dependency in one model
- predicts mesh-free solutions that can be queried at arbitrary coordinates
- shows few-shot adaptation value on unseen PDEs

UPS fit:

- extremely high
- UPS already has an any-point decoder and conditioning scaffolding
- this is the clearest path from "metadata-conditioned operator" to "equation-aware foundation model"

### 3. MORPH

Source: [MORPH: PDE Foundation Models with Arbitrary Data Modality](https://arxiv.org/abs/2509.21670)

Why it matters:

- explicitly tackles heterogeneous modality across 1D, 2D, 3D, varying resolutions, and mixed scalar/vector fields
- architecture combines local convolution, inter-field cross-attention, and axial attention
- strong transfer from pretraining, including adapter-style fine-tuning

UPS fit:

- very high
- UPS wants to be discretization-agnostic across grids, meshes, and particles
- MORPH is the strongest recent signal that this is best handled by modality-aware tokenization instead of pretending every task is the same tensor

### 4. OmniArch

Source: [OmniArch: Building Foundation Model for Scientific Computing](https://proceedings.mlr.press/v267/chen25cp.html)

Why it matters:

- unified 1D-2D-3D pretraining
- Fourier encoder/decoder for flexible multi-scale inputs
- autoregressive framing with temporal masking
- PDE-Aligner for injecting physical priors after pretraining

UPS fit:

- medium to high
- useful mainly as a systems blueprint for multi-scale pretraining and post-hoc physical alignment
- less aligned than PDEformer-2 for arbitrary query decoding, and less aligned than MORPH for multi-modality at the input layer

### 5. Latent Neural Operator / LNO

Source: [Latent Neural Operator](https://proceedings.neurips.cc/paper_files/paper/2024/file/39f6d5c2e310a5a629dcfc4d517aa0d1-Paper-Conference.pdf)

Why it matters:

- learns the operator in a compact latent space
- uses cross-attention to decouple observation locations from prediction locations
- supports forward and inverse settings naturally
- shows the efficiency benefits of encoding once, operating in latent, decoding at arbitrary points

UPS fit:

- already close conceptually
- validates the repo's latent-token + any-point-decoder direction
- suggests UPS should double down on learned query-based encoding / decoding rather than keep evaluation mostly in latent space

### 6. CFO

Source: [CFO: Learning Continuous-Time PDE Dynamics via Flow-Matched Neural Operators](https://arxiv.org/abs/2512.05297)

Why it matters:

- learns continuous-time right-hand-side dynamics without ODE-solver backpropagation
- supports arbitrary and irregular time grids
- improves long-horizon behavior and time-resolution invariance

UPS fit:

- high, but as a second-wave upgrade
- UPS currently assumes discrete stepping and a single `dt` embedding path
- CFO is the cleanest route to time-resolution invariance once the rollout-centric training path is stronger

### 7. PDE-Refiner and follow-on spectral refiners

Sources:

- [PDE-Refiner: Achieving Accurate Long Rollouts with Neural PDE Solvers](https://arxiv.org/abs/2308.05732)
- [Spectral-Refiner](https://arxiv.org/abs/2405.17211)
- [PDE-SpectralRefiner](https://arxiv.org/abs/2506.10711)

Why they matter:

- long-rollout error is often dominated by neglected high-frequency structure
- iterative refinement and spectral objectives can materially improve rollout stability
- these methods often outperform brute-force one-step training even under low data

UPS fit:

- high
- UPS already contains diffusion residual ideas, but the current system should treat "cheap spectral refinement for rollout fidelity" as more central than a generic diffusion story

### 8. Dataset and benchmark shifts

Sources:

- [The Well](https://arxiv.org/abs/2412.00568)
- [RealPDEBench](https://arxiv.org/abs/2601.01829)

Why they matter:

- PDEBench is no longer enough for a claim of generality
- The Well pushes diversity and scale
- RealPDEBench exposes the sim-to-real gap and gives a real deployment-oriented target

UPS fit:

- critical for credibility
- if UPS wants to be a foundation system, data diversity and sim-to-real evaluation matter as much as architecture

## Architecture Shortlist for UPS

### Bet 1: Semigroup-Pretrained Latent Operator

Recommendation:

- keep the latent operator abstraction
- add semigroup-consistency training and semigroup sampling for pretraining
- move evaluation focus from one-step latent error to decoded rollout metrics

Why first:

- lowest implementation risk
- strongest immediate fit to current code
- directly supported by Poseidon-style evidence

What to change:

- train on `(t0 -> t1)`, `(t0 -> t2)`, and compositional consistency `(t0 -> t1 -> t2)` versus `(t0 -> t2)`
- allow variable horizon and random temporal skips during training
- add long-rollout decoded metrics as the promotion gate

### Bet 2: Equation-Graph Conditioned UPS

Recommendation:

- add a PDE-form encoder that turns symbolic or structured PDE metadata into graph tokens
- condition the operator and decoder on those equation tokens

Why second:

- this is the biggest jump in generality
- it transforms UPS from "task-conditioned surrogate" into "equation-aware solver"

What to change:

- introduce equation graphs as first-class inputs alongside `params`, `bc`, and `geom`
- replace flat conditioning-only usage with cross-attention or token-level conditioning
- preserve AdaLN for cheap modulation, but do not rely on AdaLN alone

### Bet 3: Modality-Aware Foundation Backbone

Recommendation:

- adopt MORPH-like field-aware and modality-aware tokenization
- keep separate local stems for grid, mesh, and particle inputs before shared latent processing

Why third:

- UPS claims modality breadth; this makes that claim real
- avoids forcing all discretizations into the same front-end assumptions

What to change:

- per-modality encoder stems
- inter-field attention after modality-local processing
- shared latent backbone plus lightweight adapters for downstream tasks

### Bet 4: Continuous-Time RHS Head

Recommendation:

- add a continuous-time mode that predicts latent derivatives or vector fields
- integrate through an explicit solver only at inference

Why fourth:

- highly promising, but depends on having good rollout supervision already
- useful for irregular time data and future assimilation / control tasks

What to change:

- spline-based derivative targets or finite-difference RHS supervision
- new training mode in addition to discrete autoregressive stepping

### Bet 5: Spectral Refiner Instead of Diffusion-First Corrector

Recommendation:

- keep refinement, but prioritize high-frequency / spectral correction over generic diffusion complexity

Why fifth:

- the evidence for long-rollout improvement is strong
- cheaper and easier to justify than a large stochastic correction stack

What to change:

- add residual spectral heads or iterative deterministic refinement
- use frequency-aware losses and rollout-frequency diagnostics

## Recommended UPS vNext System

If building the most promising integrated system from the current repo, the design should be:

1. Latent any-point architecture remains the base.
2. Backbone becomes a semigroup-trained multiscale operator.
3. Equation graph tokens condition the backbone and decoder.
4. Modality-specific stems feed a shared latent backbone.
5. Decoded rollout losses become primary, latent one-step losses become auxiliary.
6. Spectral refiner becomes the main correction mechanism.
7. Continuous-time mode is added after the above is stable.

In short:

- `Poseidon` for training regime and transfer setup
- `PDEformer-2` for equation-awareness and arbitrary queries
- `MORPH` for heterogeneity across modality and field structure
- `LNO` as validation of the encode-once / latent-operator / any-point-decode pattern
- `CFO` for the future continuous-time branch
- `PDE-Refiner` for rollout stability

## What Not To Prioritize Yet

### Full text-generation multimodality

The multimodal PDE-text work is interesting, but it is not the highest leverage architecture upgrade for UPS right now. Text descriptions are useful later for retrieval, explanation, and weak supervision, not for immediate benchmark gains.

### Large MoE scaling

Mixture-of-experts may help later, but UPS does not yet have the data scale, routing structure, or benchmark harness to justify it.

### Heavy diffusion as the primary dynamics model

UPS already has diffusion ideas. The literature signal still says the deterministic or lightly corrected backbone should improve first. A strong corrector is valuable, but it should not substitute for a better core dynamics model.

## Concrete Experiment Queue

### Experiment 1: Semigroup and horizon training

Goal:

- test whether UPS benefits from Poseidon-style semigroup supervision before larger architectural surgery

Implementation:

- random temporal skip training
- consistency penalty for composed versus direct trajectories
- long-horizon decoded evaluation

Success criterion:

- rollout improvement without worse short-horizon decoded error

### Experiment 2: Any-point decoded rollout benchmark

Goal:

- make decoded rollout the primary metric surface

Implementation:

- train and evaluate through [`src/ups/io/decoder_anypoint.py`](../src/ups/io/decoder_anypoint.py)
- log horizon-specific metrics, spectral error, conservation drift, and BC violations

Success criterion:

- benchmark decisions no longer depend only on latent-space RMSE

### Experiment 3: Equation graph conditioning

Goal:

- test whether structured PDE descriptions outperform flat metadata conditioning

Implementation:

- add graph tokens for operators, coefficients, and BC relationships
- compare token conditioning against current AdaLN-only path

Success criterion:

- better few-shot transfer to held-out PDE forms or BC families

### Experiment 4: Modality-aware shared backbone

Goal:

- see whether separate stems plus shared latent processing outperform a fully shared front-end

Implementation:

- grid stem
- mesh stem
- particle stem
- shared latent backbone

Success criterion:

- better cross-task transfer and less negative transfer across modalities

### Experiment 5: Continuous-time branch

Goal:

- validate time-resolution invariance on irregularly sampled data

Implementation:

- add RHS prediction head
- train from spline-derived derivatives or sparse irregular trajectories

Success criterion:

- improved long-horizon stability under variable `dt`

### Experiment 6: Spectral refiner

Goal:

- improve high-frequency fidelity and rollout stability

Implementation:

- deterministic multi-step refiner or lightweight spectral-correction block
- compare to existing diffusion residual path

Success criterion:

- better decoded long-rollout spectral metrics at similar runtime

## Dataset Roadmap

### Near term

- PDEBench for compatibility and fast comparisons
- The Well for broader pretraining diversity

### Mid term

- RealPDEBench for sim-to-real and real-world deployment claims

### Long term

- custom multi-discretization corpora that unify grid, mesh, and particle observations under shared metadata and equation descriptions

## Bottom-Line Recommendations

If only one upgrade can be made next:

- implement semigroup rollout training and decoded rollout evaluation

If two upgrades can be made:

- add equation-graph conditioning on top of that

If building the real foundation-model version of UPS:

- combine semigroup pretraining, equation-aware conditioning, modality-aware stems, and query-based decoding

That combination is the most defensible and highest-upside path visible from the 2024-2026 literature.

## References

- Poseidon: [https://arxiv.org/abs/2405.19101](https://arxiv.org/abs/2405.19101)
- Latent Neural Operator: [https://proceedings.neurips.cc/paper_files/paper/2024/file/39f6d5c2e310a5a629dcfc4d517aa0d1-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2024/file/39f6d5c2e310a5a629dcfc4d517aa0d1-Paper-Conference.pdf)
- The Well: [https://arxiv.org/abs/2412.00568](https://arxiv.org/abs/2412.00568)
- A Multimodal PDE Foundation Model for Prediction and Scientific Text Descriptions: [https://arxiv.org/abs/2502.06026](https://arxiv.org/abs/2502.06026)
- OmniArch: [https://proceedings.mlr.press/v267/chen25cp.html](https://proceedings.mlr.press/v267/chen25cp.html)
- PDEformer-2: [https://arxiv.org/abs/2507.15409](https://arxiv.org/abs/2507.15409)
- MORPH: [https://arxiv.org/abs/2509.21670](https://arxiv.org/abs/2509.21670)
- CFO: [https://arxiv.org/abs/2512.05297](https://arxiv.org/abs/2512.05297)
- RealPDEBench: [https://arxiv.org/abs/2601.01829](https://arxiv.org/abs/2601.01829)
- PDE-Refiner: [https://arxiv.org/abs/2308.05732](https://arxiv.org/abs/2308.05732)
- Spectral-Refiner: [https://arxiv.org/abs/2405.17211](https://arxiv.org/abs/2405.17211)
- PDE-SpectralRefiner: [https://arxiv.org/abs/2506.10711](https://arxiv.org/abs/2506.10711)
- PI-GANO: [https://arxiv.org/abs/2408.01600](https://arxiv.org/abs/2408.01600)
- PINTO: [https://arxiv.org/abs/2412.09009](https://arxiv.org/abs/2412.09009)
- Learned function extensions for boundary conditions: [https://arxiv.org/abs/2602.04923](https://arxiv.org/abs/2602.04923)
