# Universal Simulator Research Landscape Snapshot

Date: 2026-06-04

Repo snapshot: `ba72f282a4c8af5a041737c120cbf3542827f6e7`

Status: research notes only. This document does not add claim evidence, does not authorize held-out test access, and does not make published-SOTA claims. Literature numbers are not comparable to this repo's `light-v1` protocol unless rerun or mapped under the frozen in-repo contract.

## North-Star Frame

The local north star is not "try every neural PDE paper." It is narrower and stronger:

- Build the strongest defensible universal simulation claim in this repository.
- Keep the claim protocol frozen and auditable: `light-v1`, PDEBench-shaped `advection1d`, `burgers1d`, `darcy2d`, 32-sample caps where recorded, 16-step decoded rollout, `decoded_rollout_nrmse`, per-task/family/horizon metrics, artifact hashes, and strict validation/test separation.
- Improve the primary contract, not only scoped or post-hoc variants.
- Treat external methods as useful only after they are measured under this repo's protocol or carefully mapped as non-comparable.

The current technical essence is clearer than the broad literature landscape: UPS has a working latent-operator/encoder/decoder stack, and its strongest unresolved model-side weakness is long-horizon transport/advection phase tracking. The repo has already shown that online context roll-persistence can crush the advection error, but that remains a changed inference contract. The next high-value path is to convert that transport signal into a causal learned mechanism that can pass the validation-only phase gate before any new held-out spend.

## Local State And Constraints

Current measured surfaces from the roadmap and claim evidence:

- Primary frozen `light-v1` CT8 claim: `ups_light_shared_context_transport_guarded`, held-out test `decoded_rollout_nrmse = 0.4165820594268877`.
- Scoped CT1 online transport-context variant: `ups_light_advection_context_transport_only_ct1_guarded`, held-out test `decoded_rollout_nrmse = 0.20177292896682064`, but it changes the inference contract and must remain side-by-side with the primary claim.
- External measured baselines under `light-v1`: FNO `0.6391747076887233`, UNO `0.5560551396226746`, PDEBench U-Net `0.6095843876848097`, CNO1d `0.5918753212407414`, local physical Fourier `0.5636730976415197`.
- Poseidon zero-shot and scalar-layer finetune were measured on validation and stopped. Scalar-only finetune reached validation `0.5453508470039229`, above the stop threshold and not worth held-out budget.
- No-context model-side advection candidate failed held-out. Validation looked good, but held-out advection h16 collapsed.
- Current validation-only phase gate requires: overall better than `0.35078329353213156`, advection rollout better than `0.4866576789288726`, and advection h16 `<= 0.44444171136384397`.

Negative evidence that should shape future work:

- Alpha sweeps around the existing checkpoint do not clear the phase gate.
- Increasing decoded rollout pressure to 16 steps did not clear the phase gate.
- Horizon-weighted rollout loss helped only marginally.
- Latest-window temporal sampling worsened the horizon-weighted candidate.
- Train-fitted fixed shift-consistency worsened the horizon-weighted candidate; train/validation shift mismatch makes fixed-shift regularization insufficient.

Code seams that already exist:

- `src/ups/models/latent_operator.py`: residual latent operator with time embedding and AdaLN conditioning.
- `scripts/train.py`: decoded field loss, task loss weights, horizon-weighted decoded rollout loss, temporal windowing, and transport shift-consistency knobs.
- `src/ups/eval/pdebench_runner.py`: decoded rollout evaluator, residual alpha/gate config, FFT/fractional roll shift, observed/prediction/context roll-shift estimators, per-task/family/horizon metrics.
- `scripts/run_light_experiment.py`: practical bounded experiment entrypoint with skip-training, stage checkpoint preference, overrides, promotion rules, and held-out ledger support.

## Executive Recommendation

The optimal next path is a causal transport-phase mechanism, implemented as the smallest train/validation-only sidecar that can explain the CT1 signal without using validation-oracle shifts or future test information.

Recommended first implementation direction:

1. Train a data-conditioned advection phase estimator on the train split only.
2. Predict a per-sample fractional shift or low-rank displacement field from allowed causal inputs: initial/current field, task/family metadata, available PDE parameters if present, and possibly model residual statistics.
3. Apply the shift through the existing differentiable FFT/fractional roll path or a small differentiable warp module.
4. Evaluate on validation only against the phase gate: overall, advection rollout, and advection h16.
5. Do not write a held-out pre-test contract unless the phase gate clears with margin and Burgers/Darcy stay stable.

Why this outranks broader foundation work:

- The CT1 result proves the transport signal exists.
- The failed fixed-shift and alpha runs show the current weakness is not a scalar blend or constant phase offset.
- Recent literature increasingly supports learned warps, hybrid local/global routing, denoising refinement, and continuous-time objectives for exactly the failure mode seen here.
- UPS already has the right seams to test this cheaply before touching the foundation backbone.

## Landscape At A Glance

| Direction | Maturity | Relevance to UPS | Recommendation |
| --- | --- | --- | --- |
| Learned transport/warp heads | Emerging, very aligned | Directly targets advection phase/h16 | Highest priority |
| PDE-Refiner / denoising correctors | Mature paper, implementable sidecar | Targets long rollouts and spectral/high-frequency loss | Second priority |
| Hybrid local/global neural operators | Active 2026 area | Addresses Fourier/global vs local/shock tradeoff | Third priority after warp sidecar |
| Continuous-time / flow-matched operators | Emerging 2025-2026 | May reduce autoregressive error and dt brittleness | Prototype after phase estimator |
| UPT-style latent set universal backbone | Strong north-star alignment | Matches arbitrary grid/mesh/particle goal | Strategic backbone upgrade, not first sprint |
| Poseidon/DPOT/AOT-POT foundation pretraining | High upside, expensive | Supports foundation claim, but transfer is currently blocked | Keep as separate train/val track |
| PINO / physics-informed losses | Mature enough | Useful if PDE residuals and parameters are reliable | Add targeted residuals, not broad PINN rewrite |
| External baseline expansion | Mature | Claim-evidence value, less model improvement | Continue only when adapter cost is low |
| Weather foundation models | Mature at scale | Good design signals for autoregression/meshes/hybrid physics | Reference patterns, not direct dependencies |

## Literature Notes

### 1. Classical Neural Operator Spine

Fourier Neural Operator (FNO) is still the canonical baseline because it made operator learning practical for PDE families by parameterizing global kernels in Fourier space. It remains the right lowest-friction external baseline, and this repo already measured FNO via NeuralOperator. FNO's weakness for this project is not that it is obsolete; it is that pure global spectral mixing can miss localized or phase-sensitive transport behavior unless the task distribution is aligned and the evaluation protocol is fixed.

U-NO adds a U-shaped hierarchy to neural operators, improving depth, memory use, and multiscale structure. This matters because UPS currently has latent tokens and a transformer core, but the advection blocker looks like a multiscale transport problem. U-NO is not necessarily a drop-in win for the current light-v1 stack, but its multiresolution skip structure is a good pattern for a decoded-side corrector or future operator backbone.

CNO is important because it argues that convolutional architectures can be made operator-consistent and robust. The repo already has a measured simplified CNO1d baseline under `light-v1`, and it lost to UPS under this protocol. The architectural lesson is still useful: continuous-respecting convolutional local branches can complement Fourier/attention branches, especially when the model needs local sharpness and stable rollouts.

PINO combines data and PDE constraints, often at different resolutions. This is relevant but should be used surgically. For the current repo, a broad physics-informed rewrite would be a distraction. A targeted advection residual, mass/conservation residual, or phase/characteristic consistency loss could be useful if the required PDE parameters and boundary assumptions are unambiguous in the current data.

GINO and geometry-informed operators matter for the long-term universal simulator north star: arbitrary geometries, point clouds, signed distance functions, graph-to-regular-latent transforms, and Fourier latent grids. This is strategic for mesh/geometry generality, not the fastest way to fix 1D advection h16.

Group-equivariant FNO and later equivariant graph/neural operators matter because physics should not depend on arbitrary coordinate choices. For this repo's immediate 1D periodic transport, translation equivariance and phase handling are especially relevant. Full rotation/SE(3) machinery is not needed for light-v1, but exact/approximate translation-equivariant transport heads are high signal.

Sources:

- FNO: https://arxiv.org/abs/2010.08895
- NeuralOperator library: https://github.com/neuraloperator/neuraloperator
- U-NO: https://arxiv.org/abs/2204.11127
- CNO: https://arxiv.org/abs/2302.01178
- CNO official code: https://github.com/camlab-ethz/ConvolutionalNeuralOperator
- PINO: https://arxiv.org/abs/2111.03794
- PINO code/docs: https://github.com/neuraloperator/physics_informed
- GINO: https://arxiv.org/abs/2309.00583
- Group Equivariant FNO: https://arxiv.org/abs/2306.05697

### 2. Long-Horizon Rollout Stability

The most relevant long-horizon paper for the current blocker is PDE-Refiner. Its core diagnosis aligns with this repo's failure: accurate neural PDE solvers must model non-dominant spatial frequencies and maintain stability over long rollouts. The proposed multistep refinement/denoising process is attractive because it can be tested as a sidecar on decoded fields without replacing UPS.

DPOT is the foundation-scale version of a similar idea: autoregressive denoising pretraining plus Fourier attention, trained on many PDE datasets. DPOT is too large to copy directly into the next sprint, but its recipe suggests a local pretraining objective: corrupt decoded or latent trajectories, train recovery over multiple horizons, then fine-tune on light-v1 before validation.

GNS and MeshGraphNets highlight two practical training lessons: long-term physical simulators often need noise/corruption during training, and stable autoregression may be more important than one-step loss. Message Passing Neural PDE Solvers make this explicit through zero-stability framed as domain adaptation. UPS has already seen the trap: small validation average gains can collapse at h16.

CFO is a newer continuous-time approach using flow matching. It attacks two weaknesses of autoregressive surrogates: uniform time-grid dependence and long-rollout error accumulation. It is not a drop-in library today, but the idea is worth prototyping if the repo has enough intermediate time samples: train a velocity/right-hand-side operator or flow-matched latent dynamics instead of only next-step residuals.

Sources:

- PDE-Refiner: https://arxiv.org/abs/2308.05732
- DPOT: https://arxiv.org/abs/2403.03542
- DPOT code: https://github.com/HaoZhongkai/DPOT
- Learning to Simulate Complex Physics with Graph Networks: https://arxiv.org/abs/2002.09405
- Message Passing Neural PDE Solvers: https://arxiv.org/abs/2202.03376
- MeshGraphNets: https://arxiv.org/abs/2010.03409
- CFO: https://arxiv.org/abs/2512.05297

### 3. Transport, Phase, Warps, And Characteristics

This is the most important cluster for the next UPS work.

Flowers is the clearest 2026 signal: build PDE solver blocks from learned coordinate warps. Each head predicts a displacement field and samples source coordinates; global interaction comes from sparse warped sampling rather than Fourier or dot-product attention. For UPS, this suggests a minimal decoded-side transport sidecar: predict a displacement/shift from current allowed state, warp persistence or the decoded prediction, then add residual correction.

Semi-Lagrangian methods are the classical numerical analog: follow characteristics backward, interpolate from source coordinates, and advance advected quantities. The existing `_roll_flattened_grid` function is already a 1D periodic special case. A learned semi-Lagrangian head would be a principled extension from fixed scalar shifts to per-sample/per-position displacement.

U-HNO is a 2026 hybrid operator paper with sparse-point adaptive routing between global Fourier and local multiscale branches. This is relevant after the first transport head because UPS likely needs both smooth global transport and local feature preservation. A small version could be a decoded corrector that blends a Fourier/roll branch with a local stencil branch using a learned gate.

AOT-POT is more foundation/pretraining oriented, but the key idea is useful locally: different PDEs benefit from different operator transformations. UPS should not force advection, Burgers, and Darcy through the same residual correction if the transport family needs a transformation/warp while elliptic Darcy does not.

Group equivariance and translation symmetry reinforce the same point. The advection problem is largely a phase/translation problem. A good model should express shifts naturally, not discover them indirectly through scalar residual blending.

Sources:

- Flowers: https://arxiv.org/abs/2603.04430
- U-HNO: https://arxiv.org/abs/2605.12965
- AOT-POT: https://arxiv.org/abs/2605.15793
- Group Equivariant FNO: https://arxiv.org/abs/2306.05697
- Semi-Lagrangian surface advection reference: https://pmc.ncbi.nlm.nih.gov/articles/PMC9395508/

### 4. PDE Foundation Models And Universal Backbones

UPT is extremely aligned with UPS's long-term architecture: a unified latent representation, no hard dependence on grid/particle latent structures, latent-space propagation, inverse encoding/decoding, and any-point queries. UPS already has several pieces of this idea: latent state, grid encoder, any-point decoder, and task conditioning. The UPT lesson is to make the latent set the common interface across grid/mesh/particle data rather than adding bespoke per-modality paths.

Poseidon is the most directly relevant open PDE foundation model because it provides code, pretraining/downstream datasets, and pretrained models. It uses a multiscale operator transformer, time-conditioned layer norms, and a semigroup-style training strategy. The repo already tested scalar adaptation and found it insufficient. That negative result should not discard Poseidon completely; it means scalar light-v1 is a bad zero-shot adapter path. A controlled unfreeze/LoRA train/validation gate is the only Poseidon path worth reopening.

DPOT proves that large-scale PDE pretraining with autoregressive denoising and Fourier attention can generalize across tasks. It is more useful as a training recipe than as a near-term dependency.

AOT-POT is a very recent direction: adaptive operator transformations before/after layers to align diverse PDE solution operators. Its local implication is strong: the next UPS model should let advection use a transport transformation while Darcy/Burgers can use different residual dynamics.

PDEformer-2, equation-graph conditioning, and symbolic PDE-conditioned models are relevant to the north star if UPS wants equation-signature conditioning beyond one-hot task IDs. But they are not the highest-ROI next step until the light-v1 advection issue is solved.

Sources:

- UPT: https://arxiv.org/abs/2402.12365
- UPT code: https://github.com/ml-jku/UPT
- Poseidon: https://arxiv.org/abs/2405.19101
- Poseidon code: https://github.com/camlab-ethz/poseidon
- Poseidon project/PDEgym: https://camlab-ethz.github.io/poseidon/
- PDEgym collection: https://huggingface.co/collections/camlab-ethz/pdegym
- DPOT: https://arxiv.org/abs/2403.03542
- AOT-POT: https://arxiv.org/abs/2605.15793
- PDEformer-2 source from existing mapping: https://github.com/functoreality/pdeformer-2

### 5. Benchmarks And Evaluation Culture

PDEBench remains the right anchor for this repo's current claim protocol. Its value is not only the data; it enforces a common discussion about task families, splits, baselines, and failure modes.

APEBench is especially relevant because it focuses on autoregressive PDE emulators and rollout behavior. It includes a differentiable pseudo-spectral solver setup and many PDEs across 1D/2D/3D. This is more aligned with UPS's actual failure mode than static one-step benchmarks. It should be considered for a future validation suite once light-v1 is stable.

LagrangeBench matters for the long-term grid/mesh/particle universal simulator north star. It includes JAX APIs, particle datasets, GNS/SEGNN baselines, and physical metrics beyond position error. UPS has particle/contact modules, so LagrangeBench is a good future gate for particle generality.

WeatherBench2/GraphCast-style evaluations are useful patterns: strict temporal holdouts, many variables, rollout lead-time metrics, and operational caveats. They are not comparable to light-v1 but are culturally useful for mature simulation claims.

Recommended local benchmark expansion:

- Keep `light-v1` frozen for the current claim.
- Add APEBench-like rollout diagnostics to validation reports: h1/h4/h8/h16 slopes, spectral band errors, conservation where applicable, and stability ratio.
- Add a future "transport-generalization split" where advection train/val/test transport parameters are intentionally aligned or intentionally OOD, so the model cannot hide split mismatch behind post-hoc corrections.
- Add a future "universal geometry" gate only after the grid light-v1 path is under control.

Sources:

- PDEBench paper: https://arxiv.org/abs/2210.07182
- PDEBench code: https://github.com/pdebench/PDEBench
- APEBench: https://arxiv.org/abs/2411.00180
- APEBench project/code pointers: https://github.com/tum-pbs/apebench-paper
- LagrangeBench: https://arxiv.org/abs/2309.16342
- LagrangeBench code/docs: https://github.com/tumaer/lagrangebench and https://lagrangebench.readthedocs.io/

### 6. Weather And Large-Scale Physical Forecasting Lessons

FourCastNet, GraphCast, GenCast, and NeuralGCM are not direct baselines for this repo, but they show what mature learned simulators look like:

- FourCastNet: AFNO/Fourier token mixing can produce fast high-resolution global forecasts.
- GraphCast: graph/mesh representations and autoregressive training can scale to global geophysical dynamics.
- GenCast: diffusion can provide probabilistic ensembles and better uncertainty surfaces.
- NeuralGCM: hybrid differentiable solvers plus ML components can be stable enough for weather and climate tasks.

Local implication:

- Do not frame UPS as a universal physics foundation model until it has stronger rollout, uncertainty, and benchmark discipline.
- Borrow architectural patterns: graph/mesh latent interfaces, typed features, lead-time metrics, uncertainty/refinement sidecars, and hybrid physics+ML where the known solver structure is useful.
- Avoid direct dependency adoption unless a benchmark gate demands it. These systems are domain-specific and heavy.

Sources:

- FourCastNet: https://arxiv.org/abs/2202.11214
- FourCastNet code: https://github.com/NVlabs/FourCastNet
- GraphCast: https://arxiv.org/abs/2212.12794
- GraphCast/GenCast code: https://github.com/google-deepmind/graphcast
- GenCast: https://arxiv.org/abs/2312.15796
- NeuralGCM: https://arxiv.org/abs/2311.07222

### 7. Physics-Informed And Solver Ecosystem

PhysicsNeMo is the most production-like open physics-ML framework in this landscape. It offers model families including neural operators, GNNs, diffusion models, transformers, scalable data pipelines, distributed training, and symbolic PDE residual computation. It is valuable as a reference and possible source of implementations, especially FNO/AFNO/MeshGraphNet/GraphCast patterns, but adopting it as a dependency would be a major stack decision.

DeepXDE and NeuralPDE.jl are mature PINN/DeepONet/scientific-ML ecosystems. They are useful for physics-informed baselines, inverse problems, and small PDE proof-of-concepts. They are not ideal for the immediate UPS training loop because the repo already has PyTorch infrastructure, evidence contracts, and data-backed neural operator workflows.

SciML/DifferentialEquations.jl/Diffrax/JAX-CFD-style differentiable solver ecosystems are relevant if UPS needs differentiable physics-generated losses or synthetic data generation. For now, the fastest path is not to replace the harness; it is to add a targeted differentiable transport/phase objective inside the current PyTorch loop.

Sources:

- PhysicsNeMo: https://github.com/NVIDIA/physicsnemo
- PhysicsNeMo examples: https://docs.nvidia.com/physicsnemo/latest/examples_catalog.html
- DeepXDE: https://github.com/lululxvi/deepxde
- DeepXDE paper: https://arxiv.org/abs/1907.04502
- NeuralPDE.jl: https://docs.sciml.ai/NeuralPDE/v4.7/
- SciML: https://sciml.ai/

## Open-Source Ecosystem Map

### Mature / Practical Today

- `neuraloperator/neuraloperator`: best PyTorch neural-operator baseline library; already used in spirit by current external FNO/UNO adapters. Good for more FNO variants, GINO, TFNO, factorized layers.
- `pdebench/PDEBench`: current benchmark anchor; use for official source attribution and external U-Net/FNO context.
- `camlab-ethz/ConvolutionalNeuralOperator`: official CNO family; already adapted for CNO1d.
- `camlab-ethz/poseidon`: official Poseidon/ScOT foundation code; only reopen with controlled unfreeze/LoRA on train/validation.
- `NVIDIA/physicsnemo`: broad production-style reference stack; useful for model patterns and large-scale training ideas.
- `google-deepmind/graphcast`: reference implementation for mesh GNN weather models and GenCast diffusion ensemble patterns.

### Research Code / Good To Study, Riskier To Depend On

- `ml-jku/UPT`: high north-star alignment, but it is a substantial architecture change.
- `HaoZhongkai/DPOT`: pretraining recipe and model code; foundation-scale rather than light-v1 quick fix.
- `tum-pbs/apebench`: future benchmark suite for autoregressive emulator behavior.
- `tumaer/lagrangebench`: future particle/Lagrangian benchmark.
- `MinkaiXu/EGNO`: relevant for 3D/particle trajectory operators, not current light-v1.
- AIRS library for G-FNO: symmetry/equivariance ideas.

### Not Immediate

- DeepXDE/NeuralPDE.jl: valuable but stack-mismatched for immediate UPS work.
- Full weather models: useful design references; too domain-specific and heavy for this repo's current claim protocol.
- New 2026 papers without stable code: Flowers, U-HNO, AOT-POT, CFO. These are high-value design signals; implement minimal local analogs rather than waiting for mature packages.

## Ranked Research-To-Implementation Queue

### P1: Data-Conditioned Transport Phase Estimator

Goal: learn the phase/shift signal that CT1 currently extracts online, without validation/test oracle selection.

Minimal design:

- Inputs: current field or initial context allowed by the target contract, task/family embedding, grid metadata, available PDE parameters, persistence statistics, raw UPS residual statistics.
- Output: scalar fractional periodic shift for 1D advection first.
- Application: use `_roll_flattened_grid(..., shift_x=predicted_shift)` before/after residual blend.
- Training: train split only. Supervise against train-derived best shifts or directly optimize next-step/rollout field loss through differentiable FFT shifting.
- Evaluation: validation only, 32 samples, 16 decoded steps, empty observed/prediction/context shift estimators unless the estimator is the trained model under test.
- Gate: clear overall, advection rollout, and advection h16 thresholds before any held-out contract.

Tradeoffs:

- Lowest implementation cost and most directly tied to evidence.
- Risk: overfits to current light-v1 advection split if input features do not explain train/val mismatch.
- Mitigation: report predicted shift distributions by task/split/horizon; require h16 improvement, not only average rollout.

### P2: Learned Warp Sidecar

Goal: move beyond one scalar shift to a per-sample or per-position displacement field.

Minimal design:

- For 1D, predict `B x 1` or `B x W` displacement from a compact CNN/MLP over the flattened field and metadata.
- Warp persistence and/or raw decoded UPS prediction using periodic interpolation.
- Blend warped persistence with UPS residual.
- Keep the module optional and default-off.

Why it matters:

- Flowers and semi-Lagrangian methods suggest warping is the natural primitive for transport.
- A scalar shift may be too weak if advection speed varies over samples, time, or spatial regions.

Tradeoffs:

- More expressive than P1 but more likely to overfit.
- Requires careful no-test evidence and small unit tests for differentiability, periodicity, identity shift, and task isolation.

### P3: PDE-Refiner-Style Decoded Corrector

Goal: reduce h16 and spectral/high-frequency errors after the base model predicts.

Minimal design:

- Inputs: current decoded prediction, persistence field, raw residual, horizon embedding, task/family embedding.
- Output: corrected field or correction residual.
- Train with small denoising/corruption around target trajectories and model predictions.
- Run 1-3 refinement steps at evaluation.

Why it matters:

- Directly addresses long rollouts and non-dominant spatial frequencies.
- Can be tested on top of existing checkpoints without changing the latent backbone.

Tradeoffs:

- May improve all tasks but can hide base-model weaknesses behind extra inference cost.
- Must report runtime/cost and ensure it does not become a post-hoc validation-tuned patch.

### P4: Hybrid Local/Global Branch Gate

Goal: let advection/Burgers use local finite-difference/stencil-like dynamics while retaining global latent/Fourier behavior for smooth fields.

Minimal design:

- Add a small local stencil residual head in decoded or latent grid space.
- Add a global branch from existing operator output.
- Gate by task/family/horizon/local contrast.
- Start on decoded side to keep blast radius low.

Why it matters:

- U-HNO and CNO both point toward local/global hybrids.
- Transport and shocks need local structure; Darcy may not benefit from the same branch.

Tradeoffs:

- More invasive than P1/P2.
- Requires task regression guard: Burgers and Darcy must not degrade materially.

### P5: Semigroup / Continuous-Time Objective

Goal: improve time consistency rather than only next-step decoded loss.

Minimal design:

- Add semigroup consistency on decoded fields: direct two-step prediction should match composed one-step predictions and target where available.
- Add random time-window pairs if the data supports it.
- Optionally train a latent velocity/right-hand-side model for continuous `dt`.

Why it matters:

- Poseidon and CFO both emphasize time conditioning/semigroup or continuous-time structure.
- This may help when h1 and h4 look acceptable but h16 collapses.

Tradeoffs:

- Needs careful data inspection; not all light-v1 data may expose enough time/parameter variation.
- Larger conceptual shift than P1/P2; should follow a cheaper phase-estimator attempt.

### P6: Foundation Backbone Work

Goal: move UPS toward a credible general/foundation simulator after light-v1 blockers are under control.

Candidate directions:

- UPT-style latent set backbone with inducing tokens and any-point query decoding.
- Poseidon-style multiscale operator transformer with time-conditioned layer norms.
- DPOT-style autoregressive denoising pretraining.
- AOT-POT-style task-adaptive operator transformations.
- Equation/PDE metadata conditioning beyond one-hot task IDs.

Tradeoffs:

- Highest north-star alignment.
- Highest cost and broadest refactor.
- Should not be started before a clean light-v1 model-side win or a deliberately separate research branch with its own harness.

## Explicitly Deprioritized Paths

- More static alpha sweeps on the existing checkpoint. Already failed the phase gate.
- More fixed train-fitted shift regularization. Train/validation shift mismatch makes this low signal.
- Held-out tests before phase-gate clearance. This violates the current evidence discipline.
- Scalar-only Poseidon transfer. Measured and stopped.
- Replacing the whole stack with PhysicsNeMo/DeepXDE/NeuralPDE. Too disruptive relative to the current narrow blocker.
- Published table comparisons without protocol mapping. Not defensible for this repo's claim.

## Suggested Next Sprint

The next sprint should be "learned causal transport phase estimator, validation-only."

Concrete work plan:

1. Add a small optional `decoded_transport_shift_model` path with no default behavior change.
2. Build unit tests for bounded shift output, identity behavior, periodic fractional roll, no effect on non-transport tasks when disabled, and summary metrics recording predicted shift stats.
3. Add a train-only fitting script or training stage that learns the shift from train trajectories.
4. Run validation only against `docs/claim_evidence/ups_advection_phase_tracking_validation_gate_contract.json`.
5. Package negative or positive validation evidence. If negative, record why it failed: h16, advection rollout, or task regression.

Success signal:

- Validation advection h16 moves meaningfully toward or below `0.44444171136384397`.
- Overall validation beats `0.35078329353213156`.
- Advection rollout beats `0.4866576789288726`.
- Burgers and Darcy stay stable.
- Predicted shift distribution is explainable from train-derived signals, not validation-oracle tuning.

If P1 fails:

- Move to P2 learned warp sidecar, not more scalar shift sweeps.
- If P2 fails, move to P3 PDE-Refiner-style corrector.
- If P3 fails, reassess the light-v1 split construction and consider a new train/val/test split contract for transport generalization before further held-out spend.

## Source Index

Primary papers and official code/docs used in this snapshot:

- Universal Physics Transformers: https://arxiv.org/abs/2402.12365
- UPT code: https://github.com/ml-jku/UPT
- Poseidon: https://arxiv.org/abs/2405.19101
- Poseidon code: https://github.com/camlab-ethz/poseidon
- Poseidon project/PDEgym: https://camlab-ethz.github.io/poseidon/
- PDEgym Hugging Face collection: https://huggingface.co/collections/camlab-ethz/pdegym
- DPOT: https://arxiv.org/abs/2403.03542
- DPOT code: https://github.com/HaoZhongkai/DPOT
- PDE-Refiner: https://arxiv.org/abs/2308.05732
- FNO: https://arxiv.org/abs/2010.08895
- NeuralOperator library: https://github.com/neuraloperator/neuraloperator
- U-NO: https://arxiv.org/abs/2204.11127
- CNO: https://arxiv.org/abs/2302.01178
- CNO code: https://github.com/camlab-ethz/ConvolutionalNeuralOperator
- PINO: https://arxiv.org/abs/2111.03794
- PINO code/docs: https://github.com/neuraloperator/physics_informed
- GINO: https://arxiv.org/abs/2309.00583
- Group Equivariant FNO: https://arxiv.org/abs/2306.05697
- Flowers: https://arxiv.org/abs/2603.04430
- U-HNO: https://arxiv.org/abs/2605.12965
- AOT-POT: https://arxiv.org/abs/2605.15793
- CFO: https://arxiv.org/abs/2512.05297
- Message Passing Neural PDE Solvers: https://arxiv.org/abs/2202.03376
- Learning to Simulate Complex Physics with Graph Networks: https://arxiv.org/abs/2002.09405
- MeshGraphNets: https://arxiv.org/abs/2010.03409
- EGNO: https://arxiv.org/abs/2401.11037
- PDEBench: https://arxiv.org/abs/2210.07182
- PDEBench code: https://github.com/pdebench/PDEBench
- APEBench: https://arxiv.org/abs/2411.00180
- APEBench project: https://github.com/tum-pbs/apebench-paper
- LagrangeBench: https://arxiv.org/abs/2309.16342
- LagrangeBench code: https://github.com/tumaer/lagrangebench
- LagrangeBench docs: https://lagrangebench.readthedocs.io/
- PhysicsNeMo: https://github.com/NVIDIA/physicsnemo
- PhysicsNeMo examples: https://docs.nvidia.com/physicsnemo/latest/examples_catalog.html
- DeepXDE: https://github.com/lululxvi/deepxde
- DeepXDE paper: https://arxiv.org/abs/1907.04502
- NeuralPDE.jl: https://docs.sciml.ai/NeuralPDE/v4.7/
- SciML: https://sciml.ai/
- FourCastNet: https://arxiv.org/abs/2202.11214
- FourCastNet code: https://github.com/NVlabs/FourCastNet
- GraphCast: https://arxiv.org/abs/2212.12794
- GraphCast/GenCast code: https://github.com/google-deepmind/graphcast
- GenCast: https://arxiv.org/abs/2312.15796
- NeuralGCM: https://arxiv.org/abs/2311.07222
