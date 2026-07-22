# Canonical Latent E8 Amortized Universal Encoder Contract

Date: 2026-07-21
Status: frozen before state-level measurement

## Question

E7 established that one ordered 52-coefficient physical function space can
represent the same smooth scalar state from regular grids and warped meshes.
Can one learned, representation-blind encoder amortize that projection while
preserving the coefficient meanings and benefiting from mixed-discretization
training?

E8 tests the encoder only. It does not instantiate a latent operator, add a
router, expose a representation label, read a reserved held-out split, or
qualify particles scientifically.

## Research basis

Current function-space guidance requires an encoder to be independent of the
input discretization and requires quadrature weights whenever a finite sum
approximates a continuous integral:
<https://www.nature.com/articles/s42256-026-01267-z>. In particular, its
representative encoder maps a sampled function to fixed-dimensional inner
products with coordinate-defined basis functions before applying a latent
map. E8 implements exactly that continuous construction.

The ICLR 2025 discretization-mismatch analysis warns that accepting different
resolutions does not itself guarantee consistent outputs:
<https://proceedings.iclr.cc/paper_files/paper/2025/hash/313829757739365201b5adb3a1cbd9bd-Abstract-Conference.html>.
E8 therefore measures paired grid/mesh mismatch and unseen-resolution
refinement directly. Function-space autoencoder work likewise motivates
defining the representation and objective at the function level before
discretization:
<https://www.jmlr.org/papers/v26/25-0035.html>.

## Frozen semantic target

Reuse `PhysicalFunctionSpace` from E7 without changing its basis, ordering,
cutoff, normalization, projection, or decoder. For every observed source, the
teacher is the exact float64 E7 quadrature-weighted least-squares coefficient
vector. The 52 output positions retain their E7 physical meanings.

The learned encoder may not access the solved teacher coefficients as inputs.
It receives only source coordinates `x_i`, scalar values `u_i`, and normalized
positive quadrature masses `w_i`.

## Frozen encoder

Evaluate the immutable E7 basis `Phi(x_i)` and form the 52 quadrature moments

`m = sum_i w_i Phi(x_i) u_i`.

This is a permutation-invariant discretization of a continuous inner-product
encoder. Apply one learned residual coefficient map

`c_hat = m + W_2 GELU(W_1 m + b_1) + b_2`,

with hidden width `64`. Initialize both arms of the residual output layer to
zero so every model starts as the same physical moment encoder. The model has
no source index, family label, resolution label, task label, graph edge, or
alternate path. Decode only as `Phi(y)c_hat`; original source samples are not
available after encoding.

This architecture intentionally learns only the finite-sampling and
non-orthogonality correction needed to amortize E7. It does not ask an opaque
tokenizer to rediscover the already-qualified latent semantics.

## Frozen data, arms, and exposure

Use the E2 analytic family with 128 training states at seed `17` and 24
disjoint validation states at seed `10017`. Train at resolutions `10` and `14`;
evaluate those plus unseen resolution `18`. Use the existing grid, warped mesh
`a`, and high-resolution remesh `b` geometries unchanged. Canonical decode
queries use resolution `18`.

Create three models by exact deep copy of one initialization:

- `shared`: alternating `(grid-low, mesh-high)` and `(mesh-low, grid-high)`;
- `grid_control`: `(grid-low, grid-high)` every epoch;
- `mesh_control`: `(mesh-low, mesh-high)` every epoch.

Every arm uses 120 epochs, batch size 16, AdamW learning rate `2e-3`, weight
decay `1e-6`, gradient norm cap `1`, 960 optimizer updates, and 30,720 scheduled
source examples. Each batch loss is the mean over its two scheduled sources.
For each source, use equal weights on coefficient-relative MSE and canonical-
decode relative MSE. All arms share data order, initialization, capacity,
optimizer, and exposure.

## Frozen evaluation

Evaluate every arm on high and unseen grid and mesh observations, including
cross-family use of both controls. Evaluate the shared encoder additionally on
low inputs and mesh remesh `b`. Record:

- coefficient NRMSE against the exact E7 teacher;
- canonical-query NRMSE and direct-interpolation comparison;
- high-frequency spectral NRMSE and amplitude ratio;
- effective coefficient rank;
- paired grid/mesh coefficient and decoded mismatch;
- within-family refinement coefficient mismatch;
- remesh coefficient and decoded mismatch;
- source-order maximum absolute coefficient and decoded error;
- per-arm parameter, update, exposure, checkpoint, config, and result hashes.

Also evaluate the shared mechanics on one deterministic 196-point uniform
Monte Carlo sample with equal masses. Record its coefficient and decoded
errors, but attach no pass gate and make no particle qualification claim.

## Frozen gates

The shared encoder clears the semantic gate only if:

1. high-resolution grid and mesh coefficient NRMSE are each `<=0.10`;
2. high-resolution grid and mesh decoded NRMSE are each `<=2x` direct
   interpolation;
3. high-frequency spectral NRMSE is `<=0.25` for each high input;
4. unseen-resolution decoded NRMSE is no more than `1.10x` high-resolution
   decoded NRMSE within each family;
5. high grid/mesh coefficient and decoded mismatch are each `<=0.10`;
6. high/unseen coefficient mismatch is `<=0.15` within each family;
7. remesh decoded NRMSE is `<=1.10x` matched high-mesh NRMSE and remesh
   coefficient mismatch is `<=0.15`;
8. source permutation changes coefficients and decoded outputs by at most
   `1e-8`; and
9. all boundary assertions hold.

Positive shared transfer requires both:

1. shared high-resolution error is no worse than `1.10x` the matched native
   family-only control for both grid and mesh; and
2. the shared macro coefficient NRMSE across high and unseen grid and mesh is
   at most `0.98x` the corresponding cross-family macro error of each
   family-only control.

Classify `amortized_universal_encoder_qualified` only if the semantic and
positive-transfer gates both pass. The next gate may then test a latent
operator acting only on these frozen coefficients.

Classify `amortized_encoder_capable_without_positive_transfer` if the semantic
gate passes but positive shared transfer fails. Keep the coefficient encoder,
but do not instantiate dynamics; first establish a scientifically meaningful
cross-representation advantage rather than merely one model accepting two
array shapes.

Classify `amortized_encoder_not_qualified` if any semantic gate fails. Diagnose
only the learned correction or training objective while keeping the E7 basis
and no-routing boundary frozen.

No threshold, arm, schedule, basis, representation, state split, particle
probe, or classification rule may change after state-level results are
visible.

## Boundary

- train/validation synthetic analytic states only; reserved held-out reads `0`;
- CPU-only, no paid provider, and no external model;
- no operator, temporal transition, rollout, router, family label, or task
  label;
- no original-source bypass after the 52 coefficients are formed;
- the exact E7 solve is a training/evaluation teacher only, not an inference
  path;
- particle input is a mechanics probe only;
- no public or claim-grade promotion from this synthetic encoder test.
