# Canonical Latent E9 Geometry-Universal Projection Contract

Date: 2026-07-23
Status: frozen before state-level measurement

## Question

E8 showed that one learned quadrature-moment encoder preserves the frozen E7
semantics on fixed grids and warped meshes, but its geometry-blind correction
fails on a new point sampling. Is the correct universal encoder the exact
coordinate-and-measure-defined function projection itself, rather than a neural
approximation to a small, well-conditioned Gram solve?

E9 tests zero-shot geometric universality of the E7 projection across regular
grids, continuously warped meshes, uniform Monte Carlo particles, and warped
Monte Carlo particles with nonuniform masses. It does not train a model or
instantiate dynamics.

## Research basis and branch decision

Current function-space guidance requires coordinate-defined operations,
quadrature weights, and discretization-independent latent interfaces:
<https://www.nature.com/articles/s42256-026-01267-z>. It explicitly permits
Monte Carlo integration for random point clouds when the sampling measure is
known.

Function Encoder work infers coefficients for a new sampled function by solving
the Hilbert-space least-squares system whose matrix is the basis Gram matrix and
whose right-hand side contains function/basis inner products:
<https://ww3.math.ucla.edu/wp-content/uploads/2025/09/2025_FunctionEncoder_OC_final.pdf>.
Basis-to-Basis operator learning likewise separates function-space projection
from the learned coefficient-to-coefficient operator:
<https://www.sciencedirect.com/science/article/pii/S0045782524009009>.

Learning an approximate inverse is justified when a large solve is the
bottleneck; recent neural-preconditioner work targets large sparse systems and
measures actual solve-time gains:
<https://papers.neurips.cc/paper_files/paper/2025/hash/23fcc63005ac1a6e460ec4e209d17607-Abstract-Conference.html>.
E7's matrix is dense but only `52 x 52`, has condition number below `10`, and
depends only on source geometry. Replacing its exact solve with a large neural
network would weaken numerical guarantees without establishing a relevant
speed bottleneck. E9 therefore qualifies the mathematical encoder directly.

## Frozen latent and candidate

Reuse E7's ordered 52-dimensional tensor Fourier plus normalized trend basis
without modification.

For basis evaluations `Phi`, normalized positive quadrature masses `w`, and
source values `u`, compute

`G = Phi^T diag(w) Phi`

and

`b = Phi^T diag(w) u`.

The candidate encoder returns `c = solve(G, b)` in float64. It must fail closed
unless `G` has rank `52`, condition number `<=10`, and finite positive masses.
The factorization is geometry-only and may be cached when multiple states share
one observation geometry. No learned parameter, exact-source bypass, family
label, or route is present.

Freeze two causal ablations:

- `moment_only`: `c = b`, matching the information available before E8's
  learned correction;
- `diagonal_gram`: `c = b / diag(G)`, testing whether per-mode scaling alone
  is sufficient.

All three paths decode only as `Phi(y)c`.

## Frozen physical states and canonical target

Use the 24 disjoint E2 validation states at seed `10017`. No training state or
reserved held-out state is read.

Define one representation-independent coefficient target per physical state by
projecting its analytic field on E7's deterministic `128 x 128` midpoint
quadrature grid. Decode every candidate and ablation on the existing canonical
`18 x 18` query grid.

## Frozen geometry distributions

Every evaluation family represents the same normalized square and scalar
field. Low/high observation budgets are:

- `grid`: midpoint grids at resolutions `10` and `18`;
- `warped_mesh`: eight deterministic warp draws at resolutions `10` and `18`;
- `uniform_particles`: eight deterministic nested iid-uniform point sets with
  `256` and `576` points and equal Monte Carlo masses;
- `warped_particles`: eight deterministic nested iid-uniform parameter-space
  point sets transformed by the same smooth warp family, with `256` and `576`
  points and Jacobian-proportional masses.

For warped families, draw `a,b` deterministically from `[-0.28, 0.28]` and
project any draw outside `|a|+|b|<=0.42` onto that boundary. The warp is

`x = u + a sin(2 pi u) sin(2 pi v) / (2 pi)`

`y = v + b sin(2 pi u) sin(2 pi v) / (2 pi)`.

Its Jacobian is used as the quadrature mass correction and must remain
positive. Geometry seeds are `20000` through `20007`; high point sets extend
their paired low set rather than resampling it.

A geometry-only preflight, performed without physical-state values, found rank
`52` and condition-number ranges `4.981` to `6.899` for uniform particles and
`5.007` to `6.983` for warped particles. These observations only establish
numerical admissibility and do not set or preview state-level performance
gates.

## Frozen metrics

For each encoder path, family, budget, geometry realization, and physical state,
record:

- coefficient NRMSE to the dense canonical E7 target;
- canonical-query decoded NRMSE;
- high-frequency spectral NRMSE and amplitude ratio;
- coefficient effective rank;
- source-order maximum absolute coefficient and decoded difference.

For the exact candidate also record:

- design rank, condition number, and singular-value extrema;
- low/high convergence within each family;
- same-state coefficient and decoded mismatch between every high-budget
  family pair;
- realization dispersion within each stochastic family;
- source node count, compression ratio, parameter count, optimizer updates,
  held-out reads, routes, and provider calls;
- complete config and result hashes.

## Frozen gates

Classify `geometry_universal_projection_qualified` only if:

1. every geometry has rank `52`, condition number `<=10`, and positive finite
   normalized masses;
2. high-budget coefficient NRMSE to the canonical target is `<=0.10` for every
   family;
3. high-budget decoded NRMSE is `<=0.10` for every family;
4. high-budget high-frequency spectral NRMSE is `<=0.25` for every family;
5. every high-budget cross-family coefficient and decoded mismatch is
   `<=0.10`;
6. high-budget coefficient NRMSE is `<=0.90x` low-budget NRMSE for both
   particle families and does not exceed `1.10x` low-budget NRMSE for either
   structured family;
7. maximum high-budget realization coefficient dispersion is `<=0.10`;
8. joint source permutation changes coefficients and decoded outputs by at most
   `1e-10`;
9. the exact candidate's high-budget macro coefficient NRMSE is at most
   `0.50x` both the `moment_only` and `diagonal_gram` macro errors; and
10. every boundary assertion holds.

If qualified, freeze the exact E7 projection as the canonical input/output
interface and permit preregistration of the first coefficient-space latent
operator gate. The qualification remains limited to this scalar function
family, normalized-square domain, basis cutoff, and the four frozen sampling
processes.

Classify `projection_identifiable_but_sampling_unstable` if every design gate
passes but another semantic, convergence, or causal gate fails. Keep dynamics
closed and repair the basis, quadrature process, or sampling budget—not a
router.

Classify `geometry_universal_projection_not_qualified` if any design fails.
Reconsider the basis or regularized projection before dynamics.

No basis, geometry seed, distribution, sample count, threshold, ablation, or
classification rule may change after state-level results are visible.

## Boundary

- validation-only analytic states; training and reserved held-out reads `0`;
- learned parameters and optimizer updates `0`;
- no operator, temporal transition, rollout, router, family label, or task
  label;
- no original-source access after coefficients are formed;
- CPU-only, no paid provider, and no external model;
- particle qualification is only for the frozen Monte Carlo observation
  processes, not particle dynamics, arbitrary densities, topologies, vector
  fields, or real solver data;
- no public or claim-grade promotion from this synthetic projection test.
