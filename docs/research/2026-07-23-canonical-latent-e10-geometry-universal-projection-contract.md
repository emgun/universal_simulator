# Canonical Latent E10 Geometry-Universal Projection Repair Contract

Date: 2026-07-23
Status: frozen before E10 state-level measurement

## E9 protocol correction

E9 is negative before valid state measurement. Its contract required
`cond(G)<=10` for `G=Phi^T W Phi`, but its first harness implementation
incorrectly gated `cond(sqrt(W)Phi)<=10`. Independent review caught the
mismatch after an uncommitted exploratory result. The repaired E9 harness
computes both quantities and stops before physical-state reads: its frozen
geometries have Gram condition numbers `22.363` to `47.596`, so E9 is
`geometry_universal_projection_not_qualified`.

E10 does not reinterpret or relax E9 after measurement. It defines a new
numerical contract from E7's original design invariant, uses fresh physical
states and fresh geometry seeds, and repairs the other review findings before
state-level access.

## Question

Does the exact E7 function projection provide one zero-shot canonical encoder
across grids, continuously warped meshes, uniform Monte Carlo particles, and
warped Monte Carlo particles when evaluated under the correctly specified E7
conditioning invariant?

## Research basis

Function-space neural-operator guidance requires coordinate-defined operations,
quadrature weights, and discretization-independent latent interfaces:
<https://www.nature.com/articles/s42256-026-01267-z>.

Function Encoders infer new-function coefficients through the Gram
least-squares system computed from measurements:
<https://ww3.math.ucla.edu/wp-content/uploads/2025/09/2025_FunctionEncoder_OC_final.pdf>.
Basis-to-Basis operator learning separates that function projection from the
learned coefficient-space operator:
<https://www.sciencedirect.com/science/article/pii/S0045782524009009>.

## Frozen numerical invariant

For weighted design `A=sqrt(W)Phi`, E7's actual fail-closed rule is

`rank(A)=52` and `cond_2(A)<=10`.

Because `G=A^T A`,

`cond_2(G)=cond_2(A)^2<=100`.

E10 must compute and record both independently and require both inequalities.
Masses must be finite, strictly positive, and normalized to one within float64
tolerance. No state value may be read if any geometry fails this preflight.

## Frozen candidate and ablations

Keep the E7 basis and coefficient ordering unchanged. Compute

`G=Phi^T W Phi`, `b=Phi^T W u`, and `c=solve(G,b)` in float64.

The geometry factorization may be cached. There are no learned parameters,
labels, routes, or source bypasses.

Freeze the same causal ablations:

- `moment_only`: `c=b`;
- `diagonal_gram`: `c=b/diag(G)`.

All paths decode only as `Phi(y)c`.

## Fresh states and geometries

Use 24 validation-only analytic states at seed `10023`. This split was not read
by E9. Training and reserved held-out reads remain zero.

Use geometry seeds `30000` through `30007`, also unread by E9. Preserve E9's
four geometry families, formulas, warp limits, nested low/high construction,
and observation budgets:

- grid resolutions `10` and `18`;
- eight warped meshes at resolutions `10` and `18`;
- eight iid-uniform particle sets at `256` and `576` points with equal masses;
- eight warped iid particle sets at `256` and `576` points with
  Jacobian-proportional masses.

The representation-independent target remains the `128 x 128` E7 projection,
and canonical decode queries remain `18 x 18`.

A geometry-only E10 preflight, before state access, found weighted-design
condition numbers `4.947` to `9.668` and Gram condition numbers `24.474` to
`93.472`. No physical field value or state-level performance metric informed
this contract.

## Frozen metrics and repaired semantics

Record E9's coefficient, decoded, spectral, convergence, rank, realization,
compression, and invariance metrics.

For every high-budget family pair, compute mismatch for every Cartesian pair
of geometry realizations. Gate the maximum same-state coefficient and decoded
mismatch, not mismatch after averaging realizations. Record pair count, mean,
and maximum.

Record:

- positive/finite/normalized mass assertions;
- weighted-design and Gram condition numbers;
- config SHA-256;
- E9 shared-runner SHA-256;
- E10 entrypoint SHA-256;
- both contract SHA-256 values;
- Git HEAD at execution;
- final result SHA-256 alongside the committed artifact.

## Frozen gates

Classify `geometry_universal_projection_qualified` only if:

1. every geometry has rank `52`, weighted-design condition `<=10`, Gram
   condition `<=100`, and admissible normalized masses;
2. high-budget coefficient NRMSE to the canonical target is `<=0.10` for every
   family;
3. high-budget decoded NRMSE is `<=0.10` for every family;
4. high-budget high-frequency spectral NRMSE is `<=0.25` for every family;
5. the maximum over every high-budget cross-family realization pair is
   `<=0.10` for both coefficient and decoded mismatch;
6. high-budget coefficient NRMSE is `<=0.90x` low-budget NRMSE for both
   particle families and `<=1.10x` for both structured families;
7. maximum high-budget realization coefficient dispersion is `<=0.10`;
8. source permutation changes coefficients and decoded outputs by at most
   `1e-10`;
9. exact high-budget macro coefficient NRMSE is at most `0.50x` both ablations;
10. provenance and every boundary assertion pass.

If qualified, freeze the exact E7 projection as the canonical encoder and
permit preregistration of the first coefficient-space operator gate.

Classify `projection_identifiable_but_sampling_unstable` if numerical design
and provenance pass but another scientific gate fails. Repair only the basis,
quadrature distribution, or sampling budget.

Classify `geometry_universal_projection_not_qualified` if design or provenance
fails. Keep dynamics closed.

No seed, geometry, basis, budget, threshold, ablation, provenance rule, or
classification may change after E10 state-level results are visible.

## Promotion-blocking implementation clarification

Independent review after the first locked numerical run found that the CLI
still accepted reduced state, calibration, and realization counts, and that
the provenance gate checked only digest length. No promotion occurred.

Before a final locked rerun, the executable must:

- accept only the exact complete frozen configuration above and reject every
  override before geometry or state access;
- prove that both contracts and both runner files byte-match the recorded Git
  HEAD;
- require a clean worktree before state access;
- retain the already-required per-state coefficient, decoded, spectral,
  design-rank, and source-order records.

The repaired implementation must be committed first. Only a subsequent
byte-identical rerun from that clean revision can support promotion. This
clarification changes no state, geometry, metric, or scientific threshold.

## Boundary

- synthetic validation states only; no training or reserved held-out reads;
- learned parameters and optimizer updates `0`;
- no operator, time transition, rollout, router, family/task label, provider,
  or original-source bypass;
- particle qualification is limited to the two frozen Monte Carlo observation
  processes on the normalized square, not particle dynamics or general
  topology/field/domain claims;
- no public or claim-grade promotion from this synthetic projection test.
