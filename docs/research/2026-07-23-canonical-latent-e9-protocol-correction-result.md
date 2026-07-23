# Canonical Latent E9 Protocol Correction Result

Date: 2026-07-23
Status: `geometry_universal_projection_not_qualified`

## Decision

E9 is negative before physical-state measurement. Its frozen contract required
the 52-dimensional quadrature Gram matrix to have condition number at most
`10`, but the initial exploratory harness accidentally checked the condition
number of the weighted design matrix instead.

Independent review caught the mismatch before any result was committed. The
repaired runner computes both quantities and fails closed before evaluating
field values. The uncommitted exploratory state metrics are discarded and do
not support a scientific claim.

## Corrected geometry-only evidence

Across 50 frozen grid, warped-mesh, uniform-particle, and warped-particle
designs:

- weighted-design condition range: `4.728984` to `6.899011`;
- Gram condition range: `22.363294` to `47.596351`;
- basis rank: `52` throughout;
- positive, finite, normalized quadrature masses: required;
- training, validation, and held-out state reads: `0`.

Because `cond(Phi^T W Phi) = cond(sqrt(W)Phi)^2`, every geometry violates the
literal E9 Gram `<=10` gate even though it satisfies the E7 weighted-design
criterion.

Compact artifact:
`docs/research/artifacts/canonical_latent_e9_geometry_universal_projection_result.json`.
The corrected full result SHA-256 is
`02c862d3ee63ba2021147e805445e2508fae3b85fe7bd196b0e7cbdff8bd2bbe`.

## Repair boundary

E9 is not relaxed after measurement. E10 is a separate preregistered
experiment with fresh geometry seeds and fresh validation states. It restores
E7's original invariant, `cond(sqrt(W)Phi)<=10`, and independently records the
equivalent Gram bound `<=100`. It also gates every cross-family realization
pair rather than comparing averages and binds source hashes plus Git revision.

No operator, optimizer, router, task/family label, provider, or held-out data
was used.
