# Canonical Latent E10 Geometry-Universal Projection Result

Date: 2026-07-23
Status: `geometry_universal_projection_qualified`

## Decision

The exact E7 function projection is now qualified as one canonical encoder for
the frozen grid, continuously warped mesh, uniform Monte Carlo particle, and
warped Monte Carlo particle observation processes.

This is the representation result the roadmap needed before dynamics. It
shows that these inputs do not require an under-the-hood family router: the
same coordinate-, value-, and quadrature-defined least-squares map produces
the same ordered 52-dimensional physical state.

The result does not qualify particle dynamics, arbitrary distributions,
domains, topologies, or public simulator claims.

## Frozen protocol

The E10 contract was frozen before state access at
`docs/research/2026-07-23-canonical-latent-e10-geometry-universal-projection-contract.md`.
It corrected E9 without reinterpreting it:

- fresh 24-state validation split at seed `10023`;
- fresh geometry seeds `30000` through `30007`;
- exact float64 solve of `G c = Phi^T W u`;
- no learned parameters, labels, routes, source bypass, or operator;
- full Cartesian comparison of high-budget geometry realizations;
- `moment_only` and diagonal-Gram causal ablations.

## Numerical admissibility

All 50 geometries had rank `52`, positive finite normalized masses, and passed
both equivalent E7 conditioning checks:

| Quantity | Observed range | Gate |
| --- | ---: | ---: |
| `cond(sqrt(W)Phi)` | `4.947116` to `9.668081` | `<=10` |
| `cond(Phi^T W Phi)` | `24.473961` to `93.471783` | `<=100` |

No state was read until this complete preflight passed.

## Encoder evidence

High-budget results:

| Observation family | Coefficient NRMSE | Decoded NRMSE | High-frequency NRMSE |
| --- | ---: | ---: | ---: |
| Grid | `0.00043151` | `0.00229844` | `0.01231419` |
| Warped mesh | `0.00043333` | `0.00229948` | `0.01231939` |
| Uniform particles | `0.00177923` | `0.00250097` | `0.01306288` |
| Warped particles | `0.00174756` | `0.00249622` | `0.01303675` |

High/low coefficient-error ratios are `0.3193`, `0.3022`, `0.6347`, and
`0.6207`, respectively, so every family improves with more observations.

The worst of every high-budget cross-family realization pair is only
`0.00443686` in coefficient relative mismatch and `0.00202842` in decoded
NRMSE, both far below `0.10`. The largest family-pair comparison evaluates all
`64` realization pairs; no averaging precedes the gate.

## Causal evidence for the Gram solve

High-budget macro coefficient NRMSE:

| Path | NRMSE |
| --- | ---: |
| Exact Gram projection | `0.00109791` |
| Moment only | `0.60348751` |
| Diagonal Gram | `0.60846429` |

The exact-to-ablation ratios are `0.0018193` and `0.0018044`. Geometry
sufficient statistics are therefore causal, not incidental bookkeeping.

For this 52-mode basis, an exact factorization is preferable to a learned
inverse: it is identifiable, deterministic, cacheable per geometry, and adds
no approximation error or family specialization. A learned preconditioner may
become worthwhile for much larger or poorly conditioned bases, but it is not
the highest-signal mechanism here.

## Reproducibility and provenance

Two complete runs in separate directories are byte-identical:

- result SHA-256:
  `3cc40e8e659f3e7f28b97103d8a72aef29aaa126f593377c6365ccb844aaade5`;
- execution Git HEAD:
  `a624df3374b4d7c0a2033fc47c2e71d46aab5390`;
- config SHA-256:
  `9bc299aa822afbfac68e3b8830986faf91b6f80bcf363fac3be74b4d665e3017`;
- both contracts, the shared runner, and the E10 entrypoint are independently
  hash-bound in the artifact and byte-match that Git HEAD;
- the worktree was clean before state access;
- the executable rejected every configuration except the exact 24-state,
  128-calibration, eight-realization frozen contract;
- all `1,200` family/budget/realization/state records per path, `3,600` total,
  retain coefficient, decoded, spectral, design-rank, and source-order
  evidence. The worst exact
  per-state coefficient/decoded NRMSE is `0.0237070` / `0.0175801`.

Maximum source-order coefficient/decoded errors are `1.12e-14` / `1.84e-14`
for exact projection, `4.44e-16` / `2.66e-15` for moment-only, and `1.89e-15`
/ `5.33e-15` for diagonal-Gram. Every path is independently permutation
invariant under the frozen `1e-10` gate.

Compact artifact:
`docs/research/artifacts/canonical_latent_e10_geometry_universal_projection_result.json`.

Training and held-out reads are zero. No operator, temporal transition,
optimizer update, provider call, route, family/task model input, or
original-source bypass exists.

## Interpretation and next gate

E8 failed because its learned moment correction omitted the sampling Gram
geometry. E10 shows that the canonical representation itself was not the
problem. Once the actual function-space projection is used, grids, meshes, and
the two particle observation processes meet in one fixed semantic latent
without routing.

Freeze this projection as the encoder for the tested function class. Next
preregister one coefficient-to-coefficient latent operator, with physical
parameters and time increment explicit, and compare shared versus matched
operator controls on paired trajectories. Keep representation identity out of
the operator input. Require positive transfer or data efficiency in addition
to rollout accuracy before claiming universal dynamics.
