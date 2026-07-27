# Canonical Latent E16 Multi-Realization Robustness Result

Date: 2026-07-27
Status: `both_practical_recovery_packages_stable`

## Decision

E16 independently classifies
`both_practical_recovery_packages_stable`.

Across the sealed E15 realization and both preregistered fresh training
realizations:

- fresh-moment deterministic schedule-weighted AdamW repairs each
  realization's own ordered-AdamW E12 checkpoint; and
- schedule-weighted componentwise strong-Wolfe L-BFGS recovers from the
  canonical neutral generator.

Every one of the six registered realization/package recovery bits is true.
The practical linear-generator recovery result is therefore stable under the
frozen multi-realization test. Stop linear optimizer archaeology. E16
authorizes preregistering the nonlinear E17 contract; it does not itself
qualify nonlinear dynamics.

## Registered results

| Realization | Control | Basis-action NRMSE | Composite rollout NRMSE | Final HF NRMSE | Recovery |
| --- | --- | ---: | ---: | ---: | --- |
| `r1` | E12 ordered-AdamW checkpoint | `0.0898734` | `0.00581277` | `0.295232` | fail |
| `r1` | deterministic AdamW restart | `0.000655134` | `0.000283697` | `0.00524745` | pass |
| `r1` | componentwise L-BFGS neutral | `1.39219e-5` | `4.94370e-6` | `1.89165e-5` | pass |
| `r2` | E12 ordered-AdamW checkpoint | `0.0943668` | `0.00547202` | `0.278676` | fail |
| `r2` | deterministic AdamW restart | `0.00326400` | `0.000164586` | `0.00443852` | pass |
| `r2` | componentwise L-BFGS neutral | `1.90208e-5` | `7.34380e-6` | `4.38561e-5` | pass |

Both fresh ordered-AdamW checkpoints repeat E12's decisive failure pattern:
generator identification and high-frequency recovery fail while average
rollout remains small. Fresh optimizer moments then repair both checkpoints
without changing the encoder, latent, generator class, data law, objective,
or thresholds. Neutral componentwise L-BFGS recovers more tightly on both
realizations.

The registered conjunction is:

| Package | Sealed E15 | `r1` | `r2` | Stable |
| --- | --- | --- | --- | --- |
| deterministic AdamW restart | pass | pass | pass | yes |
| componentwise L-BFGS neutral | pass | pass | pass | yes |

## Preflight and coverage

Both realization preflights pass before any checkpoint replay:

- all six literal elementary schedules and seeds match;
- all trajectory, parameter, schedule, and occurrence-count records are
  canonical little-endian byte records;
- all 48-dimensional nonconstant input covariances are full rank;
- all rotation-plane Gram matrices are full rank;
- both mode-tied oracle Jacobians have rank 12;
- every one of the 48,000 schedule occurrences per regime covers the full
  2,048-transition population; and
- all state/parameter tensors are finite and within the frozen ranges.

The largest input-covariance condition number is `6695.85`; this is recorded,
not post-hoc gated. E15's grouped/literal output, weighted loss, literal
schedule loss, all-ones loss, gradient, and exact separability checks pass for
both fresh checkpoints before either E16 package runs.

Each realization contains exactly:

- four generator-identification controls;
- 16 validation cells;
- 7,056 unique mode-resolved keys;
- 20 literal argmax cells; and
- three training records.

All eight literal E15 recovery gates pass for both packages on both fresh
realizations. All outputs are finite.

## Evidence and provenance

The scientific run executed from clean HEAD
`139c1c0d1ecf626b3a962cb4824337a75a6806f9` under Python `3.12.7`, PyTorch
`2.7.0`, deterministic float64 CPU execution, and one intra-op and inter-op
thread.

The canonical artifact directory contains exactly:

| Output | Bytes | Raw SHA-256 |
| --- | ---: | --- |
| Evidence bundle | `2,043,686` | `71fc490c2bc361fbf0b26d5bfcccfc460bcf5af223b5000d1e6043672504a586` |
| Compact result | `10,335,882` | `6716273a3ea980f7d24462ec3e40eb37091d229d524aec5f9a0ad89bbb9d325a` |
| Detached manifest | `1,763` | `6af927037eebeebc3a9a95842d549c633279391d28715779b7fc04c05b59720f` |

The two raw replicate files are byte-identical at SHA-256
`621e59af7dca5312963db4b32b88aafe02f3f9d03fbbf63ed971e0c059e637b1`.
Their canonical payload SHA-256 is
`2defe29b2b13839215484bc8595ec3a4d86edd2c384adf57e09fe2fa67f7c8b8`.
Removing the replication record from the complete result yields either
replicate exactly.

Independent post-result review reopened the archive and E15 seal, verified all
source and artifact bindings, deterministic gzip metadata, archive order,
member byte counts and hashes, raw/canonical replica identity, preflights,
objective integrity, finiteness, coverage, state reads, and boundary. It
recomputed every recovery bit, both stability conjunctions, the classification,
and nonlinear authorization and returned GO with no P0/P1 blocker.

Focused E16 tests pass `36/36`; the related E12-E16 suite passes `104/104`.
The complete clean post-result unit suite also passes outside the managed
sandbox; the sandboxed attempt's loopback/shared-memory denials were
environmental rather than product failures.

## Boundary

E16 reads 1,536 unique training trajectories, 256 frozen validation
trajectories, and zero held-out trajectories. It performs zero provider calls,
encoder updates, routing decisions, label inputs, or source bypasses.

The result is limited to a smooth synthetic periodic scalar linear
advection-diffusion family in the frozen 52-coefficient E10 latent. It does not
qualify nonlinear physics, particles as dynamical state, a broader basis, a
joint optimizer, or deployment-scale training. It shows that the two
registered practical recovery packages are stable across the preregistered
linear realizations.

## Next gate

Preregister E17 as a representation-preserving nonlinear closure test in the
same 52 coefficients:

1. retain the frozen E10 projection, latent semantics, routing closure, and the
   identified linear generator;
2. use a de-aliased 2-D periodic scalar viscous Burgers family that excites
   both spatial axes;
3. nest the linear model inside a constrained quadratic convection term;
4. include a frozen linear-only negative control and projected-truth ceiling;
5. gate closure/conditional derivative variance before interpreting a failed
   rollout as an operator failure; and
6. keep the claim limited to the tested quadratic nonlinear family.

Do not introduce a router, new encoder, Koopman lift, black-box neural ODE, or
neural operator in this first nonlinear gate. Those would change the
representation or capacity hypothesis before nonlinear closure in the
qualified semantic latent has been tested directly.
