# Canonical Latent E17 Truth Calibration Handoff

Date: 2026-07-27

## Status

E17 truth calibration is qualified; nonlinear closure remains untested.

The final `216/324/432` de-aliased spatial rung passes all six analytic cases.
The worst full-field error is the stress case at `6.974e-5` versus `<=2e-4`.
Its active-coefficient error is `6.726e-10`, energy mismatch `1.212e-8`, mean
drift `6.550e-15`, and the global nonlinear energy-rate residual is
`3.262e-16`.

## Boundary

- execution HEAD: `0e281dd4aa3951c7213c4cacbe402a487d98b3c6`;
- compact calibration:
  `cbabbf03d2220963523f8a9ada743dd35589ab47811dc5c3b253b8e11cb7bea2`;
- runtime: float64 CPU, one intra-op and inter-op thread;
- focused tests before launch: `16/16`;
- training reads: `0`;
- validation reads: `0`;
- held-out reads: `0`;
- provider calls: `0`;
- encoder updates: `0`;
- routing decisions: `0`.

Two lower spatial rungs failed only full-field convergence and are preserved in
the contract/result. No threshold, PDE, time step, case, population, metric, or
classification was changed.

## Next implementation slice

Completed in reviewed source without calling the registered builders:

1. exact stratified training-population construction;
2. exact 32-pair validation tail construction; and
3. split-qualified identity, uniqueness, canonical-hash, and cross-split
   overlap gates.

Complete source and tests for:

1. sealed E15 neutral componentwise L-BFGS reconstruction;
2. equality-constrained triad-supported least squares;
3. closure-pair, rollout, energy, semigroup, stress, and cross-observation
   evaluation; and
4. deterministic replicated evidence publication.

Stop before calling a population constructor until the complete runner is
committed clean and a fresh independent pre-state review returns GO.
