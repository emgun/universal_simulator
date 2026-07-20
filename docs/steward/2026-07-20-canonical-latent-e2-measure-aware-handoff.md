# Canonical Latent E2 Measure-Aware Handoff

Date: 2026-07-20

## Decision

The measure-aware learned-query Perceiver codec is `not_qualified`. Do not
freeze it, invoke a latent operator, extend epochs, or add task/family/modality
routing. The next bounded architecture test is a RIGNO-style regional
interaction encoder under the identical E2 harness.

## Implemented

- `CanonicalPointEncoder` now requires positive `geom['measure']`, uses a
  measure-weighted centroid, and combines local kernels with normalized
  quadrature weights.
- `discretization_mismatch_report` compares outputs from multiple input
  discretizations on one physical query set.
- `scripts/run_canonical_latent_e2_benchmark.py` generates disjoint analytic
  states, regular grids, Jacobian-weighted irregular remeshings, neutral query
  points, and identically initialized exposure-matched shared/grid/mesh codec
  arms.
- The gate now includes inverse-distance interpolation and refinement
  convergence, closing the false-pass hole found in the first shakedown.

## Result

Compact artifact:
`docs/research/artifacts/canonical_latent_e2_measure_aware_result.json`.

- shared canonical NRMSE: grid `0.302411`, mesh `0.306910`, remesh `0.316649`;
- controls: grid `0.282391`, mesh `0.276096`;
- interpolation: grid `0.090543`, mesh `0.098671`;
- paired retrieval `1.0`, CKA `0.9984`, physical/latent rank preserved;
- mismatch grows from `0.06636` low-resolution to `0.17469` high-resolution;
- no operator, held-out read, representation/task model input, provider call,
  or GPU occurred.

## Verification

- New and adjacent encoder/decoder/codec suite: `25 passed`.
- Final focused measure-aware/benchmark suite: `14 passed`.
- Complete `tests/unit` suite: passed outside the macOS sandbox so localhost
  and `torch_shm_manager` tests could use the required OS facilities.
- Ruff, Black check, Python compilation, compact-result JSON parse, and
  `git diff --check`: passed.
- The final exposure-fair run was reproduced from scratch at
  `/tmp/canonical-latent-e2-measure-aware-final-v4`; full result SHA-256
  `793169a871f661a02f514c970376043f403193d0841bd695aab0e1c7eea980a4`.

## Next coherent arc

Implement a compact regional interaction graph encoder as a challenger, not a
new project-wide replacement. Reuse the exact analytic states, measures,
decoder, initialization budget, training exposure, checkpoints, and evaluation
gates. Compare it with this frozen Perceiver result. Continue to E3 only if it
passes absolute reconstruction and refinement convergence as well as alignment.
