# Canonical Latent E17 Truth Calibration Result

Date: 2026-07-27

## Decision

The E17 de-aliased truth solver is qualified at the final registered
`216/324/432` primary/reference/comparison spatial rung.

All six literal analytic cases pass the unchanged spatial, temporal, energy,
mean, finiteness, mask, runtime, source, and predecessor gates. Training,
validation, and held-out state reads are all zero. E17 may proceed to
implementing its frozen population, same-latent/different-tail diagnostic, and
constrained quadratic fit. This result does not qualify nonlinear closure.

## Decisive metrics

| Case | Active coefficient NRMSE | Full-field NRMSE | Energy mismatch | Mean drift |
| --- | ---: | ---: | ---: | ---: |
| single x | `1.815e-13` | `2.676e-15` | `4.790e-15` | `2.037e-14` |
| single y | `1.161e-13` | `2.620e-15` | `4.762e-15` | `6.584e-17` |
| two-mode x | `2.078e-12` | `3.645e-12` | `4.060e-13` | `7.710e-14` |
| two-mode y | `2.069e-12` | `3.645e-12` | `4.060e-13` | `1.754e-16` |
| mixed | `7.187e-12` | `1.058e-11` | `5.280e-13` | `2.763e-15` |
| stress | `6.726e-10` | `6.974e-5` | `1.212e-8` | `6.550e-15` |

The literal gates are full-field and active-coefficient NRMSE `<=2e-4`,
relative energy mismatch `<=5e-4`, mean drift `<=1e-11`, and nonlinear
energy-rate residual `<=1e-10`. The observed maximum nonlinear energy-rate
residual is `3.262e-16`.

## Resolution history

The gate rejected two lower spatial rungs before any population construction:

- `96/144/192`: stress full-field NRMSE `0.00191923`;
- `144/216/288`: stress full-field NRMSE `0.000485588`; and
- `216/324/432`: stress full-field NRMSE `0.0000697438`, pass.

Across all three, low-mode, energy, mean, and nonlinear conservation quantities
were already far inside threshold. The failures were correctly attributed to
unresolved full-field tail energy, not to the 52-coordinate latent.

## Provenance and boundary

The final run executed once from clean HEAD
`0e281dd4aa3951c7213c4cacbe402a487d98b3c6` with one intra-op and inter-op
thread. Contract, runner, and focused-test source SHA-256 values were
`124366d9...`, `4810320f...`, and `35fd6498...`; each matched HEAD exactly.
All predecessor, sealed-E15, triad, strict-mask, source, runtime, and finiteness
preflights passed before integration.

The compact calibration record is
`docs/research/artifacts/canonical_latent_e17_truth_calibration_result.json`.
Its SHA-256 is
`cbabbf03d2220963523f8a9ada743dd35589ab47811dc5c3b253b8e11cb7bea2`.
The complete emitted calibration JSON had SHA-256
`242b4872db77b5ff0e66a766d3fa14d9e85e69eeec11e237a5e16184a712361c`.

This calibration uses six fixed analytic states only. It reads no E17 training,
validation, or held-out state and makes no encoder update, route, provider call,
or nonlinear qualification claim.

## Next gate

Implement the registered deterministic training population and 32 validation
closure pairs. Before constructing either, extend tests to prove:

- exact stratified RNG draw order and canonical hashes;
- conjugate-symmetric tail construction and identical pair latents;
- training completion before validation construction/read;
- frozen E15 linear-trunk reconstruction and exact model/matrix hashes; and
- fail-closed behavior on any population, rank, condition, or provenance
  mismatch.

Then obtain a fresh independent pre-state GO. Do not run the scientific
population from the current calibration-only runner.
