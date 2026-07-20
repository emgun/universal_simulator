# Canonical Latent E7 Function-Space Sufficiency Result

Date: 2026-07-20
Status: `function_space_latent_qualified`

## Decision

The common-latent hypothesis is now positively identified on the frozen
analytic grid/mesh benchmark. Freeze the 52 ordered physical coefficient
semantics as the target latent space. The next experiment should train one
coordinate-and-quadrature encoder to amortize this projection across input
discretizations. Do not return to abstract tokens, regional-token expansion,
family routing, or latent dynamics yet.

This is the first result in E1-E7 that demonstrates the same information-
bearing state—not merely equal tensor shape—from paired grids and warped
meshes.

## Frozen construction

The preregistered contract is
`docs/research/2026-07-20-canonical-latent-e7-function-space-contract.md`.

The latent contains `52` scalar coefficients:

- `49` tensor-product real Fourier modes through physical frequency `3` in
  each coordinate;
- normalized linear `x` and `y` trends;
- one normalized bilinear `xy` trend.

The cutoff was selected from the benchmark's stated maximum frequency before
state-level measurement. A deterministic dense component audit bounded the
localized Gaussian approximation at `0.01885191` NRMSE while representing the
other six components to numerical precision.

For each observation, one shared basis is evaluated at its physical
coordinates and a float64 quadrature-weighted least-squares projection produces
the coefficients. Decoding is direct basis evaluation at arbitrary query
coordinates. There are no learned parameters, representation labels, or source
features retained after projection.

This follows current function-space guidance: define continuous transforms in
physical coordinates, discretize integral quantities with quadrature, and keep
latent interfaces independent of observation resolution:
<https://www.nature.com/articles/s42256-026-01267-z>. It also isolates the
regular-latent-domain idea used by GINO before adding learned geometry transfer
or dynamics:
<https://proceedings.neurips.cc/paper_files/paper/2023/file/70518ea42831f02afc3a2828993935ad-Paper-Conference.pdf>.

## Reconstruction result

| Family | Low NRMSE | High NRMSE | Unseen NRMSE | High interpolation | High / interpolation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Grid | `0.001973` | `0.001921` | `0.001916` | `0.090543` | `0.02121x` |
| Mesh | `0.001946` | `0.001918` | `0.001918` | `0.098671` | `0.01944x` |

The high-resolution latent reconstruction reduces error by `97.88%` versus
grid interpolation and `98.06%` versus mesh interpolation. Both absolute gates
pass by a wide margin, and errors remain essentially unchanged when the input
resolution moves from `10` to `14` to unseen `18`.

The representation compresses `196` high-resolution samples to `52`
coefficients (`3.769x`) and `324` unseen-resolution samples to `52`
coefficients (`6.231x`). Even the low-resolution input is compressed `1.923x`.

## Spectral and semantic result

| Family | High-frequency NRMSE | HF amplitude ratio | High/unseen coefficient mismatch |
| --- | ---: | ---: | ---: |
| Grid | `0.009428` | `0.994146` | `0.000238` |
| Mesh | `0.009422` | `0.995119` | `0.000138` |

Unlike E2-E6, fine structure is preserved rather than attenuated or amplified.

Paired high-resolution grid/mesh coefficient relative mismatch is only
`0.0003716`; decoded mismatch at the common query set is `0.0001362` target-
normalized NRMSE. Linear CKA is `0.99999995`, and paired retrieval is `1.0`.
At unseen resolution the coefficient mismatch falls to `0.0002059`. A second
high-resolution mesh warp remains close at `0.0007377` coefficient mismatch and
`0.0002836` decoded mismatch.

These metrics are meaningful here because coefficient index has a fixed
physical basis definition. The same CKA or retrieval score on independently
learned opaque features would not establish semantic compatibility.

## Numerical integrity

- every design matrix has rank `52`;
- condition numbers range from `4.947` to `6.660`, below the frozen `10` gate;
- weighted source projection NRMSE is `0.00130-0.00200`;
- source permutation changes decoded output by at most `2.05e-14`;
- both complete raw results are byte-identical;
- no model parameters, optimizer, operator, held-out data, provider, router,
  label, or source-feature bypass were used.

## What this proves—and does not

E7 proves that the analytic field family has a compact common physical
coefficient representation that is stable across the tested grids, warped
meshes, resolutions, and remeshing. The negative E2-E6 codecs failed to learn
or preserve a space that demonstrably exists. That directly validates the
user's encoder-first diagnosis and removes any technical rationale for hiding
the failure behind a router.

E7 does not yet show that a neural encoder can infer these coefficients, that
the basis covers particles, complex domains, vector/tensor fields, boundary
semantics, or discontinuities, or that a shared latent operator can evolve the
coefficients. The Fourier cutoff is benchmark-informed and therefore not a
general-purpose capacity prescription.

## Reproduction and boundary

Complete results at `/tmp/canonical_latent_e7_function_space_v1/result.json`
and `/tmp/canonical_latent_e7_function_space_v2/result.json` are
byte-identical. Raw result SHA-256:
`00f6c7fdfe3312b9a7b580e69239e2a159819a6b4dd3e4eac9a5a64e19311dfe`.
Config SHA-256:
`c1637a4062d8778d8091ce204fcf4ae02111c6796064c35bd6070cf300a800ee`.

## Next coherent experiment

Freeze E8 as one universal encoder-to-coefficient test:

1. use the exact E7 projection as a teacher and the 52 coefficient positions as
   fixed semantic outputs;
2. train one coordinate-, value-, and quadrature-aware set encoder with no
   representation label on mixed grid and warped-mesh observations;
3. compare its coefficient error and decoded reconstruction with the exact E7
   projection, matched grid-only and mesh-only encoders, and direct
   interpolation;
4. require paired coefficient semantics, unseen-resolution stability, source-
   order invariance, high-frequency preservation, and positive shared transfer;
5. add a particle sampling of the same physical states as a mechanics-only
   out-of-format probe, but do not claim particle qualification without a
   separately frozen scientific distribution.

Only after one learned encoder approximates this fixed common space should a
latent operator be instantiated.
