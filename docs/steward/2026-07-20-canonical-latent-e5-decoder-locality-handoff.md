# Canonical Latent E5 Decoder-Locality Handoff

Date: 2026-07-20

## Decision

E5 passes as `decoder_locality_causal`. Freeze physical-space,
relative-coordinate, quadrature-aware local decoding. Do not return to global
query attention, another encoder swap, larger latent capacity, an operator, or
routing.

## Evidence

Compact artifact:
`docs/research/artifacts/canonical_latent_e5_decoder_locality_result.json`.

- identical E4 direct-point encoder and exact control checkpoint reproduction;
- local decoder parameters `7,010` versus global `9,089`;
- grid NRMSE `0.262032 -> 0.065206`, `75.12%` better;
- mesh `0.243033 -> 0.075839`, `68.79%` better;
- local/interpolation ratios `0.7202x` and `0.7686x`;
- unseen-resolution local NRMSE `0.037295` and `0.059645`;
- high-frequency spectral NRMSE `1.0122 -> 0.3161` grid and
  `1.0059 -> 0.3591` mesh;
- 8-30 neighbors inside radius `0.20`, zero cap truncation;
- source-order maximum error `<=9.54e-7`;
- two complete results and all checkpoints byte-identical;
- no operator, held-out read, provider call, routing, or labels.

Verification is green: `31` focused E2-E5 tests and the complete unit suite
pass; Ruff, Black check, bytecode compilation, JSON validation, source/result
hash checks, and `git diff --check` pass.

## Important boundary

This qualifies decoder locality on the no-compression ceiling, not the
universal latent. The positive arm reads every source token. Do not carry that
source-token path around a latent operator as a hidden bypass.

The next codec must contain only fixed-count, spatially anchored evolving latent
tokens plus their geometry. Abstract learned queries without physical anchors
cannot support the identified local integral mechanism.

## Next coherent arc

Freeze E6 as one compressed-locality specialist test:

1. choose fixed-count geometry-bound latent anchors;
2. determine their physical support radius from pre-training coverage geometry;
3. compare matched global versus E5-local decoding from compressed tokens only;
4. keep the E5 data, exposure, absolute, spectral, invariance, and resolution
   gates;
5. resume shared grid/mesh codec work only if both specialist families pass.

Do not instantiate the latent operator yet.
