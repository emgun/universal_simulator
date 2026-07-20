# Canonical Latent E7 Function-Space Handoff

Date: 2026-07-20

## Decision

E7 passes as `function_space_latent_qualified`. Freeze the 52 ordered physical
coefficient semantics. This is the first positive semantic common-latent result
across paired grids and warped meshes.

Do not add routing, return to opaque token spaces, or instantiate dynamics.
The next gate is whether one learned universal encoder can infer the frozen E7
coefficients without a representation label.

## Evidence

Compact artifact:
`docs/research/artifacts/canonical_latent_e7_function_space_result.json`.

- no learned parameters or optimizer;
- one coordinate-defined basis and quadrature-weighted projection for every
  representation;
- 52 coefficients versus 196 high-resolution samples (`3.769x` compression);
- grid high NRMSE `0.001921` versus interpolation `0.090543`;
- mesh high NRMSE `0.001918` versus interpolation `0.098671`;
- unseen-resolution NRMSE remains `0.001916`/`0.001918`;
- high-frequency NRMSE `0.009428`/`0.009422`, amplitude ratio near `0.995`;
- paired high grid/mesh coefficient mismatch `0.0003716`;
- paired decoded mismatch `0.0001362`, CKA `0.99999995`, retrieval `1.0`;
- all designs rank `52`, worst condition number `6.660`;
- source-order error `<=2.05e-14`;
- two complete results are byte-identical;
- no operator, held-out read, provider, routing, labels, or source bypass.

## Interpretation

The common physical latent exists on this benchmark. E2-E6 failed because the
learned encoders/tokenizers did not preserve it, not because grid and mesh
states require family routing. Equal tensor shape was never sufficient;
coordinate-defined coefficient meaning is.

The result is bounded to smooth scalar fields on the normalized square and a
benchmark-informed frequency cutoff. It does not qualify particles, complex
topologies, general fields, or dynamics.

## Next coherent arc

Freeze E8 as an amortized universal-encoder test:

1. exact E7 coefficients are the teacher and immutable semantic target;
2. one set encoder consumes `(coordinate, value, quadrature mass)` from mixed
   grids and warped meshes, with no modality label;
3. compare against matched family-only encoders under equal exposure and
   capacity;
4. gate coefficient error, decoded absolute/spectral error, paired semantics,
   remeshing/refinement stability, source-order invariance, and positive shared
   transfer;
5. keep a particle-coordinate sample as a mechanics probe only.

Do not instantiate a latent operator until the learned encoder passes.
