# Canonical Latent E8 Amortized Encoder Handoff

Date: 2026-07-21

## Decision

E8 is `amortized_encoder_capable_without_positive_transfer`.

One 6,772-parameter coordinate/value/quadrature encoder infers the fixed E7
coefficients from grids and warped meshes with about `0.7%` coefficient error,
and all semantic gates pass. Mixed-family training does not beat matched
family-only encoders, so dynamics remains closed.

## Causal evidence

- shared grid/mesh high coefficient NRMSE: `0.007144` / `0.007271`;
- shared grid/mesh high decoded NRMSE: `0.007307` / `0.007401`;
- paired high coefficient/decoded mismatch: `0.001367` / `0.001022`;
- unseen resolution, refinement, remesh, spectrum, and invariance all pass;
- shared macro coefficient NRMSE: `0.007257`;
- grid-only/mesh-only macro controls: `0.005018` / `0.006648`;
- shared-to-control macro ratios: `1.4462` / `1.0917`;
- non-gated 196-point probe: exact teacher condition `7.264`, learned
  coefficient NRMSE `0.4463`;
- two complete runs are byte-identical;
- no operator, held-out read, provider, route, label, or source bypass.

Compact artifact:
`docs/research/artifacts/canonical_latent_e8_amortized_encoder_result.json`.

## Interpretation

The encoder problem is mostly solved for the two fixed structured observation
families, but the global learned correction is blind to changes in sampling
geometry. Its input moment vector contains field information but omits the
quadrature basis Gram matrix that determines how moments map to least-squares
coefficients.

Do not reinterpret the failed positive-transfer gate as evidence for routing.
The grid-only control already transfers across the fixed mesh; the current
families are too similar to demonstrate a mixed-training advantage.

## Next coherent arc

Freeze E9 as a geometry-conditioned universal encoder test:

1. retain the exact E7 basis, coefficient ordering, decoder, and teacher;
2. expose geometry sufficient statistics such as
   `G = sum_i w_i Phi(x_i)Phi(x_i)^T` alongside
   `b = sum_i w_i Phi(x_i)u_i` to one learned correction, without labels;
3. define continuous train and held-out distributions over resolution, warp,
   remeshing, and point sampling before state measurement;
4. use matched capacity, initialization, state order, updates, and source
   exposure for shared and sampling-family controls;
5. require semantic accuracy and positive transfer on held-out sampling
   geometries, including a scientifically frozen point-set family;
6. preserve source-order invariance and prohibit exact-solve or source-token
   bypasses at learned inference.

Only after this geometry-conditioned encoder passes may a coefficient-space
latent operator be instantiated. Do not route.
