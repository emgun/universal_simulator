# Canonical Latent E7 Function-Space Sufficiency Contract

Date: 2026-07-20
Status: frozen before state-level measurement

## Question

E6 showed that eight geometry-bound regional features discard too much field
information even when decoded with the successful E5 locality mechanism. Does
a deterministic, coordinate-defined coefficient space preserve the same
physical field across grid and warped-mesh observations before any neural
encoder is asked to discover that space?

E7 tests the latent space itself. It does not train a model, instantiate an
operator, add routing, access held-out data, or retain source samples after
projection.

## Research basis

Current function-space guidance says that latent interfaces must be independent
of the observation discretization and that sums approximating continuous
transforms must use coordinates and quadrature weights:
<https://www.nature.com/articles/s42256-026-01267-z>. GINO demonstrates the
strategic pattern of mapping arbitrary input meshes to one regular latent
domain and querying outputs at arbitrary coordinates:
<https://proceedings.neurips.cc/paper_files/paper/2023/file/70518ea42831f02afc3a2828993935ad-Paper-Conference.pdf>.

E7 removes learned GNO/FNO components to ask a prior question: whether a shared
physical coefficient space is sufficient and numerically identifiable from
the frozen observations at all.

## Frozen physical basis

On the normalized domain `[0, 1]^2`, define the real one-dimensional Fourier
basis

`[1, sqrt(2) sin(2 pi k x), sqrt(2) cos(2 pi k x)]` for `k = 1, 2, 3`.

Take its full tensor product in `x` and `y`, yielding `49` modes. Add three
normalized trend functions:

- `sqrt(12) (x - 0.5)`;
- `sqrt(12) (y - 0.5)`;
- `12 (x - 0.5) (y - 0.5)`.

The latent is therefore one ordered vector of `52` scalar coefficients. The
same basis evaluation code and coefficient ordering apply to grids, meshes,
and arbitrary coordinate point sets. No representation label is available.

The cutoff is fixed from the analytic family definition: its highest stated
oscillation is frequency `3` in `x` and `2` in `y`; cutoff `3` is the smallest
tensor-product Fourier space containing it. The trend modes retain the exact
non-periodic bilinear component rather than forcing it through a periodic
boundary.

## Frozen pre-measurement calibration

On a deterministic `128 x 128` midpoint quadrature grid, project each of the
seven component functions separately into the 52-dimensional space. Component
NRMSE values, in source order, are:

`[4.71e-16, 1.17e-15, 8.53e-16, 7.54e-16, 0.01885191, 3.25e-16, 7.14e-16]`.

The maximum is `0.01885191`, from the localized Gaussian. Freeze a basis-
calibration gate of `<=0.02`.

Geometry-only weighted design matrices were also materialized before any state
measurement. Every frozen E2 grid and warped-mesh representation has rank `52`.
Condition numbers range from `4.947` to `6.660`; freeze a maximum allowed
condition number of `10`.

## Projection and decode

For source coordinates `x_i`, normalized positive quadrature masses `w_i`, and
field samples `u_i`, evaluate the common basis matrix `Phi` and compute the
deterministic weighted least-squares coefficient vector

`argmin_c sum_i w_i ||Phi(x_i)c - u_i||^2`.

Solve the corresponding float64 normal equations. The system must be full rank
and fail closed above condition number `10`.

Decode at arbitrary query coordinates `y` only as `Phi(y)c`. After projection,
the decoder cannot read source values, coordinates, measures, or source tokens.
There are no learned parameters.

## Frozen data and diagnostics

Use the exact E2 analytic family and disjoint 24-state validation split at seed
`10017`, with low/high/unseen resolutions `10/14/18` and canonical query
resolution `18`. No training states are required for the result.

Record for each family and input resolution:

- canonical-query NRMSE and direct-interpolation comparison;
- high-frequency spectral NRMSE and amplitude ratio;
- weighted source projection residual;
- coefficient norm and effective rank;
- design rank and condition number;
- source-order invariance;
- source-node/latent coefficient compression ratio.

Also record:

- paired grid/mesh coefficient relative mismatch at low and high resolution;
- paired decoded mismatch at canonical queries;
- within-family low/high and high/unseen coefficient mismatch;
- remeshed high-resolution mesh coefficient and decoded mismatch;
- complete config, boundary, source, result, and reproduction hashes.

## Frozen gates

Classify `function_space_latent_qualified` only if:

1. high-resolution grid and mesh NRMSE are each `<=2x` their direct
   interpolation baselines;
2. unseen-resolution NRMSE is no more than `1.10x` high-resolution NRMSE for
   each family;
3. high-frequency spectral NRMSE is `<=0.25` for each family;
4. high-resolution paired grid/mesh coefficient relative mismatch is `<=0.10`;
5. high-resolution paired decoded mismatch is `<=0.10` normalized by target
   RMS;
6. each high/unseen within-family coefficient mismatch is `<=0.15`;
7. source-order maximum absolute output error is `<=1e-10`;
8. all design matrices have rank `52` and condition number `<=10`;
9. basis calibration maximum component NRMSE is `<=0.02`; and
10. the high-resolution representation is compressed by at least `2x`.

Next: freeze the coefficient semantics and train one universal coordinate-and-
quadrature encoder to infer these coefficients. Still do not instantiate
dynamics.

Classify `function_space_sufficient_projection_unstable` if both absolute
reconstruction gates pass but any semantic, refinement, conditioning, or
invariance gate fails. Next: repair only the deterministic projection or basis
normalization, not the encoder or operator.

Classify `function_space_latent_not_qualified` otherwise. Next: close this
spectral-polynomial basis and reconsider the common function-space family.

No cutoff, basis function, regularizer, threshold, representation, state,
decoder, or follow-up arm may be changed after state-level results are visible.

## Boundary

- validation-only analytic states; no reserved held-out read;
- CPU-only and no paid provider;
- no learned parameters and no optimizer;
- no operator, router, task label, or representation label;
- no original-source bypass after coefficient projection;
- mechanics accept arbitrary coordinates, but E7 scientifically qualifies only
  the frozen grid and warped-mesh families, not particles or new domains;
- no public or claim-grade promotion from this synthetic codec test.
