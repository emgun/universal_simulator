# Canonical Latent E8 Amortized Universal Encoder Result

Date: 2026-07-21
Status: `amortized_encoder_capable_without_positive_transfer`

## Decision

One learned, representation-blind encoder can approximate the frozen E7
coefficient semantics from both grids and warped meshes. It passes every
semantic, reconstruction, spectral, refinement, remeshing, invariance, and
boundary gate.

It does not qualify for dynamics because mixed grid/mesh training provides no
positive transfer over matched family-only controls. The grid-only control is
best even when evaluated across both families. A mechanics-only random-point
probe also exposes a large out-of-format error despite a well-conditioned exact
E7 projection.

Preserve the E7 basis and no-routing boundary. The next encoder must condition
its learned projection on the source geometry's quadrature Gram information
and must be tested across a scientifically frozen distribution of sampling
geometries before any operator is instantiated.

## Frozen protocol

The contract was written before state-level measurement at
`docs/research/2026-07-21-canonical-latent-e8-amortized-encoder-contract.md`.

The encoder computes the 52 E7 quadrature moments
`sum_i w_i Phi(x_i)u_i`, then applies a 6,772-parameter residual MLP in the
fixed coefficient space. It receives no family, resolution, task, or routing
label. All arms start from one exact initialization and receive 960 optimizer
updates and 30,720 scheduled source examples.

## Primary evidence

Shared high-resolution validation:

| Metric | Grid | Mesh |
| --- | ---: | ---: |
| Coefficient NRMSE to exact E7 teacher | `0.00714392` | `0.00727108` |
| Canonical-query NRMSE | `0.00730714` | `0.00740108` |
| Direct interpolation NRMSE | `0.09054318` | `0.09867060` |
| High-frequency spectral NRMSE | `0.02301905` | `0.02348034` |

The shared encoder remains stable at unseen resolution 18:

- grid coefficient/canonical NRMSE: `0.00727980` / `0.00743091`;
- mesh coefficient/canonical NRMSE: `0.00733519` / `0.00746945`;
- high grid/mesh coefficient mismatch: `0.00136739`;
- high grid/mesh decoded mismatch: `0.00102185`;
- grid high/unseen coefficient mismatch: `0.00110444`;
- mesh high/unseen coefficient mismatch: `0.00121162`;
- mesh high/remesh coefficient mismatch: `0.00242566`;
- maximum observed source-order coefficient error: `4.44e-16`.

Every frozen semantic gate passes.

## Positive-transfer result

Macro coefficient NRMSE across high and unseen grid and mesh inputs:

| Arm | Macro coefficient NRMSE |
| --- | ---: |
| Shared mixed grid/mesh | `0.00725750` |
| Grid-only control | `0.00501824` |
| Mesh-only control | `0.00664777` |

The shared-to-grid-control macro ratio is `1.4462`; the shared-to-mesh-control
ratio is `1.0917`. On matched high-resolution inputs, shared/grid-control is
`1.5095` and shared/mesh-control is `1.1187`. Both preregistered transfer gates
therefore fail.

This is not evidence that a common latent requires routing. It shows the
opposite: a grid-only coefficient correction transfers to the fixed warped
mesh well enough to beat the mixed arm. The two frozen geometry families do
not create a useful positive-transfer test for this strongly structured basis
encoder, while optimization across both introduces a small compromise.

## Mechanics-only point probe

On one deterministic 196-point uniform Monte Carlo sample with equal masses:

- exact E7 teacher design rank: `52`;
- exact E7 teacher condition number: `7.26397`;
- learned coefficient NRMSE to teacher: `0.446342`;
- learned canonical-query NRMSE: `0.443530`.

No scientific gate or particle qualification attaches to this probe. It is
nevertheless diagnostic: the latent is identifiable from these points, but a
global correction of the moment vector does not know how the basis Gram matrix
changed under a new sampling measure.

## Reproducibility and boundary

Two full runs at
`/tmp/canonical_latent_e8_amortized_encoder_v1/result.json` and
`/tmp/canonical_latent_e8_amortized_encoder_v2/result.json` are byte-identical.

- result SHA-256:
  `f6b6537d3f999f73c7509c56f4407a9e3dba1e657d1e326bff37b4c0102ee75b`;
- shared checkpoint SHA-256:
  `565ef15b6516dc510c38698dfb7318d8079caf04b8ad4eb0f593212a0173e42f`;
- grid-control checkpoint SHA-256:
  `93e636b18c632a130ca66a2c30f1e1edd0fbb1fa990135b02649072dceef237e`;
- mesh-control checkpoint SHA-256:
  `62c0b1d189774cc6a45028cd42b647972193a6060f0e1d50166887ed99d46534`.

There are no held-out reads, operators, temporal transitions, task or
representation labels, provider calls, routes, or original-source bypasses.
The exact E7 solve is a teacher only and is not used by learned inference.

## Interpretation and next gate

E8 separates two claims that must not be collapsed:

1. **Learned semantic encoding succeeds on the frozen grid/mesh benchmark.**
   One model maps both representations into the ordered E7 coefficient space
   with low error and excellent paired consistency.
2. **Universal positive transfer is not established.** The mixed arm loses to
   both specialists on the preregistered comparison, and its correction does
   not extrapolate to random points.

Freeze E9 as a geometry-conditioned amortized projection test. Supply the
learned correction with coordinate/quadrature sufficient statistics, including
the E7 basis Gram matrix, while preserving one set encoder and fixed
coefficient meanings. Train across a continuous, preregistered distribution of
regular, warped, remeshed, and point-sampled observations; hold out sampling
geometries rather than only physical states. Compare equal-capacity/equal-
exposure controls and require positive transfer on held-out geometries.

Do not instantiate dynamics or add routing yet.
