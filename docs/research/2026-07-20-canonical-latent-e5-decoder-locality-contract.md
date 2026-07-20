# Canonical Latent E5 Decoder-Locality Contract

Date: 2026-07-20
Status: frozen before implementation or measurement

## Question

E4 showed that the specialist codec remains above the absolute reconstruction
gate even when every physical sample is retained as a learned source token.
Does the global latent-to-query decoder fail because it lacks an explicit
physical-space local kernel?

This is a decoder-only causal test. It is not an encoder, operator, routing,
objective, schedule, shared-representation, or held-out experiment.

## Research basis

- The current function-space construction recipe replaces index-defined graph
  neighborhoods with coordinate-defined physical neighborhoods and evaluates
  the resulting integral with quadrature weights. A fixed physical radius is
  required because a fixed number of index neighbors collapses under
  refinement: <https://www.nature.com/articles/s42256-026-01267-z>.
- GINO uses local graph neural operator layers to decode from a latent grid to
  arbitrary geometry points and reports discretization convergence:
  <https://proceedings.neurips.cc/paper_files/paper/2023/file/70518ea42831f02afc3a2828993935ad-Paper-Conference.pdf>.
- RIGNO transfers processed regional information back to physical nodes through
  an explicit regional-to-physical decoder rather than one unconstrained
  global query attention: <https://openreview.net/pdf?id=ahJfROJOYt>.

E5 adopts only the smallest mechanism common to those results: a learned local
integral decoder with relative coordinates and quadrature-aware normalization.
It does not add a regional processor, latent operator, family label, or router.

## Frozen arms

For each of the grid and warped-mesh specialist families:

1. `global_control`: the exact E4 `DirectPointCodec`, including the unchanged
   `AnyPointDecoder`;
2. `local_integral`: the same direct-point encoder and every source token,
   followed by one fixed-radius local integral decoder.

The challenger decoder uses:

- physical support radius `0.20` on the unit square;
- at most `32` nearest candidates, with points outside the fixed radius masked;
- source-to-query displacement and distance divided by the fixed radius;
- a learned scalar kernel over source-token features and relative geometry;
- normalized weights proportional to `exp(kernel_logit) * quadrature_weight`;
- a learned source-value projection, query Fourier features, and a pointwise
  output map;
- hidden width `32`, matching the control;
- no more trainable decoder parameters than `AnyPointDecoder`.

The radius, cap, normalization, width, and topology are frozen. The candidate
must fail closed if any query has no source point within the physical radius.
The candidate must be invariant to a joint permutation of source tokens,
coordinates, and measures.

## Frozen data and exposure

Reuse E4 without modification:

- seed `17`;
- `128` analytic training states and `24` disjoint validation states;
- grid and warped-mesh representations at training resolutions `10` and `14`;
- unseen input resolution `18`;
- canonical query resolution `18`;
- `120` epochs, batch size `16`, AdamW learning rate `2e-3`, weight decay
  `1e-6`, and gradient clipping at `1.0`;
- `960` optimizer updates and `30,720` scheduled source examples per arm;
- identical native-query and canonical-query normalized MSE objective;
- the E4 direct interpolation baseline and absolute `<=2x interpolation` gate.

The global control must reproduce the frozen E4 direct-point checkpoint hashes
and metrics. Both arms begin with byte-identical direct-point encoder weights
within each initial model construction. Parameter counts and checkpoint hashes
are mandatory evidence.

## Diagnostics

For both families and arms record:

- canonical-query NRMSE at low, high, and unseen input resolutions;
- high-to-interpolation ratio and absolute pass/fail;
- high-versus-unseen output mismatch and resolution-stability gate;
- high-frequency spectral NRMSE and amplitude ratio;
- prediction/target standard-deviation ratio and normalized mean bias;
- effective source-token rank;
- training loss minimum, final/minimum ratio, and trailing slope;
- fixed-radius neighbor-count minimum, mean, maximum, and truncation fraction;
- source-order invariance at absolute tolerance `1e-6`;
- source/result/config/checkpoint SHA-256 provenance.

## Causal decision

Classify exactly once after both families complete:

1. `decoder_locality_causal` only if the local decoder:
   - passes the absolute gate for both families;
   - preserves unseen-resolution stability for both families;
   - reduces high-frequency spectral NRMSE by at least `25%` versus the global
     control for both families;
   - is source-order invariant; and
   - does not exceed the control decoder parameter count.
   Next: freeze the local decoder and retest only the smallest C8 compressed
   specialist codec.
2. `decoder_locality_helpful_but_insufficient` if it does not satisfy rule 1
   but reduces high-resolution NRMSE and high-frequency spectral NRMSE by at
   least `10%` versus control for both families while preserving stability and
   invariance.
   Next: keep the all-point representation and isolate objective versus
   schedule; do not return to the encoder or operator.
3. `decoder_locality_not_causal` otherwise.
   Next: close decoder architecture work at this scale and isolate objective
   versus schedule on the frozen global control.

No threshold may be relaxed and no extra radius, seed, width, epoch, or loss
variant may be added after results are visible.

## Boundary

- validation-only analytic states; no reserved held-out read;
- CPU-only and no paid provider;
- no operator instantiated;
- no task or representation label enters either model;
- no routing path;
- no public or claim-grade promotion from this synthetic mechanism test.
