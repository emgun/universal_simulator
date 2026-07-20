# Canonical Latent E4 Capacity-Identifiability Contract

Date: 2026-07-20
Status: frozen before implementation or measurement

## Decision question

Why do both the E2 learned-query Perceiver codec and the E3 regional-
interaction codec fail absolute field reconstruction even in their specialist
grid-only and mesh-only controls?

E4 separates three mechanisms before another shared encoder or operator is
allowed:

1. fixed latent-token capacity;
2. compression architecture; and
3. the common arbitrary-point decoder/training objective.

This is a specialist codec diagnostic. It makes no universal-latent,
cross-family-transfer, or operator claim.

## Research grounding

- Universal Physics Transformers reports compound scaling of supernodes and
  latent tokens and observes a compute/performance tradeoff rather than treating
  one latent size as architecture-intrinsic:
  <https://arxiv.org/abs/2402.12365>.
- The 2025 discretization-mismatch analysis requires comparing outputs induced
  by different observations of the same function rather than inferring
  invariance from parameter shape:
  <https://proceedings.iclr.cc/paper_files/paper/2025/hash/313829757739365201b5adb3a1cbd9bd-Abstract-Conference.html>.
- The 2026 function-space design framework requires coordinate-aware continuous
  operations, quadrature weights where sums approximate integrals, and latent
  interfaces independent of observation discretization:
  <https://www.nature.com/articles/s42256-026-01267-z>.
- Phaedra argues that scientific tokenizers must preserve physical magnitudes
  and spectral properties, motivating spectral diagnostics in addition to
  spatial NRMSE:
  <https://arxiv.org/abs/2602.03915>.

## Frozen data and training

- reuse the exact E2/E3 analytic field generator, seed `17`, `128` train
  states, `24` disjoint validation states, regular grids, Jacobian-weighted
  warped meshes, and canonical query grid;
- use only specialist grid and specialist mesh arms;
- preserve `120` epochs, batch size `16`, AdamW learning rate `2e-3`, gradient
  clipping, decoder family, native/canonical reconstruction objective, and
  low/high exposure pairing;
- initialize the grid and mesh arms identically within each capacity rung;
- validation-only analytic states, local CPU, no operator, no held-out data,
  no provider calls, no task/representation labels as model inputs, and no
  routing.

## Compound latent ladder

Use the measure-aware E2 Perceiver encoder because it supplies a clean fixed-
length compression path and has frozen eight-token evidence.

| Rung | Latent tokens | Supernodes | Latent/hidden dimension |
| --- | ---: | ---: | ---: |
| C8 | `8` | `24` | `32` |
| C16 | `16` | `48` | `32` |
| C32 | `32` | `96` | `32` |

All other encoder and decoder settings remain unchanged. Each rung trains one
grid specialist and one mesh specialist for the same total source exposure.
This is a capacity curve, not parameter matching across rungs.

## Learned no-compression ceiling

Train two additional specialist codecs, one grid and one mesh, that project
every sampled `(coordinate, field value, normalized quadrature weight)` tuple
directly into a latent token and pass the complete source token set to the
unchanged `AnyPointDecoder`.

This arm has no supernode or fixed-token compression and its token count follows
the source discretization. It is not a deployable universal latent; it is a
causal ceiling for the decoder/objective. Direct four-neighbor inverse-distance
interpolation remains the nonlearned external ceiling.

## Metrics and unchanged absolute gate

For every arm and family record:

- canonical-query NRMSE from the high training resolution and unseen validation
  resolution;
- ratio to four-neighbor direct interpolation;
- native/canonical training loss and final-to-minimum loss ratio;
- target/prediction standard-deviation ratio and normalized mean bias;
- high-frequency Fourier reconstruction NRMSE on the canonical query grid;
- effective latent rank and exact parameter/exposure counts;
- boundary counters.

The existing absolute gate is unchanged: canonical-query NRMSE from the high
training resolution must be no worse than `2.0x` direct interpolation. A codec
setting is usable for renewed shared-latent work only if both its grid and mesh
specialists pass.

## Causal decision rules

1. If C16 or C32 is the smallest compound rung whose grid and mesh arms both
   pass, classify the eight-token bottleneck as causal and use that smallest
   passing capacity for one later shared-codec retest.
2. If no compressed rung passes but both learned no-compression arms pass,
   classify compression/tokenization as causal. Do not invoke an operator;
   next compare a high-fidelity scientific tokenizer mechanism under the same
   ceiling.
3. If either learned no-compression arm fails, classify the current decoder,
   objective, or schedule as the active blocker. Pause encoder architecture
   work and run a decoder/objective identifiability test.
4. If a nominal pass is accompanied by non-finite values, boundary violation,
   or validation-resolution collapse, reject it rather than promoting it.

Do not change rungs, epochs, gate thresholds, field generator, or decoder after
examining full-run metrics. A rerun is allowed only for an implementation or
protocol defect documented before replacement metrics are inspected. Reproduce
the decision-critical result once from scratch before roadmap promotion.
