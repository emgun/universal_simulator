# Canonical Latent E3 Regional-Interaction Contract

Date: 2026-07-20
Status: frozen before implementation or measurement

## Decision question

Can a compact regional-interaction graph encoder preserve one physical field
across regular grids and irregular meshes more accurately and with better
refinement consistency than the rejected learned-query Perceiver encoder,
without introducing modality routing or an operator?

E2 established that paired identity alignment is not sufficient. The E3
candidate must pass the unchanged absolute reconstruction and discretization
convergence gates as well as the alignment, rank, cross-decoding, remeshing,
and order-invariance gates.

## Research mechanism

The challenger adopts the encoder-side mechanism of RIGNO (NeurIPS 2025):

1. project physical-node field and coordinate features;
2. transfer them through learned, relative-coordinate edge messages to a
   downsampled regional mesh;
3. process regional nodes through residual message passing over multiple
   neighborhood scales; and
4. expose the processed regional nodes directly as the fixed latent sequence.

The official RIGNO implementation uses physical-to-regional, regional-to-
regional, and regional-to-physical graphs, mean-aggregated learned edge
messages, residual graph-network blocks, and a multilevel regional mesh. This
test deliberately stops after the regional encoder. It retains UPS's existing
`AnyPointDecoder`, so an operator or graph decoder cannot explain the result.

This is a compact RIGNO-style mechanism test, not a reproduction claim. It
uses deterministic geometry-only farthest-point regional selection instead of
RIGNO's random subsampling, explicit quadrature weights for the continuous
field contract, and fixed regional-node count to preserve the E2 bottleneck.

Primary sources:

- RIGNO paper and official code:
  <https://papers.nips.cc/paper_files/paper/2025/hash/dcb91f43033bb1d367d1848806dee98d-Abstract-Conference.html>
  and <https://github.com/camlab-ethz/rigno>
- Geometry-Informed Neural Operator:
  <https://arxiv.org/abs/2309.00583>

## Frozen comparison

- analytic state generator, seed, train/validation split, coordinates,
  Jacobian-derived measures, and canonical queries: unchanged from E2;
- latent length/dimension, hidden dimension, decoder, optimizer, learning
  rate, epochs, batch size, exposure schedule, alignment weight, and gradient
  clipping: unchanged from E2;
- arms: exposure-matched shared grid/mesh, grid-only control, and mesh-only
  control, initialized identically within this architecture;
- evaluation: the exact E2 gate implementation, including direct four-neighbor
  inverse-distance interpolation and target-energy-normalized resolution
  mismatch;
- boundary: validation-only analytic states, no operator, no held-out data,
  no representation/task labels as model inputs, no provider calls, and no
  routing.

The frozen E2 Perceiver artifact remains the architecture baseline. E3 also
requires its own matched controls because architecture-specific capacity can
otherwise masquerade as shared-representation cost.

## Candidate details

- eight regional nodes, matching the eight E2 latent tokens;
- deterministic coordinate-canonical farthest-point selection;
- deterministic matching of the selected regional set to fixed geometric slot
  anchors, preserving direct paired-token semantics without changing which
  regional nodes are selected;
- sixteen nearest physical neighbors per regional node;
- learned physical-to-regional edge messages using sender state, relative
  displacement, distance, and quadrature weight;
- quadrature-weighted local aggregation;
- three residual regional interaction blocks with neighborhood sizes 2, 4,
  and 7, providing local-to-global communication over the eight-node mesh;
- no attention, learned latent queries, modality-specific adapters, operator,
  or task/family router.

## Qualification and stop rules

The candidate is `qualified` only if every existing E2 gate passes. In
particular:

- shared grid and mesh reconstruction must each be no worse than `1.10x` its
  matched architecture control;
- shared grid and mesh reconstruction must each be no worse than `2.0x` direct
  interpolation;
- highest-resolution grid/mesh output mismatch must be no worse than `1.10x`
  the low-resolution mismatch;
- permutation invariance, cross-decoding, paired identity, alignment margin,
  effective rank, and remeshing gates remain unchanged.

If any gate fails, reject this exact candidate and do not invoke an operator,
read held-out data, relax gates, extend training, or add routing. A rerun is
allowed only to repair an implementation or protocol defect documented before
examining replacement metrics. If it qualifies, reproduce once from scratch
before considering an E4 freeze-based operator test.

## Pre-measurement implementation repair

The first complete mechanics run was rejected before use as experiment
evidence. At validation resolution, the selected grid and mesh regional sets
had one-way Chamfer distance `0.0021`, but FPS discovery order produced matched
slot distance `0.6101`. RIGNO's regional graph is unordered, whereas the frozen
paired-latent metric requires direct token semantics. Deterministic FPS alone
therefore did not implement the stated fixed-sequence contract.

The repair matches the unchanged selected set to fixed normalized geometric
slot anchors before message passing. It changes no selected coordinate,
parameter count, data, seed, schedule, loss, decoder, or gate. The discarded
run at `/tmp/canonical-latent-e3-regional-final-v1` is mechanics evidence only;
its metrics must not be used for the E3 decision.
