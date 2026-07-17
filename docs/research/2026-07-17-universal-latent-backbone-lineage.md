# Universal Latent Backbone Lineage

Date: 2026-07-17

## Decision

The original architectural reference is **Universal Physics Transformers
(UPT)**, but UPS implemented only the outer shape of that idea. The current
grid and mesh/particle encoders do not form one encoder and do not establish a
common physical latent basis. Before another operator experiment, converge
grids, meshes, and particles onto one coordinate-and-field point-set encoder,
then qualify that codec on paired samples of the same physical states.

This is not a router problem. Representation kind must not be an input to a
hidden expert selector in the canonical path.

Primary sources:

- [Universal Physics Transformers, NeurIPS 2024 paper](https://papers.neurips.cc/paper_files/paper/2024/file/2cd36d327f33d47b372d4711edd08de0-Paper-Conference.pdf)
- [Official UPT implementation](https://github.com/ml-jku/UPT)
- [Official project page](https://ml-jku.github.io/UPT/)

## What UPT actually specifies

UPT normalizes an input physical state into point samples: coordinates plus
field values. Its encoder then:

1. embeds the points and their positions;
2. aggregates local point information into geometry-selected supernodes;
3. processes supernodes with transformer blocks;
4. uses a Perceiver block with learned queries to produce a fixed
   `n_latent x hidden_dim` representation;
5. advances that representation without a grid or particle latent structure;
6. decodes it at arbitrary physical query positions.

The training procedure also uses inverse-encoding and inverse-decoding
reconstruction losses to separate the codec from latent dynamics. Those
losses are not incidental: they are how the paper makes latent-only rollouts
usable.

The paper's phrase "unified latent representation" means one fixed-size,
non-spatial latent interface produced by this common point-set architecture.
It does **not** establish that separately trained encoders, or two encoders
that merely emit the same tensor shape, use interchangeable coordinates.

## What the paper does not prove for our stronger north star

The reported experiments demonstrate the architecture across mesh-based,
regular-grid, and Lagrangian datasets. They do not report a paired experiment
where one trained checkpoint encodes the same physical state sampled as a grid
and as an irregular mesh, followed by latent matching and cross-decoding.

UPS therefore needs a stronger qualification than the paper reported:

- one checkpoint and one encoder path across discretizations;
- exact paired physical-state identity;
- decoding from either sampled representation at either query set;
- remeshing/resampling invariance;
- latent retrieval, rank, and collapse controls;
- no modality/task router in the encoder or operator.

## Live UPS divergence

| Contract | UPT reference | UPS before E2 foundation | Consequence |
| --- | --- | --- | --- |
| Input abstraction | points plus coordinates | grid tensors in `GridEncoder`; graphs in `MeshParticleEncoder` | two unrelated representation paths |
| Supernodes | geometry-selected, local message aggregation | contiguous storage-order chunks | mesh encoding changes when point order changes |
| Second hierarchy | transformer blocks | absent | no learned interaction among pooled mesh tokens |
| Perceiver pooling | learned latent queries with cross-attention | adaptive average pooling | fixed shape without canonical token semantics |
| Decoder training | inverse encode/decode reconstruction | D6 standalone decoder omitted Darcy solutions | codec and dynamics became entangled |
| Cross-discretization evidence | not reported by paper | not measured | universal-basis claim remains unqualified |

The existing `GridEncoder` is also a grid-specific pixel-unshuffle codec. It
can remain as a frozen baseline, but it is the wrong canonical interface for a
grid/mesh/particle north star.

## Implemented E2 foundation

`src/ups/io/enc_canonical_point.py` adds an opt-in
`CanonicalPointEncoder`. It has no modality switch: regular grids, irregular
meshes, and particle samples all enter as coordinates plus an exact field
schema. It uses deterministic geometry-based farthest-point supernodes,
local distance-weighted aggregation, transformer processing, and learned
Perceiver queries.

The learned queries establish shared latent token slots. This makes direct
paired alignment meaningful in a way that it is not for two independently
co-adapted codecs. The encoder is invariant to point storage order, including
through its reduced-supernode path, and fails closed on field-semantic or
channel mismatches.

`src/ups/eval/latent_qualification.py` adds representation diagnostics and the
full encoder-source/query-discretization cross-decoding matrix. These mechanics
do not declare the encoder qualified; no training or physical benchmark has
run.

## Tradeoffs

- A single point-set encoder gives up the grid convolution's cheap local
  inductive bias. It buys a truthful common interface and avoids having to
  align two independently rotating latent bases.
- Geometry-aware supernodes and Perceiver queries cost more than average
  pooling, but scale much better than full attention over every source point.
- Exact field schemas are intentionally strict. Cross-PDE universality will
  require explicit physical variable semantics and units; silently padding or
  reordering channels would undermine the common-space claim.
- Direct latent alignment is useful only because token slots are shared.
  Reconstruction and cross-decoding remain decisive because a collapsed or
  co-adapted representation can still look geometrically similar.

## Recommendation

Keep the old grid and mesh encoders only as matched controls. Next, wire the
canonical point encoder and one any-point decoder into a codec-only training
path, create paired analytic grid/mesh states with frozen units and identity,
and run the qualification contract below before touching the latent operator.
