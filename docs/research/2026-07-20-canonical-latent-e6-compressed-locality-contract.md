# Canonical Latent E6 Compressed-Locality Contract

Date: 2026-07-20
Status: frozen before implementation or measurement

## Question

E5 proved that a physical-space local integral decoder reconstructs grid and
warped-mesh fields when every source sample is retained. Can the same mechanism
qualify a fixed eight-token, spatially anchored latent without passing original
source features around the bottleneck?

This is the smallest compressed specialist test licensed by E5. It does not
train a shared grid/mesh codec, instantiate dynamics, add routing, or access
held-out data.

## Architecture choice

Reuse the exact E3 `RegionalInteractionEncoder`:

- eight processed regional nodes, each `32` dimensional;
- deterministic farthest-point regional selection;
- deterministic normalized geometric slot semantics;
- measure-aware physical-to-regional messages and multiscale regional
  processing;
- `27,872` encoder parameters.

The encoder must expose, without adding trainable parameters:

1. the eight latent feature vectors;
2. the actual selected regional coordinates in semantic slot order;
3. a positive regional quadrature mass obtained by assigning every physical
   sample to its nearest regional coordinate and summing its normalized source
   measure.

The masses must sum to one. After encoding, neither decoder may read original
field values, original source tokens, original source coordinates, or original
source measures. The eight feature/coordinate/mass triples are the complete
state crossing the codec bottleneck.

This follows RIGNO's downsampled regional mesh and regional-to-physical decode
pattern: <https://arxiv.org/abs/2501.19205>. It deliberately tests a
geometry-bound alternative to UPT's abstract learned latent queries and global
Perceiver decoder: <https://proceedings.neurips.cc/paper_files/paper/2024/file/2cd36d327f33d47b372d4711edd08de0-Paper-Conference.pdf>.

## Frozen arms

For grid and warped-mesh specialists:

1. `global_control`: exact E3 regional encoder plus unchanged
   `AnyPointDecoder`;
2. `local_integral`: the identically initialized regional encoder plus the E5
   relative-coordinate, quadrature-aware local integral decoder.

Both use hidden width `32`. The local decoder must not exceed the global
decoder's `9,089` parameters.

The global arm must reproduce the frozen E3 specialist checkpoint hashes:

- grid `8ffe13f62cb5bacd5925dd0a62279183a3f4b6100af6225c6086e12dfe1e9041`;
- mesh `5bdd8d865d01ce774ebbfad6a9dd1b9a794df9df245e24a4d4b5714c37f359b6`.

## Frozen physical support

Before training, the eight-token regional geometry was materialized for every
E3 low, high, remeshed, and unseen-resolution representation. Against the
canonical `18 x 18` query grid, the worst nearest-anchor distance was
`0.44444445`. Freeze decoder support radius to `0.47`, the observed covering
radius plus more than a `5%` margin rounded upward to two decimal places.

The decoder considers at most all eight tokens and masks tokens outside that
fixed physical radius. It must fail closed if any query lacks support. The
radius may not be retuned from reconstruction results.

## Frozen data, objective, and exposure

Reuse E3 exactly:

- seed `17`;
- `128` analytic training states and `24` disjoint validation states;
- grid and warped-mesh inputs at training resolutions `10` and `14`;
- unseen input resolution `18` and canonical query resolution `18`;
- `120` epochs, batch size `16`, AdamW learning rate `2e-3`;
- `960` optimizer updates and `30,720` scheduled source examples per arm;
- equally weighted native-query and canonical-query normalized MSE;
- low/high latent alignment weight `0.10` within each specialist;
- the E3/E5 direct interpolation baseline and absolute
  `<=2x interpolation` gate.

No schedule extension, radius sweep, token-count rung, seed, loss variant, or
shared arm may be added after results are visible.

## Required diagnostics

For both arms and families record:

- canonical-query NRMSE from low, high, and unseen input resolutions;
- direct-interpolation error, ratio, and absolute gate;
- high-versus-unseen decoded mismatch and stability gate;
- high-frequency spectral NRMSE and amplitude ratio;
- prediction/target standard-deviation ratio and normalized mean bias;
- effective latent rank;
- encoder, decoder, and total parameter counts;
- initial and final checkpoint hashes;
- final/minimum loss ratio and trailing loss slope;
- regional mass minimum, maximum, and sum error;
- decoder neighbor minimum, mean, maximum, and truncation fraction;
- joint latent/coordinate/mass source-order invariance at absolute tolerance
  `1e-6`;
- matched-slot regional-coordinate distance between paired high-resolution grid
  and mesh representations;
- complete config, source, result, and reproduction hashes.

## Causal decision

1. `compressed_spatial_latent_qualified` only if local decoding:
   - passes the absolute reconstruction gate for grid and mesh;
   - preserves unseen-resolution stability for both;
   - reduces high-frequency spectral NRMSE by at least `25%` versus each global
     control;
   - is source-order invariant;
   - uses valid regional masses and complete physical support; and
   - does not exceed the global decoder parameter count.
   Next: freeze this specialist codec and run one shared grid/mesh codec
   qualification with the same spatial token contract; still no dynamics.
2. `compressed_locality_helpful_but_insufficient` if rule 1 fails but local
   decoding reduces high-resolution and high-frequency NRMSE by at least `10%`
   for both families while preserving stability and invariance.
   Next: keep spatial tokens and run one preregistered anchor-count
   identifiability comparison before shared training.
3. `compressed_locality_not_qualified` otherwise.
   Next: close this compact regional codec and reconsider the representation
   contract before any shared model or operator.

## Boundary

- validation-only analytic states; no reserved held-out read;
- CPU-only and no paid provider;
- no operator instantiated;
- no task or representation label enters either model;
- no router;
- no source-feature bypass;
- no public or claim-grade promotion from this synthetic codec test.
