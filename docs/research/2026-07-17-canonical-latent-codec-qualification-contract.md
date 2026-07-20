# Canonical Latent Codec Qualification Contract

Date: 2026-07-17
Status: measure-aware analytic E2 completed negative on 2026-07-20

## Claim under test

One encoder checkpoint maps different discretizations of the same physical
state into an operator-usable canonical latent basis, and one decoder can query
that basis at coordinates from any supported discretization.

Equal tensor shapes, high CKA alone, or good within-modality reconstruction do
not pass this contract.

## Frozen boundary before learning

The first E2 benchmark is codec-only and validation-only. It must freeze:

- physical-state identity and group identity;
- analytic/source field definition, variables, channels, units, coordinate
  frame, domain, parameters, boundary conditions, and time;
- regular-grid sampling and at least two independently generated irregular
  remeshings per state;
- a representation-neutral canonical query set not reused as either encoder
  input;
- train/validation group allocation before writing samples;
- source, manifest, config, and checkpoint hashes;
- positive quadrature/cell measures for every encoder sample, normalized per
  physical domain rather than inferred from point count;
- one `CanonicalPointEncoder` and one `AnyPointDecoder` checkpoint, with no
  modality/task selector or expert router.

Particle samples may be added only when the physical field admits a truthful
Lagrangian representation. Relabeling irregular Eulerian points as particles
does not count.

## Required controls

1. Grid-only and mesh-only matched codecs with the same latent budget and
   training exposure.
2. Identity resampling: the same points in a different storage order.
3. Same-modality resolution/remeshing controls.
4. Negative physical-state pairing and a fixed permutation null.
5. Constant/collapsed latent control.
6. Preserved initialization, selected codec, and post-training checkpoints.

## Qualification matrix

For encoder sources `grid` and `mesh`, decode at all three query sets:
`grid`, `mesh`, and `canonical`. Report global and per-state NRMSE for every
cell. The diagonal is within-discretization reconstruction; off-diagonal cells
are cross-decoding. The canonical-query column prevents either native sampling
from defining the truth surface.

## Fail-closed gates

Thresholds are relative to frozen controls so they cannot be relaxed after
seeing E2:

1. **Identity:** point-order permutations change every latent and decoded
   output by at most the numeric-determinism tolerance established in a
   no-training smoke.
2. **Within codec:** canonical grid and mesh reconstruction are each no worse
   than `1.10x` their matched single-discretization codec NRMSE.
3. **Cross codec:** every off-diagonal cross-decoding NRMSE is no worse than
   `1.10x` the worse corresponding within-codec NRMSE.
4. **Canonical queries:** both encoder sources decode the neutral query set no
   worse than `1.10x` the better native-query within NRMSE.
5. **Paired identity:** symmetric top-1 retrieval is at least `0.90` and at
   least `10x` chance; its paired-label permutation p-value is below `0.01`.
6. **Alignment margin:** standardized paired latent RMSE is at most `0.50x`
   the median fixed-negative-pair RMSE. CKA is diagnostic and cannot rescue a
   failed retrieval or cross-decoding gate.
7. **Rank:** each representation's effective rank is at least `0.80x` its
   matched codec control and strictly above the constant/collapse control.
8. **Remeshing:** worst independent-remesh canonical-query NRMSE is no worse
   than `1.10x` the base irregular-mesh result.
9. **Boundary:** no latent operator is instantiated, no held-out object is
   opened, and no representation/task label enters the encoder or decoder.
10. **Absolute reconstruction:** shared grid and mesh canonical-query NRMSE
    are each at most `2.0x` deterministic four-neighbor inverse-distance
    interpolation from the same input samples. This prevents equally weak
    shared and matched codecs from passing a purely relative gate.
11. **Resolution convergence:** grid-versus-mesh output mismatch at the unseen
    highest resolution is no worse than `1.10x` mismatch at the lowest
    training resolution.

All gates must pass. A missing cell, identity, hash, control, or preserved
checkpoint yields `not_qualified`, not an inferred pass.

## Interpretation

- Poor within reconstruction: repair codec capacity/training first.
- Good within reconstruction but poor cross-decoding/retrieval: add an explicit
  paired alignment objective while keeping one encoder; do not add routing.
- Good cross-decoding but collapsed rank: increase information preservation;
  do not pass based on output averages.
- Full pass: freeze the codec and proceed to E3 operator-only shared versus
  specialized dynamics.

## 2026-07-20 protocol repair and result

An initial mechanics shakedown exposed that the original relative-only gates
could pass while every codec had about `0.70` absolute NRMSE. That shakedown is
rejected as decision evidence. Gates 10 and 11 were added before the final
default run using a deterministic nonlearned baseline and a refinement
criterion rather than a threshold selected to the model result.

The final 128-train/24-validation-state, 120-epoch run is `not_qualified`.
Details are in
`docs/research/2026-07-20-canonical-latent-e2-measure-aware-result.md`.
