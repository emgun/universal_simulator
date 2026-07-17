# Universal Latent Encoder Audit Plan

Date: 2026-07-17

## Decision

The next architecture gate is the representation, not a family router. D6 is a
valid negative result for its end-to-end grid candidate, but it did not isolate
the shared latent operator and did not test a universal grid/mesh/particle
encoder. Family-specific routing is therefore paused.

The project north star remains one physical latent space in which a shared
operator can act on grids, meshes, particles, and other discretizations. A
common tensor shape is an interface property, not evidence of a common latent
space.

## Why D6 does not resolve the encoder question

The source-bound contract audit is
`docs/research/artifacts/strat_v1_d6_universal_latent_contract_audit.json`.
It establishes:

- D6 used only `data.kind: grid`; no mesh or particle sample entered the run.
- UPS has separate `GridEncoder` and `MeshParticleEncoder` implementations and
  does not share their parameters or align their representations.
- The 12-epoch operator stage optimizes only the operator. Its encoder is
  materialized and then used as a fixed feature map for latent pairs.
- The 6-epoch decoder stage loads and freezes that encoder.
- Encoder, decoder, and operator are jointly optimized only in the final
  4-epoch stage, with two rollout steps and reconstruction weight `0.25`.
- D6 reports end-to-end rollout quality, not codec-only reconstruction,
  latent geometry, paired cross-discretization alignment, cross-decoding, or
  resolution invariance.
- D6's routed modules consume `task_id` inside the operator. Their matched-arm
  advantage can therefore reflect encoder, decoder, dynamics, or joint
  optimization interference; it is not an operator-only causal result.

## Gate sequence

### E0 — contract and evidence inventory (complete)

Run the fail-closed audit against the merged D6 config:

```bash
PYTHONPATH=src:. python scripts/audit_universal_latent_contract.py \
  --config configs/d6_strat_v1_modular_shared_trunk.yaml \
  --output docs/research/artifacts/strat_v1_d6_universal_latent_contract_audit.json
```

E0 classifies the universal-encoder and common-latent claims as `not_tested`,
codec-versus-dynamics causality as `unresolved`, and family routing as not
authorized. It accesses no provider and no held-out data.

### E1 — codec-only grid diagnosis (complete)

Using the immutable D6 joint and matched checkpoints plus the exact locked
validation shards, measure `decode(encode(x))` separately for Advection,
Burgers, and Darcy. Record reconstruction NRMSE, spectral error, effective
rank, per-dimension variance, covariance condition, latent RMS/norm, and a
held-out-within-validation task probe. Compare joint and matched codecs while
holding the latent operator out of the path.

This gate must use the exact `strat-v1` validation objects. The older local
`pdebench.oct2025_backup/*_val.h5` files are a different protocol and must not
be substituted. If the locked validation shards are not locally staged, E1 is
blocked rather than silently run on mismatched data.

E1 finds codec-path negative transfer on all tasks before the operator is
called. Joint-to-matched global reconstruction NRMSE ratios are `2.4922x` for
Advection, `1.2396x` for Burgers, and `2.9357x` for the Darcy coefficient.
Both codecs poorly reconstruct the Darcy solution target, which standalone
decoder training did not supervise directly. See
`docs/research/2026-07-17-universal-latent-encoder-e1-result.md`.

E1 establishes that representation/decoding contributes to D6. It does not
establish cross-modality universality or cleanly separate co-adapted encoder
and decoder bases.

### E2 — paired discretization benchmark

The original UPT lineage and the live UPS divergence are now documented in
`docs/research/2026-07-17-universal-latent-backbone-lineage.md`. UPS previously
had separate grid and mesh/particle paths; the mesh path's storage-order chunk
pooling and adaptive average reduction were not the geometry-aware supernode +
transformer + learned-query Perceiver encoder described by UPT.

The opt-in canonical point-set encoder and qualification metrics are now
implemented, but no scientific E2 run has occurred. The preregistered gate is
`docs/research/2026-07-17-canonical-latent-codec-qualification-contract.md`.

Construct a validation-only benchmark in which the same physical states are
sampled as at least grid and irregular mesh representations, with particle
representations where physically meaningful. Freeze physical-state identity,
coordinates, field semantics, units, boundary conditions, parameters, and
train/validation grouping before learning.

Measure:

- within-representation reconstruction;
- paired latent alignment and retrieval;
- cross-encoding/cross-decoding;
- invariance under resolution, remeshing, and particle resampling;
- latent effective rank and collapse;
- task and representation leakage, reported diagnostically rather than
  assumed to be intrinsically good or bad.

The decisive comparison is the same physical state under different
discretizations. Unpaired datasets cannot adjudicate a common physical latent
space.

### E3 — codec/dynamics causal split

Only after E1/E2 produce a viable representation, freeze the codec and compare
one shared latent operator with matched task/family operators under identical
latent inputs and update accounting. Then reverse the freeze: hold the
operator fixed and compare shared versus specialized codecs. This factorial
split localizes interference instead of attributing all end-to-end error to
the trunk.

## Stop rules

- Do not add hidden family or task routing before E1/E2.
- Do not claim universality from equal latent dimensions or token counts.
- Do not use unpaired modalities as alignment evidence.
- Do not access held-out test data in E0-E3.
- Do not rerun D6, add seeds, extend schedules, or relax its frozen gates.
- If codec-only reconstruction or latent rank collapses, repair the encoder
  before another shared-operator experiment.
- If paired representations reconstruct well but do not align, test an
  explicit alignment objective before routing.
- If a frozen viable codec aligns and the shared operator still loses under
  E3, then family-specific dynamics become evidence-backed rather than a
  default workaround.

## Immediate next move

Wire the canonical point-set encoder and one any-point decoder into a codec-only
training path. Materialize the smallest frozen paired analytic grid/mesh
benchmark, then run the preregistered cross-decoding, retrieval, rank, and
remeshing gates. Do not skip directly to routing or a new shared-operator run.
