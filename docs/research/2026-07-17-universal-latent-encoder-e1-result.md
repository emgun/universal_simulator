# Universal Latent Encoder E1 Result

Date: 2026-07-17

## Decision

E1 confirms that D6 contains substantial codec-path negative transfer before
the latent operator is called. The joint grid codec is worse than the matched
codec on every task. D6 therefore cannot support a family-router conclusion;
part of its end-to-end failure is already present in representation and
decoding.

This is a post-hoc validation diagnostic, not a preregistered promotion gate.
It localizes a failure mechanism and changes the next engineering step. It
does not establish grid/mesh/particle alignment or authorize held-out access.

## Boundary and provenance

- Training lock SHA-256:
  `5666fa8bb646b270d23077d1936c8ee80fa1cfeaf8f5870c55103fc58dd80afd`
- Exact validation objects staged and rehashed: Advection `47,288,356`
  bytes, Burgers `32,289,060` bytes, Darcy `3,918,783` bytes; total
  `83,496,199` bytes.
- D6 joint and three matched `encoder_joint.pt` / `decoder_joint.pt`
  checkpoints came from immutable archive SHA-256
  `3e58f7fea593f46e05389c9260a13ac33f60eca44e157cdb06234a9c1eaf9bcc`.
- The audit instantiated no operator, made no parameter update, performed no
  training, and read no held-out or measurement-lock object.
- Source-bound result:
  `docs/research/artifacts/strat_v1_d6_universal_latent_codec_audit.json`.

The audited state is the first physical input state, matching the standalone
reconstruction path. For Darcy this is the coefficient field. Darcy's target
solution is also audited separately because it is the state that the operator
must ultimately decode.

## Codec-only result

Global NRMSE aggregates squared error and target energy over all validation
states. `J/J` is joint encoder plus joint decoder; `M/M` is the task-matched
encoder plus decoder.

| Physical state | J/J global NRMSE | M/M global NRMSE | J/J to M/M |
| --- | ---: | ---: | ---: |
| Advection input | `0.275626` | `0.110594` | `2.4922x` |
| Burgers input | `0.284255` | `0.229305` | `1.2396x` |
| Darcy coefficient input | `1.403196` | `0.477970` | `2.9357x` |
| Darcy solution target | `0.903793` | `0.833640` | `1.0842x` |

The Darcy solution row is poor for both codecs. Per-sample relative errors are
strongly scale-sensitive across beta regimes: median NRMSE is `5.8903` for
J/J and `0.9092` for M/M, while the global metric is `0.9038` and `0.8336`.
The result therefore supports a codec-training defect, not a new regime gate.

## Encoder/decoder swap matrix

Cross-swapping checkpoints is diagnostic because separately trained latent
bases may be co-adapted. It cannot cleanly assign all error to one component.
Still, Darcy is decisive:

| Darcy coefficient path | Global NRMSE |
| --- | ---: |
| Joint encoder / joint decoder | `1.403196` |
| Joint encoder / matched decoder | `0.571853` |
| Matched encoder / joint decoder | `1.429776` |
| Matched encoder / matched decoder | `0.477970` |

Changing the decoder moves the result far more than changing the encoder. The
standalone decoder stage currently reconstructs `fields` only. For Darcy,
`fields` is the coefficient; the solution target enters decoder supervision
only through coupled operator rollout stages. That makes the output codec an
uncontrolled confound in the D6 Darcy result.

Advection and Burgers cross-swaps are worse than either aligned pair, showing
strong encoder-decoder basis co-adaptation. They demonstrate why separately
initialized latent tensors with the same shape are not automatically one
interchangeable latent space.

## Latent geometry

Joint-versus-matched linear CKA is very high: Advection `0.999166`, Burgers
`0.998460`, and Darcy coefficient `0.999998`. This means sample geometry is
similar up to linear transformations; it does not mean the latent bases are
cross-decodable.

Effective rank is preserved approximately for the two temporal inputs:
Advection `2.458` physical to `2.310` joint latent and Burgers `2.226` to
`2.102`. Darcy coefficient rank contracts from `7.245` physical to `2.806`
joint latent. Joint and matched Darcy latent ranks are nearly identical, so
this contraction is a shared codec limitation rather than the source of the
joint-versus-matched gap.

The deterministic within-validation task probe reaches `0.56` accuracy versus
`0.333` chance, with Darcy perfectly separable and Advection not separable
under this simple probe. This is descriptive only; task identity in a physical
latent is neither inherently desirable nor inherently invalid.

## Implementation repair

Two forward-path defects are repaired without altering or rerunning D6:

1. Standalone decoder training now includes both coefficient and solution for
   canonical steady-operator tasks. Temporal behavior is unchanged.
2. Joint training now preserves `pre_joint/operator.pt`,
   `pre_joint/encoder.pt`, and `pre_joint/decoder.pt` below the checkpoint
   directory before overwriting compatibility checkpoints. Keeping evidence in
   a subdirectory prevents deployment-size accounting from counting duplicate
   models. D6's archive contains tensor-identical base and joint checkpoints
   because the old path overwrote the base files, so its true pre-joint
   parameter delta cannot be recovered.

## Forward path

Do not train another shared operator yet. First make the codec a separately
qualified object:

1. Define a canonical latent-basis contract and codec-only validation gates
   for coefficient fields, solution fields, and temporal states.
2. Train and freeze a codec under those gates, preserving pre/post checkpoints
   and the full cross-decoding matrix.
3. Run E2 on paired physical states sampled as grid and irregular mesh (then
   particles where meaningful), with explicit latent alignment and
   remeshing/resampling invariance.
4. Only after E2 passes, run the E3 frozen-codec shared-versus-specialized
   operator comparison.
