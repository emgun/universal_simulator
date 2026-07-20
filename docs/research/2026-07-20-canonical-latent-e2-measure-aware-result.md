# Canonical Latent E2 Measure-Aware Result

Date: 2026-07-20

## Decision

The current learned-query Perceiver codec is **not qualified** as the canonical
physical latent basis. It learns representation-invariant state identity, but
it does not preserve enough field information for accurate or convergent
decoding. Do not freeze this codec, invoke the latent operator, or add routing.

The next architecture challenger should replace the encoder's local
supernode-to-token compression with a RIGNO-style regional interaction path
while retaining the same measure-aware inputs, any-point decoder, exposure
accounting, and E2 evaluation.

This conclusion follows the current discretization-invariance literature:

- [ICLR 2025 discretization mismatch](https://proceedings.iclr.cc/paper_files/paper/2025/hash/313829757739365201b5adb3a1cbd9bd-Abstract-Conference.html)
  requires comparing outputs produced from different input discretizations.
- [NeurIPS 2025 RIGNO](https://papers.nips.cc/paper_files/paper/2025/hash/dcb91f43033bb1d367d1848806dee98d-Abstract-Conference.html)
  uses a downsampled regional mesh and explicitly evaluates unseen resolution
  robustness.
- [Principled function-space architectures, 2026](https://www.nature.com/articles/s42256-026-01267-z)
  warns that encoder-decoder bottlenecks can lose high-resolution information
  and emphasizes principled discretization and domain-level receptive fields.

## Protocol

The final deterministic CPU run used 128 analytic training states and 24
disjoint validation states. Each state is a seven-coefficient two-dimensional
field containing smooth, localized, and higher-frequency components. Regular
grids and two independent smooth irregular remeshings carry positive normalized
quadrature measures derived from their mapping Jacobian.

The shared, grid-only, and mesh-only arms:

- start from the identical checkpoint state;
- train 120 exact epochs, 960 optimizer updates, and 30,720 scheduled source
  examples each;
- use one `CanonicalPointEncoder` and one `AnyPointDecoder` per arm;
- train only inverse codec reconstruction and paired alignment;
- never instantiate a latent operator or receive task/representation labels.

The first mechanics shakedown used only relative shared-versus-control gates.
It incorrectly returned `qualified` while all codecs had about `0.70` absolute
NRMSE. That result is rejected as decision evidence. Before the final run, the
gate was repaired with a deterministic inverse-distance interpolation baseline
and an explicit resolution-convergence requirement. This protocol repair is
reported rather than hidden; E2 remains validation-only mechanism evidence,
not a claim-grade promotion.

Compact source-bound result:
`docs/research/artifacts/canonical_latent_e2_measure_aware_result.json`.

Full local result SHA-256:
`793169a871f661a02f514c970376043f403193d0841bd695aab0e1c7eea980a4`.

## What passed

- Point storage-order invariance: maximum latent difference `7.15e-7` and
  decoded difference `1.07e-6`.
- Paired grid/mesh identity: symmetric top-1 retrieval `1.0` versus chance
  `0.0417`, permutation `p=0.005`.
- Latent alignment: CKA `0.9984`; standardized paired RMSE `0.0517` versus
  fixed-negative `0.6967`.
- Rank preservation: physical-state effective rank `3.374`; shared grid
  `3.686`, shared mesh `3.596`.
- Cross-decoding symmetry: cross-to-within NRMSE ratio `1.0019`.

These results show that the shared learned queries provide stable token slots
and recognize the same state across discretizations. They do not establish an
operator-usable physical representation.

## What failed

| Canonical-query codec | NRMSE |
| --- | ---: |
| Shared, grid source | `0.302411` |
| Shared, mesh source | `0.306910` |
| Shared, independent remesh | `0.316649` |
| Grid-only control | `0.282391` |
| Mesh-only control | `0.276096` |
| Grid inverse-distance interpolation | `0.090543` |
| Mesh inverse-distance interpolation | `0.098671` |

The fair alternating-resolution schedule substantially narrows the relative
gap: the shared codec is `1.071x` the grid control and `1.112x` the mesh
control, narrowly missing the `1.10x` mesh bound. It remains `3.340x` and
`3.110x` worse than direct interpolation. Independent-remesh error is only
`1.032x` the base mesh error and passes.

Most importantly, refinement does not converge. Grid/mesh output mismatch
increases from `0.06636` at the low resolution to `0.17469` at the unseen high
resolution, a `2.633x` increase. The worst pairwise mismatch is `0.23832`.

The final gate vector is:

- pass: identity, alignment margin, paired identity, rank, cross-decoding,
  canonical-query relative quality, remeshing, boundary;
- fail: absolute reconstruction, within-codec parity, resolution convergence.

## Mechanism interpretation

The failure is not simply that grid and mesh latent coordinates rotate apart:
retrieval, CKA, alignment margin, and cross-decoding all pass. The information
bottleneck instead learns a common but lossy state code. The shared arm's final
loss is still improving, so this run does not prove that schedule extension
could never help. It does establish that the exact equal 120-epoch candidate is
not qualified, remains more than `3x` worse than direct interpolation, and
worsens rather than converges under resolution refinement. Do not extend this
candidate outside a new frozen contract.

A RIGNO-style regional interaction graph is the highest-signal challenger
because it targets the missing domain-level information flow and resolution
behavior without abandoning one discretization-neutral encoder. Transolver++
local adaptivity remains a later scalability option; mixture-of-experts routing
would not repair the measured codec loss.
