# Canonical Latent E6 Compressed-Locality Result

Date: 2026-07-20
Status: `compressed_locality_not_qualified`

## Decision

Close the compact eight-token regional codec. A physical-space local decoder
does not rescue the E3 regional representation after a real no-bypass
compression boundary. Do not add a router, instantiate an operator, enlarge
this same regional-token model, sweep its radius, or resume shared training.

This result does not reject a common latent space. It rejects this particular
realization: eight pointwise regional tokens whose learned features have an
effective rank below four and expose only one to four tokens to each query.
The next representation should be defined as coefficients of one physical
function space, not as another collection of abstract or sparsely sampled
tokens.

## Frozen comparison

The preregistered contract is
`docs/research/2026-07-20-canonical-latent-e6-compressed-locality-contract.md`.

Both grid and warped-mesh specialists used the exact E3 regional encoder,
analytic state split, mixed low/high input resolutions, latent-alignment loss,
`120` epochs, `960` updates, `30,720` scheduled source examples, and the same
validation metrics. The frozen E3 global controls reproduce exactly:

- grid checkpoint `8ffe13f62cb5bacd5925dd0a62279183a3f4b6100af6225c6086e12dfe1e9041`;
- mesh checkpoint `5bdd8d865d01ce774ebbfad6a9dd1b9a794df9df245e24a4d4b5714c37f359b6`.

Only the decoder changed. The global arm used the `9,089`-parameter
`AnyPointDecoder`; the local arm used the `7,010`-parameter E5 local integral
decoder. Both received only eight latent feature vectors, their selected
physical coordinates, and aggregated Voronoi masses. Original values,
coordinates, measures, and source tokens were unavailable after encoding.

The eight tokens compress the high-resolution input by `24.5x`. Their
pre-training worst-case covering radius was `0.44444445`; support was frozen at
`0.47`, a `5.75%` margin. The radius was not tuned from results.

## Result

| Family | Global NRMSE | Local NRMSE | Interpolation | Local / interpolation | Relative change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Grid | `0.278208` | `0.304720` | `0.090543` | `3.3655x` | `9.53%` worse |
| Mesh | `0.263731` | `0.323675` | `0.098671` | `3.2804x` | `22.73%` worse |

Neither local specialist passes the absolute `<=2x interpolation` gate. At
unseen input resolution `18`, local NRMSE degrades to `0.526729` grid and
`0.637470` mesh, so neither resolution-stability gate passes.

The negative result is also spectral:

| Family | Global HF NRMSE | Local HF NRMSE | Relative change | Local HF amplitude ratio |
| --- | ---: | ---: | ---: | ---: |
| Grid | `1.008671` | `1.186041` | `17.58%` worse | `3.9047` |
| Mesh | `1.021147` | `1.228073` | `20.26%` worse | `4.2973` |

The local decoder does not merely fail to recover detail; it amplifies the
high-frequency component while increasing phase-aware error.

## What was ruled out

The implementation-side causal checks pass:

- every query has physical support, with one to four of eight tokens used;
- the neighbor cap never truncates support;
- regional masses are positive and sum to one;
- joint latent/coordinate/mass permutation changes outputs by at most
  `2.38e-7`;
- the challenger has fewer parameters than the control;
- paired grid/mesh anchors remain close in semantic slot order: mean distance
  `0.01197`, maximum `0.07537`;
- the original source field is not available to either decoder.

The active failure is therefore representational. E5 established that local
field information exists when all physical samples cross the bottleneck. E6
shows that the eight processed regional features do not preserve enough of
that information for the same locality mechanism to use. The high-resolution
effective latent ranks (`3.88` grid and `3.89` mesh) reinforce that diagnosis,
although they do not by themselves prove why the learned encoder collapses.

This is consistent with the current function-space view of neural operators:
discretization convergence requires a common continuous-domain representation,
physical-coordinate neighborhoods, and quadrature-consistent integral maps,
not merely a fixed tensor shape. See
<https://www.nature.com/articles/s42256-026-01267-z>. RIGNO supplies a useful
regional-mesh transfer mechanism, but this experiment shows that an eight-node
regional state is not an adequate information-bearing space for the frozen
field family: <https://arxiv.org/abs/2501.19205>. UPT's learned abstract tokens
and global decoder had already failed the same absolute codec gate:
<https://proceedings.neurips.cc/paper_files/paper/2024/file/2cd36d327f33d47b372d4711edd08de0-Paper-Conference.pdf>.

## Reproduction and boundary

Complete results at
`/tmp/canonical_latent_e6_compressed_locality_v1/result.json` and
`/tmp/canonical_latent_e6_compressed_locality_v2/result.json` are
byte-identical, as are all four checkpoint files. A third full run from the
final formatted source at
`/tmp/canonical_latent_e6_compressed_locality_v3/result.json` is also
byte-identical. Raw result SHA-256:
`cc7fd10af42eabc91148173f3c467416e0f5741fadd055e013a1ac6571066393`.
The config SHA-256 is
`1ff6844badd65a62509c5037d6f54d7608e309a91fe3e960a491d4fda273d5e2`.

No operator, held-out read, provider call, routing path, task label,
representation label, or source-feature bypass occurred.

## Next coherent experiment

Before another learned encoder, define E7 as a function-space latent
sufficiency test:

1. choose one modality-agnostic multiresolution physical basis whose functions
   are evaluated from coordinates and whose coefficients form the latent;
2. obtain coefficients by deterministic quadrature-weighted projection from
   grids, meshes, or particles, with no modality label or learned router;
3. reconstruct from those coefficients only and require the same grid/mesh
   absolute, spectral, refinement, and paired-state gates;
4. freeze the coefficient budget from the analytic field bandwidth and
   pre-training approximation error, not from downstream reconstruction
   results;
5. train a universal encoder to approximate that qualified projection only if
   the deterministic latent space first preserves the required information.

This separates the question "is the proposed common latent space sufficient?"
from "can a neural encoder discover it?" It also gives a future latent
operator an actual common physical state to evolve, rather than eight opaque
features or a hidden source-token path.
