# Canonical Latent E5 Decoder-Locality Result

Date: 2026-07-20
Status: `decoder_locality_causal`

## Decision

Explicit physical-space decoder locality is causal for the repeated E2-E4
information-preservation failure. Freeze the coordinate-aware local integral
decoder mechanism. Do not return to global `AnyPointDecoder`, enlarge the
latent, swap the encoder again, invoke an operator, or add routing.

The next valid question is whether a spatially anchored compressed latent can
retain this gain. E5 does not yet qualify a fixed-size universal latent because
its positive arm deliberately retains every source point.

## Frozen comparison

The preregistered contract is
`docs/research/2026-07-20-canonical-latent-e5-decoder-locality-contract.md`.

Both grid and warped-mesh specialists used the exact E4 direct-point encoder,
analytic state split, low/high mixed-resolution inputs, objective, `120`
epochs, `960` updates, `30,720` scheduled source examples, interpolation
baseline, and absolute `<=2x interpolation` gate.

Only the decoder changed:

- global control: unchanged `AnyPointDecoder`, `9,089` decoder parameters;
- local challenger: fixed physical radius `0.20`, relative-coordinate learned
  kernel, quadrature-aware normalized aggregation, `7,010` decoder parameters.

The challenger therefore has `22.9%` fewer decoder parameters. Both arms begin
with the same direct-point encoder hash. The global checkpoints exactly
reproduce E4, closing data, initialization, exposure, and evaluation drift.

The design follows current function-space guidance: graph neighborhoods must be
defined in physical coordinates and their integrals evaluated with quadrature
weights, rather than using index neighborhoods that collapse under refinement:
<https://www.nature.com/articles/s42256-026-01267-z>. It is also the minimal
decoder-side mechanism shared by GINO's local geometry decoder and RIGNO's
regional-to-physical transfer:
<https://proceedings.neurips.cc/paper_files/paper/2023/file/70518ea42831f02afc3a2828993935ad-Paper-Conference.pdf>,
<https://openreview.net/pdf?id=ahJfROJOYt>.

## Result

| Family | Global NRMSE | Local NRMSE | Interpolation | Local / interpolation | Relative gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| Grid | `0.262032` | `0.065206` | `0.090543` | `0.7202x` | `75.12%` |
| Mesh | `0.243033` | `0.075839` | `0.098671` | `0.7686x` | `68.79%` |

The local decoder passes the absolute gate for both families and outperforms
the four-neighbor inverse-distance interpolation baseline on this analytic
validation distribution. This is plausible because it learns a smooth local
kernel across the training state family; it is not evidence that it dominates
general numerical interpolation outside this frozen distribution.

At unseen input resolution `18`, local grid/mesh NRMSE improves further to
`0.037295`/`0.059645`, so both preregistered resolution-stability gates pass.
Low-resolution input remains harder at `0.107361`/`0.122162`, but still well
below the global control.

## Spectral and topology evidence

| Family | Global HF spectral NRMSE | Local HF spectral NRMSE | Relative gain |
| --- | ---: | ---: | ---: |
| Grid | `1.012150` | `0.316104` | `68.77%` |
| Mesh | `1.005935` | `0.359076` | `64.30%` |

The gain is not merely mean or variance correction: the local path recovers the
high-frequency component that global attention lost. High-frequency amplitude
ratios are still high (`1.62` grid, `1.88` mesh), so spectral calibration is not
perfect even though phase-aware error improves decisively.

Every high-resolution query has 8-30 source points inside the fixed radius.
The `32`-neighbor implementation cap never truncates the physical support.
Joint source-token/coordinate/measure permutation changes output by at most
`9.54e-7`, satisfying the frozen invariance tolerance.

## Causal interpretation and limitation

E5 proves that the previous codec ceiling was decoder-limited. The source
samples and learned token features already contained sufficient local detail;
unconstrained global cross-attention failed to retrieve it. A coordinate-aware,
quadrature-consistent inductive bias repairs both grid and mesh without a
representation label or extra capacity.

E5 is not yet a universal latent result. The positive decoder reads all source
tokens and their coordinates. Attaching it as a skip path around a future
latent operator would bypass the latent state and would not provide future-time
source features. The compressed follow-up must therefore place evolving latent
features at explicit spatial anchors; it must not secretly retain the original
source field under the decoder.

## Reproduction and boundary

Complete results at `/tmp/canonical-latent-e5-locality-final-v1/result.json`
and `/tmp/canonical-latent-e5-locality-final-v2/result.json` are byte-identical,
as are all four checkpoint files. Raw result SHA-256:
`eecdab66a7f74faea169ce95049c3ad72c3cf174df06873cb187e257b195b3a9`.
The config SHA-256 is
`5f9823dbc3093ee7559a6e43db61622cc62728790a20e51820dfa0159b304425`.

No operator, held-out read, provider call, routing path, task label, or
representation label occurred.

## Verification

- `31` focused E2-E5 encoder, decoder, and qualification tests pass;
- the complete `tests/unit` suite passes with its required localhost and
  PyTorch multiprocessing access;
- Ruff, Black check, Python bytecode compilation, JSON validation, and
  `git diff --check` pass;
- the two raw results and every corresponding checkpoint are byte-identical;
- the E5 global control state hashes exactly match the frozen E4 direct-point
  controls.

## Next coherent experiment

Freeze one E6 compressed-locality contract before implementation:

- encode the field into fixed-count spatially anchored latent tokens;
- decode only from those compressed tokens and their anchor coordinates with
  the E5 local integral mechanism—no original-source bypass;
- compare against the matched compressed global-decoder control;
- retain identical state split, exposure, absolute/spectral gates, cross-
  resolution checks, and grid/mesh specialists;
- preregister the anchor covering radius from geometry before training rather
  than tuning it from reconstruction results.

Only if a compressed specialist passes both families should shared grid/mesh
training resume. The latent operator remains downstream of codec qualification.
