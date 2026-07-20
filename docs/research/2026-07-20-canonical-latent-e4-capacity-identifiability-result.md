# Canonical Latent E4 Capacity-Identifiability Result

Date: 2026-07-20
Status: `decoder_objective_or_schedule_blocker`

## Decision

Pause encoder architecture and latent-capacity work. Do not enlarge the latent
again, retest a shared encoder, invoke an operator, add routing, relax the
absolute gate, or extend these runs.

Compound scaling from eight to sixteen and thirty-two latent tokens does not
improve specialist reconstruction. Removing fixed-token compression entirely
improves fidelity and discretization stability, but the unchanged global
`AnyPointDecoder` still fails the absolute interpolation gate and reconstructs
almost none of the high-frequency structure. The active causal gate is decoder
locality, reconstruction objective, or schedule—not latent-token count.

## Research-grounded design

The frozen contract is
`docs/research/2026-07-20-canonical-latent-e4-capacity-identifiability-contract.md`.

- UPT's own capacity study scales supernodes and latent tokens together, so E4
  tested `8/24`, `16/48`, and `32/96` token/supernode pairs rather than changing
  tokens alone: <https://arxiv.org/abs/2402.12365>.
- Current function-space guidance requires coordinate-aware continuous
  operations, quadrature weights, and explicit discretization-convergence
  measurement: <https://www.nature.com/articles/s42256-026-01267-z>.
- Phaedra's 2026 tokenizer study motivates checking spectral and magnitude
  fidelity, not spatial NRMSE alone: <https://arxiv.org/abs/2602.03915>.

Every arm used the exact E2/E3 analytic states, measures, decoder family,
`120` epochs, `960` optimizer updates, `30,720` scheduled source examples,
and direct-interpolation gate. Only specialist grid and mesh codecs ran.

## Compound capacity result

| Rung | Grid NRMSE | Mesh NRMSE | Grid / interpolation | Mesh / interpolation |
| --- | ---: | ---: | ---: | ---: |
| C8: 8 tokens / 24 supernodes | `0.282391` | `0.276096` | `3.1189x` | `2.7982x` |
| C16: 16 / 48 | `0.289961` | `0.328263` | `3.2025x` | `3.3269x` |
| C32: 32 / 96 | `0.299443` | `0.309111` | `3.3072x` | `3.1328x` |

The unchanged absolute thresholds are grid `<=0.181086` and mesh
`<=0.197341`. No rung passes either family, and the curve is not monotonic.
Larger latent sequences therefore do not identify the eight-token bottleneck
as causal.

C8 independently reproduces the frozen E2 specialist checkpoints exactly:
grid SHA-256
`e8424c54ebe05d6ce40fe4740a71a872da340e4c05631448ac92e45eb0215499`
and mesh
`fc18f2266789cfd62f7b7da5b219fd4fd0c7e1b787d9831585ddcfe52f9fc685`.
This confirms the ladder did not drift from the E2 control protocol.

## Learned no-compression ceiling

The ceiling projects every sampled coordinate, field value, and normalized
quadrature weight into a source token and gives all source tokens to the same
decoder. At the high training resolution it uses `196` tokens instead of
compressing to `8/16/32`.

| Family | High-source NRMSE | Unseen-resolution NRMSE | Ratio to interpolation | Gate |
| --- | ---: | ---: | ---: | --- |
| Grid | `0.262032` | `0.261981` | `2.8940x` | fail |
| Mesh | `0.243033` | `0.243565` | `2.4631x` | fail |

Removing compression improves the C8 grid error by `7.21%` and mesh error by
`11.98%`, and makes unseen-resolution error essentially flat. Its high-versus-
validation output mismatch is only `0.00971` grid and `0.00950` mesh. Thus
compression contributes to the error and cross-resolution instability, but it
is not the active absolute blocker.

The grid ceiling still needs a further `30.89%` error reduction to pass; mesh
needs `18.80%`. Final training losses are only `4.8%` and `7.4%` above their
observed minima, so ordinary final-epoch variation is too small to explain the
required gap.

## Spectral and magnitude diagnosis

The direct-point ceiling preserves gross magnitude: prediction/target standard-
deviation ratios are `0.9953` grid and `0.9511` mesh, with small normalized mean
bias. Yet high-frequency spectral NRMSE remains `1.0122` and `1.0059`, close to
complete failure on the high-frequency component. Every compressed rung is
similarly near `1.0`.

The source samples contain the missing detail—four-neighbor interpolation
reconstructs it at `0.09054/0.09867`—but global latent-to-query cross-attention
does not recover it even when all source points are retained. This supports a
decoder-locality hypothesis: the current decoder has no explicit relative-
coordinate local kernel or regional-to-point graph edge.

## Reproduction and boundary

Complete results at `/tmp/canonical-latent-e4-capacity-final-v1/result.json`
and `/tmp/canonical-latent-e4-capacity-final-v2/result.json` are byte-identical.
SHA-256:
`c79befefadb6f6d5da72077d4df4aaa2bbaf0f9b0c4ff2936787bc930999467d`.
The config SHA-256 is
`947243fddf8a8149014a04662b52d409e8a047a5af7de1eb0a7f115d1427d4ca`.
Source and checkpoint hashes are in the compact artifact.

No operator, held-out read, task/representation model input, provider call,
routing path, or GPU occurred.

## Verification

- `27` focused E2-E4 encoder, decoder, and qualification tests pass;
- the complete `tests/unit` suite passes outside the filesystem/network sandbox
  required for its localhost and PyTorch multiprocessing tests;
- Ruff, Black check, Python bytecode compilation, JSON validation, and
  `git diff --check` pass;
- both raw result files and all bound source files re-hash to the values above.

## Next coherent experiment

Run one decoder-locality ablation on the learned no-compression ceiling:

- control: the current global `AnyPointDecoder`;
- challenger: a measure-aware relative-coordinate local kernel or RIGNO-style
  regional-to-point message decoder with the same source tokens, hidden width,
  exposure, targets, and absolute/spectral gates;
- retain direct interpolation as the external ceiling.

If explicit locality passes both families, freeze that decoder mechanism and
then retest the smallest compressed codec. If locality still fails, isolate the
objective and schedule before returning to tokenization. This remains an
encoder-free, operator-free causal test.
