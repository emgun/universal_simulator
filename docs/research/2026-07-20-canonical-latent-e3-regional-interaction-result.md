# Canonical Latent E3 Regional-Interaction Result

Date: 2026-07-20
Status: `not_qualified`

## Decision

Reject this exact compact regional-interaction codec. Do not freeze it, invoke
an operator, extend its schedule, relax gates, or add modality/task/family
routing.

The RIGNO-style mechanism improves the specific refinement-consistency failure
seen in E2 and preserves strong shared identity after a required slot-ordering
repair. It does not preserve the field accurately enough. Both shared arms are
about four times worse than direct interpolation and substantially worse than
their architecture-matched controls.

## Research grounding and isolated mechanism

The frozen contract is
`docs/research/2026-07-20-canonical-latent-e3-regional-interaction-contract.md`.
The challenger follows the encoder-side structure of RIGNO (NeurIPS 2025):
learned physical-to-regional messages, a downsampled regional mesh, and
residual multiscale regional message passing. The same `AnyPointDecoder` from
E2 remains in place. No regional-to-physical graph decoder or dynamics
operator was introduced, so the result isolates encoder-side compression.

The official RIGNO paper and code are
<https://papers.nips.cc/paper_files/paper/2025/hash/dcb91f43033bb1d367d1848806dee98d-Abstract-Conference.html>
and <https://github.com/camlab-ethz/rigno>.

## Pre-measurement repair

The first complete mechanics run was discarded before being treated as E3
evidence. FPS selected nearly identical grid and mesh regional sets at
resolution 18 (one-way Chamfer `0.0021`), but their discovery order gave a
matched-slot distance of `0.6101`. That violated the fixed latent-sequence
contract and made tokenwise alignment ill-posed.

The encoder now deterministically assigns the unchanged selected nodes to
fixed normalized geometric slots. The repair changes no selected node,
parameter count, seed, state, exposure, loss, decoder, or gate. Focused tests
cover regular-grid ties, arbitrary point order, and warped-mesh slot semantics.
The discarded `/tmp/canonical-latent-e3-regional-final-v1` metrics are not
experiment evidence.

## Frozen run

- seed `17`;
- `128` analytic train states and `24` disjoint validation states;
- `120` epochs, `960` optimizer updates, and `30,720` scheduled source
  examples per arm;
- eight `32`-dimensional regional latent nodes;
- regional encoder `27,872` parameters, unchanged decoder `9,089`, total
  `36,961`;
- identically initialized shared, grid-control, and mesh-control arms;
- alternating low/high shared exposure and the unchanged E2 evaluation;
- CPU-only, provider-free, validation-only, and no operator.

## Result

| Metric | Shared grid | Shared mesh | Grid control | Mesh control |
| --- | ---: | ---: | ---: | ---: |
| Canonical-query NRMSE | `0.385641` | `0.386519` | `0.278208` | `0.263731` |
| Shared/control ratio | `1.3862x` | `1.4656x` | - | - |
| Direct interpolation NRMSE | `0.090543` | `0.098671` | - | - |
| Shared/interpolation ratio | `4.2592x` | `3.9173x` | - | - |

The shared remesh NRMSE is `0.407992`, or `1.0556x` the base shared-mesh
error, so remeshing passes. Cross-decoding and canonical-query relative gates
also pass, but they compare already-lossy decoded fields.

Paired semantics recover after the slot repair:

- symmetric top-1 retrieval `1.0` versus chance `0.04167`;
- permutation p-value `0.005`;
- linear CKA `0.99776`;
- standardized paired RMSE `0.07548` versus fixed-negative `0.63603`;
- exact storage-order invariance: latent and decoded maximum absolute
  differences both `0.0`.

Unlike E2, the formal resolution-convergence gate passes: grid/mesh output
mismatch falls from `0.15343` at the low resolution to `0.08746` at the
highest resolution, a `0.5700x` ratio. This is a real improvement in agreement
between representations, but the corresponding highest-resolution prediction
NRMSE remains about `0.46`; agreement between two inaccurate reconstructions
does not qualify the latent.

Failed gates:

- `within_codec`: the shared candidate is `38.6%` worse than the grid control
  and `46.6%` worse than the mesh control;
- `absolute_reconstruction`: shared grid and mesh are roughly four times the
  direct interpolation errors.

Every other gate passes. The compact evidence is
`docs/research/artifacts/canonical_latent_e3_regional_interaction_result.json`.

## Reproduction and provenance

The repaired run was executed independently at
`/tmp/canonical-latent-e3-regional-final-v2` and
`/tmp/canonical-latent-e3-regional-final-v3`. Their complete `result.json`
files are byte-identical with SHA-256
`46bc50abe900973c5cacb74a44a7846b9ba8da3dd3273b7d36c3dc9580d1ad25`.
The config SHA-256 is
`9e5b434810a9a3f3fb77d5a3ff9e837b3a9df294ba45ab6191b2f1ed7ff87433`.
Source and checkpoint hashes are in the compact artifact.

## Verification

- regional encoder, benchmark, prior canonical encoder, latent qualification,
  and arbitrary-point decoder focused suite: `23 passed`;
- complete `tests/unit` suite: passed outside the macOS sandbox so localhost
  staging and `torch_shm_manager` loader tests could use required OS
  facilities;
- Ruff, Black check, Python compilation, compact-result JSON parse, and
  `git diff --check`: passed;
- both repaired full benchmark runs produced byte-identical complete JSON.

## Roadmap implication

Do not immediately test another encoder architecture. E2's Perceiver controls
(`0.2824` grid, `0.2761` mesh) and E3's regional controls (`0.2782`, `0.2637`)
are all roughly three times worse than interpolation. The repeated absolute
failure is therefore not identified as an encoder-only defect; the fixed
eight-token bottleneck, arbitrary-point decoder, objective, or schedule may be
the limiting mechanism.

The next highest-signal experiment is a codec capacity-identifiability ladder
on specialist controls before any new shared candidate: vary latent length
while keeping the data, decoder family, exposure, and absolute gate fixed, and
include a no-compression/direct-query reconstruction ceiling. Continue shared
latent work only if some bounded codec setting reaches the interpolation gate.
