# Canonical Latent E3 Regional-Interaction Handoff

Date: 2026-07-20

## Decision

The compact RIGNO-style regional-interaction codec is `not_qualified`. It fixes
paired slot semantics and the formal E2 refinement-mismatch failure, but fails
absolute reconstruction and shared-versus-control parity. Do not invoke an
operator, add routing, extend this run, or swap immediately to another encoder.

## Implemented

- `RegionalInteractionEncoder`: one measure-aware point-set path with learned
  physical-to-regional edge messages, deterministic FPS regions, fixed
  geometric slot assignment, and three residual regional graph scales;
- E2 harness selection between the frozen Perceiver and regional challenger,
  with architecture metadata and unchanged gates;
- focused tests for schema/measure failures, grid and irregular inputs,
  storage-order invariance including grid ties, slot semantics across a warped
  mesh, measure sensitivity, and the no-attention/no-query boundary.

## Final result

Compact artifact:
`docs/research/artifacts/canonical_latent_e3_regional_interaction_result.json`.

- shared canonical NRMSE: grid `0.385641`, mesh `0.386519`, remesh `0.407992`;
- controls: grid `0.278208`, mesh `0.263731`;
- interpolation: grid `0.090543`, mesh `0.098671`;
- paired retrieval `1.0`, CKA `0.99776`, exact order invariance;
- grid/mesh mismatch falls from `0.15343` low-resolution to `0.08746` high-
  resolution;
- no operator, held-out read, representation/task model input, provider call,
  routing, or GPU occurred.

## Reproduction

The complete repaired results at
`/tmp/canonical-latent-e3-regional-final-v2/result.json` and
`/tmp/canonical-latent-e3-regional-final-v3/result.json` are byte-identical.
SHA-256:
`46bc50abe900973c5cacb74a44a7846b9ba8da3dd3273b7d36c3dc9580d1ad25`.

## Verification

- Focused regional/canonical encoder, benchmark, latent qualification, and
  decoder suite: `23 passed`.
- Complete `tests/unit` suite: passed outside the macOS sandbox; the in-
  sandbox run reached completion with only expected localhost and
  `torch_shm_manager` permission failures.
- Ruff, Black check, Python compilation, artifact JSON parse, and
  `git diff --check`: passed.

## Next coherent arc

Run a small codec capacity-identifiability ladder on specialist controls, not
another shared encoder. Hold the analytic states, measures, decoder family,
training exposure, and interpolation gate fixed; vary latent length and add a
no-compression/direct-query reconstruction ceiling. This distinguishes an
eight-token/decoder bottleneck from encoder failure. Reopen shared-encoder work
only if a bounded specialist codec passes the absolute reconstruction gate.
