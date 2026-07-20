# Canonical Latent E4 Capacity-Identifiability Handoff

Date: 2026-07-20

## Decision

E4 classifies the active blocker as `decoder_objective_or_schedule_blocker`.
Compound C8/C16/C32 scaling does not improve absolute reconstruction, and the
learned all-point/no-compression ceiling still fails. Pause encoder, capacity,
operator, and routing work.

## Final evidence

Compact artifact:
`docs/research/artifacts/canonical_latent_e4_capacity_identifiability_result.json`.

- C8 grid/mesh NRMSE: `0.282391` / `0.276096`;
- C16: `0.289961` / `0.328263`;
- C32: `0.299443` / `0.309111`;
- direct-point ceiling: `0.262032` / `0.243033`;
- interpolation: `0.090543` / `0.098671`;
- direct-point unseen-resolution NRMSE: `0.261981` / `0.243565`;
- direct-point high-frequency spectral NRMSE: `1.0122` / `1.0059`;
- C8 checkpoint hashes exactly reproduce frozen E2 controls;
- no operator, held-out read, provider call, routing, or GPU.

Two complete results are byte-identical at SHA-256
`c79befefadb6f6d5da72077d4df4aaa2bbaf0f9b0c4ff2936787bc930999467d`.

Verification is green: `27` focused E2-E4 codec tests pass; the complete unit
suite passes with its required localhost and PyTorch multiprocessing access;
Ruff, Black check, bytecode compilation, JSON validation, source/result hash
checks, and `git diff --check` pass.

## Implemented

- `scripts/run_canonical_latent_e4_capacity_ladder.py`;
- compound Perceiver capacity rungs with identical specialist exposure;
- a measure-aware direct-point codec retaining every source token;
- unseen-resolution, output-mismatch, effective-rank, magnitude, mean-bias,
  high-frequency spectral, convergence, parameter, checkpoint, and boundary
  evidence;
- focused tests for direct-token retention, quadrature sensitivity, source-
  order invariance, spectral identity, compact result materialization, and
  operator/provider/held-out boundaries.

## Next coherent arc

Freeze and run one decoder-locality E5 ablation on the direct-point ceiling:
current global `AnyPointDecoder` versus an explicit measure-aware relative-
coordinate local kernel or RIGNO-style regional-to-point message decoder.
Keep source tokens, hidden width, state split, exposure, direct interpolation,
absolute gate, and spectral diagnostics unchanged. If locality passes both
families, only then return to the smallest compressed codec.
