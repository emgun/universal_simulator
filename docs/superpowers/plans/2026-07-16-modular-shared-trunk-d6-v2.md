# D6 v2 Modular Shared-Trunk Validation Plan

Date: 2026-07-16

Status: superseded before execution by D6 v3 after a second independent
fail-closed review; executable plan self-hash
`f00f2031be4138954cffc21fe5793aeeb0edbf9197b11e6290534e176897267d`
must never be launched; no D6 run occurred

## Supersession

The original D6 executable plan with self-hash
`ec36aead4c537267fae78c71de8d14156fba253899f90ec72fe867dd6bce80e8`
is abandoned before execution. Independent pre-merge review found that its
adapter-placement wording was not precise enough for the implementation and
that its result chain did not independently recompute parameter-shuffle
degradation or cryptographically bind stage and resumable resource evidence.
It must never be launched.

This replacement preserves the original scientific question, seed, data,
training schedule, arms, metrics, U1/U2 thresholds, cost cap, and stop rules.
Only protocol-integrity repairs are allowed before the new executable plan is
generated from a clean implementation commit.

## Exact architectural variable

Each task receives one bottleneck-16 input adapter and one bottleneck-16 output
adapter. The input adapter is applied after time features and AdaLN
conditioning, immediately before `PDETransformerBlock`. The output adapter is
applied after the shared output normalization and before the outer latent-state
residual addition. Both output projections are zero-initialized, so the initial
function remains identical to the D5 trunk.

The shared grid encoder, AnyPoint decoder, conditioning schema,
`PDETransformerBlock`, output normalization, and all non-adapter weights remain
unchanged. The four arms and full three-task adapter inventory are unchanged.

## Evidence repairs

- The six-object train/validation stage report is self-hashed, checked against
  the frozen training lock, bound into the self-hashed summary, and independently
  rechecked by the materializer.
- The materializer recomputes shuffled-parameter degradation from the joint and
  shuffled macro NRMSE values and rejects any inconsistent supplied scalar.
- Each arm persists a self-hashed attempt record binding its summary digest,
  orchestrator wall time, and cumulative child-process RSS high-water mark.
  Resume must load this record; missing or inconsistent evidence fails closed.

## Unchanged boundaries and gates

The run remains seed 17 and validation-only. Held-out and measurement-lock
access are forbidden. D5 is not retrained. No extra seed, replacement run,
epoch extension, threshold relaxation, dataset change, optimizer change, or
new dependency is permitted. U1 and U2 and their fail-closed interpretation are
identical to the original plan.
