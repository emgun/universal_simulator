# D6 v3 Modular Shared-Trunk Validation Plan

Date: 2026-07-16

Status: superseded before execution by D6 v4 after final independent review;
executable plan self-hash
`017af9eab7c60250d0825f91eea83f25a212e49c09368e96a3037e889f70290e`
must never be launched; no D6 run occurred

## Supersession

D6 v3 inherits the exact scientific question, architecture, seed, data,
training schedule, arms, metrics, U1/U2 thresholds, cost cap, stop rules, and
protocol-integrity repairs from D6 v2. D6 v2 self-hash
`f00f2031be4138954cffc21fe5793aeeb0edbf9197b11e6290534e176897267d`
was abandoned before execution after independent review found two remaining
fail-closed gaps. Neither v1 nor v2 was launched.

## Final integrity repairs

- Stage evidence must contain exactly six entries, exactly six unique frozen
  object IDs, exact train/valid roles, and exact SHA-256 algorithm/value pairs.
- The runner requires the exact
  `strat-v1-modular-shared-trunk-d6-v3` plan identity.
- The local launcher verifies the plan self-hash and validation-only boundary,
  rejects the retired v1/v2 hashes, and requires the selected Git ref to
  descend from the plan-bound implementation before network/provider work.

## Exact architecture and evidence chain

The input adapter is applied after time features and AdaLN conditioning,
immediately before `PDETransformerBlock`. The output adapter is applied after
shared output normalization and before the outer latent-state residual. Both
output projections are zero-initialized.

The stage report, per-arm attempt records, run summary, and independently
materialized result are self-hashed and transitively bound. The materializer
recomputes shuffled-parameter degradation from the reference and shuffled
macro NRMSE values. Resume requires positive, self-hashed wall-time and
cumulative child-process RSS evidence bound to each arm summary.

## Unchanged boundaries

The run remains seed 17 and validation-only. Held-out and measurement-lock
access are forbidden. D5 is not retrained. No extra seed, replacement run,
epoch extension, threshold relaxation, dataset change, optimizer change,
architecture change, or new dependency is permitted. U1 and U2 retain their
original fail-closed interpretation.
