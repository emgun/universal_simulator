# Frozen `strat-v1` Evaluation Contract

Date frozen: 2026-07-13

Status: frozen before any `strat-v1` baseline or candidate metric was produced.

## Purpose and authority

`strat-v1` is the primary three-task protocol for honest in-distribution,
regime-balanced generalization. The immutable universal release is
`docs/data/releases/strat-v1/universal/9d43d283f04f5b8d17cf6126ad189075c53307e715d7d4f61af440c2fed155c1/`.
Its training lock is
`5666fa8bb646b270d23077d1936c8ee80fa1cfeaf8f5870c55103fc58dd80afd`;
its separately authorized measurement lock is
`04a7936327611af3879d095bfc706ad03a60ac6fabda0a64671319b7b6db3fa2`.
Object hashes, split membership, and source identities may not change under
this name. A change requires a new protocol and release identity.

## Tasks and fixed splits

| Task | Physical parameter | Train | Validation | Reserved test |
| --- | --- | ---: | ---: | ---: |
| Advection 1D | `beta`, 8 regimes | 256 | 64 | 64 |
| Burgers 1D | `nu`, 12 regimes | 288 | 72 | 72 |
| Darcy 2D | `beta`, 5 regimes | 260 | 65 | 65 |

Every regime is exactly balanced within each task split. Provenance groups are
disjoint across train, validation, and test. For Darcy, all beta variants of a
matched coefficient realization stay in one split. Physical parameters are
allowed conditioning inputs only when the run record declares their use.

## Metrics and reporting

The primary temporal metric is decoded physical-space rollout NRMSE over 16
predicted steps for Advection and Burgers. Darcy is steady, so its primary
metric is decoded solution-field NRMSE; it has no artificial rollout horizon.
Every report must include:

- macro overall and per-task primary NRMSE;
- per-regime NRMSE for every task;
- per-horizon NRMSE for temporal tasks, including horizon 16;
- the exact data-lock, code, checkpoint, normalization, and selection hashes;
- whether physical-parameter conditioning or inferred-parameter context was used.

Aggregate scores are macro averages over tasks, not sample-count-weighted
averages. A candidate with any validation regime NRMSE greater than `1.5x` its
task mean is ineligible for held-out promotion.

## Selection and normalization

Training may use only the training lock's train objects. Hyperparameters,
epochs, checkpoints, variants, stopping, and promotion are selected using
training and validation only. Normalization is fitted on train objects and is
bound to the training-lock and selection hashes. Test metrics, test-derived
statistics, or prior `strat-v1` test results may not influence selection.

The baseline wall is measured first: persistence plus FNO, UNO, U-Net, and CNO
under this exact protocol. Candidate held-out promotion requires validation to
beat the best applicable re-measured baseline overall, protect every task, and
pass the regime-spread gate. Conditional branches must be declared before the
run that resolves them.

## Held-out access and ledger rules

Test bytes exist only in the measurement lock and a physically separate cache.
A held-out evaluation requires a pre-registered measurement contract containing
the selected artifact identity, exact command, metrics, gates, and unique
measurement key. Each key executes once. The result is appended to the
experiment ledger regardless of sign; failed or negative measurements are not
silently replaced. Model code never receives the measurement lock or test
cache during training or selection.

The publication-verification measurement performed while freezing this
contract transferred and checksummed test objects but ran no model, computed no
metric, and informed no selection.

## Reporting tracks

1. **Primary `strat-v1`:** disjoint, balanced, in-distribution generalization
   under this contract. Only this track can support the new three-task baseline
   claim.
2. **Frozen legacy/scoped:** `light-v1`, `medium-v1`, and the beta-head pretest
   retain their historical labels and interpretations. They may diagnose
   reproduction or regime extrapolation, but their numbers are never pooled or
   compared in the same claim sentence as `strat-v1`.

This freeze opens baseline measurement. It does not itself authorize a
candidate held-out evaluation or a broad universality claim.
