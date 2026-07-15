# Strat-v1 shared tier-b D5 plan

Date: 2026-07-15

Status: implementation contract; validation-only; no provider run or held-out access yet.

## Question

Can one physically conditioned native UPS `tier_b` model match three
same-architecture single-task controls closely enough to earn further shared-model
work, while remaining materially cheaper than storing and serving all three controls?

This is an architecture/interference measurement, not claim-grade specialist
adequacy. D5 deliberately excludes low-data adaptation, leave-one-regime transfer,
additional seeds, and held-out measurement.

## Why tier-b precedes Poseidon

The native UPS path already carries train-bound beta/nu transforms, parameter values,
presence signatures, task identity, equation semantics, and steady-operator support in
its latent data layer. Poseidon's current adapter accepts task IDs only, omits the
separate Darcy solution target, and reports teacher-forced one-step predictions as
rollouts. Repairing Poseidon first would change both candidate and measurement harness.
Tier-b lets D5 repair the shared harness once and ask one architectural question.

## Autoresearch scope contract

- Run tag: `2026-07-15-strat-v1-shared-tier-b-d5`
- Branch: `codex/shared-conditioning-d5`
- Primary metric: equal-task macro of Advection/Burgers autoregressive h16 rollout
  NRMSE and Darcy coefficient-to-solution NRMSE.
- Live arms: one shared three-task `tier_b` model and three single-task `tier_b`
  controls.
- Seed: `17` only.
- Data: the six immutable `strat-v1` training-lock objects, roles `train` and
  `valid` only.
- Mutable surface: conditioning schema, steady/sample-balanced training helpers,
  strict UPS validation metrics, D5 config/planner/runner/materializer, and focused
  tests.
- Fixed surface: shard bytes, split selection, train-fitted parameter transforms,
  metric addendum, seed, architecture sizes, stage recipe, optimizer recipe, and
  validation gates.
- Forbidden moves: measurement lock or test-object access, legacy capacity wrapper,
  explicit conditioning maps that omit physical parameters, validation-oracle
  correction, extra seeds, Poseidon/DPOT execution, dependency changes, or public
  claim updates.

## Matched architecture and exposure

Every arm uses one frozen universal input schema even when only one task is selected:

- task vocabulary: `advection1d`, `burgers1d`, `darcy2d`;
- parameter vocabulary: `beta`, `nu`;
- train-bound log10-zscore parameter values plus value-presence signatures;
- equation signature, resolution, and spatial dimensionality;
- identical `tier_b` latent/operator/decoder sizes.

The latent-operator stage uses batch size `16`. Decoder, decoded-operator, and
joint stages use batch size `2` because each source sample expands into as many
as 201 decoded frames; the smaller decoded-stage batch bounds peak memory
without changing examples, optimizer recipes, or the matched exposure between
the shared and specialist arms.

The training objective averages transitions within each source sample before averaging
samples, preventing temporal trajectories from outweighing one-shot Darcy mappings.
Steady samples are exactly coefficient-to-solution mappings and receive no invented
time trajectory. One shared epoch sees one pass over each task; the three controls
together see the same total examples. The result must report exact optimizer steps,
examples, parameters, checkpoint bytes, GPU time, and peak memory.

## Gates

All gates are frozen before execution:

1. Held-out reads equal `0`.
2. Shared macro NRMSE is no greater than `1.05x` the equal-task macro of the three
   live controls.
3. No shared task metric exceeds `1.10x` its matched live control.
4. Shared beats frozen persistence on every task.
5. Shared Darcy NRMSE is no greater than `0.1403299866`, or `1.20x` the frozen D3
   conditioned-FNO value.
6. Maximum corrected global-scale regime spread is at most `1.5` for every task.
7. A deterministic within-task cyclic shuffle of physical parameters worsens the
   shared primary metric by at least `5%`. D1-D4 already establish nonzero
   counterfactual prediction sensitivity for the specialist interface; D5 tests that
   the shared interface retains metric-relevant parameter use.
8. The one shared checkpoint uses fewer parameters and checkpoint bytes than the
   three-control ensemble.

Failure closes this native shared recipe at the frozen scale. Passing U1/U2 permits a
separate D5b low-data transfer contract; it does not authorize held-out access.

## Harness and stop conditions

The implementation must fail before training if arm schemas differ, a source binding
is dirty or uncommitted, a test-capable path is present, the training lock contains a
test role, or a validation shard lacks complete regime coverage. It must publish a
self-hashed plan, resumable checkpoint evidence, an independently materialized result,
and an immutable archive with read-back SHA verification and guaranteed Vast teardown.

The baseline is not yet runnable locally because the six data objects are external.
Setup therefore ends when the synthetic mechanics harness, focused tests, full unit
suite, frozen plan generator, and dry-run remote contract pass. The first paid run is
the baseline and requires that clean implementation snapshot.
