# P2 Post-DPOT Validation Branch Check

Date: 2026-06-23

Status: branch decision after negative DPOT Tiny validation. No provider work
ran for this branch check, no held-out test was used, no claim evidence
changed, and no public language changed.

## Trigger

DPOT Tiny `channel_lift` passed the local CPU mechanics smoke but missed the
bounded validation-only GPU gate:

- Aggregate decoded rollout NRMSE: `0.7136888249949349`
- Advection decoded rollout NRMSE: `0.8575561454613253`
- Burgers decoded rollout NRMSE: `0.588255711789389`
- Darcy decoded rollout NRMSE: `0.28923145953251056`

This is worse than the Poseidon `channel_lift` validation aggregate
`0.35782889238675264` and repeats the transport/advection weakness seen in the
Poseidon held-out pretest. It is not a held-out result and is not
claim-comparable evidence.

## Options

### A. Escalate DPOT

Possible moves:

- try a larger DPOT checkpoint;
- train more than the 13-parameter lift/readout adapter;
- add task-conditioned channel gains/biases or a temporal adapter.

Pros:

- DPOT has a native temporal input contract, which was the main reason to test
  it after Poseidon failed on transport generalization.
- Darcy improved relative to the tiny CPU smoke, suggesting the runner path is
  functional and not uniformly broken.

Cons:

- Tiny `channel_lift` missed aggregate and advection gates badly after full
  train/validation.
- Escalating checkpoint size or adapter scope would add provider cost before
  there is evidence that DPOT is close to the validation frontier.
- The observed failure is not a small threshold miss; advection `0.8576`
  approaches the kind of transport collapse this phase is trying to avoid.

Decision: do not spend more provider budget on DPOT escalation until a
no-provider design identifies a specific mechanism likely to address
advection, with a bounded validation gate and kill condition. DPOT remains a
secondary branch, not the next GPU run.

### B. Poseidon Option B / Task Modulation

Possible moves:

- keep the pretrained Poseidon embedding/recovery intact;
- add task-conditioned adapter parameters on top of `channel_lift`;
- make the validation gate advection-aware before any future held-out pretest.

Pros:

- Poseidon Option A already cleared aggregate validation G2a and was much
  closer than DPOT Tiny on train/validation.
- The failure mode is now sharply identified: validation aggregate alone is not
  enough, and transport/advection must be protected.
- Task modulation is a small, testable adapter change and can be implemented
  without immediately using held-out test.

Cons:

- Poseidon Option A failed the single held-out pretest, so Option B must not be
  treated as a claim path unless it clears a stricter validation gate and earns
  a new pretest contract later.
- Task modulation may overfit validation if the gate is not explicit about
  advection and per-task stability.

Decision: this is the primary next engineering branch, but only as
validation-only design/implementation first. The next useful work is to design
or implement a task-conditioned channel-lift adapter and validator that gates
aggregate, advection, Burgers, and Darcy before any GPU run.

### C. UPS-Side Transport/Refiner Work

Possible moves:

- revisit a decoded-space refiner or transport-aware correction path;
- strengthen validation metrics around phase/shift error;
- return to the learned-operator roadmap with transport-specific pressure.

Pros:

- Directly targets the project north star and the recurring failure mode.
- Avoids dependence on external backbone source/checkpoint behavior.

Cons:

- Previous local sidecar/shift-estimator work was closed because it risked
  reopening a broad local branch without enough evidence.
- It is likely a larger engineering branch than Poseidon Option B.

Decision: keep this as the fallback if small adapter/backbone follow-ups fail
strict validation. Do not reopen it just because DPOT Tiny failed.

## Decision

Primary next branch: Poseidon Option B / task modulation, validation-only.

Required before any provider spend:

- define the adapter parameterization and trainable count;
- keep pretrained embedding/recovery intact;
- keep held-out test forbidden;
- define a strict validation gate:
  - aggregate decoded rollout NRMSE `<= 0.363424243629033`;
  - advection decoded rollout NRMSE `<= 0.4866576789288726`;
  - Burgers must not materially regress from Poseidon Option A validation
    `0.15674926288225416`;
  - Darcy must not materially regress from Poseidon Option A validation
    `0.2071060212271272`;
  - no task may approach collapse near `1.0`;
- specify the exact command, artifact schema, expected runtime, and stop
  conditions.

Secondary branch: DPOT escalation only after a no-provider design states why a
larger checkpoint or richer temporal adapter should fix advection and what
validation result would kill it.

Closed for now: standalone transport sidecar/shift-estimator work and any
held-out pretest.

## Next Best Path

Prepare a no-provider Poseidon Option B task-modulation design and, if small
enough, implement runner scaffolding/tests before any new GPU run. The goal is
not another aggregate-only improvement; the goal is a validation gate that
would have caught the earlier held-out transport failure.
