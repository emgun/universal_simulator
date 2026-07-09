# P2 Post-Option-B Transport Branch Check

Date: 2026-06-23

Status: branch decision after mixed Poseidon Option B validation. No provider
work ran for this branch check, no held-out test was used, no claim evidence
changed, and no public language changed.

## Trigger

Poseidon Option B `channel_lift_task_modulated` completed a bounded
validation-only GPU run:

- Aggregate decoded rollout NRMSE: `0.3566052737393018`
- Advection decoded rollout NRMSE: `0.4967493071208899`
- Burgers decoded rollout NRMSE: `0.14460934384484475`
- Darcy decoded rollout NRMSE: `0.18262014873452226`

The result is mixed. Aggregate clears G2a, and Burgers/Darcy improve versus
Poseidon Option A validation, but advection misses the strict transport gate
`<= 0.4866576789288726` and slightly regresses from Option A validation
`0.4937043430599529`.

This confirms the current failure is not a generic adapter-capacity problem.
Small task modulation can improve non-transport families and aggregate score,
but it does not repair the transport/advection failure mode that caused the
Option A held-out pretest miss.

## Options

### A. Continue Poseidon With A Transport-Aware Adapter

Possible moves:

- add temporal conditioning beyond the scalar `time_value`;
- add a transport-specific displacement/phase head around the Poseidon output;
- try shallow nonlinear lift/readout or LoRA only with an explicit advection
  mechanism and no held-out access.

Pros:

- Poseidon remains the strongest external backbone track on aggregate
  validation.
- The frozen pretrained embedding/recovery path is now proven mechanically
  sound under `channel_lift` and `channel_lift_task_modulated`.
- A transport-aware output correction could target the actual missed gate
  without discarding the working non-transport improvements.

Cons:

- Option A and Option B both failed the advection-specific threshold.
- More adapter capacity without a transport mechanism risks another
  aggregate-only GPU run.
- LoRA or a nonlinear adapter would add trainable capacity/provider cost before
  explaining the transport miss.

Decision: keep Poseidon alive only as a secondary design branch. The next
Poseidon proposal must be transport-aware on paper before any GPU run:
specific phase/displacement/temporal mechanism, exact trainable scope, and a
kill gate that treats advection as first-class.

### B. Escalate DPOT

Possible moves:

- try a larger DPOT checkpoint;
- add a richer temporal adapter;
- replace repeat-current history with a better causal history initialization.

Pros:

- DPOT has a native temporal input contract, which is theoretically relevant
  to transport.
- A richer history adapter could test whether the Tiny result was a poor
  interface rather than a poor backbone.

Cons:

- DPOT Tiny missed validation badly: aggregate `0.7136888249949349`,
  advection `0.8575561454613253`, Burgers `0.588255711789389`.
- The current DPOT evidence is farther from the frontier than both Poseidon
  variants.
- Escalating checkpoint size or adapter scope would spend provider budget
  without a concrete mechanism for the observed advection failure.

Decision: do not run DPOT GPU/provider work next. DPOT escalation remains
blocked until a no-provider mechanism note explains why a specific temporal
interface should fix advection and what validation result would kill it.

### C. Return To UPS-Side Transport Mechanism Work

Possible moves:

- use the existing train/validation-only transport-shift gates and scorecards
  as the next diagnostic surface;
- design a small causal transport phase or displacement sidecar that predicts
  from allowed train-fitted/current-context features;
- strengthen validation around advection h16, phase/shift error, and
  train/validation shift support before any held-out contract.

Pros:

- This directly targets the north-star blocker: decoded physical-space rollout
  quality, specifically transport/advection drift.
- Prior local landscape notes already selected causal transport-phase work as
  the highest-signal local mechanism after fixed alpha/shift and global recipe
  sweeps failed.
- The repo already has relevant implementation seams and tests:
  `scripts/run_source_conditioned_transport_shift_gate.py`,
  `scripts/run_inferred_transport_transfer_scorecard.py`,
  `scripts/diagnose_transport_temporal_windows.py`, and associated unit tests.
- It avoids another paid external-backbone run until a mechanism-level
  validation plan exists.

Cons:

- Prior transport-sidecar work was kept closed to avoid broad local churn.
- It risks changing the inference contract unless the next step is explicitly
  train-fitted, default-off, validation-only, and claim-safe.
- It may produce scoped diagnostic evidence rather than a direct universal
  model improvement on the first pass.

Decision: this is now the primary branch, but not as an open-ended sidecar
restart. The next work should be a no-provider design/probe that uses existing
transport diagnostics to answer one question: what causal transport mechanism
is most likely to reduce advection without hiding behind aggregate metrics or
validation-oracle shifts?

## Decision

Primary next branch: UPS-side transport mechanism design/probe,
validation-only and no-provider.

Do not run another GPU experiment until this branch check is converted into a
bounded validation plan with:

- the exact causal inputs allowed at inference;
- whether the mechanism changes the inference contract or remains a diagnostic
  gate;
- the train-fitted parameters or learned parameters;
- validation command and artifact schema;
- aggregate, advection, advection h16, Burgers, and Darcy gates;
- a kill condition that prevents another aggregate-only pass.

Secondary branch: Poseidon transport-aware adapter or temporal modulation, but
only after a no-provider design identifies the phase/displacement mechanism and
parameter scope. A shallow nonlinear lift/readout or LoRA run without that
mechanism should not be the next provider spend.

Blocked branch: DPOT escalation. It needs a specific temporal-interface
hypothesis before any more spend.

Closed for now: held-out pretest, public claim updates, and claim-evidence
changes.

## Next Best Path

Prepare a no-provider transport mechanism design note that starts from the
existing UPS transport scripts and prior phase-estimator plan. The note should
choose one small candidate to implement or revalidate next, likely a
train-fitted causal transport/displacement diagnostic that reports advection
h16 and per-task regressions. The purpose is to convert the observed external
backbone failure into a local mechanism-level validation plan, not to run
another foundation-adapter sweep.

Reassess after that note only if it gives a concrete, bounded validation
command. If it does not, stop and ask for a strategic choice between local
transport mechanism work and pausing Phase 2.
