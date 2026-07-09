# P2 Post-Held-Out Branch Check

Date: 2026-06-23

Status: complete. No held-out test, provider spend, credential use, claim
evidence update, or public-language change was performed by this branch check.

## Trigger

Poseidon ScOT `channel_lift` Option A cleared the aggregate train/validation
gate but failed the single ledger-protected held-out pre-test:

- Validation aggregate decoded rollout NRMSE: `0.35782889238675264`
- Validation advection decoded rollout NRMSE: `0.4937043430599529`
- Held-out aggregate decoded rollout NRMSE: `0.5551415687535287`
- Held-out advection decoded rollout NRMSE: `0.7840223655431167`
- Held-out measurement key:
  `b487e8841f7631554248fcaeedf9dd3a1fba1faa7f003f0e6304a2b96375516a`

Evidence:

- `docs/research/2026-06-23-p2-poseidon-channel-lift-heldout-result.md`
- `docs/research/2026-06-23-p2-poseidon-channel-lift-heldout-evidence-manifest.json`
- `reports/research/sota_loop/external_baselines/poseidon_scot_channel_lift_test_light_v1_e30_lr1e2_roll4/summary.json`

The result should not be treated as a generic runtime failure: Burgers and
Darcy stayed near their validation behavior, while advection/transport broke.

## Branches Considered

### A. DPOT Probe

DPOT remains the cleanest next challenger branch. The earlier P2 adapter design
already identified it as the fallback when Poseidon stalls:
`docs/research/2026-06-11-p2-poseidon-adapter-design.md`.

Why it moves the roadmap:

- It tests a different pretrained PDE backbone rather than adding more capacity
  around the same Poseidon failure mode.
- It is aligned with the foundation/backbone hypothesis rather than another
  local correction around the current UPS operator.
- It can be kept validation-only, with the same train/validation discipline and
  artifact schema used for Poseidon.

Costs and risks:

- No DPOT runner is currently implemented in this repo.
- Source/checkpoint availability and input/output contract need a local import
  and adapter audit before GPU spend.
- A DPOT validation run should not advance to held-out unless it clears an
  advection-aware gate, not only aggregate G2a.

Best bounded next action:

Write a DPOT adapter/readiness design and CPU/import smoke runbook, then
implement only enough runner surface to produce a 2-sample validation smoke.
Do not spend GPU until the import/checkpoint/schema path is locally verified.

### B. Transport-Aware Poseidon Adapter Or Backbone Change

Poseidon is not conclusively dead, but Option A is. The key lesson is that the
aggregate validation gate was too permissive: Option A validation advection
`0.4937043430599529` was already weak, and held-out advection became
`0.7840223655431167`.

Viable Poseidon follow-ups:

- Option B shallow nonlinear lift/readout from the original design.
- Task-modulated channel gains/biases from the original Option A extension.
- A very small controlled unfreeze/LoRA path only after a stronger validation
  result.

Why it is secondary:

- It continues the same backbone family immediately after a failed held-out
  transfer.
- It risks optimizing the aggregate again unless the gate is changed first.
- It likely needs GPU validation to be informative, while DPOT still needs a
  no-provider readiness step.

Gate required before any future held-out pre-test:

- Aggregate validation must clear G2a: `decoded_rollout_nrmse <=
  0.363424243629033`.
- Advection validation must improve materially over Option A and clear at least
  the prior model-side phase reference:
  `task_advection1d_decoded_rollout_nrmse <= 0.4866576789288726`.
- Prefer also requiring an h16/advection horizon metric when the runner exposes
  it, because prior held-out misses were dominated by long-horizon transport.
- No task may collapse near `1.0`.

Best bounded next action:

Keep Poseidon Option B/C as secondary. Revisit only after DPOT readiness is
known or if a no-provider implementation review shows a very small patch can
add task modulation plus an advection-aware validator without GPU spend.

### C. Local Transport-Sidecar Lessons

The prior transport-sidecar/phase track showed that explicit transport
parameter information can make advection easy under validation-only protocols,
but that track is closed as a standalone branch:
`docs/claim_evidence/universal_sota_roadmap.md`.

Useful lesson to carry forward:

- Any future backbone/adapter gate must protect transport explicitly; aggregate
  validation NRMSE alone is insufficient.
- Parameter-conditioned transport signals are useful as diagnostic inspiration,
  but not as another standalone sidecar or roll-shift-estimator loop.
- Reopening this work requires a new successor model contract, such as
  model-side parameter conditioning inside a credible learned operator.

Why it should not be the next branch:

- The prior track is saturated at validation and explicitly deprioritized for
  more sidecar variants.
- More transport-only pretest contracts would violate the current roadmap
  direction unless tied to a concrete successor model.

Best bounded next action:

Use these lessons to define DPOT/Poseidon validation gates and diagnostics. Do
not reopen standalone sidecar or shift-estimator work.

## Decision

Primary next branch: **DPOT readiness and 2-sample validation smoke design**.

Secondary branch: Poseidon Option B/task modulation only after the DPOT
readiness result, and only under the stricter advection-aware validation gate.

Retired for now: standalone transport sidecar, roll-shift estimator, or
transport-only held-out pretest work.

## Stop Rules

- Do not rerun the Poseidon held-out key.
- Do not run any new held-out test from this branch check.
- Do not spend GPU/provider budget until a DPOT or Poseidon validation-only
  plan names source, checkpoint, command, artifact schema, expected runtime,
  gate, and stop conditions.
- Do not update claim evidence or public docs from this research note.

## Next Best Path

Prepare a DPOT adapter/readiness note that answers:

1. Which DPOT source/checkpoint is the live candidate?
2. What input/output tensor contract must be adapted to `light-v1`?
3. What parameters are trainable versus frozen?
4. What local CPU/import smoke can run with 2 samples?
5. What validation gate includes aggregate plus advection/transport protection?

Only after those are explicit should a bounded validation GPU run be considered.
