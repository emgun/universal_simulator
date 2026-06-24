# P2 Model-Side Parameter-Conditioned Transport Head Design

Date: 2026-06-24

Status: no-provider design complete. No GPU/provider work ran, no held-out
test was used, no claim evidence changed, and no public language changed.

## Purpose

The previous transport probe selected parameter-conditioned causal
shift/displacement as the strongest available mechanism for advection drift.
This note turns that mechanism into a model-side implementation target, so the
next work does not become another evaluator-only sidecar or generic adapter
capacity sweep.

The design question:

> What is the smallest model-side transport head that can absorb the validated
> beta-conditioned phase mechanism while preserving strict validation/test
> separation and task-family safeguards?

## Decision

Primary design: a default-off decoder-side transport head that predicts a
periodic displacement from causal PDE metadata and applies it to decoded
advection fields during rollout training/evaluation.

The first implementation should be scoped to `advection1d` and should not
modify the existing public `light-v1` default path unless explicitly enabled by
configuration. It should be treated as a candidate mechanism, not as claim
evidence or public copy.

## Input Contract

Required causal inputs:

- `task_name`: used to restrict the first head to `advection1d`.
- `task_family`: optional guard; active only for the transport family.
- `params.beta`: required for the first non-smoke advection candidate.
- `horizon` or rollout step: allowed because rollout horizon is known at
  inference time.

Behavior when beta is absent:

- Default candidate behavior: skip the transport head for that sample and
  record `transport_head_beta_missing_count`.
- Optional smoke behavior: allow a zero-shift fallback only in synthetic CPU
  tests, and mark it as `fallback = zero_shift`.
- Do not infer beta from validation targets, source identity as a learned key,
  or observed future context transitions.

This input contract intentionally keeps the candidate scoped: it requires beta
provenance for advection and therefore cannot silently become the standard
primary public protocol.

## Trainable Parameterization

First implementation target:

```text
shift = clamp(a_beta * beta + a_horizon * horizon_norm + bias, min_shift, max_shift)
```

Recommended defaults:

- `a_beta` initialized from the validated sidecar coefficient
  `10.236877359639507`.
- `bias` initialized from the validated sidecar coefficient
  `-0.08098891730605368`.
- `a_horizon` initialized to `0.0`.
- `min_shift` / `max_shift` set wide enough for the observed train/val shift
  distribution, then recorded in the artifact.

This is deliberately smaller than a full warp module. The validated evidence
already says beta explains the advection phase; the first model-side test should
check whether making that mechanism part of the rollout path preserves the
signal without adding unrelated capacity.

If this fails mechanically but the learned coefficients stay near the sidecar
values, the next variant can add fractional interpolation. If it fails because
the beta relation is unstable, escalate to a tiny periodic warp only after a
new branch check.

## Placement

Chosen placement: decoded field path after the decoder and before decoded
rollout loss/metric computation.

Why this placement:

- It matches the already validated decoded sidecar mechanism.
- It minimizes changes to the latent operator and encoder.
- It can be trained with decoded rollout loss and inspected directly via
  predicted shift statistics.
- It keeps Burgers/Darcy protected because the head can be task-scoped and
  default-off.

Rejected for first implementation:

- Latent-operator displacement: more aligned long term, but harder to validate
  because latent spatial semantics are less direct.
- Evaluator-only diagnostic: already exists as the sidecar path and does not
  move the mechanism into the learned simulator.
- External-backbone adapter capacity: recent Poseidon/DPOT evidence shows
  aggregate adapter capacity does not reliably fix advection phase drift.

## Proposed Config Surface

Use a new disabled-by-default configuration block:

```yaml
model_side_transport_head:
  enabled: true
  tasks: ["advection1d"]
  required_params: ["beta"]
  features: ["param:beta", "horizon_norm", "bias"]
  mode: "periodic_roll"
  apply_at: "decoded_rollout"
  trainable: true
  init:
    param:beta: 10.236877359639507
    horizon_norm: 0.0
    bias: -0.08098891730605368
  clamp:
    min_shift: -64.0
    max_shift: 64.0
  missing_param_policy: "skip"
```

The implementation can map this into `stages.decoded_operator_finetune` or a
top-level `operator.transport_head` block, but the persisted summary must echo
the resolved transport-head config so evidence validators can prove what ran.

## Training And Validation Command Shape

No provider run is authorized by this design. The first implementation should
add CPU/synthetic tests and a validation-only plan. When ready, the candidate
command should have this shape:

```bash
python scripts/run_light_experiment.py \
  --config configs/train_multitask_heterogeneous_light_best.yaml \
  --run-name ups_p2_model_side_beta_transport_head_val_light_v1 \
  --split val \
  --max-samples 32 \
  --rollout-steps 16 \
  --eval-override model_side_transport_head.enabled=true \
  --eval-override model_side_transport_head.tasks='["advection1d"]' \
  --eval-override model_side_transport_head.required_params='["beta"]' \
  --eval-override model_side_transport_head.missing_param_policy=skip
```

The exact command should be updated by the implementation owner after the
config path exists. It must remain train/validation only until the validation
artifact clears the gates below and a separate held-out contract exists.

## Artifact Schema

The summary or companion evidence JSON must include:

```json
{
  "candidate": "ups_p2_model_side_beta_transport_head",
  "held_out_test_used": false,
  "held_out_test_data_read": false,
  "model_side_transport_head": {
    "enabled": true,
    "mode": "periodic_roll",
    "apply_at": "decoded_rollout",
    "tasks": ["advection1d"],
    "required_params": ["beta"],
    "missing_param_policy": "skip",
    "trainable_parameter_count": 3,
    "coefficients": {
      "param:beta": 10.236877359639507,
      "horizon_norm": 0.0,
      "bias": -0.08098891730605368
    }
  },
  "transport_head_metrics": {
    "shift_mean": 0.0,
    "shift_std": 0.0,
    "beta_missing_count": 0,
    "applied_sample_count": 0,
    "skipped_sample_count": 0
  }
}
```

Validator requirements:

- reject any `split=test` command;
- reject missing or true held-out flags;
- reject absent resolved config;
- reject task scope broader than `advection1d` for the first candidate;
- reject missing beta provenance unless the run is explicitly marked synthetic
  smoke;
- recompute aggregate, advection rollout, advection h16, Burgers, and Darcy
  gates from the summary;
- fail if a context-roll estimator or observed/prediction roll estimator is
  also active.

## Validation Gates

The first real validation candidate must clear all of:

- aggregate decoded rollout NRMSE `< 0.35078329353213156`;
- advection decoded rollout NRMSE `< 0.4866576789288726`;
- advection h16 NRMSE `<= 0.44444171136384397`;
- Burgers decoded rollout NRMSE `<= 0.15674926288225416`;
- Darcy decoded rollout NRMSE `<= 0.2071060212271272`;
- `held_out_test_used = false`;
- `held_out_test_data_read = false`;
- no context-conditioned, observed, or prediction roll-shift estimator enabled.

If the candidate beats advection but regresses Burgers/Darcy, it is not a
general rollout-quality step. If it passes aggregate but misses advection h16,
it repeats the Option A/Option B failure mode and should stop.

## First Implementation Slice

Next safe work package:

1. Add synthetic CPU tests for a default-off `ModelSideTransportHead` or
   equivalent helper.
2. Implement the minimal linear beta/horizon/bias shift predictor and periodic
   roll application for 1D decoded fields.
3. Integrate it behind a disabled-by-default config path in decoded rollout
   training/evaluation only.
4. Emit resolved config and shift statistics in summaries.
5. Add an evidence validator stub that rejects held-out use and incompatible
   estimator combinations.
6. Run focused CPU tests and `git diff --check`.

Stop before any GPU/provider run. A provider plan is only useful after the CPU
implementation proves the head is default-off, beta-gated, and summary-visible.

## Recommendation

Implement the decoder-side linear beta transport head first. It is the best
tradeoff between scientific signal and engineering risk: it carries the
validated causal mechanism into the model path, avoids changing the latent
operator before the displacement semantics are proven, and gives future steward
ticks concrete validation gates instead of another broad architecture fork.
