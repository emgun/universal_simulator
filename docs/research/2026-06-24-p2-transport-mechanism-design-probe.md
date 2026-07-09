# P2 Transport Mechanism Design Probe

Date: 2026-06-24

Status: no-provider design/probe complete. No GPU/provider work ran, no
held-out test was used, no claim evidence changed, and no public language
changed.

## Purpose

The post-Option-B branch check selected UPS-side transport mechanism work as
the primary next branch. This note converts that branch decision into a
concrete mechanism path and uses existing validation-only evidence to avoid
another aggregate-only external-backbone loop.

The question for this tick:

> Which smallest causal transport mechanism is most likely to reduce advection
> drift while preserving Burgers/Darcy, and what gate should it face next?

## Context

Recent external-backbone results:

- Poseidon Option A validation aggregate cleared G2a at
  `0.35782889238675264`, but advection was weak at
  `0.4937043430599529` and the single held-out pretest failed with held-out
  advection `0.7840223655431167`.
- Poseidon Option B task modulation improved aggregate, Burgers, and Darcy,
  but missed the strict advection gate: aggregate `0.3566052737393018`,
  advection `0.4967493071208899`, Burgers `0.14460934384484475`, Darcy
  `0.18262014873452226`.
- DPOT Tiny `channel_lift` missed validation badly: aggregate
  `0.7136888249949349`, advection `0.8575561454613253`.

This makes the next path mechanism-specific transport work, not more generic
adapter capacity.

## Candidate Chosen

Primary mechanism: parameter-conditioned causal transport shift/displacement.

Existing validated evidence:

- Evidence JSON:
  `docs/claim_evidence/ups_advection_p2_parameter_canonical_root_sidecar_val_evidence.json`
- Summary:
  `docs/claim_evidence/artifacts/ups_advection_p2_parameter_canonical_root_sidecar_val_summary.json`
- Validator:
  `scripts/validate_p2_parameter_canonical_root_sidecar_evidence.py`

The mechanism uses the decoded evaluator's
`evaluation.decoded_data_conditioned_roll_shift_estimator` with:

```text
feature_names = ["param:beta", "bias"]
coefficients = {"param:beta": 10.236877359639507, "bias": -0.08098891730605368}
mode = "roll_persistence"
tasks = ["advection1d"]
calibration_scope = "p2_parameter_conditioned_train_fit"
```

This is causal with respect to the PDE parameter metadata: it does not use
validation-oracle shifts, held-out data, source identity as a learned key, or
observed context transitions. It does require beta provenance for the advection
root, so it is a scoped mechanism/protocol candidate rather than a replacement
for the current public `light-v1` primary claim.

## Probe Verification

Commands run:

```bash
python scripts/validate_p2_parameter_canonical_root_sidecar_evidence.py
python -m pytest tests/unit/test_validate_p2_parameter_canonical_root_sidecar_evidence.py tests/unit/test_validate_p2_parameter_mixed_root_sidecar_evidence.py tests/unit/test_run_inferred_transport_transfer_scorecard.py tests/unit/test_diagnose_transport_temporal_windows.py -q
```

Results:

- Evidence validator returned `passed = true` with no errors.
- Focused tests passed: `15 passed`.

Validated metrics from the canonical-root summary:

| Metric | Value |
| --- | ---: |
| Aggregate decoded rollout NRMSE | `0.11122069865007121` |
| Advection decoded rollout NRMSE | `0.0017868130908052495` |
| Advection h16 NRMSE | `0.0017842800879688658` |
| Burgers decoded rollout NRMSE | `0.14738121412908425` |
| Darcy decoded rollout NRMSE | `0.188979512124482` |
| Data-conditioned roll-shift mean | `3.501918154754239` |
| Data-conditioned roll-shift std | `2.3455632165989537` |

These numbers clear the active transport phase gate by a wide margin:

- aggregate `< 0.35078329353213156`;
- advection rollout `< 0.4866576789288726`;
- advection h16 `<= 0.44444171136384397`.

They also preserve the non-transport families relative to the recent external
adapter gates:

- Burgers `0.14738121412908425` is better than the Option A validation guard
  `0.15674926288225416`;
- Darcy `0.188979512124482` is better than the Option A validation guard
  `0.2071060212271272`.

## Tradeoff

The mechanism is high-signal because it directly explains the advection phase
failure with a causal physical parameter. It is also not yet a direct public
claim replacement because the current primary `light-v1` protocol does not
universally require beta provenance in the standard data root.

There are two clean ways forward:

1. Formalize a canonical beta-provenance validation protocol for advection and
   keep the result scoped.
2. Move the parameter-conditioned transport behavior into model-side training
   or a default-off model head so the mechanism becomes part of the learned
   simulator rather than a decoded evaluator sidecar.

The second path is more aligned with the north star, but it needs a bounded
implementation plan before any provider run.

## Next Plan

Prepare a model-side parameter-conditioned transport head design before any
new GPU work.

Required design contents:

- exact input contract: `param:beta` plus task/family metadata, with behavior
  when beta is absent;
- trainable parameterization: scalar shift head, fractional roll/displacement
  head, or tiny periodic warp module;
- where it lives: decoder-side, latent-operator side, or evaluator-only
  diagnostic;
- whether it changes the public inference contract;
- train/validation command;
- artifact schema and validator;
- stop conditions.

Minimum validation gates for the first no-held-out implementation:

- aggregate decoded rollout NRMSE `< 0.35078329353213156`;
- advection decoded rollout NRMSE `< 0.4866576789288726`;
- advection h16 NRMSE `<= 0.44444171136384397`;
- Burgers decoded rollout NRMSE `<= 0.15674926288225416`;
- Darcy decoded rollout NRMSE `<= 0.2071060212271272`;
- `held_out_test_used = false`;
- `held_out_test_data_read = false`;
- no context-roll estimator unless the design explicitly stays scoped as a
  context-conditioned diagnostic.

## Decision

Primary next branch: model-side parameter-conditioned transport mechanism
design, no-provider.

Do not run another external-backbone GPU experiment, held-out pretest, or
public claim update from this evidence alone. The current best signal is that
transport/advection needs a causal phase/displacement mechanism, and beta
conditioning is the strongest validated handle available in the repo.
