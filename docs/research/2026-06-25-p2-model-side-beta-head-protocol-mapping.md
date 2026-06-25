# P2 Model-Side Beta Head Protocol Mapping

Date: 2026-06-25

Status: no-provider protocol/evidence mapping complete. No held-out test,
claim-evidence update, public-language change, or provider work was run.

## Inputs

- Current public protocol surface:
  `docs/claim_evidence/universal_sota_roadmap.md` and
  `docs/public/README.md`.
- Model-side beta transport-head validation result:
  `docs/research/2026-06-25-p2-model-side-transport-head-remote-result.md`.
- Remote result artifact:
  `b2://pdebench/remote-runs/model-side-transport-head/model_side_transport_head_real_shard_20260625T022059Z.tar.gz`.
- Local artifact SHA256:
  `9778317b2942728e0d5e9bd503baadbecd66ee08ef44968e9ed60eb2dff9e905`.

## Decision

The model-side beta transport head is not the same inference contract as the
current public `light-v1` CT8/shared-context primary claim. It is also not the
same contract as the already recorded CT1 online transport-context scoped
variant.

The result should be treated as a scoped
`light-v1 model-side beta-parameter transport-head UPS variant` until the
project explicitly decides that beta provenance is part of a public inference
contract. It is accepted validation evidence for that scoped branch only.

## Contract Comparison

| Surface | Contract | Current claim role |
| --- | --- | --- |
| CT8/shared-context primary | Frozen public `light-v1` claim protocol with the existing shared-context transport configuration, held-out ledger discipline, and no new beta-provenance requirement. | Primary public claim surface. |
| CT1 online transport-context | Scoped variant that changes context timing and online transport-context behavior. | Scoped held-out variant, not a primary replacement. |
| Data-conditioned context-phase | Scoped variant that uses one observed transition to infer phase. | Scoped held-out variant, not a no-context autonomous rollout claim. |
| Model-side beta transport head | Decoder-side model/evaluator hook applies periodic displacement using required `param:beta` provenance for `advection1d`; Burgers/Darcy are evaluated under the same full-task validation root without that head. | Validation-passing scoped branch only; not public claim evidence yet. |

## Beta Provenance

The head requires beta metadata for `advection1d`. In the successful validation
run, beta was derived through the official Advection beta-provenance root rather
than the unchanged standard `data/pdebench` root. The validator requires:

- `model_side_transport_head.required_params` includes `beta`.
- `model_side_transport_head_metrics.beta_missing_count = 0`.
- `model_side_transport_head_metrics.applied_count > 0`.
- held-out flags remain false.
- incompatible evaluator roll-shift sidecars are absent.

If beta is absent, the current implementation can skip the head, but a
non-smoke candidate must fail closed rather than silently becoming a different
contract.

## Pretest Readiness

The validation evidence is strong enough to draft a pre-registered held-out
pretest contract for the scoped beta-provenance variant. It is not enough to run
held-out, update claim evidence, or change public language in this tick.

No additional validation-only check is required before drafting that scoped
contract because the validated run cleared the active aggregate and per-task
schema gates with:

- aggregate decoded rollout NRMSE `0.11122069837659315`
- advection rollout `0.0017868115829009724`
- advection h16 `0.001784282965734058`
- Burgers `0.14738121133726986`
- Darcy `0.18897951477635447`
- `512` applied shifts
- zero skipped samples
- zero beta-missing samples
- no held-out data read

If the goal is to replace the current CT8/shared-context primary claim under
the same public inference contract, this branch is not ready. That would require
either removing the beta-provenance dependency or explicitly defining and
validating a new public protocol that includes beta provenance as an allowed
input.

## Required Gates For A Scoped Held-Out Contract

A future scoped pretest contract should be written before any held-out command
and should include:

- exact run command, git ref, artifact path, artifact SHA256, and intended
  measurement key.
- explicit variant label:
  `light-v1 model-side beta-parameter transport-head UPS variant`.
- explicit statement that this is not the CT8 primary contract, not CT1, not an
  external-paper reproduction, and not public claim evidence until the contract
  is executed and audited.
- beta provenance source and fail-closed behavior for missing beta.
- no held-out test data download, sharding, ledger write, or artifact access
  before the pretest begins.
- `held_out_test_used = true` only inside the pre-registered held-out run, and
  all pretest artifact manifests must show no repeated key or extra test access.
- summary validator must pass with top-level
  `extra.model_side_transport_head` and
  `extra.model_side_transport_head_metrics`.
- metrics to record and gate: aggregate decoded rollout NRMSE, advection
  rollout, advection h16, Burgers, Darcy, applied count, skipped count,
  beta-missing count, held-out/test data flags, and artifact traceability.

Recommended stop conditions:

- stop if beta provenance cannot be derived for every held-out advection sample.
- stop if the run would silently skip advection transport-head application.
- stop if the command needs a new public protocol claim before evidence exists.
- stop if the result misses CT8 primary held-out aggregate or regresses any
  non-transport family enough to erase the transport gain.
- stop if the artifact cannot be validated and hashed before any claim-evidence
  discussion.

## Next Path

Draft the scoped held-out pretest contract and validator/audit wiring next, but
do not execute the held-out command without explicit user direction for that
irreversible test measurement. Keep claim evidence and public language unchanged
until a schema-passing held-out result exists and the scoped-variant language is
reviewed separately.
