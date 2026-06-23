# Current State

Updated: 2026-06-23

## Project

Universal Physics Stack is research software for latent-space neural simulation
of PDE-style physical systems. The durable repository is
`/Users/emerygunselman/Code/universal_simulator`; Codex worktrees may be used
for execution, but roadmap synthesis should resolve back to that canonical
workspace when paths disagree.

## North Star

Improve decoded physical-space rollout quality across task families while
preserving strict validation/test separation, frozen protocols, artifact
traceability, and claim-safe public language. The current technical blocker is
long-horizon rollout stability, especially transport/advection phase tracking
and drift on persistence-friendly tasks.

## Current Evidence

- Public `light-v1` claim: UPS primary held-out decoded rollout NRMSE is
  `0.4165820594268877`, beating the measured matched-protocol external
  baselines recorded in `docs/claim_evidence/universal_sota_roadmap.md`.
- Medium confirmation: `ups_medium_shared_context_transport` held-out
  `medium-v1` decoded rollout NRMSE is `0.30616533327650614` versus
  persistence `0.5725109200102603`; see
  `docs/claim_evidence/medium_v1_confirmation_evidence.json`.
- Phase 1 capacity sweep: scaling the current learned operator from small
  through tier-d did not beat persistence on validation; tier-b was best but
  still far above persistence. See
  `docs/research/2026-06-10-p1-capacity-sweep-results.md`.
- Phase 1 recipe sweep: longer rollout pressure, horizon weighting, semigroup
  weighting, and longer training did not fix h16 collapse. E1 is killed at this
  scale and the Phase 1 fallback is active. See
  `docs/research/2026-06-11-p1-recipe-sweep-results.md`.
- Phase 2 adapter design exists: `docs/research/2026-06-11-p2-poseidon-adapter-design.md`
  specifies the Poseidon channel-lift/readout path and DPOT probe. The
  `channel_lift` runner path and unit tests are implemented in
  `scripts/run_external_poseidon_scot_finetune.py` and
  `tests/unit/test_external_poseidon_scot_finetune.py`; targeted tests passed
  on 2026-06-22 with `python -m pytest tests/unit/test_external_poseidon_scot_finetune.py -q`.
- P2.2 Poseidon `channel_lift` Option A cleared validation gate G2a on
  `light-v1` train/val only: decoded rollout NRMSE
  `0.35782889238675264` versus gate `0.363424243629033`; held-out test was
  not used. Evidence:
  `reports/research/sota_loop/external_baselines/poseidon_scot_channel_lift_val_light_v1_e30_lr1e2_roll4/summary.json`
  and B2 artifact
  `b2://pdebench/remote-runs/poseidon-channel-lift/poseidon_channel_lift_light-v1_20260623T015718Z.tar.gz`.

## Roadmap Implication

Treat the current in-house core as below the minimum learned-operator bar until
new evidence says otherwise. P2.2 now has a validation-positive challenger:
Poseidon `channel_lift` Option A cleared G2a on aggregate validation while
preserving the pretrained embedding/recovery path and using only 13 trainable
adapter parameters. This authorized drafting a held-out pre-test contract and
evidence manifest. The user later removed the approval blocker for the bounded
contract path on 2026-06-23, but the run must still follow the pre-registered
command, ledger guard, evidence manifest, and claim-language boundaries
exactly; it does not authorize broader public claims.

A no-provider branch check on 2026-06-22 concluded that source restore plus CPU
smoke remains the preferred path; DPOT is a fallback, and local decoded
refiner/transport-sidecar work should not be reopened merely because source
setup is blocked. See
`docs/research/2026-06-22-p2-source-blocker-branch-check.md`.

A source-restore and CPU-smoke runbook is recorded at
`docs/research/2026-06-22-p2-poseidon-source-restore-smoke-runbook.md`; it was
executed after user approval.

User approval on 2026-06-22 allowed the source restore and CPU smoke. Official
Poseidon source was restored to `/tmp/poseidon-official` at commit
`b8fa28f59bd7f7673323f28d11a12c6f3a215c61`; `scOT` imported successfully; the
2-sample validation-only `channel_lift` smoke wrote
`reports/research/sota_loop/external_baselines/poseidon_scot_channel_lift_smoke_val_light_v1/summary.json`
with `decoded_rollout_nrmse = 0.31116372295004086`, `held_out_test_used =
false`, `embedding_recovery_replaced = false`, and 13 trainable parameters.
That smoke unblocked the bounded P2.2 train/validation GPU measurement; no
held-out test was authorized.

The exact GPU validation plan is prepared at
`docs/research/2026-06-22-p2-channel-lift-gpu-validation-plan.md`. User
approval allowed bounded Vast execution on 2026-06-22/23. The valid final run
used `train` for fitting and `val` for selection, never used held-out test, and
wrote
`reports/research/sota_loop/external_baselines/poseidon_scot_channel_lift_val_light_v1_e30_lr1e2_roll4/summary.json`.
It passed the external-baseline validator with:
`decoded_rollout_nrmse = 0.35782889238675264`,
`task_advection1d_decoded_rollout_nrmse = 0.4937043430599529`,
`task_burgers1d_decoded_rollout_nrmse = 0.15674926288225416`,
`task_darcy2d_decoded_rollout_nrmse = 0.2071060212271272`,
`held_out_test_used = false`, `adapter_mode = channel_lift`,
`embedding_recovery_replaced = false`, source commit
`b8fa28f59bd7f7673323f28d11a12c6f3a215c61`, checkpoint SHA256
`e97428c93a16cbb52a41bc4794eb71be3aed436fb9cc547d9eeebb20f3940fb2`, and 13
trainable parameters. Advection/transport remains the weakest task family and
should be reviewed explicitly in the pre-test contract.

## Steward Starting Points

Start each steward tick from these files, then verify live git state:

- `docs/current-state.md`
- `docs/experiments/ledger.md`
- `README.md`
- `docs/public/README.md`
- `docs/claim_evidence/universal_sota_roadmap.md`
- `docs/superpowers/plans/2026-06-09-universal-simulator-north-star-roadmap.md`
- latest files under `docs/research/`

Use the global `roadmap-steward` skill as the operating guide. Keep recurring
work focused on one highest-signal safe move per tick, use lightweight branch
checks when one branch dominates too long, and update this file only when
project knowledge, active work, canonical paths, or stop conditions change.

## Active Work

| Owner | Objective | Scope | Stop condition | Next check |
| --- | --- | --- | --- | --- |
| `universal-simulator-roadmap-steward` | Keep the roadmap moving toward the north star | Repo docs, safe local analysis, bounded held-out contract execution | Contract drift, repeat held-out key, broader public claims, or strategic fork | Validate the Poseidon held-out pre-test contract, then run the exact registered command once |
