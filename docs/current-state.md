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
- DPOT readiness runner scaffolding is implemented in
  `scripts/run_external_dpot_finetune.py` with unit coverage in
  `tests/unit/test_external_dpot_finetune.py`; targeted tests passed on
  2026-06-23 with
  `python -m pytest tests/unit/test_external_dpot_finetune.py -q`. The runner
  blocks `split=test`, preserves validation-only claim boundaries, freezes the
  DPOT backbone, and exposes the 13-parameter `channel_lift` adapter needed
  for a later 2-sample CPU/import/checkpoint smoke.
- DPOT Tiny `channel_lift` CPU/import/checkpoint smoke passed on 2026-06-23:
  summary
  `reports/research/sota_loop/external_baselines/dpot_tiny_channel_lift_smoke_val_light_v1/summary.json`
  has aggregate decoded rollout NRMSE `0.4056234877403711`, advection
  `0.46377715332535735`, Burgers `0.31818835051545247`, Darcy
  `0.5280264566767598`, `held_out_test_used = false`, SHA-verified Tiny
  checkpoint
  `074c337f9b3a3c70253f8022ce6be7e7dfb809a91a7b00e46fbfedf9611d767f`,
  pinned DPOT source commit `dcd2f9a9359765e19ad63e2f3f879a2a8ce1aa17`,
  `history_steps = 10`, `history_init = repeat_current`, and 13 trainable
  adapter parameters. This is a mechanics smoke, not claim-comparable
  evidence.
- DPOT Tiny `channel_lift` validation-only GPU run completed on 2026-06-23
  and missed the gate: aggregate decoded rollout NRMSE
  `0.7136888249949349` versus gate `<= 0.363424243629033`, advection
  `0.8575561454613253` versus gate `<= 0.4866576789288726`, Burgers
  `0.588255711789389`, Darcy `0.28923145953251056`, `held_out_test_used =
  false`, 13 trainable parameters, source/checkpoint provenance matched. See
  `docs/research/2026-06-23-p2-dpot-gpu-validation-result.md` and B2 artifact
  `b2://pdebench/remote-runs/dpot-channel-lift/dpot_channel_lift_light-v1_20260623T221057Z.tar.gz`.
- Poseidon Option B task modulation is implemented in
  `scripts/run_external_poseidon_scot_finetune.py` with coverage in
  `tests/unit/test_external_poseidon_scot_finetune.py`. The adapter mode
  `channel_lift_task_modulated` keeps the frozen ScOT backbone and pretrained
  embedding/recovery intact, initializes task conditioning to identity, and
  trains 43 parameters: the 13-parameter base `channel_lift` plus per-task
  affine gain/bias before the backbone and after scalar readout. The 2-sample
  validation-only CPU smoke passed on 2026-06-23 with summary
  `reports/research/sota_loop/external_baselines/poseidon_scot_task_modulated_channel_lift_smoke_val_light_v1/summary.json`,
  aggregate decoded rollout NRMSE `0.3422384139670503`, advection
  `0.3623806561813393`, Burgers `0.22147624438612523`, Darcy
  `0.7516511659226303`, `held_out_test_used = false`, source commit
  `b8fa28f59bd7f7673323f28d11a12c6f3a215c61`, and checkpoint SHA256
  `e97428c93a16cbb52a41bc4794eb71be3aed436fb9cc547d9eeebb20f3940fb2`.
  This is a mechanics smoke, not a gate or claim-comparable result.
- Poseidon Option B task modulation validation-only GPU run completed on
  2026-06-23 and produced a mixed result: aggregate decoded rollout NRMSE
  `0.3566052737393018` cleared G2a `<= 0.363424243629033`, Burgers
  `0.14460934384484475` and Darcy `0.18262014873452226` improved versus
  Option A validation, but advection `0.4967493071208899` missed the strict
  transport gate `<= 0.4866576789288726` and slightly regressed from Option A
  validation `0.4937043430599529`. No held-out test was used; trainable
  parameters were `43`; source/checkpoint provenance matched. See
  `docs/research/2026-06-23-p2-poseidon-option-b-gpu-validation-result.md`
  and B2 artifact
  `b2://pdebench/remote-runs/poseidon-channel-lift/poseidon_channel_lift_light-v1_20260623T235710Z.tar.gz`.

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

The held-out pre-test contract was then executed once under measurement key
`b487e8841f7631554248fcaeedf9dd3a1fba1faa7f003f0e6304a2b96375516a`.
It produced negative/mixed transfer: held-out aggregate decoded rollout NRMSE
`0.5551415687535287`, with advection `0.7840223655431167`, Burgers
`0.18316455707528173`, and Darcy `0.21459086990463278`. The run used held-out
test, recorded the ledger once, preserved `channel_lift`, 13 trainable
parameters, the pinned Poseidon source commit, and the checkpoint SHA256. See
`docs/research/2026-06-23-p2-poseidon-channel-lift-heldout-result.md` and
`docs/research/2026-06-23-p2-poseidon-channel-lift-heldout-evidence-manifest.json`.
Do not promote Poseidon `channel_lift` Option A or rerun this held-out key.

A post-held-out branch check on 2026-06-23 compared DPOT, transport-aware
Poseidon follow-ups, and local transport-sidecar lessons. Decision: make DPOT
readiness and a 2-sample validation smoke design the primary next branch; keep
Poseidon Option B/task modulation secondary and only under an advection-aware
validation gate; do not reopen standalone transport sidecar or shift-estimator
work. See
`docs/research/2026-06-23-p2-post-heldout-branch-check.md`.

The first DPOT implementation step is complete: `scripts/run_external_dpot_finetune.py`
supports the pinned Tiny DPOT source/checkpoint contract, deterministic
repeat-current 10-frame history, scalar-to-four-channel lift/readout, frozen
backbone parameter discipline, validation-only summary validation, and
autoregressive rollout evaluation. The local CPU smoke has now passed; no
GPU/provider run, held-out test, claim evidence update, or public language
change was performed. The next useful step is the bounded validation-only DPOT
GPU plan in `docs/research/2026-06-23-p2-dpot-gpu-validation-plan.md` if
provider work is allowed. The gate must protect aggregate, advection, Darcy,
and Burgers validation metrics; the CPU smoke suggests Darcy can be weak under
this adapter even when advection does not immediately collapse.

The DPOT Tiny validation-only GPU run is now negative. Do not rerun it, do not
move it to held-out, and do not update claim evidence. The result supports a
branch check before any more provider spend. That branch check is recorded at
`docs/research/2026-06-23-p2-post-dpot-validation-branch-check.md`: primary
next branch is Poseidon Option B/task modulation, validation-only; DPOT
escalation is secondary; UPS-side transport/refiner work remains fallback.

Poseidon Option B task modulation now has runner scaffolding, focused unit
coverage, a successful 2-sample validation-only CPU smoke, and a completed
validation-only GPU run. The GPU run is mixed: aggregate, Burgers, and Darcy
are positive versus the gate, but advection/transport misses the strict gate.
Do not move Option B to held-out or claim evidence. The next useful step is a
no-provider branch check focused on transport mechanism, not another aggregate
validation run.

The post-Option-B transport branch check is recorded at
`docs/research/2026-06-23-p2-post-option-b-transport-branch-check.md`.
Decision: primary next branch is UPS-side transport mechanism design/probe,
validation-only and no-provider. Poseidon remains secondary only if the next
design names a concrete phase/displacement/temporal mechanism. DPOT escalation
is blocked until a no-provider temporal-interface hypothesis exists. No held-out
pretest or claim-evidence update is authorized.

Project-specific steward lessons are tracked in
`docs/steward/self-improvement.md`. Current lessons: future provider plans
must include explicit transport/advection gates, adapter-capacity escalation
must name the transport mechanism it is expected to fix, and Vast teardown must
be verified manually because auto-shutdown can fail after successful wrapper
exit.

The first no-provider transport mechanism design/probe is recorded at
`docs/research/2026-06-24-p2-transport-mechanism-design-probe.md`. It selects
parameter-conditioned causal transport shift/displacement as the primary next
mechanism because existing canonical-root validation evidence already clears
the active phase gate by a wide margin: aggregate `0.11122069865007121`,
advection rollout `0.0017868130908052495`, advection h16
`0.0017842800879688658`, Burgers `0.14738121412908425`, and Darcy
`0.188979512124482`, with no held-out test. The result remains scoped because
it requires beta provenance for advection; it does not authorize claim-language
changes or held-out access. The next useful step is a no-provider model-side
parameter-conditioned transport-head design.

The model-side parameter-conditioned transport-head design is recorded at
`docs/research/2026-06-24-p2-model-side-transport-head-design.md`. It selects a
default-off decoder-side linear beta/horizon/bias periodic-displacement head as
the smallest useful model-side bridge from the validated beta sidecar. The
design requires beta provenance for non-smoke advection candidates, skips the
head when beta is absent, emits resolved config plus shift statistics, and gates
aggregate, advection rollout, advection h16, Burgers, Darcy, and held-out flags.
It does not authorize provider work, held-out access, claim-evidence updates, or
public-language changes. The next safe move is a CPU-only implementation slice
with synthetic tests and summary/validator plumbing.

The first CPU-only model-side beta transport-head scaffold is implemented in
`src/ups/models/transport_head.py` and wired into decoded evaluation in
`src/ups/eval/pdebench_runner.py`, with tests in
`tests/unit/test_model_side_transport_head.py`,
`tests/unit/test_pdebench_runner_eval.py`, and
`tests/unit/test_validate_model_side_transport_head_summary.py`. The scaffold is
default-off, scoped to `advection1d`, predicts a linear
`param:beta`/`horizon_norm`/`bias` periodic displacement, skips when required
beta metadata is absent, and emits resolved config plus shift counts in
`MetricReport.extra`. The validator stub at
`scripts/validate_model_side_transport_head_summary.py` rejects held-out use,
missing resolved config, broad task scope, missing beta, incompatible roll-shift
estimators, and failed aggregate/advection/advection-h16/Burgers/Darcy gates.
This is mechanics evidence only, not a validation result. The next safe move is
a no-provider synthetic or tiny real-shard smoke plan that proves the summary
schema end to end before any provider/GPU run.

The no-provider end-to-end synthetic smoke/schema exercise for the model-side
beta transport head is implemented in
`scripts/run_model_side_transport_head_smoke.py` with test coverage in
`tests/unit/test_run_model_side_transport_head_smoke.py`. It wrote the local
ignored summary
`reports/research/sota_loop/model_side_transport_head_smoke_val_light_v1/summary.json`
and `scripts/validate_model_side_transport_head_summary.py` returned
`passed = true` with no errors. Synthetic smoke metrics were aggregate `0.0`,
advection rollout `0.0`, advection h16 `0.0`, Burgers `0.0`, Darcy `0.0`,
mean model-side shift `1.0`, 16 applied shifts, zero skipped shifts, zero
missing-beta count, and held-out flags false. This proves summary/validator
plumbing only; it is not validation evidence. The next safe move is a
no-provider real-shard validation plan for the model-side beta transport head,
with no GPU/provider run until that plan names data root, command, artifact
schema, gates, and stop conditions.

The no-provider real-shard validation plan is recorded at
`docs/research/2026-06-24-p2-model-side-transport-head-real-shard-validation-plan.md`.
It found that the current checkout has standard `data/pdebench` train/val
shards, but the standard advection shards lack `source_file_index` and
`source_paths`, so `params.beta` cannot be derived and the model-side head would
skip advection. The prior beta-provenance root
`data/pdebench_official_advection_light`, generated canonical validation root,
and decoded checkpoint source
`reports/research/sota_loop/learned_capacity_gate/ups_light_local_joint_rollout4_residual_ft_val`
are not present locally. The plan defines the full-task root build, CPU
validation command, summary validator call, gates, and stop conditions, but does
not authorize hydration/download, provider work, held-out access, claim-evidence
updates, or public-language changes. The next safe move is to restore or hydrate
the missing beta root/checkpoint under a bounded no-held-out plan, then run the
CPU validation command.

User approval for the bounded hydration/restoration action is now present, but
local disk is insufficient. The live preflight found only about `5.7 GiB` free,
while even sequential official Advection hydration needs about `9.47 GiB` for
one raw train file plus safety margin; full selected official train-file
hydration is about `61.34 GiB`. The no-local-storage remote route is recorded at
`docs/research/2026-06-24-p2-model-side-transport-head-remote-no-local-storage-plan.md`
with wrappers `scripts/run_remote_model_side_transport_head_real_shard.sh` and
`scripts/launch_remote_model_side_transport_head_vast.sh`. It uses remote
scratch, sequential official Advection hydrate-convert-delete, standard
`light-v1` validation shards from B2 for Burgers/Darcy, the small checkpoint
archive `remote-runs/checkpoints/ups_light_task_signature_trained_residual_20260526T1928Z.tar.gz`,
and the model-side summary validator. This route still forbids held-out data,
claim-evidence updates, public-language changes, and evaluator roll-shift
sidecars.

The DPOT readiness note is now recorded at
`docs/research/2026-06-23-p2-dpot-readiness-smoke-design.md`. It pins the live
source candidate to `HaoZhongkai/DPOT` main
`dcd2f9a9359765e19ad63e2f3f879a2a8ce1aa17`, the Tiny checkpoint to
`hzk17/DPOT` repo SHA `2adec1cf9a55942f1456aa7463cd7ade908398d0` /
`model_Ti.pth` SHA256
`074c337f9b3a3c70253f8022ce6be7e7dfb809a91a7b00e46fbfedf9611d767f`, and a
no-held-out 2-sample CPU/import smoke target. No DPOT source clone,
checkpoint download, GPU work, or held-out test has been run.

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
| `universal-simulator-roadmap-steward` | Keep the roadmap moving toward the north star | Repo docs, safe local analysis, validation-only challenger design | Held-out repeat request, broader public claims, unknown billing/top-up, teardown failure, or strategic fork | Dry-run and then execute the remote no-local-storage model-side real-shard validation route if Vast/B2 readiness passes |
